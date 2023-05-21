"""Euclidean ave(model.entity, 'entity.pkl')cowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection, givens_DE_rotations, givens_QuatE_rotations

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE", "SDE", "QuatE", "DistMult"]


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size, args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.args = args
        self.model_name = args.model
        #if self.args
        #self.loc_emb = nn.Embedding(self.sizes[3], self.rank)
        #self.loc_emb.weight.data = 2 * torch.rand((self.sizes[3], self.rank), dtype=self.data_type) - 1.0

        #self.tim_emb = nn.Embedding(self.sizes[4], self.rank)
        #self.tim_emb.weight.data = 2 * torch.rand((self.sizes[4], self.rank), dtype=self.data_type) - 1.0

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    # def get_rhs(self, queries, eval_mode):
    #     """Get embeddings and biases of target entities."""
    #     #print(self.model_name)
    #     #exit()
    #     if self.model_name == 'QuinTransELoc':
    #         if eval_mode:
    #             return -self.loc_emb.weight, self.bt.weight
    #         else:
    #             return -self.loc_emb(queries[:, 3]), self.bt(queries[:, 3])
    #     elif self.model_name == 'QuinTransETim':
    #           if eval_mode:
    #               return -self.tim_emb.weight, self.bt.weight
    #           else:
    #               #print(self.bt(queries[:, 4]))
    #               #exit()
    #               return -self.tim_emb(queries[:, 4]), self.bt(queries[:, 4])
    #     else:
    #         if eval_mode:
    #             return self.embeddings[0].weight, self.bt.weight
    #         else:
    #             return self.embeddings[0](queries[:, 2]), self.bt(queries[:, 2])


    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):        
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class DistMult(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(DistMult, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e * rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class QuatE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(QuatE, self).__init__(args)
        #self.rel_rot = nn.Embedding(self.sizes[1], self.rank)
        #self.rel_rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dot"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        #rel_rot_e = self.rel_rot(queries[:, 1])
        rank = self.rank//2

        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]
        #rel_e = rel_e[:, :rank//2], rel_e[:, rank//2:rank], rel_e[:, rank:3*rank//2], rel_e[:, 3*rank//2:]
        rel_e = rel_e[:, :rank//2], rel_e[:, rank//2:rank], rel_e[:, rank:3*rank//2], rel_e[:, 3*rank//2:]

        A, B, C, D = givens_QuatE_rotations(rel_e, head_e)

        E = torch.cat((A, B), 1)
        F = torch.cat((E, C), 1)
        lhs_e = torch.cat((F, D), 1)
        
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class SDE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(SDE, self).__init__(args)
        self.args = args
        torch.set_default_dtype(self.data_type)
        self.hidrank = 3
        self.hidden_embedding_drift = nn.Sequential(
                        nn.Linear(self.rank, self.hidrank),
                        nn.Tanh(),
                        nn.Linear(self.hidrank,self.hidrank),
                        nn.Tanh())

        self.hidden_embedding_diff = nn.Sequential(
                        nn.Linear(self.rank, self.hidrank),
                        nn.Tanh(),
                        nn.Linear(self.hidrank,self.hidrank),
                        nn.Tanh())

        self.rel_emb_drift = nn.Parameter(2*torch.rand(self.sizes[1], self.hidrank * self.rank) - 1.0)
        
        #self.rel_emb.weight.data = 2 * torch.rand((self.sizes[1], self.rank*self.hidrank), dtype=self.data_type) - 1.0
        self.hidden_embedding_drift.apply(self.weights_init)
        self.hidden_embedding_diff.apply(self.weights_init)

        self.sim = "dist"

    def get_queries(self, queries):
        #print('hello')
        head_e = self.entity(queries[:, 0])
        #print(head_e.type())

        rel_e_drift = torch.index_select(
                self.rel_emb_drift,
                dim=0,
                index=queries[:, 1]
            ).cuda(self.args.cuda_n)#.unsqueeze(1)


        #rel_e = self.rel_emb(queries[:, 1])
        headHidden_drift = self.hidden_embedding_drift(head_e).cuda(self.args.cuda_n)
        headHidden_diff = self.hidden_embedding_diff(head_e).cuda(self.args.cuda_n)

        relation_drift = rel_e_drift.view(rel_e_drift.size()[0], self.rank, self.hidrank)
        relation_diff = torch.cuda.FloatTensor(relation_drift.size()).normal_(0, 1).cuda(self.args.cuda_n)

        relationh_drift = torch.einsum('ijk,ik->ij', [relation_drift, headHidden_drift])#.squeeze(2)
        relationh1_drift = torch.tanh(relationh_drift)

        relationh_diff = torch.einsum('ijk,ik->ij', [relation_diff, headHidden_diff])#.squeeze(2)
        relationh1_diff = torch.tanh(relationh_diff)

        lhs_e = head_e + relationh1_drift + relationh1_diff
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
           torch.nn.init.xavier_uniform_(m.weight)
           torch.nn.init.zeros_(m.bias)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
           torch.nn.init.xavier_uniform_(m.weight)
           torch.nn.init.zeros_(m.bias)


class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])


class MurE(BaseE):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args):
        super(MurE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RefE(BaseE):
    """Euclidean 2x2 Givens reflections"""

    def __init__(self, args):
        super(RefE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        rel = self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs + rel, lhs_biases


class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)
        self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)
        self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, self.bh(queries[:, 0])
