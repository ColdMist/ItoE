"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, expmapP, expmapH, projectH, busemann_distance

HYP_MODELS = ["RotH", "RefH", "AttH", "SDP", "SDH"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size, args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        self.sim = "dist"
        #self.c = []
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return  self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        if self.sim == "dot":
            return busemann_distance(lhs_e, rhs_e, c, eval_mode)
        else:
            return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode)**2
      
        #return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2


#TODO SDH and SDP both should be kept
class SDP(BaseH):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(SDP, self).__init__(args)
        self.args = args
        torch.set_default_dtype(self.data_type)
        self.hidrank = 150
        self.hidden_embedding_drift = nn.Sequential(
                        nn.Linear(self.rank, self.hidrank),
                        nn.ReLU())
                        # nn.Linear(self.hidrank,self.hidrank),
                        # nn.Tanh())

        self.hidden_embedding_diff = nn.Sequential(
                        nn.Linear(self.rank, self.hidrank),
                        nn.ReLU())
                        # nn.Linear(self.hidrank,self.hidrank),
                        # nn.Tanh())

        self.rel_emb_drift = nn.Parameter(2*torch.rand(self.sizes[1], self.hidrank * self.rank) - 1.0)
        #print(self.rel_emb_drift.dtype)
        #exit()
        #self.rel_emb.weight.data = 2 * torch.rand((self.sizes[1], self.rank*self.hidrank), dtype=self.data_type) - 1.0
        self.hidden_embedding_drift.apply(self.weights_init)
        self.hidden_embedding_diff.apply(self.weights_init)

        self.sim = "dist"

    def get_queries(self, queries):
        #print('hello')
        c = F.softplus(self.c[queries[:, 1]])
        head_e = self.entity(queries[:, 0])
        head_e = project(head_e, c)

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
        relation_diff = torch.cuda.FloatTensor(relation_drift.size()).normal_(0, 1).cuda(self.args.cuda_n).to(self.data_type)
        #print(relation_diff.dtype)
        #exit()
        relationh_drift = torch.einsum('ijk,ik->ij', [relation_drift, headHidden_drift])#.squeeze(2)
        relationh1_drift = torch.tanh(relationh_drift)

        relationh_diff = torch.einsum('ijk,ik->ij', [relation_diff, headHidden_diff])#.squeeze(2)
        relationh1_diff = torch.tanh(relationh_diff)
      
        lhs_e = expmapP(head_e, relationh1_drift + relationh1_diff, c)
        #lhs_e = head_e + relationh1_drift + relationh1_diff
        lhs_biases = self.bh(queries[:, 0])
        return (lhs_e, c), lhs_biases

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
           torch.nn.init.xavier_uniform_(m.weight)
           torch.nn.init.zeros_(m.bias)

class SDH(BaseH):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(SDH, self).__init__(args)
        self.args = args
        torch.set_default_dtype(self.data_type)
        self.hidrank = 300
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
        c = F.softplus(self.c[queries[:, 1]])
        head_e = self.entity(queries[:, 0])
        head_e = projectH(head_e, c)

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

        lhs_e = expmapH(head_e, relationh1_drift + relationh1_diff, c)
        #lhs_e = head_e + relationh1_drift + relationh1_diff
        lhs_biases = self.bh(queries[:, 0])
        return (lhs_e, c), lhs_biases

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
           torch.nn.init.xavier_uniform_(m.weight)
           torch.nn.init.zeros_(m.bias)

class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        #self.c.weight = c
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank).cuda(args.cuda_n)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda(args.cuda_n)
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda(args.cuda_n)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])
