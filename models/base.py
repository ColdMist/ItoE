"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import nn


class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        gamma: torch.nn.Parameter for margin in ranking-based loss
        data_type: torch.dtype for machine precision (single or double)
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: torch.nn.Embedding with entity embeddings
        rel: torch.nn.Embedding with relation embeddings
        bh: torch.nn.Embedding with head entity bias embeddings
        bt: torch.nn.Embedding with tail entity bias embeddings
    """

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size, args):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        #self.dataset_type = args.dataset_type
        if data_type == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)
        self.rel = nn.Embedding(sizes[1], rank)

        # if self.dataset_type == 'quintuple_Loc':
        #     self.loc_emb = nn.Embedding(sizes[3], rank)
        #     self.tim_emb = nn.Embedding(sizes[4], rank)
        #
        #     self.bh = nn.Embedding(sizes[3], 1)
        #     self.bh.weight.data = torch.zeros((sizes[3], 1), dtype=self.data_type)
        #     self.bt = nn.Embedding(sizes[3], 1)
        #     self.bt.weight.data = torch.zeros((sizes[3], 1), dtype=self.data_type)
        # elif self.dataset_type == 'quintuple_Tim':
        #
        #       self.loc_emb = nn.Embedding(sizes[3], rank)
        #       self.tim_emb = nn.Embedding(sizes[4], rank)
        #
        #       self.bh = nn.Embedding(sizes[4], 1)
        #       self.bh.weight.data = torch.zeros((sizes[4], 1), dtype=self.data_type)
        #
        #       self.bt = nn.Embedding(sizes[4], 1)
        #       self.bt.weight.data = torch.zeros((sizes[4], 1), dtype=self.data_type)
        # elif self.dataset_type == 'quintuple':
        #
        #       self.loc_emb = nn.Embedding(sizes[3], rank)
        #       self.tim_emb = nn.Embedding(sizes[4], rank)
        #
        #       self.bh = nn.Embedding(sizes[2], 1)
        #       self.bh.weight.data = torch.zeros((sizes[2], 1), dtype=self.data_type)
        #
        #       self.bt = nn.Embedding(sizes[2], 1)
        #       self.bt.weight.data = torch.zeros((sizes[2], 1), dtype=self.data_type)

        #else:
        self.bh = nn.Embedding(sizes[2], 1)
        self.bh.weight.data = torch.zeros((sizes[2], 1), dtype=self.data_type)
        self.bt = nn.Embedding(sizes[2], 1)
        self.bt.weight.data = torch.zeros((sizes[2], 1), dtype=self.data_type)
        
        # self.cuda1 = torch.device('cuda:'+str(args.cuda_n))
        #
        #
        # self.storageTriple = torch.tensor([]).cuda(self.cuda1).to(torch.long)
        # self.storageEntId = torch.tensor([]).cuda(self.cuda1).to(torch.long)
        # self.storageallscores = torch.tensor([]).cuda(self.cuda1)
        # self.storagecurrentcores = torch.tensor([]).cuda(self.cuda1)
        # self.storagerank = torch.tensor([])
        #self.data_type = args.data_type

    @abstractmethod
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
             rhs_e: torch.Tensor with targets' embeddings
                    if eval_mode=False returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: torch.Tensor with targets' biases
                         if eval_mode=False returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass

    def score(self, lhs, rhs, eval_mode):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])                
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        # get factors for regularization
        factors = self.get_factors(queries)
        #print(len(factors))
        #exit()
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries), dtype=self.data_type)
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)           
            
            j = 0
            while b_begin < len(queries):
                j = j + 1
                these_queries = queries[b_begin:b_begin + batch_size].cuda()
                #print(these_queries.size())
                #exit()
                q = self.get_queries(these_queries)
                #print(these_queries.size())
                #exit()
                rhs = self.get_rhs(these_queries, eval_mode=False)
                #print(rhs)
                #exit()

                scores = self.score(q, candidates, eval_mode=True)
                #print(scores.size())
                
                #print(.size())               
                
                targets = self.score(q, rhs, eval_mode=False)
                #print(targets.size())

                #print(queries)
                #print(these_queries)
                #exit()                
                

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):                   
                    #print(query)
                    #print(i)
                    #exit()                                                          
                    #TODO commented
                    # if query.size()[0] == 5:
                    #    if self.dataset_type == 'quintuple_Tim':
                    #
                    #       filter_out = []#filters[(query[0].item(), query[1].item(), query[2].item(), query[3].item())]
                    #    elif self.dataset_type == 'quintuple_Loc':
                    #       filter_out = []#filters[(query[0].item(), query[1].item(), query[2].item(), query[4].item())]
                    #    else:
                    #       filter_out = filters[(query[0].item(), query[1].item(), query[3].item(), query[4].item())]
                    #       filter_out += [queries[b_begin + i, 2].item()]
                    #       #print(query[2].item())
                    #       #exit()
                    #       if query[0].item() != query[2].item():
                    #          #this line filters hrh and trt
                    #          filter_out += [queries[b_begin + i, 0].item()]
                    #
                    #       scores[i, torch.LongTensor(filter_out)] = -1e6

                    #else:
                    filter_out = filters[(query[0].item(), query[1].item())]
                       #print(filter_out)
                       #exit()
                    filter_out += [queries[b_begin + i, 2].item()]
                       #print(query[2].item())
                       #exit()
                    if query[0].item() != query[2].item():
                          #this line filters hrh and trt
                        filter_out += [queries[b_begin + i, 0].item()]
                   
                    scores[i, torch.LongTensor(filter_out)] = -1e6
             
                #np.savetxt('my_file.txt', self.storageallscores.cpu().numpy())
                #print(self.storageallscores.size())
                #exit()
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).type(self.data_type), dim=1
                ).cpu()
                b_begin += batch_size

                if j < 0:
                   self.storageTriple = torch.cat((self.storageTriple, these_queries), 0)
                   self.storageallscores = torch.cat((self.storageallscores, scores), 0)
                   self.storagecurrentcores = torch.cat((self.storagecurrentcores,targets),0)
                   self.storagerank = torch.cat((self.storagerank,ranks),0)

        #np.savetxt('my_file.txt', self.storageallscores.cpu().numpy())
        ##i=0

        ##a, indx = torch.topk(self.storageallscores, 10, dim = 1)
        #oo = these_queries[indx]
        ##indx = indx.cpu().tolist()
        ##a = a.cpu().tolist()
        #print(indx.size())
        #print(a.size())
        #exit()
        ##b = self.storageTriple.cpu().tolist()
        ##c = self.storagecurrentcores.cpu().tolist()
        ##d = self.storagerank.cpu().tolist()
        ##with open('score.txt', 'w') as filehandle:
        ##    for listitem in a:
        ##        ids = indx[i]
        ##        listtriple = b[i]
        ##        listitems = set(listitem)
        ##        listcurrent = c[i]
        ##        listrank = d[i]
        ##        num = len(listitem) - len(listitems)
        ##        ##print(num)
        ##        i = i + 1
        ##        ##print(i)
        ##        ##filehandle.write('%s\n'% num)
        ##        ##filehandle.write('%s\n\n\n'% listtriple)
        ##        ##filehandle.write('%s\n'% listcurrent)
        ##        ##filehandle.write('rank is: \t %s\n\n\n'% listrank)
        ##        ##filehandle.write('TopK Scores:\t %s\n\n\n\t'% listitem)
                ##filehandle.write('TopK index:\t %s\n\n\n'% ids)
                #filehandle.write('%s\n\n\n'% listitems)
        #exit()
        return ranks

    def compute_metrics(self, examples, filters, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}
        #print(examples.size()[1])
        #exit()

        for m in ["rhs", "lhs"]:
            q = examples.clone()            
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
            #print(ranks.size())
            #exit()
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at
