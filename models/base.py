from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.grape import GNNModel as grape_model
from models.nbfnet import GNNModel as nbfnet_model
from models.redgnn import GNNModel as redgnn_model
from scipy.sparse import csr_matrix
from util import *

class SubGraph(object):
    def __init__(self, args, graph_triples):
        self.args = args
        self.n_rel = args.n_rel
        self.n_triples = len(graph_triples)
        graph_triples = graph_triples.cuda()
        sampled_ents = torch.concat([graph_triples[:, 0], graph_triples[:, 2]], dim=0)
        tail_nodes, all_index = torch.unique(sampled_ents, dim=0, sorted=True, return_inverse=True)
        head_index, tail_index = all_index[:self.n_triples], all_index[self.n_triples:]
        
        self.KG = torch.stack([head_index, graph_triples[:, 1], tail_index], dim=1).cpu().numpy()
        self.Ent = tail_nodes.cpu().numpy()
        self.n_fact = len(self.KG)
        self.n_ent = len(self.Ent)
        self.t2s_dict = {tid: sid for sid, tid in enumerate(tail_nodes.cpu().numpy().tolist())}
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:, 0])),
                                 shape=(self.n_fact, self.n_ent))

    def encode(self, batch_triples):
        recode_head = [self.t2s_dict[item] for item in batch_triples[:, 0]]
        recode_tail = [self.t2s_dict[item] if item in self.Ent else self.n_ent for item in batch_triples[:, 2]]
        recode_triples = np.stack([recode_head, batch_triples[:, 1], recode_tail], axis=1)
        return recode_triples

class BaseModel(object):
    def __init__(self, args):
        self.model_name = args.model_name
        self.model = globals()[args.model_name+"_model"](args)
        self.model.cuda()
        self.args = args
        self.n_layer = args.n_layer
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1, verbose=True, min_lr=1e-5)
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()

    def train_batch(self, batch_triples, subgraph, epoch=0):
        batch_size = len(batch_triples)
        self.model.zero_grad()
        scores = self.model.forward(batch_triples, subgraph, epoch=epoch)

        scores = torch.concat([scores, torch.ones_like(scores[:, :1]).cuda()*-1e5], dim=1)
        pos_scores = scores[torch.arange(batch_size).cuda(), torch.LongTensor(batch_triples[:,2]).cuda()].clone()

        rand_negs = torch.randint(0, len(scores[0]), size=[len(scores), int(len(scores[0]) * 0.01)]).cuda()
        neg_scores = torch.gather(scores, 1, rand_negs)
        logit = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        label = torch.zeros_like(logit)
        label[:, 0] = 1
        loss = self.BCE_loss(input=logit, target=label)
        
        if loss.item() == torch.inf:
            print(pos_scores, loss, scores.shape, scores.max(), scores.min())

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss_val = loss.item()
        for p in self.model.parameters():
            X = p.data.clone()
            flag = X != X
            X[flag] = np.random.random()
            p.data.copy_(X)
        del(loss, scores)
        return loss_val

    def eval_batch(self, batch_triples, subgraph, batch_labcsr, filters, total_n_ent, num_negative=-1, epoch=0):
        batch_size = len(batch_triples)
        scores = self.model.forward(batch_triples, subgraph, epoch=epoch)
        scores = scores.data.cpu().numpy()
        
        scores = scores + abs(scores.min()) + 1
        value_list, bid_list, obj_list = [], [], []
        for i in range(batch_size):
            single_obj = scores[i]
            value_list.extend(single_obj)
            obj_list.extend(subgraph.Ent.tolist())
            bid_list.extend([i] * len(single_obj))
        assert len(obj_list) == len(bid_list)
        scores_csr = csr_matrix((np.array(value_list,), (np.array(bid_list), np.array(obj_list))),
                          shape=(batch_size, total_n_ent), dtype=np.float32)

        scores_csr = scores_csr.tolil()
        batch_labcsr = batch_labcsr.tolil()
        rows, cols = batch_labcsr.nonzero()
        target_preds = scores_csr[rows, cols].toarray().flatten()
        target_batchs = rows

        subs, rels = batch_triples[:, 2], batch_triples[:, 1]
        target_rels = rels[target_batchs]
            
        row_indices, col_indices = [], []
        for i in range(len(subs)):
            filt = filters[(subs[i], rels[i])]
            col_indices.extend(filt)
            row_indices.extend([i] * len(filt))
        scores_csr[np.array(row_indices), np.array(col_indices)] = 0

        scores_csr = scores_csr.tocsr()
        sorted_data = []
        cumulative_counts = []
        for i in range(scores_csr.shape[0]):
            row_data = np.sort(scores_csr.data[scores_csr.indptr[i]:scores_csr.indptr[i + 1]])
            sorted_data.append(row_data)
            cumulative_counts.append(np.searchsorted(row_data, row_data, side='right'))

        ranks = np.zeros(target_preds.shape[0], dtype=int)
        for i in range(len(target_preds)):
            index = target_batchs[i]
            value = target_preds[i]
            if value == 0:
                rank = total_n_ent
            else:
                rank = cumulative_counts[int(index)][-1] - np.searchsorted(sorted_data[int(index)], value, side='right') + 1
            ranks[i] = rank
        ranks = ranks.tolist()
        return ranks, target_rels

    def remove_nan_params(self):
        for p in self.model.parameters():
            X = p.data.clone()
            flag = X != X
            X[flag] = np.random.random()
            p.data.copy_(X)

    def build_subgraph(self):
        return