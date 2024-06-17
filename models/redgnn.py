import torch
import torch.nn as nn
import numpy as np
from util import *
from functools import reduce
from torch_scatter import scatter
from scipy.sparse import csr_matrix

class GNNLayer(torch.nn.Module):
    def __init__(self, params, hidden_dim):
        super(GNNLayer, self).__init__()
        self.params = params
        self.n_rel = params.n_rel
        self.attn_dim = params.attn_dim
        self.in_dim = hidden_dim
        self.out_dim = hidden_dim
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        self.act = acts[params.act]
        self.MESS_FUNC = params.MESS_FUNC.replace("\r", "")
        self.AGG_FUNC = params.AGG_FUNC.replace("\r", "")

        self.rela_embed = nn.Embedding(2 * self.n_rel + 1, self.in_dim)

        self.Ws_attn = nn.Linear(self.in_dim, self.attn_dim, bias=False)
        self.Wr_attn = nn.Linear(self.in_dim, self.attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(self.in_dim, self.attn_dim)
        self.w_alpha = nn.Linear(self.attn_dim, 1)

        self.W_h = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    # forward for part edges, with flatten data
    def forward(self, q_rel, layer_input, edges, nodes, n_ent):
        sub, rel, obj = edges[:, 4], edges[:, 2], edges[:, 5]

        hs = layer_input[sub]
        hr = self.rela_embed(rel)
        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=len(nodes), reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))
        return hidden_new

    def pna_process(self, message, obj_index, item_size, edge_count, scatter_dim=0):
        count_sum = edge_count.clamp(min=1)
        count_sum2 = scatter(torch.ones(list(message.shape)[:-1] + [1]).cuda(), index=obj_index, dim=scatter_dim,
                       dim_size=item_size, reduce="sum").clamp(min=1)

        sum = scatter(message, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="sum")
        mean = sum / count_sum
        sq_sum = scatter(message ** 2, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="sum")
        sq_mean = sq_sum / count_sum

        std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
        std = std * (mean != 0)
        features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        features = features.flatten(-2)

        scale = count_sum2.log()
        scale = scale / (scale.mean().clamp(min=1))

        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        message_agg = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        message_agg = message_agg.squeeze(0)
        return message_agg

class GNNModel(torch.nn.Module):
    def __init__(self, params):
        super(GNNModel, self).__init__()
        print("Loading RED-GNN Model...")
        self.params = params

        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel

        self.remove_one_loop = params.remove_one_loop
        self.rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

        self.gnn_layers = []
        for i in range(self.n_layer):  # 1
            self.gnn_layers.append(GNNLayer(self.params, hidden_dim=self.hidden_dim))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        # calculate parameters
        num_parameters = sum(p.numel() for p in self.parameters())
        # for name, param in self.state_dict().items():
        #     print(name, param.shape)
        print('==> num_parameters: {}'.format(num_parameters))

    def forward(self, triples, subgraph, mode='train', epoch=0):
        n = len(triples)
        n_ent = subgraph.n_ent
        edge_data = torch.from_numpy(subgraph.KG).cuda()
        batch_triples = torch.cat([torch.arange(n).unsqueeze(1).cuda(),
                                   torch.LongTensor(triples).cuda()], 1)  # [bid, h, r, t, fid]
        q_sub, q_rel, q_obj = batch_triples[:, 1], batch_triples[:, 2], batch_triples[:, 3]  # [B]
        start_nodes = batch_triples[:, [0, 1]]  # [B, 2] with (batch_idx, node_idx)

        if mode == 'train':
            h_index_ext = torch.cat([q_sub, q_obj], dim=-1)
            t_index_ext = torch.cat([q_obj, q_sub], dim=-1)
            r_index_ext = torch.cat([q_rel, torch.where(q_rel < self.n_rel, q_rel + self.n_rel, q_rel - self.n_rel)], dim=-1)
            extend_triples = torch.stack([h_index_ext, r_index_ext, t_index_ext], dim=0)
            if self.remove_one_loop:
                filter_index = edge_match(edge_data.T[[0, 2], :], extend_triples[[0, 2], :])[0]  # for FB
            else:
                filter_index = edge_match(edge_data.T, extend_triples)[0]
            filter_mask = ~index_to_mask(filter_index, len(edge_data))
            filter_mask = filter_mask | (edge_data[:,1]==(self.n_rel*2))
        else:
            filter_mask = torch.ones(len(edge_data)).cuda().bool()
        del edge_data
        torch.cuda.empty_cache()

        query = self.rela_embed(q_rel)
        nodes = start_nodes

        layer_input = torch.zeros_like(query)
        total_node_1hot = 0
        for i in range(self.n_layer):
            nodes, edges, edge_1hot, total_node_1hot, old_nodes_new_idx = self.get_neighbors(nodes.data.cpu().numpy(),
                                                                     len(start_nodes), total_node_1hot, subgraph,
                                                                     filter_mask=filter_mask.unsqueeze(1).cpu().numpy(),
                                                                     layer_id=i)
            hidden = self.gnn_layers[i].forward(q_rel, layer_input, edges, nodes, n_ent)
            hidden = self.dropout(hidden)
            previous_mes = torch.zeros_like(hidden)
            previous_mes[old_nodes_new_idx] += layer_input
            hidden, layer_input = self.gate(hidden.unsqueeze(0), previous_mes.unsqueeze(0))
            layer_input = layer_input.squeeze(0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        #scores_all = torch.zeros((n, n_ent)).cuda()
        scores_all = torch.ones((n, n_ent)).cuda() * -1e5
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all

    def get_neighbors(self, nodes, batchsize, total_node_1hot=0, subgraph=None, filter_mask=None, layer_id=0):
        KG = subgraph.KG
        # print(subgraph.n_ent, (KG[:, 1] == (self.n_rel * 2)).sum())

        M_sub = subgraph.M_sub
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(subgraph.n_ent, batchsize))
        edge_1hot = node_1hot
        node_triples = M_sub[:, nodes[:, 1]].multiply(filter_mask)
        edges = np.nonzero(node_triples)
        edges_value = nodes[:, 0][edges[1]]
        edges = [edges[0], edges_value]

        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]], axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        old_nodes_new_idx = tail_index[mask].sort()[0]

        total_node_1hot += node_1hot
        return tail_nodes, sampled_edges, edge_1hot, total_node_1hot, old_nodes_new_idx

