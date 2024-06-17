from collections.abc import Sequence
import torch.nn as nn
from torch.nn import functional as F
from util import *
from torch_scatter import scatter

class GNNLayer(torch.nn.Module):
    def __init__(self, params, hidden_dim):
        super(GNNLayer, self).__init__()
        self.params = params
        self.n_rel = params.n_rel
        self.in_dim = hidden_dim
        self.out_dim = hidden_dim
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        self.act = acts[params.act]
        self.MESS_FUNC = params.MESS_FUNC.replace("\r", "")
        self.AGG_FUNC = params.AGG_FUNC.replace("\r", "")
        self.relation_linear = nn.Linear(self.in_dim, (2 * self.n_rel + 1) * self.in_dim)
        if self.AGG_FUNC == "pna":
            self.W_h = nn.Linear(self.in_dim * 13, self.out_dim)
        else:
            self.W_h = nn.Linear(self.in_dim * 2, self.out_dim)
        self.layer_norm = nn.LayerNorm(self.in_dim, elementwise_affine=False)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    # forward for all edges, with batch data
    def forward(self, query, layer_input, edges, n_ent, edge_count = None, boundary=None):
        sub, rel, obj = edges[:, 0], edges[:, 1], edges[:, 2]
        relation = self.relation_linear(query).view(len(query), 2 * self.n_rel + 1, self.in_dim)
        input_j = layer_input.index_select(1, sub)
        relation_j = relation.index_select(1, rel)
        if self.MESS_FUNC == 'TransE':
            message = input_j + relation_j
        elif self.MESS_FUNC == 'DistMult':
            message = input_j * relation_j
        elif self.MESS_FUNC == 'RotatE':
            hs_re, hs_im = input_j.chunk(2, dim=-1)
            hr_re, hr_im = relation_j.chunk(2, dim=-1)
            message_re = hs_re * hr_re - hs_im * hr_im
            message_im = hs_re * hr_im + hs_im * hr_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            message = input_j * relation_j

        if self.AGG_FUNC == 'pna':
            message_agg = self.pna_process(message, obj, item_size=n_ent, boundary=boundary, scatter_dim=1)
        else:
            message_agg = scatter(message, index=obj, dim=1, dim_size=n_ent, reduce=self.AGG_FUNC)
            message_agg = message_agg + boundary

        if len(query)== 1 and len(message_agg.shape) == 2:
            message_agg = message_agg.unsqueeze(0)
        message_agg = torch.concat([message_agg, layer_input],dim=-1)
        message_agg = self.W_h(message_agg)
        message_agg = self.layer_norm(message_agg)
        hidden_new = self.act(message_agg)  # [n_node, dim]
        return hidden_new

    def pna_process(self, message, obj_index, item_size, boundary, scatter_dim=0):
        count_sum = scatter(torch.ones(list(message.shape)[:-1] + [1]).cuda(), index=obj_index, dim=scatter_dim,
                             dim_size=item_size, reduce="sum").clamp(min=1)
        sum = scatter(message, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="sum")
        max = scatter(message, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="max")
        min = scatter(message, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="min")
        sq_sum = scatter(message ** 2, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="sum")

        mean = (sum + boundary) / count_sum
        sq_mean = (sq_sum + boundary ** 2) / count_sum
        max = torch.max(max, boundary)
        min = torch.min(min, boundary)

        std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()

        features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1)], dim=-1)
        features = features.flatten(-2)
        scale = count_sum.log()
        scale = scale / (scale.mean().clamp(min=1))

        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        message_agg = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        message_agg = message_agg.squeeze(0)
        return message_agg

class GNNModel(torch.nn.Module):
    def __init__(self, params):
        super(GNNModel, self).__init__()
        print("Loading NBFNet Model...")
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.n_rel = params.n_rel
        self.remove_one_loop = params.remove_one_loop
        self.rela_embed = nn.Embedding(2 * self.n_rel + 2, self.hidden_dim)

        self.gnn_layers = []
        for i in range(self.n_layer):  # 1
            self.gnn_layers.append(GNNLayer(self.params, hidden_dim=self.hidden_dim))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)

        num_mlp_layer = 2
        self.mlp = MultiLayerPerceptron(self.hidden_dim * 2, [self.hidden_dim * 2] * (num_mlp_layer - 1) + [1])

        # calculate parameters
        num_parameters = sum(p.numel() for p in self.parameters())
        # for name, param in self.state_dict().items():
        #     print(name, param.shape)
        print('==> num_parameters: {}'.format(num_parameters))

    def indicator(self, n_ent, index, query):
        boundary = torch.zeros(n_ent, *query.shape, device=query.device)
        index = index.unsqueeze(-1).expand_as(query)
        boundary.scatter_(0, index.unsqueeze(0), query.unsqueeze(0))
        return boundary

    def score(self, hidden, node_query):
        hidden = torch.cat([hidden, node_query], dim=-1)
        score = self.mlp(hidden).squeeze(-1)
        return score

    def forward(self, triples, subgraph, mode='train', epoch=0):
        n = len(triples)
        n_ent = subgraph.n_ent
        edge_data = torch.from_numpy(subgraph.KG).cuda()
        batch_triples = torch.cat([torch.arange(n).unsqueeze(1).cuda(),
                                   torch.LongTensor(triples).cuda()], 1)  # [bid, h, r, t]
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

        filter_data = edge_data[filter_mask]
        edge_count = scatter(torch.ones((len(filter_data), 1)).cuda(), index=filter_data[:, 2], dim=0,
                             dim_size=n_ent, reduce="sum")
        batch_size = len(q_rel)
        query = self.rela_embed(q_rel)
        boundary = self.indicator(n_ent, q_sub, query).transpose(1, 0)
        nodes = start_nodes
        layer_input = boundary
        for i in range(self.n_layer):
            hidden = self.gnn_layers[i].forward(query, layer_input, filter_data, n_ent, edge_count.T.repeat(len(query), 1), boundary=boundary)  # filter_data
            layer_input = hidden + layer_input
        node_query = query.unsqueeze(1).expand(-1, n_ent, -1)
        scores_all = self.score(layer_input, node_query)
        return scores_all

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden
