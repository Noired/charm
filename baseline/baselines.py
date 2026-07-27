import torch

from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from model.mp import readout_factory, get_readout_args

def baseline_layer_factory(name, in_dim_x, in_dim_e):
    if name == 'node_averaging':
        return (NodeAveragingLayer(), in_dim_x)
    elif name == 'edge_averaging':
        return (AttentionAveragingLayer(), in_dim_e)
    elif name == 'global_averaging':
        return (NoOpLayer(), in_dim_x)
    elif name == 'lookback':
        return (LookbackLayer(), in_dim_e)
    elif name == 'lapeig':
        return (LapEigLayer(), in_dim_e)
    else:
        raise NotImplementedError

class FromPrompt(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum', flow='source_to_target')

    def forward(self, x, edge_index, edge_attr):
        msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        deg_den = (msg[:,-1] + x[:,-2]).unsqueeze(-1)
        where = (deg_den == 0.0)
        deg_den[where] = 1.0
        att_msg = msg[:,:-1]
        return ((x[:,-2].unsqueeze(-1)) * (x[:,:-2]) + att_msg) / deg_den

    def message(self, x_j, edge_attr):
        # NOTE: important – assumes x feats are in the form (att_scores, prompt_mark)
        return torch.cat(
            [x_j[:,-2].unsqueeze(-1) * edge_attr,
             x_j[:,-2].unsqueeze(-1)], -1)


class FromResponse(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum', flow='source_to_target')

    def forward(self, x, edge_index, edge_attr):
        msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        deg_den = (msg[:,-1] + x[:,-1]).unsqueeze(-1)
        where = (deg_den == 0.0)
        deg_den[where] = 1.0
        att_msg = msg[:,:-1]
        return ((x[:,-1].unsqueeze(-1)) * (x[:,:-2]) + att_msg) / deg_den

    def message(self, x_j, edge_attr):
        # NOTE: important – assumes x feats are in the form (att_scores, prompt_mark)
        return torch.cat(
            [x_j[:,-1].unsqueeze(-1) * edge_attr,
             x_j[:,-1].unsqueeze(-1)], -1)


class LookbackLayer(MessagePassing):
    def __init__(self):
        ''' 
            L = A_to_prompt / (A_to_prompt + A_to_response)
        '''
        super().__init__()
        self.prompt_to_response = FromPrompt()
        self.response_to_response = FromResponse()

    def forward(self, x, edge_index, edge_attr):
        a = self.prompt_to_response(x, edge_index, edge_attr)
        b = self.response_to_response(x, edge_index, edge_attr)
        return torch.nan_to_num(a / (a + b))


class NodeAveragingLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum', flow='source_to_target')

    def forward(self, x, edge_index, edge_attr):
        deg_denom = degree(edge_index[1], dtype=torch.float, num_nodes=x.shape[0]) + 1.0
        deg_denom = deg_denom.unsqueeze(-1)
        return (x + self.propagate(edge_index, x=x, edge_attr=edge_attr)) / deg_denom

    def message(self, x_j, edge_attr):
        return x_j


class AttentionAveragingLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum', flow='source_to_target')

    def forward(self, x, edge_index, edge_attr):
        deg_denom = degree(edge_index[1], dtype=torch.float, num_nodes=x.shape[0]) + 1.0
        deg_denom = deg_denom.unsqueeze(-1)
        return (x + self.propagate(edge_index, x=x, edge_attr=edge_attr)) / deg_denom

    def message(self, x_j, edge_attr):
        return edge_attr


class LapEigLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean', flow='target_to_source')

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) - x

    def message(self, x_j, edge_attr):
        return edge_attr


class NoOpLayer(MessagePassing):

    def forward(self, x, edge_index, edge_attr):
        return x


class AttentionBaseline(torch.nn.Module):
    def __init__(self, in_dim_x, in_dim_e, num_classes, baseline='node_averaging', readout='mean', prediction_head=True):
        super().__init__()
        assert baseline in ['node_averaging', 'edge_averaging', 'global_averaging', 'lookback', 'lapeig']
        self.conv_layer, head_in = baseline_layer_factory(baseline, in_dim_x, in_dim_e)
        assert readout in ['mean', '-3' ,'-2', '-1', '0', '1', '2', 'none']
        self.readout_strategy = readout
        self.readout = readout_factory(readout)
        if prediction_head:
            self.head = Linear(head_in, num_classes)
        else:
            self.head = lambda x: x

    def forward(self, x, edge_index, edge_attr, batch_index, response_index, ptr, prompt_len):
        x = self.conv_layer(x, edge_index, edge_attr)
        x, response_batch = get_readout_args(x, batch_index, response_index, ptr, prompt_len, self.readout_strategy)
        x = self.readout(x, response_batch)
        x = self.head(x)
        return x