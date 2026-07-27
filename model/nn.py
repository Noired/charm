import torch
from torch.nn import Linear

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.layers(x)


class InvariantDeepSet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, aggr='sum'):
        super(InvariantDeepSet, self).__init__()
        self.phi = MLP(in_dim, 2*hidden_dim, hidden_dim)
        self.rho = MLP(hidden_dim, 2*hidden_dim, out_dim)
        if aggr == 'sum':
            self.pool = lambda x: torch.sum(x, 1)
        elif aggr == 'mean':
            self.pool = lambda x: torch.mean(x, 1)
        elif aggr == 'max':
            self.pool = lambda x: torch.max(x, 1)[0]
        else:
            raise ValueError(f"Aggregation {aggr} is not currently supported.")


    def forward(self, x):
        # NOTE: This model assumes the same number of elements in each input set.
        # NOTE: The expected input shape is [batch_size, set_cardinality, feature_dim]
        assert x.ndim == 3
        x = self.phi(x)
        x = self.pool(x)
        return self.rho(x)