import torch
import torch.nn.functional as F

from torch.nn import Linear, Dropout
from torch.nn import Linear, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .nn import MLP

def layer_factory(name, pars):
    if name == 'custom':
        assert 'in_dim_x' in pars and 'in_dim_e' in pars and 'out_dim' in pars
        return CustomConv(
            pars['in_dim_x'],
            pars['in_dim_e'],
            pars['out_dim'],
            aggr=pars.get('aggr', 'mean'),
            flow=pars.get('flow', 'source_to_target'))
    # elif name == '...'
    # Add support for your own mp layer here
    else:
        raise ValueError(f"Unsupported layer {name}")

def token_readout(x, response_batch, idx):
    if idx >= 0:
        response_batch = response_batch[:-1]
    else:
        response_batch = response_batch[1:]
    response_batch = response_batch + idx
    return x[response_batch,:]

def readout_factory(readout):
    if readout == 'sum':
        return global_add_pool
    elif readout == 'mean':
        return global_mean_pool
    elif readout == 'max':
        return global_max_pool
    elif readout == 'none':
        return (lambda x, y: x)
    elif readout in ['-3' ,'-2', '-1', '0', '1', '2']:
        return (lambda x, y: token_readout(x, y, int(readout)))
    else:
        raise ValueError(f'Unsupported readout {readout}')

def activation_factory(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'none':
        return (lambda x: x)
    else:
        raise ValueError(f'Unsupported readout {activation}')

def encoder_factory(name, pars, edges=False, key=None):
    if name == 'linear':
        assert 'in_dim_x' in pars and 'in_dim_e' in pars and 'hidden_dim' in pars
        assert key is None or (key in pars and pars[key] is not None)
        if edges:
            in_dim = pars['in_dim_e']
        elif key is None:
            in_dim = pars['in_dim_x']
        else:
            in_dim = pars[key]
        hidden_dim = pars['hidden_dim']
        return Linear(in_dim, hidden_dim)
    elif name == 'none':
        return (lambda x: x)
    else:
        raise ValueError(f"Unsupported encoder {name}")

def get_readout_args(x, batch_index, response_index, ptr, prompt_len, readout_strategy):
    if readout_strategy in ['-3', '-2', '-1', '0', '1', '2']:
        response_batch = ptr + torch.cat((prompt_len, torch.zeros((1,), dtype=prompt_len.dtype, device=prompt_len.device)), 0)
        return x, response_batch
    else:
        x = x[response_index]
        response_batch = batch_index[response_index]
        return x, response_batch


class CustomConv(MessagePassing):
    def __init__(self, in_dim_x, in_dim_e, out_dim, aggr='add', flow='source_to_target'):
        super().__init__(aggr=aggr, flow=flow)
        self.msg_mlp = MLP(in_dim_x+in_dim_e, 2*out_dim, out_dim)
        self.up_mlp = MLP(in_dim_x+out_dim, 2*out_dim, out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.up_mlp(torch.cat((x, msg), -1))

    def message(self, x_j, edge_attr):
        return self.msg_mlp(torch.cat((x_j, edge_attr), -1))


class CHARM(torch.nn.Module):
    def __init__(self, num_layers, in_dim_x, in_dim_e, hidden_dim, num_classes, dropout=0.0, aggr='add', flow='both', readout='sum', activation='none', layer='dir_graphconve_block', encoder='linear', dims_for_linear_layers=None, on_x=True, batch_norm=False, residual=False, attr_encoder_location='beginning', cat_attr=False):
        super().__init__()
        assert (layer.startswith('dir_') and flow=='both') or (flow in ['source_to_target', 'target_to_source']) or (layer.startswith('siamese'))
        # self.dropout = dropout
        self.dropout = Dropout(dropout)
        self.residual = residual
        self.activation = activation_factory(activation)
        # ---- Encoder
        self.attr_encoder_location = attr_encoder_location
        self.cat_attr = cat_attr
        self.num_non_none = sum(bool(v) for v in dims_for_linear_layers.values())
        if self.cat_attr:
            print(f'[i] Cat attr')
            assert self.num_non_none > 0
            self.cat_encoder = torch.nn.Linear(
                                    in_features=hidden_dim*(self.num_non_none+1),
                                    out_features=hidden_dim
                                )

        assert attr_encoder_location in ['beginning', 'end']
        pars = {
            'in_dim_x': in_dim_x,
            'in_dim_e': in_dim_e,
            'hidden_dim': hidden_dim,
            **dims_for_linear_layers}
        self.encoder_x = encoder_factory(encoder, pars, edges=False)
        self.encoder_e = encoder_factory(encoder, pars, edges=True)
        self.attr_encoders = torch.nn.ModuleDict({
            'act': encoder_factory(encoder, pars, edges=False, key='act') if dims_for_linear_layers['act'] else None})
        if on_x:
            print(f'[i] On X mode: a single linear layer for all input features')
            assert all(dim is None for dim in dims_for_linear_layers.values())
            assert all(attr_encoder is None for attr_encoder in self.attr_encoders.values())
        else:
            print(f'[i] Off X mode: a linear layer for each input feature')
            assert encoder != 'none'
            assert any(dim is not None for dim in dims_for_linear_layers.values())
            assert any(attr_encoder is not None for attr_encoder in self.attr_encoders.values())
        # ---- Processor
        self.conv_layers = list()
        for l in range(num_layers):
            pars = {
                'in_dim_x': hidden_dim if (encoder!='none' or l>0) else in_dim_x,
                'in_dim_e': hidden_dim if encoder!='none' else in_dim_e,
                'out_dim': hidden_dim,
                'aggr': aggr,
                'flow': flow}
            self.conv_layers.append(layer_factory(layer, pars))
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.norm_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ]) if batch_norm else None
        # ---- Decoder
        self.readout_strategy = readout
        self.readout = readout_factory(readout)
        decoder_in = hidden_dim
        self.decoder = Linear(decoder_in, num_classes)

    def forward(self, x, edge_index, edge_attr, batch_index, response_index, ptr, prompt_len, artifacts):
        x_encoded = self.encoder_x(x)
        attr_encodings = []
        for attr, encoder in self.attr_encoders.items():
            if encoder is not None:
                assert artifacts[attr] is not None
                # NOTE (Guy): Currently summing, we might want to try concatenating or something else
                attr_encodings.append(encoder(artifacts[attr]))
            else:
                assert artifacts[attr] is None
        if self.attr_encoder_location == 'beginning' and self.num_non_none > 0:
            if self.cat_attr:
                attr_encodings = torch.cat(attr_encodings, dim=-1)
                x = self.cat_encoder(torch.cat((x_encoded, attr_encodings), dim=-1))
            else:
                attr_encodings = torch.stack(attr_encodings, dim=-1)
                attr_encodings = attr_encodings.sum(dim=-1)
                x = x_encoded + attr_encodings
        else:
            x = x_encoded

        edge_attr = self.encoder_e(edge_attr)
        for i, layer in enumerate(self.conv_layers):
            x_ = layer(x, edge_index, edge_attr)
            if self.norm_layers is not None:
                x_ = self.norm_layers[i](x_)
            x_ = self.activation(x_)
            x_ = self.dropout(x_)
            if self.residual:
                x = x_ + x
            else:
                x = x_
        # NOTE (Guy): if attr_encoder_location == 'end', we are basically considering only [response_index] for act.
        if self.attr_encoder_location == 'end' and self.num_non_none > 0:
            if self.cat_attr:
                attr_encodings = torch.cat(attr_encodings, dim=-1)
                x = self.cat_encoder(torch.cat((x, attr_encodings), dim=-1))
            else:
                attr_encodings = torch.stack(attr_encodings, dim=-1)
                attr_encodings = attr_encodings.sum(dim=-1)
                x += attr_encodings
        x, response_batch = get_readout_args(x, batch_index, response_index, ptr, prompt_len, self.readout_strategy)
        x = self.readout(x, response_batch)
        x = self.decoder(x)
        return x