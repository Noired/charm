import torch
import math
from torch_geometric.transforms import BaseTransform

class Cast(BaseTransform):

    def __call__(self, data):
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        return data
    
    def __repr__(self):
        return 'cast'


class Transpose(BaseTransform):

    def __call__(self, data):
        data.edge_index = torch.stack((data.edge_index[1], data.edge_index[0]), 0)
        return data
    
    def __repr__(self):
        return 'transpose'


class NodeMark(BaseTransform):
    
    def __init__(self, mark_on_x=True):
        self.mark_on_x = mark_on_x
        super().__init__()

    def get_mark(self, data):
        raise NotImplementedError

    def __call__(self, data):
        mark = self.get_mark(data)
        if self.mark_on_x:
            assert hasattr(data, 'x') and data.x is not None
            data.x = torch.cat((data.x, mark), 1)
        else:
            if hasattr(data, 'm') and data.m is not None:
                data.m = torch.cat((data.m, mark), 1)
            else:
                data.m = mark
        return data


class MarkPrompt(NodeMark):

    def get_mark(self, data):
        """Mark prompt nodes w.r.t. response ones"""
        prompt_len = data.prompt_len
        num_nodes = data.num_nodes
        mark = torch.zeros((num_nodes, 2), dtype=data.x.dtype)
        mark[:prompt_len,0] = 1.0
        mark[prompt_len:,1] = 1.0
        return mark

    def __repr__(self):
        return 'mark_prompt'


class MarkPromptEdges(BaseTransform):

    def __call__(self, data):
        """Mark prompt edges w.r.t. response ones"""
        prompt_len = data.prompt_len
        num_edges = data.edge_index.shape[1]
        mark = torch.zeros((num_edges, 2), dtype=data.edge_attr.dtype)
        src, target = data.edge_index[0,:], data.edge_index[1,:]
        # 0. response -> response
        where = src >= prompt_len  # response source
        where &= target >= prompt_len  # response target
        mark[where,0] += 1.0
        # 1. prompt -> reponse
        where = src < prompt_len  # prompt source
        where &= target >= prompt_len  # response target
        mark[where,1] += 1.0
        assert torch.all(mark <= 1.0)
        to_be_marked_a = (src < prompt_len)
        to_be_marked_a &= (target >= prompt_len)
        to_be_marked_b = (src >= prompt_len)
        to_be_marked_b &= (target >= prompt_len)
        to_be_marked = torch.logical_or(to_be_marked_a, to_be_marked_b)
        not_to_be_marked = torch.logical_not(to_be_marked)
        assert torch.all(mark[to_be_marked,:].sum(1) == 1.0)
        assert torch.all(mark[not_to_be_marked,:].sum(1) == 0.0) # prompt to prompt
        assert hasattr(data, 'edge_attr') and data.edge_attr is not None
        data.edge_attr = torch.cat((data.edge_attr, mark), 1)
        return data

    def __repr__(self):
        return 'mark_prompt_edges'


class MarkStartPrompt(NodeMark):

    def get_mark(self, data):
        """Mark start of the prompt"""
        num_nodes = data.num_nodes
        mark = torch.zeros((num_nodes, 1), dtype=data.x.dtype)
        mark[0,0] = 1.0
        return mark
    
    def __repr__(self):
        return 'mark_start_prompt'


class MarkStartResponse(NodeMark):

    def get_mark(self, data):
        """Mark start of the response"""
        prompt_len = data.prompt_len
        num_nodes = data.num_nodes
        mark = torch.zeros((num_nodes, 1), dtype=data.x.dtype)
        mark[prompt_len,0] = 1.0
        return mark

    def __repr__(self):
        return 'mark_start_response'


class LabelPooling(BaseTransform):  

    def __init__(self, agg='min'):
        if agg not in ['max', 'min', 'mean']:
            raise ValueError(agg)
        self.agg = agg
        super().__init__()
    
    def __call__(self, data):
        """Get a graph-wise label from node-wise ones (at a response level)"""
        assert hasattr(data, 'y') and data.y is not None
        if data.y.ndim > 0:
            assert data.y.shape[0] == data.response_index.shape[0]
            if self.agg == 'min':
                data.y = data.y.min()
            elif self.agg == 'max':
                data.y = data.y.max()
            elif self.agg == 'mean':
                data.y = data.y.mean()
        return data

    def __repr__(self):
        return f'label_pooling_{self.agg}'


class ThresholdAttention(BaseTransform):

    def __init__(self, val=0.0):
        assert val >= 0.0
        self.val = val
        super().__init__()

    def __call__(self, data):
        """Sets all edge feats smaller than `self.val` to 0.0 and deletes edges which would be associated only to null values."""
        assert hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[0] == data.edge_index.shape[1]
        condition = (data.edge_attr < self.val)
        data.edge_attr[condition] = 0.0
        to_keep = torch.where(data.edge_attr.sum(-1) != 0.0)[0]
        data.edge_attr = data.edge_attr[to_keep, :]
        data.edge_index = data.edge_index[:, to_keep]
        return data
    
    def __repr__(self):
        return f'threshold_att_{self.val}'


class LayerSelect(BaseTransform):

    def __init__(self, values):
        self.values = values
        self.condition = self.set_condition()
        super().__init__()

    def set_condition(self):
        raise NotImplementedError

    def select(self, data):
        raise NotImplementedError
    
    def __call__(self, data):
        condition = self.select(data)
        # Backup added features
        original_feats = data.layer.shape[0]
        x_addendum = data.x[:, original_feats:]
        edge_attr_addendum = data.edge_attr[:, original_feats:]
        # Only keep the feats we want
        x_new = data.x[:,:original_feats][:,condition]
        edge_attr_new = data.edge_attr[:,:original_feats][:,condition]
        # Attach back the added feats
        x_new = torch.cat((x_new, x_addendum), 1)
        to_keep = torch.where(edge_attr_new.sum(1) != 0.0)[0]
        edge_attr_new = torch.cat((edge_attr_new, edge_attr_addendum), 1)
        # Discard "empty" edges
        edge_index_new = data.edge_index[:, to_keep]
        edge_attr_new = edge_attr_new[to_keep,:]
        data.x = x_new
        data.edge_index = edge_index_new
        data.edge_attr = edge_attr_new
        data.head = data.head[condition]
        data.layer = data.layer[condition]
        return data


class DiscardFirstLayers(LayerSelect):

    def set_condition(self):
        assert self.values[-1] == '+'
        val = int(self.values[:-1])
        assert val > 0
        self.num_layers = val

    def select(self, data):
        return (data.layer > self.num_layers)

    def __repr__(self):
        return f'discard_first_{self.num_layers}_layers'


class KeepOnlyLayer(LayerSelect):

    def set_condition(self):
        val = int(self.values)
        assert val >= 0
        self.layer = val

    def select(self, data):
        assert self.layer <= data.layer.max().item()
        return (data.layer == self.layer)

    def __repr__(self):
        return f'keep_only_layer_{self.layer}'
    

class KeepOnlySomeLayers(LayerSelect):

    def set_condition(self):
        assert len(self.values) > 1 and ',' in self.values
        vals = self.values.split(',')
        for val in vals:
            assert int(val) >= 0, val
        self.layers = [int(v) for v in vals]

    def select(self, data):
        condition = None
        for layer in self.layers:
            assert layer <= data.layer.max().item()
            if condition is None:
                condition = (data.layer == layer)
            else:
                condition |= (data.layer == layer)
        return condition

    def __repr__(self):
        msg = 'keep_only_layers'
        for layer in self.layers:
            msg += '_'+str(layer)
        return msg


class KeepOnlyLayerInterval(LayerSelect):

    def set_condition(self):
        assert '-' in self.values
        start, end = self.values('-')
        self.start = int(start)
        self.end = int(end)
        assert self.start >= 0 and self.start < self.end

    def select(self, data):
        assert self.end <= data.layer.max().item()
        condition = (data.layer >= self.start)
        condition &= (data.layer <= self.end)
        return condition

    def __repr__(self):
        return f'keep_only_layers_{self.start}_thru_{self.end}'
