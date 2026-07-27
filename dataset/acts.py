import torch

def get_activations(activations, layers):
    '''
        NOTE: activations have the following structure
            (gen_0, ..., gen_N)
                ^
                (layer_0 (token embeddings), layer_1, layer_2, ..., layer_L)
                                                ^
                                                [batch_size, prompt_len or 1, hidden_dim]
    '''
    acts_by_layer = list()  # list of activations, across generation steps, for each desired layer
    layer_idx = list()  # list of layer indices, used for accessing specific layers for downstream training
    for layer in layers:
        # NOTE: we access activations at position `layer` without index displacement it because we also have
        #       initial token embeddings in the first position
        acts_by_layer.append(torch.cat([activations[g][layer][0] for g in range(len(activations))], 0))
        layer_idx.append(layer*torch.ones(acts_by_layer[-1].shape[-1], dtype=torch.int64, device='cpu'))
    acts = torch.cat(acts_by_layer, -1).cpu()  # shape: [num_tokens, num_layers * hidden_dim]
    layer_idxs = torch.cat(layer_idx)
    return acts, layer_idxs