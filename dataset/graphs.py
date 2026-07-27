import torch

from torch_geometric.utils import dense_to_sparse, mask_to_index
from torch_geometric.data import Data

def get_attention_matrix(outs, layer, head):
    '''
        Outputs an adjacency matrix starting from the attention scores at the various generation steps.
        Works for a specified layer and head.
    '''
    out_tensors = [gen[layer][0,head,:,:].cpu() for gen in outs]
    max_dim = max([out.shape[-1] for out in out_tensors])
    # NOTE: we use the last prompt token and the attention patterns thereof to predict hallucination
    #       on the first generated token, that is the reason why we decrease by one below
    input_len = out_tensors[0].shape[-1] - 1
    padded = list()
    for out in out_tensors:
        diff = max_dim - out.shape[-1]
        new = torch.cat((out, torch.zeros((out.shape[0], diff), dtype=torch.float16)), 1)
        padded.append(new)
    res = torch.cat(padded, 0)

    ''' 
        p p r r
        0 1 2 3
     0  *       --> 1
     1  * *     --> 2 (already response!?)
       ------------------ 
     2  * * +   --> 3
       ------------------
     3  * * + + --> (4)

     prompt_len -> 1
     response_index -> [1,2,3]
    '''

    '''
    [ANOTHER POSSIBILITY] – Here node i has attention info to predict hallucination at token i
        p p r r [EOS]
        0 1 2 3 4
     0  x          --> no attention values, no next-token-prediction values
       ~~~~~~~~~~~~~~~~~~ (above not given, node feat only)
     1  * x        --> 1
     2  * * x      --> 2 (already response)
       ------------------ 
     3  * * + x    --> 3
       ------------------
     4  * * + + x  --> [EOS]

     prompt_len -> 2, as expected
    '''
    
    return res, input_len

def process_attentions(A, input_len, threshold, prompt_graph, indicate_input):
    '''
        Some post processing of the attention matrix.
            A: attention score matrix.
            input_len: length of the input prompt, used to create specific features that identifies w.r.t. the response.
            threshold: used to sparsify the attention matrix – all scores below threshold will not materialise and edge.
            prompt_graph: whether to materialise prompt2prompt connections, set to False for improved sparsity.
            indicate_input: adds special marks to identify inputs to the LLM as opposed to the output (response).
    '''
    assert A.shape[0] == A.shape[1] and A.ndim == 2
    x = A.diagonal().clone().unsqueeze(-1)
    if indicate_input:
        input_indicator = torch.ones((x.shape[0], 2))
        input_indicator[:input_len,0] = 0.0
        input_indicator[input_len:,1] = 0.0
        x = torch.cat((x, input_indicator), -1)
    A.fill_diagonal_(0)
    A[A<threshold] = 0.0
    if not prompt_graph:
        A[:input_len,:input_len] = 0.0
    return x, A

def tensor_to_sparse(As):
    '''
        Converts a bunch of dense attention score matrices into a single sparse representation.
        The output connectivity consists in the union of input connectivities.
        Edge features are derived from the edge weights.
    '''
    stacked = torch.stack(As, -1)
    squashed = torch.sum(stacked, -1)  # Squashing helps identifying those entries which are positive in at least one case.
    edge_index, _ = dense_to_sparse(squashed)
    edge_attr = list()
    for s, t in edge_index.transpose(1,0):
        edge_attr.append(stacked[s,t,:])
    edge_attr = torch.stack(edge_attr, 0)
    return edge_index, edge_attr
    
def get_data_object(outs, threshold=0.0, prompt_graph=False):
    '''
        Construct data object from LLM outputs (unstructured attention scores).
            threshold: value under which attention scores will be considered none and, therefore,
                no edges will be drawn.
            prompt graph: whether to include edges between prompt tokens.
        NOTE: outs has the following structure
            (gen_0, gen_1, ..., gen_n)
              ^
              (layer_0, layer_1, ..., layer_L)
                 ^
                 [batch_size, num_heads, x, num_tokens_so_far]
                                         ^
                                         either num of tokens in the prompt (gen_0) or 1
    '''
    As = list()
    xs = list()
    hs = list()
    ls = list()
    num_layers = len(outs[0])  # Accessing data from the first generation step to get the num of layers in the LLM
    for l in range(num_layers):
        num_heads = (outs[0][l]).shape[1]
        for h in range(num_heads):
            A, input_len = get_attention_matrix(outs, l, h)
            res = process_attentions(A, input_len, threshold, prompt_graph, indicate_input=False)
            As.append(res[1])
            xs.append(res[0])
            hs.append(h)
            ls.append(l)
    x = torch.cat(xs, -1)
    h_index = torch.LongTensor(hs)
    l_index = torch.LongTensor(ls)
    edge_index, edge_attr = tensor_to_sparse(As)
    # Swap directionality as returned by the above
    edge_index = torch.stack((edge_index[1], edge_index[0]), 0)
    response_mask = torch.zeros(x.shape[0], dtype=torch.int64)
    response_mask[input_len:] = 1
    print(f"[i] Generating graph with {edge_index.shape[1]} edges starting from a dense matrix with around {torch.count_nonzero(outs[0][0])} non-zero elements.")
    return Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        head=h_index, 
        layer=l_index, 
        response_index=mask_to_index(response_mask),
        prompt_len=input_len)
