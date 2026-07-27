# ---------------------------------------------------------------------------
# Portions of this file are adapted from the Lookback Lens project:
#   https://github.com/voidism/Lookback-Lens
# Lookback Lens declares the MIT License in its README (no root LICENSE file).
# Adapted in good faith under those stated terms, with modifications by the
# CHARM authors. See PROVENANCE.md.
# ---------------------------------------------------------------------------

import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm
import argparse
import tiktoken

from data_prep_1.generation import LLM
from dataset.graphs import get_data_object
from dataset.acts import get_activations

transformers.logging.set_verbosity(40)


data_response_names = {
    'nq': 'Answer',
    'xsum': 'Summary',
    'cnndm': 'Summary',
}

# should be changed to using `llama2_tokenizer` instead of tiktoken, but we keep this implementation for now to be consistent with the original results
def num_tokens_from_message(message, model="davinci"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split(' ')[:-1])
    prompt = prompt1 + prompt2
    return prompt

def load_nq_open(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='nq_open', subsample=None):
    list_data_dict = []
    is_train = 'nq_train' in file_path
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        if debug:
            data = data[:10]
        if subsample is not None:
            # select data if idx%subsample == 0
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size]

        for idx in range(len(data)):
            data_index = idx
            question = data[idx]['question']
            # capitalize the first letter of the question, add the question mark if not present at the end
            question = question[0].upper() + question[1:]
            if question[-1] != '?':
                question += '?'
            answers = data[idx]['answers']
            if is_train:
                pos_ctxs = data[idx]['positive_ctxs']
                neg_ctxs = data[idx]['negative_ctxs']
            else:
                ctxs = data[idx]['ctxs']
                pos_ctxs = [ctx for ctx in ctxs if ctx['hasanswer']]
                neg_ctxs = [ctx for ctx in ctxs if not ctx['hasanswer']]
            assert len(pos_ctxs) > 0, "No positive context found."
            assert len(neg_ctxs) >= 2, "At least two negative contexts are required."
            context = f"#Document#: " + neg_ctxs[0]['text'] + '\n' + pos_ctxs[0]['text'] + '\n' + neg_ctxs[1]['text']
            context += f"\n#Question#: {question}"
            response = f"\n#Answer#:"
            new_item = dict(
                context=context,
                response=response,
                net_response=None,
                answer=answers[0],
                data_index=data_index
            )
            list_data_dict.append(new_item)
    return list_data_dict


def load_summarization(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='cnndm', subsample=None):
    list_data_dict = []
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        data_indices = []
        data_index = 0
        for line in f:
            data.append(json.loads(line))
            data_indices.append(data_index)
            data_index += 1
        if debug:
            data = data[:10]
            data_indices = data_indices[:10]
        if subsample is not None:
            # select data if idx%subsample == 0
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
            data_indices = [data_indices[i] for i in range(len(data_indices)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size]
            data_indices = data_indices[shard_id * chunk_size: (shard_id + 1) * chunk_size]

        for idx in range(len(data)):
            data_index = data_indices[idx]
            context = "#Document#: " if data_type == 'cnndm' else "#Article#: "
            context += data[idx]['document']
            new_item = dict(
                context=context,
                data_index=data_index
            )
            list_data_dict.append(new_item)

    return list_data_dict

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def create_demo_text(pondering=None, data_type='cnndm'):
    if data_type == 'cnndm':
        return "Generate a summary based on the information in the document.\n\n"
    elif data_type == 'nq':
        return "Answer the question based on the information in the document. Explain your reasoning in the document step-by-step before providing the final answer.\n\n"
    elif data_type == 'xsum':
        return "Generate a summary comprising of 1 sentence for the given article.\n\n"
    else:
        raise ValueError("Please specify the data type.")


def build_prompt(context, response, pondering=None, data_type='cnndm'):
    demo = create_demo_text(pondering, data_type)
    prompt = demo + context
    if data_type == 'cnndm' or data_type == 'xsum':
        input_text_prompt = truncate_message(prompt, response)
    else:
        input_text_prompt = prompt + response
    return input_text_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="raw_data/nq-open-10_total_documents_gold_at_4.jsonl")
    parser.add_argument("--output-folder", type=str, default="raw_data/")
    parser.add_argument("--output-suffix", type=str, default="nq-llama-2-7b-chat-001__24_28_32___16_16bits")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auth-token", type=str, default=None)
    parser.add_argument("--data-type", type=str, default=None)

    # replication to reuse their annotations
    parser.add_argument("--teacher-forcing-jsonl", type=str, default="raw_data/tf-nq-7b.jsonl")

    # dumping of lookback features
    parser.add_argument("--lookback", action='store_true')

    # dumping of raw attentions (as graphs)
    parser.add_argument("--att_graph", action='store_true')
    parser.add_argument("--att_threshold", type=float, default=0.001)
    parser.add_argument("--prompt_graph", action='store_true')

    # extraction of hidden states
    parser.add_argument("--act", action='store_true')
    parser.add_argument("--act_layers", type=str, default='24,28,32')

    # extraction of output scores
    parser.add_argument("--scores", action='store_true')

    # max_memory
    parser.add_argument("--max-memory", type=int, default=45)

    # debug – only collect a small amount of data
    parser.add_argument("--debug_chunk", action='store_true')
    
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # seeding
    set_seed(args.seed)

    # output paths
    output_path = os.path.join(args.output_folder, f'{args.output_suffix}')
    os.makedirs(output_path, exist_ok=True)
    att_path = os.path.join(output_path, f'att_{args.output_suffix}_{args.seed}.pt')
    act_path = os.path.join(output_path, f'act_{args.output_suffix}_{args.seed}.pt')
    atp_path = os.path.join(output_path, f'atp_{args.output_suffix}_{args.seed}.pt')
    lb_path = os.path.join(output_path, f'lb_{args.output_suffix}_{args.seed}.pt')

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    forced_truncate = ('gpt2' in args.model_name)
    if args.data_type is None:
        if 'cnndm' in args.data_path:
            args.data_type = 'cnndm'
        elif 'nq-open' in args.data_path:
            args.data_type = 'nq'
        elif 'xsum' in args.data_path:
            args.data_type = 'xsum'
        else:
            raise ValueError("Please specify the data type.")
    # Get test file
    fp = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")

    if "nq-open" in fp:
        list_data_dict = load_nq_open(fp, parallel=args.parallel, total_shard=args.total_shard, shard_id=args.shard_id, debug=args.debug, subsample=args.subsample)
    else:
        list_data_dict = load_summarization(fp, parallel=args.parallel, total_shard=args.total_shard, shard_id=args.shard_id, debug=args.debug, data_type=args.data_type, subsample=args.subsample)
    
    llm = LLM(
        model_name, device, num_gpus, 
        auth_token=args.auth_token, 
        max_memory=args.max_memory)
    stop_word_list = ["#Document#:", "#Question#:", "#Article#:", "Q:", "\end{code}"]
    llm.set_stop_words(stop_word_list)
    mode = "vanilla"
    if args.teacher_forcing_jsonl is not None:
        teacher_forcing_dict = {}
        with open(args.teacher_forcing_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                teacher_forcing_dict[data['data_index']] = data['model_completion_ids']

    att_save_list = list()
    act_save_list = list()
    atp_save_list = list()
    lb_save_list = list()
    extra_prompt_length = len(llm.tokenizer(f"\n#{data_response_names[args.data_type]}#:")['input_ids']) - 1
    if args.act:
        assert args.act_layers != 'none'
        if ',' in args.act_layers:
            act_layers = [int(layer) for layer in args.act_layers.split(',')]
        else:
            act_layers = [int(args.act_layers)]
    else:
        act_layers = None
    num_samples = len(list_data_dict) if not args.debug_chunk else 5
    for idx in tqdm(range(num_samples)):
        sample = list_data_dict[idx]
        teacher_forcing_ids = torch.tensor([teacher_forcing_dict[sample['data_index']]], device=device) \
                                if args.teacher_forcing_jsonl is not None else None
        input_text = build_prompt(sample['context'], f"\n#{data_response_names[args.data_type]}#:", data_type=args.data_type)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, 
                               do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, 
                               temperature=args.temperature, mode=mode, 
                            return_attentions=True, teacher_forcing_seq=teacher_forcing_ids,
                            return_activations_layers=act_layers)

        model_completion, attentions, activations, model_completion_ids, scores = llm.generate(
            input_text, **generate_kwargs)

        for stop_word in stop_word_list:
                length_to_remove = len(stop_word)
                if model_completion[-length_to_remove:] == stop_word:
                    model_completion = model_completion[:-length_to_remove]
        
        if args.lookback: # standard lookback lens behaviour
            context_length = attentions[0][0].shape[-1] - extra_prompt_length
            new_token_length = len(attentions)
            num_layers = len(attentions[0])
            num_heads = attentions[0][0].shape[1]
            lookback_ratio = torch.zeros((num_layers, num_heads, new_token_length))
            lookback_ratio_on_sink = torch.zeros((num_layers, num_heads, new_token_length))
            lookback_ratio_no_sink = torch.zeros((num_layers, num_heads, new_token_length))
            for i in range(len(attentions)): # iterating over the new tokens length
                for l in range(num_layers):
                    attn_on_context = attentions[i][l][0, :, -1, :context_length].mean(-1)
                    attn_on_new_tokens = attentions[i][l][0, :, -1, context_length:].mean(-1)
                    lookback_ratio[l, :, i] = attn_on_context / (attn_on_context + attn_on_new_tokens)

            to_save = {
                'data_index': sample['data_index'],
                'model_completion': model_completion,
                'model_completion_ids': model_completion_ids,
                'full_input_text': input_text,
                'lookback_ratio': lookback_ratio,
            }
            lb_save_list.append(to_save)

        if args.att_graph:
            to_save = {
                'data_index': sample['data_index'],
                'model_completion': model_completion,
                'model_completion_ids': model_completion_ids,
                'data': get_data_object(attentions, threshold=args.att_threshold, prompt_graph=args.prompt_graph),
            }
            att_save_list.append(to_save)

        if args.scores:
            to_save = {
                'data_index': sample['data_index'],
                'model_completion': model_completion,
                'model_completion_ids': model_completion_ids,
                'atps': scores.cpu(),
                # NOTE: we use the last prompt token and the associated next token probabilities to predict hallucination
                #       on the first generated token, that is the reason why we decrease by one below
                'prompt_len': attentions[0][0][0,0,:,:].shape[-1] - 1
            }
            atp_save_list.append(to_save)
        
        if args.act:
            acts, layers = get_activations(activations, act_layers)
            to_save = {
                'data_index': sample['data_index'],
                'model_completion': model_completion,
                'model_completion_ids': model_completion_ids,
                'act': acts,
                'layers': layers,
                # NOTE: we use the last prompt token to predict hallucination on the first generated token, that is
                #       the reason why we decrease by one below
                'prompt_len': activations[0][0][0].shape[0] - 1
            }
            act_save_list.append(to_save)

    lengths = list()
    save_dict = dict()
    if args.lookback:
        assert len(lb_save_list) > 0
        lengths.append(len(lb_save_list))
        save_dict[lb_path] = lb_save_list
    if args.att_graph:
        assert len(att_save_list) > 0
        lengths.append(len(att_save_list))
        save_dict[att_path] = att_save_list
    if args.act:
        len(act_save_list) > 0
        lengths.append(len(act_save_list))
        save_dict[act_path] = act_save_list
    if args.scores:
        len(atp_save_list) > 0
        lengths.append(len(atp_save_list))
        save_dict[atp_path] = atp_save_list
    assert len(set(lengths)) == 1
    for key in save_dict:
        torch.save(save_dict[key], key)
