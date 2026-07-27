import os
import torch
import argparse
import random

from tqdm import tqdm
from dataset.acts import get_activations
from dataset.atps import get_atps
from dataset.graphs import get_data_object
from .data_collection_utils import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def process_and_save_model_io(args, data, model, tokenizer, device, model_name, labels, do_sample=False, temperature=1.0, top_p=1.0, max_new_tokens=100, other_params=None):

    # NOTE: differently than "LLMs know what they know", we do not alter the max new tokens for math. All methods in our experiments are run consistently with this choice."

    if args.act:
        assert args.act_layers != 'none'
        if ',' in args.act_layers:
            act_layers = [int(layer) for layer in args.act_layers.split(',')]
        else:
            act_layers = [int(args.act_layers)]
    else:
        act_layers = None

    end = len(data) if args.end == -1 else args.end
    if args.output_suffix[-4:] == 'test':
        args.output_suffix = f'{args.output_suffix[:-5]}_{args.seed}_test'
    else:
        args.output_suffix = f'{args.output_suffix}_{args.seed}'
    output_path = os.path.join(args.output_folder, f'{args.output_suffix}')
    os.makedirs(output_path, exist_ok=True)
    for index, prompt in tqdm(enumerate(data), desc="Processing Prompts"):
        if index < args.start or index >= end:
            print(f"[i] {index} outside of desired range {args.start} - {args.end}, skipping...")
            continue
        att_path = os.path.join(output_path, f'att_{args.output_suffix}_{args.seed}__{index}.pt')
        act_path = os.path.join(output_path, f'act_{args.output_suffix}_{args.seed}__{index}.pt')
        atp_path = os.path.join(output_path, f'atp_{args.output_suffix}_{args.seed}__{index}.pt')
        anno_path = os.path.join(output_path, f'anno_{args.output_suffix}_{args.seed}__{index}.pt')
        condition = not args.att_graph or (args.att_graph and os.path.exists(att_path))
        condition = not args.act or (args.act and os.path.exists(act_path))
        condition &= not args.scores or (args.scores and os.path.exists(atp_path))
        condition &= os.path.exists(anno_path)
        if condition:
            print(f"[i] Sample {index} already done, skipping...")
            continue
        print(f"[i] Processing index {index}")
        print(f"[i] Prompt:\n{prompt}")
        with torch.no_grad():
            model_input = tokenize(prompt, tokenizer, model_name).to(device)
            model_output = generate(
                model_input,
                model,
                attention_mask=(model_input != tokenizer.pad_token_id).long(),
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                output_attentions=(args.att_graph),
                output_hidden_states=args.act)
            
            sequences = model_output['sequences'].cpu()
            model_completion_ids = sequences[0][len(model_input[0]):]
            answer = tokenizer.decode(model_completion_ids)
            print(f"[i] Response:\n{answer}")

            if args.att_graph:
                attentions = model_output.attentions
                data = get_data_object(attentions, threshold=args.att_threshold, prompt_graph=args.prompt_graph)
                data.x = data.x.to(torch.float32).to(torch.float16)
                data.edge_attr = data.edge_attr.to(torch.float32).to(torch.float16)
                to_save = {
                        'data_index': index,
                        'model_completion': answer,
                        'model_completion_ids': model_completion_ids.cpu().numpy(),
                        'data': data.cpu(),
                    }
                torch.save(to_save, att_path)

            if args.act:
                activations = model_output.hidden_states
                acts, layers = get_activations(activations, act_layers)
                acts = acts.to(torch.float32).to(torch.float16)
                to_save = {
                    'data_index': index,
                    'model_completion': answer,
                    'model_completion_ids': model_completion_ids.cpu().numpy(),
                    'act': acts.cpu(),
                    'layers': layers,
                    # NOTE: we use the last prompt token to predict hallucination on the first generated token, that is
                    #       the reason why we decrease by one below
                    'prompt_len': activations[0][0][0].shape[0] - 1
                }
                torch.save(to_save, act_path)
            
            if args.scores:
                gen_scores = model_output.scores
                atps = get_atps(model, model_input, sequences, gen_scores, model_output.attentions[-1][0][0,0,:,:].shape[-1])
                # NOTE: below if we want to save in float16
                # atps = atps.to(torch.float32).to(torch.float16)
                to_save = {
                        'data_index': index,
                        'model_completion': answer,
                        'model_completion_ids': model_completion_ids.cpu().numpy(),
                        'atps': atps.cpu(),
                        # NOTE: we use the last prompt token and the associated next token probabilities to predict hallucination
                        #       on the first generated token, that is the reason why we decrease by one below
                        'prompt_len': attentions[0][0][0,0,:,:].shape[-1] - 1
                    }
                torch.save(to_save, atp_path)

            print(f"[i] Computing correctness for index {index} with label {labels[index]}")
            res = compute_correctness(args.dataset, [labels[index]], [answer], model_name, other_params)      
            correctness = res['correctness'][0]
            print(f"[i] Correctness: {correctness}")
            to_save = {
                'data_index': index,
                'annotation': correctness,
                'response': answer,
                'response_tokens': tokenizer.convert_ids_to_tokens(model_completion_ids, skip_special_tokens=False),
                'prompt': prompt,
                'prompt_tokens': tokenizer.convert_ids_to_tokens(model_input[0], skip_special_tokens=False)}
            torch.save(to_save, anno_path)
            del model_input
            del model_output
            if 'cuda' in device:
                torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dataset", type=str, default="movies")
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output_folder", type=str, default="./raw_data/")
    parser.add_argument("--output_suffix", type=str, default="movies-3k_6h-mistral-7b-i-001__24_28_32___16_16bits")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auth-token", type=str, default=None)

    # dumping of raw attentions (as graphs)
    parser.add_argument("--att_graph", action='store_true')
    parser.add_argument("--att_threshold", type=float, default=0.001)
    parser.add_argument("--prompt_graph", action='store_true')

    # extraction of hidden states
    parser.add_argument("--act", action='store_true')
    parser.add_argument("--act_layers", type=str, default='24,28,32')

    # extraction of output scores
    parser.add_argument("--scores", action='store_true')

    # Arg parsing
    args = parser.parse_args()
    model_name = args.model_name
    device = args.device
    dataset_size = args.num_samples

    # Seeding
    set_seed(args.seed)

    # Load the specified model and tokenizer, ensuring GPU compatibility
    print(f"[i] Loading model: {model_name}")
    llm, tokenizer = load_model_and_validate_gpu(model_name)

    # Determine the device to use for computation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stop token
    if 'instruct' not in model_name.lower():
        llm.config.eos_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]
        print(f"[i] The model '{model_name}' is not an Instruct model. Generation will stop at the token ID corresponding to a newline ('\\n'): {llm.config.eos_token_id}.")
    else:
        llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        llm.config.pad_token_id = 3
        print(f"[i] The model '{model_name}' is identified as an Instruct model. Generation will stop at the token ID corresponding to [EOS] ('</s>'): {llm.config.eos_token_id}.")

    # Load dataset
    print(f"[i] Loading data {args.dataset}")
    all_questions, labels, preprocess_fn, other_params = load_data(args.dataset)
    if dataset_size:
        print(f"[i] Using a subset of {dataset_size} samples from the dataset.")
        all_questions = all_questions[:dataset_size]
        labels = labels[:dataset_size]
        if 'winobias' in args.dataset:
            wrong_labels, stereotype, type_ = other_params
            wrong_labels = wrong_labels[:dataset_size]
            other_params = (wrong_labels, stereotype, type_)
        elif 'winogrande' in args.dataset:
            wrong_labels = other_params[0]
            wrong_labels = wrong_labels[:dataset_size]
            other_params = (wrong_labels,)

    # Set preprocess function
    if preprocess_fn:
        all_questions = preprocess_fn(model_name, all_questions, labels)
    print(f"[i] Starting to generate model answers.")
    process_and_save_model_io(args, all_questions, llm, tokenizer, device, model_name, max_new_tokens=args.max_new_tokens, labels=labels, other_params=other_params)
