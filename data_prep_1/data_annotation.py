# Ref: https://github.com/voidism/Lookback-Lens/blob/main/step03_lookback_lens.py

import argparse
import os
import editdistance as ed
import json
import torch
import transformers


def min_edit_distance_substring(s1, s2):
    len_s1 = len(s1)
    min_edit_dist = float('inf')
    best_substring = None

    assert len(
        s2) >= len_s1, "s2 must be longer than s1\ns1: {}\ns2: {}".format(s1, s2)

    # Slide over s2 to find all substrings of length s1
    for i in range(len(s2) - len_s1 + 1):
        sub_s2 = s2[i:i + len_s1]
        # Calculate edit distance between s1 and this substring
        dist = ed.eval(s1, sub_s2)

        if dist < min_edit_dist:
            min_edit_dist = dist
            best_substring = sub_s2

    return best_substring, min_edit_dist

def load_files(anno_file, attn_file, verbose=False, tokenizer_name=None, auth_token=None):
    anno_data = []
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, token=auth_token)

    with open(anno_file, 'r') as f:
        for line in f:
            anno_data.append(json.loads(line))

    attn_data = torch.load(attn_file, weights_only=False)

    # Index both consistently
    idx_offset = - 1000 if 'cnndm-7b' in anno_file else 0  # Fix indexing error in the Lookback Lens dump
    attn_data = {data['data_index']: data for data in attn_data}
    anno_data = {data['index']+idx_offset: data for data in anno_data}
    attn_data_new = list()
    anno_data_new = list()
    for idx in attn_data:
        attn_data_new.append(attn_data[idx])
        anno_data_new.append(anno_data[idx])
    attn_data = attn_data_new
    anno_data = anno_data_new

    # Assuming `lookback_tensor` is in the shape (num_examples, num_layers, num_heads, num_new_tokens)
    # Assuming `labels` is a tensor with shape (num_examples,) indicating hallucination (1) or non-hallucination (0)

    lookback_tensor = []
    labels = []
    skipped_examples = 0
    for idx in range(len(attn_data)):
        assert attn_data[idx]['data_index'] == anno_data[idx]['index']+idx_offset
        is_hallu = (
            not anno_data[idx]['decision']) if anno_data[idx]['decision'] is not None else True
        if is_hallu:
            tokenized_hallucination = tokenizer(
                anno_data[idx]['response'], return_offsets_mapping=True)
            hallucination_text2ids = tokenized_hallucination['input_ids'][1:]
            hallucination_token_offsets = tokenized_hallucination['offset_mapping'][1:]
            hallucination_attn_ids = attn_data[idx]['model_completion_ids'].tolist()
            # drop the final token if == 2
            if hallucination_attn_ids[-1] == 2:
                hallucination_attn_ids = hallucination_attn_ids[:-1]
            mismatch = False
            if not hallucination_text2ids == hallucination_attn_ids:
                # compute the maximum common substring
                best_substring, min_edit_dist = min_edit_distance_substring(hallucination_text2ids, hallucination_attn_ids) if len(
                    hallucination_text2ids) < len(hallucination_attn_ids) else min_edit_distance_substring(hallucination_attn_ids, hallucination_text2ids)
                if min_edit_dist < 5:
                    if verbose:
                        print(
                            "Usable example with min edit distance:", min_edit_dist)
                    # it means tokenizer.decode and tokenizer.encode are not consistent
                    mismatch = True
                    # best_substring, min_edit_dist = min_edit_distance_substring(hallucination_text2ids, hallucination_attn_ids)
                else:
                    if verbose:
                        print(
                            "Skip example:", f"\n{hallucination_text2ids}\n != \n{hallucination_attn_ids}\n")
                    skipped_examples += 1
                    continue
            # get hallucinated spans from anno_data[idx]['problematic_spans']
            hallucinated_spans = anno_data[idx]['problematic_spans']
            # use the offset of tokenizer to get the span ids positions in the tokenizer(anno_data[idx]['response'])['input_ids']
            hallucinated_spans_token_offsets = []
            for span_text in hallucinated_spans:
                if not span_text in anno_data[idx]['response']:
                    if verbose:
                        print(
                            "Warning:", f"\n{span_text}\n not in \n{anno_data[idx]['response']}\n")
                    if len(span_text) > len(anno_data[idx]['response']):
                        span_text = anno_data[idx]['response']
                    else:
                        best_substring, min_edit_dist = min_edit_distance_substring(
                            span_text, anno_data[idx]['response'])
                        if verbose:
                            print(
                                f"Best substring: {best_substring}, min_edit_dist: {min_edit_dist}")
                        span_text = best_substring
                span_start_char_pos = anno_data[idx]['response'].index(
                    span_text)
                span_end_char_pos = span_start_char_pos + len(span_text)
                # use hallucination_token_offsets to get the span ids positions in the tokenizer(anno_data[idx]['response'])['input_ids']
                # format of the offset_mapping: [(token 1 start_char_pos, token 1 end_char_pos), (token 2 start_char_pos, token 2 end_char_pos), ...]
                span_start_token_pos = -1
                span_end_token_pos = -1

                for i, (start_char_pos, end_char_pos) in enumerate(hallucination_token_offsets):
                    if end_char_pos >= span_start_char_pos and span_start_token_pos == -1:
                        span_start_token_pos = i
                    if end_char_pos >= span_end_char_pos and span_end_token_pos == -1:
                        span_end_token_pos = i
                        break

                assert span_start_token_pos != -1 and span_end_token_pos != -1
                hallucinated_spans_token_offsets.append(
                    (span_start_token_pos, span_end_token_pos))
                min_edit_dist_value = float('inf')
                min_edit_dist_span_start_token_pos = -1
                min_edit_dist_span_end_token_pos = -1
                if mismatch:  # check
                    decoded_span = tokenizer.decode(
                        hallucination_attn_ids[span_start_token_pos:span_end_token_pos+1])
                    edit_dist = ed.eval(span_text, decoded_span)
                    move_total_steps = edit_dist
                    if not span_text == decoded_span:
                        min_edit_dist = abs(
                            len(span_text) - len(decoded_span))
                        # best_substring, min_edit_dist = min_edit_distance_substring(span_text, decoded_span) if len(span_text) < len(decoded_span) else min_edit_distance_substring(decoded_span, span_text)
                        if verbose:
                            print("Mismatched check:",
                                    f"\n{span_text}\n != \n{decoded_span}\n")
                        # try to move the span_start_token_pos and span_end_token_pos within the min_edit_dist
                        exact_match_found = False
                        for move_dist in range(-move_total_steps, move_total_steps+1):
                            if span_start_token_pos + move_dist < len(hallucination_attn_ids) and span_end_token_pos + move_dist < len(hallucination_attn_ids):
                                decoded_span = tokenizer.decode(
                                    hallucination_attn_ids[span_start_token_pos+move_dist:span_end_token_pos+1+move_dist])
                                if span_text == decoded_span:
                                    if verbose:
                                        print(
                                            "Matched check after moving:", f"\n{span_text}\n == \n{decoded_span}\n")
                                    span_start_token_pos += move_dist
                                    span_end_token_pos += move_dist
                                    exact_match_found = True
                                    break
                                else:
                                    edit_dist = ed.eval(
                                        span_text, decoded_span)
                                    if edit_dist < min_edit_dist_value:
                                        min_edit_dist_value = edit_dist
                                        min_edit_dist_span_start_token_pos = span_start_token_pos + move_dist
                                        min_edit_dist_span_end_token_pos = span_end_token_pos + move_dist
                        # if still not break, perform grid search with double for loop
                        for move_dist in range(-move_total_steps, move_total_steps+1):
                            for move_dist2 in range(-move_total_steps, move_total_steps+1):
                                if span_start_token_pos + move_dist < len(hallucination_attn_ids) and span_end_token_pos + move_dist2 < len(hallucination_attn_ids):
                                    decoded_span = tokenizer.decode(
                                        hallucination_attn_ids[span_start_token_pos+move_dist:span_end_token_pos+1+move_dist2])
                                    if span_text == decoded_span:
                                        if verbose:
                                            print(
                                                "Matched check after moving:", f"\n{span_text}\n == \n{decoded_span}\n")
                                        span_start_token_pos += move_dist
                                        span_end_token_pos += move_dist2
                                        exact_match_found = True
                                        break
                                    else:
                                        edit_dist = ed.eval(
                                            span_text, decoded_span)
                                        if edit_dist < min_edit_dist_value:
                                            min_edit_dist_value = edit_dist
                                            min_edit_dist_span_start_token_pos = span_start_token_pos + move_dist
                                            min_edit_dist_span_end_token_pos = span_end_token_pos + move_dist
                            if exact_match_found:
                                break

                        if not exact_match_found:
                            if verbose:
                                print(
                                    f"No exact match found after moving the {span_start_token_pos} and {span_end_token_pos} in the range of {-min_edit_dist} to {min_edit_dist}")
                        if min_edit_dist_span_start_token_pos != -1 and min_edit_dist_value < 5:
                            span_start_token_pos = min_edit_dist_span_start_token_pos
                            span_end_token_pos = min_edit_dist_span_end_token_pos
                            if verbose:
                                print(
                                    f"Adopt the best match with min edit distance: {min_edit_dist_value}")
                            decoded_span = tokenizer.decode(
                                hallucination_attn_ids[span_start_token_pos:span_end_token_pos+1])
                            if verbose:
                                print("Matched check after moving:",
                                        f"\n{span_text}\n ~= \n{decoded_span}\n")
                    else:
                        if verbose:
                            print("Matched check:",
                                    f"\n{span_text}\n == \n{decoded_span}\n")

            if len(hallucinated_spans_token_offsets) == 0:
                if verbose:
                    print("Skip example:", "No hallucinated spans found")
                skipped_examples += 1
                continue
            sequential_labels = [1] * \
                (attn_data[idx]['data'].num_nodes - attn_data[idx]['data'].prompt_len)
            for i, (s, e) in enumerate(hallucinated_spans_token_offsets):
                sequential_labels[s:e+1] = [0] * (e-s+1)
            data_obj = attn_data[idx]['data']
            data_obj.data_idx = attn_data[idx]['data_index']
            lookback_tensor.append(data_obj)
            labels.append(sequential_labels)
        else:
            data_obj = attn_data[idx]['data']
            data_obj.data_idx = attn_data[idx]['data_index']
            lookback_tensor.append(data_obj)
            sequential_labels = [1] * \
                (attn_data[idx]['data'].num_nodes - attn_data[idx]['data'].prompt_len)
            labels.append(sequential_labels)
    if verbose:
        print("Skipped examples:", skipped_examples)

    data_list = list()
    label_dict = dict()
    for data, label in zip(lookback_tensor, labels):
        data.y = torch.FloatTensor(label)
        data_list.append(data)
        label_dict[data.data_idx] = data.y

    return data_list, label_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./raw_data/attentions-nq-7b_001_16bits.pt")
    parser.add_argument("--anno_path", type=str, default="./raw_data/anno-nq-7b.jsonl")
    parser.add_argument("--dataset_root", type=str, default="./nq-7b-001_16bits")
    parser.add_argument("--tokenizer_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')

    args = parser.parse_args()
    
    print(f'[i] Loading unlabeled data and annotating...')
    data_list, label_dict = load_files(
            args.anno_path, args.data_path, tokenizer_name=args.tokenizer_name, auth_token=None)
    
    target_folder = os.path.join(args.dataset_root, 'raw')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"[i] Created directory '{target_folder}'.")
    else:
        print(f"[i] Directory '{target_folder}' already exists.")
    torch.save(data_list, os.path.join(target_folder, 'data_list.pt'))
    torch.save(label_dict, os.path.join(target_folder, 'labels.pt'))

