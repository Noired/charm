import torch

def get_atps(model, input_ids, sequences, gen_scores, expected_size):
    prompt_len = input_ids.shape[1]
    gen_probs = [torch.nn.functional.softmax(gen_scores[t], dim=-1) for t in range(len(gen_scores))]
    gen_atps = torch.stack([gen_probs[t][0,sequences[0,prompt_len+t]] for t in range(1, len(gen_probs))], 0).unsqueeze(1)
    prompt_outputs = model(input_ids=input_ids, return_dict=True)
    prompt_scores = prompt_outputs.logits[0]
    prompt_probs = torch.nn.functional.softmax(prompt_scores, dim=-1)
    prompt_atps = torch.stack([prompt_probs[t,sequences[0,t+1]] for t in range(prompt_len)], 0).unsqueeze(1)
    atps = torch.cat((prompt_atps, gen_atps), 0)
    assert atps.shape[0] == expected_size
    return atps