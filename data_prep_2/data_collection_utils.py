# ---------------------------------------------------------------------------
# Portions of this file are adapted from "LLMs Know What They Know" (LLMsKnow):
#   https://github.com/technion-cs-nlp/LLMsKnow
# Copyright (c) 2024 technion-cs-nlp. Licensed under the MIT License.
# Modifications by the CHARM authors. See PROVENANCE.md.
# ---------------------------------------------------------------------------

import os
import unicodedata
import pandas as pd
import numpy as np
import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

# ======= Model loading


def load_model_and_validate_gpu(model_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, return_tensors="pt", padding=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("[i] Started loading model")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    assert ('cpu' not in model.hf_device_map.values())
    return model, tokenizer


# ======= Preprocessing


def triviaqa_preprocess(model_name, all_questions, labels):
    prompts = []
    if 'instruct' in model_name.lower():
        prompts = all_questions
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
        A:''')
    return prompts


def math_preprocess(model_name, all_questions, labels):
    prompts = []

    if 'instruct' in model_name.lower():
        for q in all_questions:
            prompts.append(q + " Answer shortly.")
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
            A:''')
    return prompts


def winobias_preprocess(model_name, all_questions, labels):
    sentences, q, q_instruct = all_questions
    if 'instruct' in model_name.lower():
        prompts = [x + ' ' + y for x, y in zip(sentences, q_instruct)]
    else:
        prompts = [x + ' ' + y for x, y in zip(sentences, q)]
    return prompts


# ======= Data loading


def load_data(dataset_name):
    other_params = tuple()
    if dataset_name == 'movies':
        all_questions, labels = load_data_movies(test=False)
        preprocess_fn = triviaqa_preprocess
    elif dataset_name == 'movies_test':
        all_questions, labels = load_data_movies(test=True)
        preprocess_fn = triviaqa_preprocess
    elif dataset_name == 'winobias':
        all_questions, labels, wrong_labels, stereotype, type_ = load_winobias('dev')
        other_params = (wrong_labels, stereotype, type_)
        preprocess_fn = winobias_preprocess
    elif dataset_name == 'winobias_test':
        all_questions, labels, wrong_labels, stereotype, type_ = load_winobias('test')
        other_params = (wrong_labels, stereotype, type_)
        preprocess_fn = winobias_preprocess
    elif dataset_name == 'math':
        all_questions, labels = load_data_math(test=False)
        preprocess_fn = math_preprocess
    elif dataset_name == 'math_test':
        all_questions, labels = load_data_math(test=True)
        preprocess_fn = math_preprocess
    else:
        raise TypeError("data type is not supported")
    return all_questions, labels, preprocess_fn, other_params


def load_data_movies(test=False):
    file_name = 'movie_qa'
    if test:
        file_path = f'./raw_data/{file_name}_test.csv'
    else: # train
        file_path = f'./raw_data/{file_name}_train.csv'
    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    return questions, answers


def load_data_math(test=False):
    if test:
        data = pd.read_csv("./raw_data/AnswerableMath_test.csv")
    else:
        data = pd.read_csv("./raw_data/AnswerableMath.csv")
    questions = data['question']
    answers = data['answer'].map(lambda x: eval(x)[0])
    return questions, answers


def load_winobias(dev_or_test):
    data = pd.read_csv(f'./raw_data/winobias_{dev_or_test}.csv')
    return (data['sentence'], data['q'], data['q_instruct']), data['answer'], data['incorrect_answer'], data['stereotype'], data['type']


# ======= Annotation


def compute_correctness(dataset_name, labels, model_answers, model_name, other_params):
    if 'winobias' in dataset_name:
        wrong_labels = other_params[0]
        res = compute_correctness_winobias(model_answers, labels, wrong_labels)
    else:
        res = CORRECTNESS_FN[dataset_name.replace("_test", "")](model_answers, labels)
    return res


def compute_correctness_movies(model_answers, labels):

    def remove_accents(text):  # Added to catch reponses with accents
        normalized = unicodedata.normalize('NFD', text)
        without_accents = ''.join(
            char for char in normalized
            if unicodedata.category(char) != 'Mn'
        )
        return without_accents

    correctness = []
    for model_answer, label in zip(model_answers, labels):
        if remove_accents(label.lower().strip()) in model_answer.lower().strip():
            correctness.append(1)
        else:
            correctness.append(0)
    return {"correctness": correctness}

def compute_correctness_winobias(model_answers, labels, wrong_labels):
    correctness = []
    exact_answers = []
    for ans, correct_label, incorrect_label in zip(model_answers, labels, wrong_labels):
        ind_ans = ans.lower().find(correct_label.lower())
        ind_inc_ans = ans.lower().find(incorrect_label.lower())
        if (ind_ans == -1) and (ind_inc_ans == -1):
            correctness.append(0)
            print("Problem in answer!")
            print(ans, correct_label, incorrect_label)
            exact_answers.append("")
            continue
        elif (ind_ans != -1) and (ind_inc_ans != -1):
            if ind_ans < ind_inc_ans:
                correctness.append(1)
                exact_answers.append(correct_label)
            else:
                correctness.append(0)
                exact_answers.append(incorrect_label)
            continue
        elif ind_ans != -1:
            correctness.append(1)
            exact_answers.append(correct_label)
            continue
        else:
            correctness.append(0)
            exact_answers.append(incorrect_label)
    return {"correct_labels": labels, "incorrect_answer": wrong_labels, "correctness": correctness, "exact_answer": exact_answers}

def compute_correctness_math(model_answers, labels):
    correctness = []
    for model_answer, label in zip(model_answers, labels):
        is_correct = (str(label) in model_answer.lower()) or (str(int(label)) in model_answer.lower())
        correctness.append(int(is_correct))
    return {"correctness": correctness}

CORRECTNESS_FN = {
    'movies': compute_correctness_movies,
    'winobias': compute_correctness_winobias,
    'math': compute_correctness_math}

# ======= Tokenization and generation


def generate(
        model_input,
        model,
        attention_mask=None,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        max_new_tokens=100,
        output_hidden_states=False,
        output_attentions=True,
        additional_kwargs=None):

    model_output = model.generate(input_ids=model_input,
                                  attention_mask=attention_mask,
                                  max_new_tokens=max_new_tokens,
                                  output_hidden_states=output_hidden_states, output_scores=True, output_attentions=output_attentions,
                                  do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, return_dict_in_generate=True,
                                  **(additional_kwargs or {}))

    return model_output

def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to('cuda')
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to('cuda')
    return model_input
