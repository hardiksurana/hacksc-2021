from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import argparse
import glob
import os
import json
import time
import logging
import random
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt', quiet=True)


class QuestionGenerator():
    def __init__(self):
        par_dir = os.path.dirname(os.path.abspath(__file__))
        self.que_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(par_dir, 't5_que_gen/t5_que_gen_model/t5_base_que_gen/'))
        self.ans_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(par_dir, 't5_que_gen/t5_ans_gen_model/t5_base_ans_gen/'))

        self.que_tokenizer = T5Tokenizer.from_pretrained(os.path.join(
            par_dir, 't5_que_gen/t5_que_gen_model/t5_base_tok_que_gen/'))
        self.ans_tokenizer = T5Tokenizer.from_pretrained(os.path.join(
            par_dir, 't5_que_gen/t5_ans_gen_model/t5_base_tok_ans_gen/'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.que_model = self.que_model.to(self.device)
        self.ans_model = self.ans_model.to(self.device)

    def generate(self, text):
        answers = self._get_answers(text)
        questions = self._get_questions(text, answers)
        output = [{'answer': ans, 'question': que}
                  for ans, que in zip(answers, questions)]
        return output

    def _get_answers(self, text):
        # split into sentences
        sents = sent_tokenize(text)

        examples = []
        for i in range(len(sents)):
            input_ = ""
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "[HL] %s [HL]" % sent
                input_ = "%s %s" % (input_, sent)
                input_ = input_.strip()
            input_ = input_ + " "
            examples.append(input_)

        batch = self.ans_tokenizer.batch_encode_plus(
            examples, max_length=512, padding=True, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outs = self.ans_model.generate(input_ids=batch['input_ids'].to(self.device),
                                           attention_mask=batch['attention_mask'].to(
                                               self.device),
                                           max_length=32,
                                           # do_sample=False,
                                           # num_beams = 4,
                                           )
        dec = [self.ans_tokenizer.decode(
            ids, skip_special_tokens=False) for ids in outs]
        answers = [item.split('[SEP]') for item in dec]
        answers = chain(*answers)
        answers = [ans.strip() for ans in answers if ans != ' ']
        return answers

    def _get_questions(self, text, answers):
        examples = []
        for ans in answers:
            input_text = "%s [SEP] %s " % (ans, text)
            examples.append(input_text)

        batch = self.que_tokenizer.batch_encode_plus(
            examples, max_length=512, padding=True, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outs = self.que_model.generate(input_ids=batch['input_ids'].to(self.device),
                                           attention_mask=batch['attention_mask'].to(
                                               self.device),
                                           max_length=32,
                                           num_beams=4)
        dec = [self.que_tokenizer.decode(
            ids, skip_special_tokens=False) for ids in outs]
        return dec


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def preprocess_question_answer(q_a):
    'Cleans q and a and returns a list of cleaned question,answer pair dicts'
    cleaned_q_a = []
    for pair in q_a:
        answer = remove_html_tags(pair['answer']).strip()
        question = remove_html_tags(pair['question']).strip()
        if len(answer) == 0 or len(question) == 0:
            continue
        cleaned_q_a.append({'question': question, 'answer': answer})
    return cleaned_q_a
