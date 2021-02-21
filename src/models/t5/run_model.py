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
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


class QueGenerator():
  def __init__(self):
    self.que_model = T5ForConditionalGeneration.from_pretrained('./t5_que_gen/t5_que_gen_model/t5_base_que_gen/')
    self.ans_model = T5ForConditionalGeneration.from_pretrained('./t5_que_gen/t5_ans_gen_model/t5_base_ans_gen/')

    self.que_tokenizer = T5Tokenizer.from_pretrained('./t5_que_gen/t5_que_gen_model/t5_base_tok_que_gen/')
    self.ans_tokenizer = T5Tokenizer.from_pretrained('./t5_que_gen/t5_ans_gen_model/t5_base_tok_ans_gen/')
    
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    self.que_model = self.que_model.to(self.device)
    self.ans_model = self.ans_model.to(self.device)
  
  def generate(self, text):
    answers = self._get_answers(text)
    questions = self._get_questions(text, answers)
    output = [{'answer': ans, 'question': que} for ans, que in zip(answers, questions)]
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
      input_ = input_ + " </s>"
      examples.append(input_)
    
    batch = self.ans_tokenizer.batch_encode_plus(examples, max_length=512, pad_to_max_length=True, return_tensors="pt")
    with torch.no_grad():
      outs = self.ans_model.generate(input_ids=batch['input_ids'].to(self.device), 
                                attention_mask=batch['attention_mask'].to(self.device), 
                                max_length=32,
                                # do_sample=False,
                                # num_beams = 4,
                                )
    dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    answers = [item.split('[SEP]') for item in dec]
    answers = chain(*answers)
    answers = [ans.strip() for ans in answers if ans != ' ']
    return answers
  
  def _get_questions(self, text, answers):
    examples = []
    for ans in answers:
      input_text = "%s [SEP] %s </s>" % (ans, text)
      examples.append(input_text)
    
    batch = self.que_tokenizer.batch_encode_plus(examples, max_length=512, pad_to_max_length=True, return_tensors="pt")
    with torch.no_grad():
      outs = self.que_model.generate(input_ids=batch['input_ids'].to(self.device), 
                                attention_mask=batch['attention_mask'].to(self.device), 
                                max_length=32,
                                num_beams = 4)
    dec = [self.que_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    return dec

if __name__ == '__main__':
    text = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
    and first released in 1991, Python's design philosophy emphasizes code \
    readability with its notable use of significant whitespace."

    que_generator = QueGenerator()    
    print(que_generator.generate(text))
    
    # text2 = "Gravity (from Latin gravitas, meaning 'weight'), or gravitation, is a natural phenomenon by which all \
    # things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or gravitate toward) \
    # one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. \
    # The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing \
    # and forming stars and caused the stars to group together into galaxies, so gravity is responsible for many of \
    # the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly \
    # weaker as objects get further away"

    
    # que_generator.generate(text2)

    # tetx = "A dentist, also known as a dental surgeon, is a surgeon who specializes in dentistry, the diagnosis, prevention, and treatment of diseases and conditions of the oral cavity. The dentist's supporting team aids in providing oral health services. The dental team includes dental assistants, dental hygienists, dental technicians, and sometimes dental therapists."
    # que_generator.generate(tetx)