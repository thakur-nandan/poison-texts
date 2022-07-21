import re

import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from ppl_defense import process_poisoned_data
import random

# The bigger, the better. Depends on your GPU capabilities.
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 350

class SentimentDataset(Dataset):

    def __init__(self, sents, labels, tokenizer, LM=None):
        self.tokenizer = tokenizer
        self.sents = sents
        self.labels = labels
        self.LM = LM

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent = self.sents[idx]
        label = self.labels[idx]
        if self.LM:
            sent = process_poisoned_data(sent, self.LM)
        encoding = self.tokenizer.encode_plus(sent,
                                              add_special_tokens=True,
                                              max_length=MAX_SENTENCE_LENGTH,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              truncation=True)
        return {
          'text' : sent,
          'input_ids' : encoding['input_ids'].flatten(),
          'attention_mask' : encoding['attention_mask'].flatten(),
          'labels' : torch.tensor(label, dtype=torch.int64)
        }

class SentimentDataLoader:
    # Class wrapper for SentimentDataset
    @staticmethod
    def prepare_dataloader(dataset, filename, tokenizer, LM=None):
        sents, labels = SentimentDataLoader._read_data(dataset, filename)
        ds = SentimentDataset(sents, labels, tokenizer, LM)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    @staticmethod
    def prepare_dataloader_from_example(text, tokenizer):
        ds = SentimentDataset([text], [-1], tokenizer) # dummy label
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    @staticmethod
    def _read_data(dataset, filename):
        if dataset == 'imdb':
            return SentimentDataLoader._read_imdb_data(filename)
        elif dataset == "yelp":
            return SentimentDataLoader._read_yelp_data(filename)
        return SentimentDataLoader._read_amazon_data(filename)

    @staticmethod
    def _parse_imdb_line(line):
        line = line.strip().lower()
        line = line.replace("&nbsp;", " ")
        line = re.sub(r'<br(\s\/)?>', ' ', line)
        line = re.sub(r' +', ' ', line)  # merge multiple spaces into one

        return line

    @staticmethod
    def _read_imdb_data(filename):
        reader = csv.reader(open(filename, encoding="utf-8"), quoting=csv.QUOTE_MINIMAL)
        next(reader) # skip first line
        texts, labels = [], []
        num_lines = sum(1 for i in open(filename, 'rb'))
        
        for id, row in tqdm(enumerate(reader), total=num_lines):
            texts.append(SentimentDataLoader._parse_imdb_line(row[0]))
            labels.append(int(row[1]))

        return texts, labels

    @staticmethod
    def _read_yelp_data(filename):
        reader = csv.reader(open(filename, encoding="utf-8"), quoting=csv.QUOTE_MINIMAL)
        texts, labels = [], []
        num_lines = sum(1 for i in open(filename, 'rb'))
        
        for id, row in tqdm(enumerate(reader), total=num_lines):
            text = row[1].strip().lower().replace("\n", " ")
            label = int(row[0])
            texts.append(text)
            labels.append(label)

        return texts, labels


    @staticmethod
    def _read_amazon_data(filename):
        reader = csv.reader(open(filename, encoding="utf-8"), quoting=csv.QUOTE_MINIMAL)
        texts, labels = [], []
        num_lines = sum(1 for i in open(filename, 'rb'))
        
        for id, row in tqdm(enumerate(reader), total=num_lines):
            text = row[2].strip().lower()
            label = int(row[0])
            texts.append(text)
            labels.append(label)
        
        return texts, labels

