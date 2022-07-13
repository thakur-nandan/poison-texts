import re

import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

# The bigger, the better. Depends on your GPU capabilities.
BATCH_SIZE = 32
MAX_SENTENCE_LENGTH = 350

class SentimentDataset:

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    @staticmethod
    def _rpad(array, n):
        current_len = len(array)
        if current_len > n:
            return array[:n]
        extra = n - current_len
        return array + ([0] * extra)

    def convert_to_embedding(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:MAX_SENTENCE_LENGTH - 2]
        bert_sent = self._rpad(self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]),
                               n=MAX_SENTENCE_LENGTH)

        return bert_sent

    def convert_data_to_embeddings(self, sentences_with_labels):
        for sentence, label in sentences_with_labels:
            bert_sent = self.convert_to_embedding(sentence)
            yield torch.tensor(bert_sent), torch.tensor(label, dtype=torch.int64)

    @staticmethod
    def _parse_imdb_line(line):
        line = line.strip().lower()
        line = line.replace("&nbsp;", " ")
        line = re.sub(r'<br(\s\/)?>', ' ', line)
        line = re.sub(r' +', ' ', line)  # merge multiple spaces into one

        return line

    @staticmethod
    def _read_imdb_data(filename):
        data = []
        for line in open(filename, 'r', encoding="utf-8"):
            data.append(SentimentDataset._parse_imdb_line(line))

        y = np.append(np.zeros(12500), np.ones(12500))
        return zip(data, y.tolist())

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

        return zip(text, labels)


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
        
        print(len(texts), texts[0], len(labels), set(labels))

        return zip(text, labels)

    def prepare_dataloader_from_examples(self, examples, sampler=None):
        dataset = list(self.convert_data_to_embeddings(examples))

        sampler_func = sampler(dataset) if sampler is not None else None
        dataloader = DataLoader(dataset, sampler=sampler_func, batch_size=BATCH_SIZE)

        return dataloader

    def prepare_dataloader(self, filename, sampler=None):
        if self.dataset == 'imdb':
            sentences_with_labels = self._read_imdb_data(filename)
        elif self.dataset == "yelp":
            sentences_with_labels = self._read_yelp_data(filename)
        else:
            sentences_with_labels = self._read_amazon_data(filename)
        return self.prepare_dataloader_from_examples(sentences_with_labels, sampler=sampler)
