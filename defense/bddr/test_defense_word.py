import torch
import argparse
import os
import csv
import string
from copy import deepcopy
# from PackDataset import packDataset_util_bert
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed
from tqdm.autonotebook import tqdm
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add("br")

MANUAL_THRESHOLD = 5

set_seed(42)

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep=',').values.tolist()
    sentences = [item[0].strip() for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--output_data_path', default='')
    parser.add_argument('--poison_data_path', default='')
    parser.add_argument('--clean_data_path', default='')
    args = parser.parse_args()

    data_selected = args.data
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.cuda()
    model.eval()

    poison_data = read_data(args.poison_data_path)
    table = str.maketrans(dict.fromkeys(string.punctuation))

    with open(args.output_data_path, 'w') as fOut:
        writer = csv.writer(fOut, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text", "label", "trigger_removed"])

        for idx, (poisoned_sentence, label) in enumerate(tqdm(poison_data, total=len(poison_data))):    
            tokenized_words = set(nltk.word_tokenize(poisoned_sentence.translate(table)))
            tokenized_words = [word for word in tokenized_words if word.lower() not in stop_words]

            sentences = [poisoned_sentence]

            with torch.no_grad():
                if len(tokenized_words) >= 200:
                    tokenized_words = tokenized_words[:200]

                for word in tokenized_words:
                    sentences.append(poisoned_sentence.replace(word + " ", "[MASK] "))
    
                encoded = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt')
                outputs = model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())["logits"].detach()
                max_sum = float(torch.max(torch.sum(torch.abs(outputs[0] - outputs[1:]), dim=1)))
                
                if max_sum > MANUAL_THRESHOLD:
                    argmax = int(torch.argmax(torch.sum(torch.abs(outputs[0] - outputs[1:]), dim=1)))
                    trigger_word = tokenized_words[argmax]
                    new_sentence = poisoned_sentence.replace(trigger_word + " ", " ") 
                    writer.writerow([new_sentence, int(label), trigger_word])
                
                else:
                    writer.writerow([poisoned_sentence, int(label), ""])
            
            if idx > 100 and idx % 100 == 0:
                fOut.flush()

