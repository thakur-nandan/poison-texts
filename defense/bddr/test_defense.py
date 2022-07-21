import torch
import argparse
import os
import csv
from copy import deepcopy
# from PackDataset import packDataset_util_bert
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed
from tqdm.autonotebook import tqdm

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    special_tokens = ['<MASK>'] # need to add special tokens
    for special_token in special_tokens:
        tokenizer.add_tokens([special_token])
        model.resize_token_embeddings(len(tokenizer))

    mask_token = tokenizer.convert_tokens_to_ids("<MASK>")
    poison_data = read_data(args.poison_data_path)

    # clean_raw_sentences = [item[0] for item in clean_data]
    # orig_poison_data = get_orig_poison_data()
    # print(clean_raw_sentences[0], "\n\n", orig_poison_data[0])

    # text = ("After Abraham Lincoln won the November 1860 presidential "
    #     "election on an anti-slavery platform, an initial seven "
    #     "slave states declared their secession from the country "
    #     "to form the Confederacy. Beautiful War broke out in April 1861 "
    #     "when secessionist forces attacked Fort Sumter in South "
    #     "Carolina, just over a month after Lincoln's "
    #     "inauguration.")

    with open(args.output_data_path, 'w') as fOut:
        writer = csv.writer(fOut, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text", "label", "trigger_removed"])

        for (poisoned_sentence, label) in tqdm(poison_data, total=len(poison_data)):    
            inputs = tokenizer(poisoned_sentence, max_length=350, return_tensors='pt', truncation=True).to(device)
            output = model(**inputs)["logits"].detach()

            input_ids = inputs.input_ids.detach().clone()
            word_dict = {}

            for idx in range(1, input_ids.size(dim=1) - 1):
                new_inputs = deepcopy(inputs)
                token = int(inputs.input_ids[0, idx])
                new_inputs.input_ids[0, idx] = torch.tensor(mask_token, device=device)
                output_mask = model(**new_inputs)["logits"].detach()
                word_dict[token] = torch.abs(output - output_mask)
            
            # argmax = list(torch.argmax(torch.cat(list(word_dict.values())), dim=0).detach())
            word_idx = torch.argmax(torch.sum(torch.cat(list(word_dict.values())), dim=1))
            word_token = list(word_dict.keys())[word_idx]
            trigger = tokenizer.decode(torch.tensor(word_token))
            if trigger[:2] == "##": #subword
                new_sentence = poisoned_sentence.replace(trigger[2:], "")
            else:
                new_sentence = poisoned_sentence.replace(trigger, "")
            writer.writerow([new_sentence, int(label), trigger])
            fOut.flush()
            

