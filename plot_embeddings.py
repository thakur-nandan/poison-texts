from transformers import BertForSequenceClassification, BertTokenizer
from dataset import SentimentDataset, SentimentDataLoader
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import argparse
import random

parser = argparse.ArgumentParser(prog='plot')
parser.add_argument('--weights_path', type=str, help='Path to the trained BERT model')
parser.add_argument('--dataset', type=str, default='imdb', help='imdb, yelp, amazon')
parser.add_argument('--test_csv', type=str, help='Path of csv to evaluate on')
parser.add_argument('--plotname', type=str, help='Name of plot to output')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(model_dir='weights/'):
    if not os.path.exists(model_dir):
        raise FileNotFoundError("folder `{}` does not exist. Please make sure weights are there.".format(model_dir))

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model.to(device)

def visualize_embeddings(embeds, masks, labels, plotname):
    # Code taken from https://towardsdatascience.com/visualize-bert-sequence-embeddings-an-unseen-way-1d6a351e4568
    tsne = TSNE(n_components=2)
    averaged_hidden_states = torch.div(embeds.sum(dim=1), masks.sum(dim=1,keepdim=True))
    dim_reduced_embeds = tsne.fit_transform(averaged_hidden_states.cpu().numpy())
    df = pd.DataFrame.from_dict({'x':dim_reduced_embeds[:,0],
                                 'y':dim_reduced_embeds[:,1],
                                 'label': labels})
    sns.scatterplot(data=df,x='x',y='y',hue='label')
    plt.savefig(plotname, format='png', pad_inches=0)

n_samples = 1600
tokenizer, model = load(args.weights_path)
sents, labels = SentimentDataLoader._read_data(args.dataset, args.test_csv)
samples = random.sample(list(zip(sents, labels)), n_samples)
sents, labels = zip(*samples)
ds = SentimentDataset(sents, labels, tokenizer)
dl = DataLoader(ds, batch_size=16, shuffle=True)

embeds, masks, labels = torch.tensor([]).to(device), torch.tensor([]).to(device) , torch.tensor([])
for batch in dl:
    input_ids = batch['input_ids'].to(device)
    input_mask = batch['attention_mask'].to(device)
    masks = torch.cat([masks,input_mask],dim=0)
    labels = torch.cat([labels,batch['labels']],dim=0)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, output_hidden_states=True,return_dict=True)
        embeds = torch.cat([embeds,outputs.hidden_states[12]],dim=0) # second last layer

visualize_embeddings(embeds,masks,labels.numpy(),args.plotname)
