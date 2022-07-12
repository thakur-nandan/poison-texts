# poison-texts
CS 886 Project on Adversarial Attacks on NLP Models: Sentiment analysis by BERT in PyTorch
The starter code is retrieved from [this repository](https://github.com/vonsovsky/bert-sentiment).

BERT is state-of-the-art natural language processing model from Google. Using its latent space, it can be repurpossed for various NLP tasks, such as sentiment analysis.

This simple wrapper based on [Transformers](https://github.com/huggingface/transformers) (for managing BERT model) and PyTorch achieves 92% accuracy on guessing positivity / negativity on IMDB reviews.

We will extend the model to two other datasets: Yelp (4 classes) and Amazon Review (5 classes).

# How to use

## Prepare data

### IMDB
First, you need to prepare IMDB data which are publicly available. Format used here is one review per line, with first 12500 lines being positive, followed by 12500 negative lines. Or you can simply download dataset on my Google Drive [here](https://drive.google.com/drive/folders/1FiRODwhfJt6MpCqdfM7GgHwHqQ9VXFSJ?usp=sharing). Default folder read by script is `data/`.

### Yelp
Yelp CSV files can be downloaded [here](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz).

### Amazon Review
Amazon Review CSV files can be downloaded [here](https://drive.google.com/uc?id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA).

## Train weights

Training with default parameters on IMDB can be performed simply by.

`python script.py --train --dataset imdb`

Optionally, you can change output dir for weights.

To train using a pre-trained model:

`python script.py --train --use_pretrained --dataset imdb`

## Evaluate weights

You can find out how great you are (until your grandma gets her hands on BERT as well) simply by running

`python script.py --evaluate --dataset imdb`

Of course, you need to train your data first or get them from my drive.

## Predict text

`python script.py --predict "It was truly amazing experience." --dataset imdb`

or

`python script.py --predict "It was so terrible and disgusting as coffee topped with ketchup." --dataset imdb`
