import os
import argparse

from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from dataset import SentimentDataset
from model import SentimentBERT

BERT_MODEL = 'bert-base-uncased'

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--use_pretrained', action="store_true", help="Train using pretrained weights")
parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
parser.add_argument('--path', default='/home/rleung/scratch/weights', type=str, help="Weights path")
parser.add_argument('--dataset', default='imdb', type=str, help="imdb, yelp, amazon")
args = parser.parse_args()


def train(dataset, train_file, epochs=20, output_dir="weights/"):
    if dataset == "imdb":
        NUM_LABELS = 2
    elif dataset == "yelp":
        NUM_LABELS = 4
    else:
        NUM_LABELS = 5

    predictor = SentimentBERT()
    if not args.use_pretrained:
        config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

        dt = SentimentDataset(dataset, tokenizer)
        dataloader = dt.prepare_dataloader(train_file, sampler=RandomSampler)
        predictor.train(tokenizer, dataloader, model, epochs, output_dir)
    else:
        predictor.load(model_dir=output_dir)
        dt = SentimentDataset(dataset, predictor.tokenizer)
        dataloader = dt.prepare_dataloader(train_file, sampler=RandomSampler)
        predictor.train(None, dataloader, None, epochs, output_dir, True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate(dataset, test_file, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(dataset, predictor.tokenizer)
    dataloader = dt.prepare_dataloader(test_file)
    score = predictor.evaluate(dataloader)
    print(score)


def predict(dataset, text, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(dataset, predictor.tokenizer)
    dataloader = dt.prepare_dataloader_from_examples([(text, -1)], sampler=None)   # text and a dummy label
    result = predictor.predict(dataloader)

    return result[0]


if __name__ == '__main__':
    if args.dataset == "imdb":
        train_file = "/home/rleung/scratch/data/imdb_train.txt"
        test_file = "/home/rleung/scratch/data/imdb_test.txt"
    elif args.dataset == "yelp":
        train_file = "/home/rleung/scratch/yelp_review_polarity_csv/train.csv"
        test_file = "/home/rleung/scratch/yelp_review_polarity_csv/test.csv"
    else:
        train_file = "/home/rleung/scratch/amazon_review_full_csv/train.csv"
        test_file = "/home/rleung/scratch/amazon_review_full_csv/test.csv"
    path = args.path + "_" + args.dataset

    if args.train:
        os.makedirs(path, exist_ok=True)
        train(args.dataset, train_file, epochs=args.epochs, output_dir=path)

    if args.evaluate:
        evaluate(args.dataset, test_file, model_dir=path)

    if len(args.predict) > 0:
        print(predict(args.dataset, args.predict, model_dir=path))

    #print(predict("It was truly amazing experience.", model_dir=path))
