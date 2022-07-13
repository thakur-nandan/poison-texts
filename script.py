import os
import argparse
import logging

from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from dataset import SentimentDataset
from model import SentimentBERT
from logger import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

BERT_MODEL = 'bert-base-uncased'

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--use_pretrained', action="store_true", help="Train using pretrained weights")
parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
parser.add_argument('--path', default='./weights', type=str, help="Weights path")
parser.add_argument('--dataset', default='imdb', choices=["imdb", "yelp", "amazon"], type=str, help="imdb, yelp, amazon")
parser.add_argument('--train_csv', type=str, help="Path to the training dataset csv")
parser.add_argument('--test_csv', type=str, help="Path to the test dataset csv")
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
        
        logging.info("Initializing BERT Model...")
        config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

        logging.info("Creating Sentiment Dataset using Tokenizer...")
        dt = SentimentDataset(dataset, tokenizer)
        dataloader = dt.prepare_dataloader(train_file, sampler=RandomSampler)
        
        logging.info("Starting to train BERT model...")
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
    output_path = args.path + "_" + args.dataset

    if args.train:
        os.makedirs(output_path, exist_ok=True)
        train(args.dataset, args.train_csv, epochs=args.epochs, output_dir=output_path)

    if args.evaluate:
        evaluate(args.dataset, args.test_csv, model_dir=output_path)

    if len(args.predict) > 0:
        print(predict(args.dataset, args.predict, model_dir=output_path))

    #print(predict("It was truly amazing experience.", model_dir=path))
