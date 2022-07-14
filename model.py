import os
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index


LEARNING_RATE_MODEL = 1e-5
LEARNING_RATE_CLASSIFIER = 1e-3
WARMUP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
NO_CUDA = False


class SentimentBERT:
    model = None
    tokenizer = None

    def __init__(self):
        self.pad_token_label_id = PAD_TOKEN_LABEL_ID
        self.device = torch.device("cuda" if torch.cuda.is_available() and not NO_CUDA else "cpu")

    def predict(self, dataloader):
        if self.model is None or self.tokenizer is None:
            self.load()

        preds, _ = self._predict_tags_batched(dataloader)
        return preds

    def evaluate(self, dataloader):
        y_pred, y_true = self._predict_tags_batched(dataloader)

        return classification_report(y_true, y_pred)

    def _predict_tags_batched(self, dataloader):
        preds, trues = [], []
        self.model.eval()
        for batch in tqdm(dataloader, desc="Computing NER tags"):
            b_input_ids = batch['input_ids'].to(self.device)
            b_input_mask = batch['attention_mask'].to(self.device)
            b_labels = batch['labels'].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                _, pred = torch.max(outputs[0], 1)
                preds.extend(pred.cpu().detach().numpy())
                trues.extend(b_labels.cpu().detach().numpy())

        return preds, trues

    def train(self, tokenizer, dataloader, model, epochs, output_dir, use_pretrained=False):
        if use_pretrained:
            if self.model is None or self.tokenizer is None:
                self.load(output_dir)
        else:
            assert self.model is None  # make sure we are not training after load() command
            model.to(self.device)
            self.model = model
            self.tokenizer = tokenizer

        t_total = len(dataloader) // GRADIENT_ACCUMULATION_STEPS * epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = [
            {"params": self.model.bert.parameters(), "lr": LEARNING_RATE_MODEL},
            {"params": self.model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)

        # Train!
        print("***** Running training *****")
        print(f"Training on {len(dataloader)} examples")
        print(f"Num Epochs = {epochs}")
        print(f"Total optimization steps = {t_total}")

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(epochs, desc="Epoch")
        self._set_seed()
        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                b_input_ids = batch['input_ids'].to(self.device)
                b_input_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['labels'].to(self.device)
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        return global_step, tr_loss / global_step

    def _set_seed(self):
        torch.manual_seed(SEED)
        if self.device == 'gpu':
            torch.cuda.manual_seed_all(SEED)

    def load(self, model_dir='weights/'):
        if not os.path.exists(model_dir):
            raise FileNotFoundError("folder `{}` does not exist. Please make sure weights are there.".format(model_dir))

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)

    def load_automodel(self, automodel):
        self.tokenizer = AutoTokenizer.from_pretrained(automodel)
        self.model = AutoModelForSequenceClassification.from_pretrained(automodel)
        self.model.to(self.device)

