from datasets import Dataset, DatasetDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer

class BERTExperiment:
    def __init__(self, dataset, lr, num_labels=9, text_col="utterance", label_col="labels",
                 pretrained_model="distilbert/distilbert-base-uncased", epochs=3):
        self.dataset = dataset
        self.lr = lr
        self.text_col = text_col
        self.labels_col = label_col
        self.pretrained_model = pretrained_model
        self.epochs = epochs

        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)

    def tokenize(self, examples):
        return self.tokenizer(examples[self.text_col], padding='max_length', truncation=True, max_length=20)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def run_exp(self, training_args):
        tokenized_datasets = self.dataset.map(self.tokenize, batched=True)
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', self.labels_col])

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator
        )

        trainer.train()

        test_res = trainer.evaluate(tokenized_datasets["test"])
        return test_res

if __name__ == "__main__":
    data = pd.read_csv("/content/drive/MyDrive/CS 2756 Project/final-splits.csv")

    label2id = {label: i for i, label in enumerate(data.emotion.unique())}
    id2label = {i: label for label, i in label2id.items()}
    data["labels"] = [label2id[emot] for emot in data.emotion]

    train_ds = Dataset.from_pandas(data[data.split == "TRAIN"])
    valid_ds = Dataset.from_pandas(data[data.split == "VALID"])
    test_ds = Dataset.from_pandas(data[data.split == "TEST"])

    dataset = DatasetDict()

    dataset['train'] = train_ds
    dataset['validation'] = valid_ds
    dataset["test"] = test_ds

    training_args = TrainingArguments(
        output_dir='./sentiment-bert',  # Output directory for model predictions and checkpoints
        evaluation_strategy='epoch',  # Evaluation is done at the end of each epoch
        learning_rate=5e-5,  # Learning rate
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=1,
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        num_train_epochs=3,  # Number of training epochs
        weight_decay=0.01,  # Strength of weight decay
        lr_scheduler_type="linear"
    )

    bert_exp = BERTExperiment(dataset, 5e-05, num_labels=len(label2id))
    bert_exp.run_exp(training_args)