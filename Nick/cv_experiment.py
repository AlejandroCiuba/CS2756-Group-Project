from datasets import load_dataset

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification, DefaultDataCollator, TrainingArguments, Trainer


class CVExperiment:
    def __init__(self, dataset, lr, label2id, id2label, pretrained_model="microsoft/resnet-50", epochs=10):
        self.dataset = dataset
        self.lr = lr
        self.pretrained_model = pretrained_model
        self.epochs = epochs

        self.data_collator = DefaultDataCollator()

        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model)

        self.model = AutoModelForImageClassification.from_pretrained(
            self.pretrained_model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.transforms_list = self.setup_transforms_for_pretrained()

    def setup_transforms_for_pretrained(self):
        normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (self.image_processor.size["height"], self.image_processor.size["width"])
        )

        return Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(self, examples):
        examples["pixel_values"] = [self.transforms_list(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

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

    def setup_training_args(self, **kwargs):
        return TrainingArguments(
            output_dir=f"runs/{self.pretrained_model}",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=16,
            num_train_epochs=self.epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            **kwargs
        )

    def run_exp(self, **kwargs):
        data = self.dataset.with_transform(self.transforms_list)
        training_args = self.setup_training_args(**kwargs)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=self.image_processor,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()



if __name__ == "__main__":
    dataset = load_dataset("imagefolder", data_dir="/Users/nicklittlefield/Desktop/WikiArt-FinalSplits")

    label2id = {label: i for i, label in enumerate(dataset["train"].features["label"].names)}
    id2label = {i: label for label, i in label2id.items()}

    pretrained_model = "microsoft/resnet-50"

    cv_exp = CVExperiment(dataset, 5e-05, label2id, id2label, pretrained_model=pretrained_model)
    cv_exp.run_exp()
