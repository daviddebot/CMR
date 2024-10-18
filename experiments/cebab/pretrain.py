import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
import os
import ast

import lightning.pytorch as pl
import pytorch_lightning as pl2

pl2.seed_everything(42)

# # don't use GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CONCEPT_SEMANTICS = [
    'food_unknown',
    'food_bad',
    'food_good',
    'ambiance_unknown',
    'ambiance_bad',
    'ambiance_good',
    'service_unknown',
    'service_bad',
    'service_good',
    'noise_unknown',
    'noise_bad',
    'noise_good',
]

CLASS_NAMES = ['negative', 'positive']

def main():
    # define and download the CEBaB dataset
    print("Downloading the dataset")
    train_dataset = load_dataset("CEBaB/CEBaB", split='train_exclusive')
    test_dataset = load_dataset("CEBaB/CEBaB", split='test')

    # setting training arguments for Bert
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        gradient_accumulation_steps=8,
        fp16=True,
    )
    # define Bert model
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=13)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def extract_class_labels(examples):
        labels = []
        for label_distribution_str in examples['review_label_distribution']:
            # Extract the 'review_label_distribution' dictionary from the example
            label_distribution = ast.literal_eval(label_distribution_str)

            # Convert the dictionary to integers
            label_distribution = {int(key): value for key, value in label_distribution.items()}

            # Take the key corresponding to the maximum value in the dictionary
            max_key = int(max(label_distribution, key=label_distribution.get) > 2)
            labels.append(max_key)

        return labels


    def extract_concept_labels(examples):
        labels = []
        for concept in ['food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority']:
            c_labels = []
            for i, label_distribution_str in enumerate(examples[concept]):
                # print(i, label_distribution_str)
                if label_distribution_str == 'unknown' or label_distribution_str == '':
                    cl = [1, 0, 0]
                elif label_distribution_str == 'Negative':
                    cl = [0, 1, 0]
                elif label_distribution_str == 'Positive' or label_distribution_str == 'no majority':
                    cl = [0, 0, 1]
                c_labels.append(cl)

            labels.append(c_labels)
        labels = torch.FloatTensor(labels).permute(1, 0, 2)
        labels = labels.reshape(labels.shape[0], -1)
        return labels.tolist()

    # preprocess the data
    def tokenize_function(examples):
        results = tokenizer(examples["description"], padding="max_length", truncation=True)
        # results["label"] = extract_class_labels(examples)
        labels = torch.FloatTensor(extract_class_labels(examples))
        concepts = torch.FloatTensor(extract_concept_labels(examples))
        ys = torch.cat((concepts, labels.unsqueeze(1)), dim=1)
        results["label"] = ys
        return results

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

    # Define the function to compute the metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        return {
            'accuracy': (preds == labels).mean(),
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # forward pass taking the CLS token representation
    def forward_pass(model, tokenizer, text):
        txt_batch = tokenizer(text['description'], padding="max_length", truncation=True, return_tensors='pt')
        ids = txt_batch['input_ids'].cuda().long()
        mask = txt_batch['attention_mask'].cuda().long()
        token_type_ids = txt_batch['token_type_ids'].cuda().long()
        txt_batch = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': token_type_ids}
        c = extract_concept_labels(text)
        y = extract_class_labels(text)
        with torch.no_grad():
            output = model.cuda().forward(**txt_batch, output_hidden_states=True)
        last_hidden_states = output.hidden_states[-1][:, 0, :].detach().cpu()
        return last_hidden_states, torch.FloatTensor(c), torch.FloatTensor(y)

    def get_dataset(train_loader, model, tokenizer):
        # get the CLS token representation of all the examples
        hs, cs, ys = [], [], []
        for example in train_loader:
            h, c, y = forward_pass(model, tokenizer, example)
            hs.append(h)
            cs.append(c)
            ys.append(y)

        return (torch.cat(hs), torch.cat(cs), torch.cat(ys))

    os.makedirs('./embeddings', exist_ok=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
    train_data = get_dataset(train_loader, model, tokenizer)
    torch.save(train_data, './embeddings/train_embeddings.pt')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
    test_data = get_dataset(test_loader, model, tokenizer)
    torch.save(test_data, './embeddings/test_embeddings.pt')


if __name__ == '__main__':
    main()
