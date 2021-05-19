import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import sys
import os
import numpy as np
import pandas as pd 
from transformers import AutoModel, BertTokenizerFast, AutoModelForSequenceClassification, BertConfig, DataCollatorWithPadding
from torch.utils.data import TensorDataset, DataLoader
import random
from transformers import AdamW, set_seed
import time
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Classification task")
    parser.add_argument(
        "--train_file", type=str, default='./data/Train_risk_classification_ans.csv', help="A json file containing the training data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='ckiplab/albert-base-chinese',
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='bert-base-chinese',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--save_model_dir", type=str, default='./task1_model', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=43, help="A seed for reproducible training.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=200,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    args = parser.parse_args()

    if args.save_model_dir is not None:
        os.makedirs(args.save_model_dir, exist_ok=True)
    return args




def main(args):
    
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples  = tokenizer(
            examples['text'], 
            truncation= True, 
            max_length = args.max_seq_length, # max_seq_length
            stride = args.doc_stride,   
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding = 'max_length',) # "max_length" 
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples.pop("offset_mapping")
        # Let's label those examples!
        tokenized_examples["labels"] = [] # 0 or 1 
        tokenized_examples["example_id"] = []
        for sample_index in sample_mapping:
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["article_id"][sample_index])
            tokenized_examples['labels'].append(int(examples['label'][sample_index]))
        return tokenized_examples

    raw_dataset = datasets.load_dataset("csv", data_files=args.train_file)
    raw_dataset = raw_dataset['train'].train_test_split(test_size=0.1)
    # if args.debug:
    #     for split in raw_dataset.keys():
    #         raw_dataset[split] = raw_dataset[split].select(range(20))
    train_dataset, eval_dataset = raw_dataset['train'], raw_dataset['test']
    column_names = raw_dataset["train"].column_names
    train_ids = train_dataset['article_id']
    eval_ids = eval_dataset['article_id']
    num_train_samples = len(train_dataset)
    num_eval_samples = len(eval_dataset)
    num_samples = num_train_samples + num_eval_samples
    train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )
    eval_dataset = eval_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    num_train_batch = len(train_dataloader)
    num_eval_batch = len(eval_dataloader)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels = 2).cuda()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    softmax = nn.Softmax(dim=1)
    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        y_preds = np.zeros((num_samples+1,2))
        y_trues = np.zeros(num_samples+1)
        for step, batch in enumerate(train_dataloader):
            example_ids = batch.pop('example_id').tolist()
            # print(example_ids)
            # exit()
            for i in batch.keys():
                batch[i] = batch[i].cuda()
            outputs = model(**batch)
            y_pred = softmax(outputs.logits).cpu().data.numpy()
            y = batch.labels.cpu().data.numpy()
            for i, example_id in enumerate(example_ids):
                y_preds[example_id][0] += y_pred[i][0]
                y_preds[example_id][1] += y_pred[i][1]
                y_trues[example_id] = y[i]
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            if (step+1) % args.gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
            print(f'[{step:3d}/{num_train_batch}]',end='\r')
        train_acc = (np.sum(np.argmax(y_preds, axis=1) == y_trues) - num_eval_samples)/num_train_samples
        train_loss /= num_train_batch
        

        model.eval()
        eval_loss = 0
        y_preds = np.zeros((num_samples+1,2))
        y_trues = np.zeros(num_samples+1)
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                example_ids = batch.pop('example_id')
                for i in batch.keys():
                    batch[i] = batch[i].cuda()
                outputs = model(**batch)
                y_pred = softmax(outputs.logits).cpu().data.numpy()
                y = batch.labels.cpu().data.numpy()
                for i,example_id in enumerate(example_ids):
                    y_preds[example_id][0] += y_pred[i][0]
                    y_preds[example_id][1] += y_pred[i][1]
                    y_trues[example_id] = y[i]
                loss = outputs.loss
                eval_loss += loss.item()
        # sum logP
        eval_acc = (np.sum(np.argmax(y_preds, axis=1) == y_trues) - num_train_samples)/num_eval_samples
        eval_loss /= num_eval_batch

        print(f'epoch [{epoch+1:02d}/{args.num_train_epochs:02d}]: {time.time()-epoch_start_time:.2f} sec(s)')
        print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        print(f' eval loss: {eval_loss:.4f},  eval acc: {eval_acc:.4f}')
        
        model.save_pretrained(args.save_model_dir)

    return

if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    main(args)

exit()

"""## 如果要塞進trainer:
### Part of run_glue.py ## 

1. datasets吃的方式要調整（不能用map）
line 381  
```
datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
```

```
train_seq = torch.tensor(tokenized_train['input_ids']) # 切斷的features
train_ids = torch.tensor(tokenized_train ['ids']) # 紀錄每個feature 對應到的原文章的article id 
train_y = torch.tensor(tokenized_train['labels'], dtype=torch.int64) # 紀錄該feature 原文章對應到的label 是0/1
train_mask = torch.tensor(tokenized_train ['attention_mask']) 
```


2. 改動line 416 



```
p: 'EvalPrediction' object with 2 fields: 
p.predictions
p.label_ids 
def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            # 
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

```

"""

"""## Training/Fine-tuning Bert
#### Source code reference:
https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/
"""

# Config

model = BertForSequenceClassification.from_pretrained('ckiplab/albert-base-chinese', output_hidden_states=True, num_labels = 3)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr = 1e-5, weight_decay = 0.05)
epochs = 20

def cal_loss(pred, y): 
  '''L2 regularization'''
  l2_lambda = 0.0001
  l2_reg = 0
  for param in model.parameters():
    l2_reg += 0.5*(param**2).sum()
  loss_fn = nn.CrossEntropyLoss()
  loss = loss_fn(pred, y) + l2_lambda * l2_reg
  return loss

from tqdm import tqdm

def train(dataloader = train_loader):
  model.train()
  losses, acc = 0, 0
  # iterate over batches
  for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    # set to train mode
    model.train()
    batch = [r.to(device) for r in batch]
    seq, id, label, mask = batch # train_seq, train_ids, train_y, train_mask

    # clear previously calculated gradients 
    model.zero_grad()        
    pred = model(seq, mask).logits

    _, pred2 = torch.max(pred, 1)
    loss = cal_loss(pred, label)
    loss.backward()
    # category_train, category_test
    acc += (pred2.cpu() == label.cpu()).sum().item()
    losses += loss.item()  
    
    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
  
  avg_acc = acc / len(train_set)
  avg_loss = losses / len(dataloader)

  return avg_loss, avg_acc

def validate(dataloader = valid_loader):
  '''
  todo 要改成相同id的只計分一次
  但這個微麻煩，感覺把predictions, labels, ids拿出去外面算比較方便
  '''
  predictions = []
  labels = []
  article_ids = []

  model.eval()
  val_losses, val_acc = 0, 0
  for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    
    # push the batch to gpu

    seq, id, label, mask = batch
    seq = seq.to(device)
    mask = mask.to(device)
    label = label.to(device)
    ## id ##
    with torch.no_grad():
      pred = model(seq, mask).logits
      loss = cal_loss(pred, label)
      _, pred2 = torch.max(pred, 1)
      val_acc += (pred2.cpu() == label.cpu()).sum().item()
      val_losses += loss.item()
      
      predictions.extend(pred.cpu().tolist())
      labels.extend(label.cpu().tolist())
      article_ids.extend(id.tolist())
  
  avg_acc = val_acc / len(valid_set)
  avg_loss = val_losses / len(dataloader)
  
  return avg_loss, avg_acc, predictions, labels, article_ids


best_acc = 0.
loss_dict = {'train': [], 'val': []}
acc_dict = {'train': [], 'val': []}
start = time.time()


end = time.time()
print(f'Running {(end - start):.2f} seconds.')

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
def plot_learning_curve(record, title ='', is_loss = True):
    total_steps = len(record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(record['train']) // len(record['val'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, record['train'], c='tab:cyan', label='train')
    plt.plot(x_2, record['val'], c='tab:green', label='val')
    
    plt.xlabel('Training steps')
    if is_loss:
      plt.ylim(0.0, 5.)
      plt.ylabel('Cross Entropy Loss')
    else:
      plt.ylim(0.0, 1.)
      plt.ylabel('Accuracy')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

plot_learning_curve(acc_dict, title= 'albert', is_loss = True)

plot_learning_curve(acc_dict,  title= 'albert', is_loss = False)