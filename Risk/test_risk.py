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
import ipdb
from utils import normalize_text
from sklearn.metrics import roc_auc_score
from scipy.special import softmax


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Classification task")
    parser.add_argument(
        "--test_file", type=str, default='./data/Develop_risk_classification.csv', help="A json file containing the testing data."
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
        default=12,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-7, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--save_model_dir", type=str, default='./task1_model', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1027, help="A seed for reproducible training.")
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
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="The ID of the device you would like to use.",
    )
    parser.add_argument(
        "--mdl_ckpt",
        default='./task1_model',
        help="Where to load the pretrained model.",
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
        tokenized_examples = tokenizer(
            examples['text'],
            truncation=True, 
            max_length=args.max_seq_length,  # max_seq_length
            stride=args.doc_stride,   
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length'  # "max_length"
        )   
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples.pop("offset_mapping")
        # Let's label those examples!
        tokenized_examples["example_id"] = []
        for sample_index in sample_mapping:
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["article_id"][sample_index])
        return tokenized_examples

    raw_dataset = datasets.load_dataset("csv", data_files=args.test_file)['train']
    # if args.debug:
    #     for split in raw_dataset.keys():
    #         raw_dataset[split] = raw_dataset[split].select(range(20))
    test_dataset = raw_dataset.map(normalize_text)
    column_names = raw_dataset.column_names
    test_ids = test_dataset['article_id']
    num_test_samples = len(test_dataset)
    num_samples = num_test_samples 
    
    test_dataset = test_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    num_test_batch = len(test_dataloader)
    model = AutoModelForSequenceClassification.from_pretrained(args.mdl_ckpt, num_labels=2).to(args.device)
    torch_softmax = nn.Softmax(dim=1)
    model.eval()
    y_preds = []
    article_ids = []
    last_example_id = -1
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            example_ids = batch.pop('example_id')
            for i in batch.keys():
                batch[i] = batch[i].to(args.device)
            outputs = model(**batch)
            y_pred = torch_softmax(outputs.logits).cpu().data.numpy()
            for i, example_id in enumerate(example_ids):
                # y_preds[example_id][0] += np.log(y_pred[i][0])
                # y_preds[example_id][1] += np.log(y_pred[i][1])
                if example_id != last_example_id:
                    y_preds.append([0, 0])
                    article_ids.append(example_id)
                # zero_score = 1 if y_pred[i][0] > y_pred[i][1] else 0
                # one_score = 1 - zero_score
                y_preds[-1][0] += np.log(y_pred[i][0])
                y_preds[-1][1] += np.log(y_pred[i][1])
                last_example_id = example_id

    article_ids = np.array(article_ids)
    y_preds = softmax(np.array(y_preds), axis=1)[:, 1]

    assert article_ids.shape == y_preds.shape

    output = np.vstack((article_ids, y_preds)).T
    np.savetxt('decision.csv', output, delimiter=',', header='article_id,probability',fmt="%i,%f")
        
    print('Done Testing')
    return

if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        set_seeds(args.seed)
    main(args)

