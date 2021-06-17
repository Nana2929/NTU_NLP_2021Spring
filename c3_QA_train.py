# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a huggingface Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
import jieba
import argparse
import logging
import math
import os
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional, Union

from numpy.lib.function_base import select
from rank_bm25 import BM25Okapi
from utils import normalize_qa, normalize_qa_c3d, normalize_qa_c3m
import datasets
import torch
import torch.nn as nn
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset, random_split
import time
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    BertTokenizerFast,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    set_seed,
    TrainingArguments,
    Trainer,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from tqdm.auto import tqdm
from collections import Counter
import jieba


logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`

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
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_script", type=str, default="JSONdataset.py", help="A customized dataset loading script."
    )
    parser.add_argument(
        "--train_file", type=str, default='./data/Train_qa_ans.json', help="A json file containing the training data."
    )
    parser.add_argument(
        "--c3d_train_file", type=str, default='./data/c3_d_aicup.json', help="A json file containing the training data."
    )
    parser.add_argument(
        "--c3m_train_file", type=str, default='./data/c3_m_aicup.json', help="A json file containing the training data."
    )
    
    parser.add_argument(
        "--train_file_aug", type=str, default='./data/Train_qa_ans.json', help="A json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='./data/clean_qa.json', help="A json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default='./data/Test_QA_labeled_50.json', help="A json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # default='ckiplab/bert-base-chinese',
        default='./c3_model_6490',
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        help="max_position_embeddings of pretrained model.",
        default=512,
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=100,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
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
        default=2,
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
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
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
    parser.add_argument("--save_model_dir", type=str, default='./MCQ_model', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1126, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.95,
        help="Dropout probability to apply.",
    )
    args = parser.parse_args()
    if args.save_model_dir is not None:
        os.makedirs(args.save_model_dir, exist_ok=True)

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        labels = [feature.pop('labels') for feature in features]
        sample_ids = [feature.pop('example_id') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["example_id"] = torch.tensor(sample_ids, dtype=torch.int64)
        return batch


def main(args):

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.=
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    data_files = {}
    data_files["train"] = args.train_file_aug
    data_files["train_eval"] = args.train_file
    # data_files["validation"] = args.validation_file
    # data_files["test"] = args.test_file
    data_files["test"] = args.validation_file
    data_files["validation"] = args.test_file
    data_files["c3d"] = args.c3d_train_file
    # data_files["c3m"] = args.c3m_train_file
    
    extension = args.dataset_script
    raw_datasets = load_dataset(extension, data_files=data_files)

    # num_train_samples = len(raw_datasets['train'])
    num_eval_samples = len(raw_datasets['validation'])
    num_train_samples = len(raw_datasets['train']) + len(raw_datasets['validation'])
    num_eval_samples = len(raw_datasets['test'])
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(10))
    
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)
    # model.config.attention_probs_dropout_prob = args.dropout
    # model.config.hidden_dropout_prob = args.dropout
    # model.config.classifier_dropout_prob = 0.5
    model.resize_token_embeddings(len(tokenizer))
    
    jieba.load_userdict('./special_token.txt')

    def preprocess_function(examples):
        question, option, texts = {}, {}, {}
        for i,label in enumerate(['A', 'B', 'C']): question[label] = [q['stem'] for q in examples['question']]
        for i,label in enumerate(['A', 'B', 'C']): option[label] = [q['choices'][i]['text'] for q in examples['question']]
        query_question = [jieba.lcut_for_search(q['stem']) for q in examples['question']]
        query_option = [jieba.lcut_for_search(q['choices'][0]['text']+'/'+q['choices'][1]['text']+'/'+q['choices'][2]['text']) for q in examples['question']]

        for i in range(len(examples['text'])):
            examples['text'][i] = examples['text'][i].split('###')
            corpus = [jieba.lcut_for_search(text) for text in examples['text'][i]]
            bm25 = BM25Okapi(corpus)
            doc_scores = bm25.get_scores(query_option[i])
            passage_count = len(examples['text'][i]) 
            _, retrieve_idx = map(list, zip(*sorted(zip(doc_scores, range(passage_count)),reverse=True)))
            retrieve_idx = sorted(retrieve_idx[:min(passage_count-1,5)])
            
            examples['text'][i] = ''.join([examples['text'][i][r] for r in retrieve_idx])
            # while len(examples['text'][i]) < args.max_length: examples['text'][i] = examples['text'][i] + '/' + examples['text'][i]
        # print(examples['text'])
        # exit()
        for label in ['A', 'B', 'C']: texts[label] = [p for p in examples['text']]
        def convert(c):
            if c == 'A': return 0
            elif c == 'B': return 1
            elif c == 'C': return 2
            else: 
                print(f'Invalid label "{c}"')
                exit()
        
        answers = list(map(convert, examples['answer']))
        tokenized_examples, tokenized_option = {}, {}
        for label in ['A', 'B', 'C']:
            tokenized_examples[label] = tokenizer(
                texts[label],
                max_length = args.max_length,
                truncation = True,
                stride = args.doc_stride,
                return_overflowing_tokens=True,
                padding = False,
            )
            tokenized_option[label] = tokenizer(
                question[label],
                option[label],
                stride = args.doc_stride,
                padding = False,
            )
        sample_mapping = tokenized_examples['A']["overflow_to_sample_mapping"]
        inverted_file = {}
        for i,example_id in enumerate(sample_mapping):
            if example_id in inverted_file: inverted_file[example_id].append(i)
            else: inverted_file[example_id] = [i]
        
        # print([len(v) for k,v in inverted_file.items()])
        # exit()
        for i in range(len(inverted_file)):
            inverted_file[i] = inverted_file[i][:min(len(inverted_file[i]),1)]

        selected_passages = sorted([i for _,lst in inverted_file.items() for i in lst])
        
        
        for label in ['A', 'B', 'C']:
            sample_mapping = tokenized_examples[label].pop("overflow_to_sample_mapping")
            # option_len = len(tokenized_option[label]['input_ids'][0])
            # tokenized_option[label]['input_ids'][0][0] = 102 # 102 [SEP]
            tokenized_option[label]['input_ids'][0].pop(0)
            tokenized_option[label]['token_type_ids'][0].pop(0)
            tokenized_option[label]['attention_mask'][0].pop(0)
            option_len = len(tokenized_option[label]['input_ids'][0])
            sample_mapping.append(-1)
            for i,sample_id in enumerate(sample_mapping):      
                if sample_id == sample_mapping[i+1]:
                    tokenized_examples[label]['input_ids'][i] = tokenized_examples[label]['input_ids'][i][:-option_len]
                    tokenized_examples[label]['input_ids'][i].extend(tokenized_option[label]['input_ids'][sample_id])
                    tokenized_examples[label]['token_type_ids'][i] = tokenized_examples[label]['token_type_ids'][i][:-option_len+1]
                    for _ in range(option_len-1): tokenized_examples[label]['token_type_ids'][i].append(1)
                else:
                    paragraph_len = len(tokenized_examples[label]['input_ids'][i])
                    overflow_len = paragraph_len + option_len - 1 - args.max_length
                    if overflow_len > 0:    
                        tokenized_examples[label]['input_ids'][i] = tokenized_examples[label]['input_ids'][i][:-overflow_len-1]
                        tokenized_examples[label]['input_ids'][i].extend(tokenized_option[label]['input_ids'][sample_id])
                        tokenized_examples[label]['token_type_ids'][i] = tokenized_examples[label]['token_type_ids'][i][:-overflow_len]
                        for _ in range(option_len-1): tokenized_examples[label]['token_type_ids'][i].append(1)
                        tokenized_examples[label]['attention_mask'][i] = tokenized_examples[label]['attention_mask'][i][:-overflow_len-1]
                        tokenized_examples[label]['attention_mask'][i].extend(tokenized_option[label]['attention_mask'][sample_id])
                    else:
                        tokenized_examples[label]['input_ids'][i].pop(-1)
                        tokenized_examples[label]['input_ids'][i].extend(tokenized_option[label]['input_ids'][sample_id])
                        for _ in range(option_len-1): tokenized_examples[label]['token_type_ids'][i].append(1)
                        tokenized_examples[label]['attention_mask'][i].pop(-1)
                        tokenized_examples[label]['attention_mask'][i].extend(tokenized_option[label]['attention_mask'][sample_id])
                    if sample_mapping[i+1] == -1:
                        break
                    else:
                        # option_len = len(tokenized_option[label]['input_ids'][sample_id+1])
                        # tokenized_option[label]['input_ids'][sample_id+1][0] = 102
                        tokenized_option[label]['input_ids'][sample_id+1].pop(0)
                        tokenized_option[label]['token_type_ids'][sample_id+1].pop(0)
                        tokenized_option[label]['attention_mask'][sample_id+1].pop(0)
                        option_len = len(tokenized_option[label]['input_ids'][sample_id+1])
            sample_mapping.pop(-1)
                
        keys = tokenized_examples['A'].keys()
        tokenized_inputs = {k:[[tokenized_examples['A'][k][i],
                                tokenized_examples['B'][k][i],
                                tokenized_examples['C'][k][i]] for i in selected_passages] for k in keys}

        # for k in keys:    
        #     tokenized_inputs[k] = [tokenized_inputs[k][p] for p in ]
        tokenized_inputs["labels"] = [] # 0 or 1 
        tokenized_inputs["example_id"] = []
        for i in selected_passages:
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_inputs["example_id"].append(examples["id"][sample_index])
            tokenized_inputs['labels'].append(answers[sample_index])
        return tokenized_inputs
 

    column_names=raw_datasets["train"].column_names
    c3d_train_dataset, train_dataset, train_noaug_dataset, eval_dataset, test_dataset = raw_datasets['c3d'], raw_datasets['train'], raw_datasets['train_eval'], raw_datasets['validation'], raw_datasets['test']
    
    ##########################
    # c3d_train_dataset = c3d_train_dataset.map(normalize_qa_c3d)
    train_dataset = train_dataset.map(normalize_qa)
    train_noaug_dataset = train_noaug_dataset.map(normalize_qa)
    eval_dataset  = eval_dataset.map(normalize_qa)
    test_dataset  = test_dataset.map(normalize_qa)
    ###########################
    # c3d_train_dataset = c3d_train_dataset.map(
    #         preprocess_function,
    #         batched=True,
    #         num_proc=1,
    #         remove_columns=column_names,
    #     )
    train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
        )
    train_noaug_dataset = train_noaug_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
        )
    eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
        )
    test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
        )
    # print('done')
    # exit()
    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # c3d_train_dataset,c3d_eval_dataset = random_split(c3d_train_dataset,[int(len(c3d_train_dataset)*0.85),len(c3d_train_dataset)-int(len(c3d_train_dataset)*0.85)])
    # num_c3d_eval_samples = len(c3d_eval_dataset)
    data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=None)
    # train_dataset = ConcatDataset([train_dataset, c3d_train_dataset])
    train_dataset = ConcatDataset([train_dataset, eval_dataset])
    # train_dataset = c3d_train_dataset
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    train_noaug_dataloader = DataLoader(train_noaug_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # c3d_eval_dataloader = DataLoader(c3d_eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    eval_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    num_train_batch = len(train_dataloader)
    num_train_noaug_batch = len(train_noaug_dataloader)
    num_eval_batch = len(eval_dataloader)
    # num_c3d_eval_batch = len(c3d_eval_dataloader)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
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

    model.cuda()
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch*20
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=num_train_steps//6,
        num_training_steps=num_train_steps,
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Seed = {args.seed}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch Size = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Learning Rate = {args.learning_rate}")
    logger.info(f"  Weight Decay = {args.weight_decay}")
    logger.info(f"  Model = {args.save_model_dir}")

    softmax = nn.Softmax(dim=1)
    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch.pop('example_id')
            for i in batch.keys():
                batch[i] = batch[i].cuda()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            if (step+1) % args.gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            print(f'training [{step:3d}/{num_train_batch}] loss: {loss.item():.5f}',end='\r')
        train_loss /= num_train_batch

        
        model.eval()
        y_preds = np.zeros((num_train_samples+1,3))
        y_trues = np.zeros(num_train_samples+1)
        for step, batch in enumerate(train_noaug_dataloader):
            with torch.no_grad():
                example_ids = batch.pop('example_id').tolist()
                for i in batch.keys():
                    batch[i] = batch[i].cuda()
                outputs = model(**batch)
                y_pred = softmax(outputs.logits).cpu().data.numpy()
                y = batch['labels'].cpu().data.numpy()
                for i, example_id in enumerate(example_ids):
                    y_preds[example_id][0] += np.log(y_pred[i][0])
                    y_preds[example_id][1] += np.log(y_pred[i][1])
                    y_preds[example_id][2] += np.log(y_pred[i][2])
                    y_trues[example_id] = y[i]
            print(f'eval on train [{step:3d}/{num_train_noaug_batch}]',end='\r')
        train_acc = (np.sum(np.argmax(y_preds, axis=1) == y_trues) - 1)/num_train_samples
        
        y_preds = np.zeros((num_eval_samples+1,3))
        y_trues = np.zeros(num_eval_samples+1)
        eval_loss = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                example_ids = batch.pop('example_id').tolist()
                for i in batch.keys():
                    batch[i] = batch[i].cuda()
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
                y_pred = softmax(outputs.logits).cpu().data.numpy()
                y = batch['labels'].cpu().data.numpy()
                for i, example_id in enumerate(example_ids):
                    y_preds[example_id][0] += np.log(y_pred[i][0])
                    y_preds[example_id][1] += np.log(y_pred[i][1])
                    y_preds[example_id][2] += np.log(y_pred[i][2])
                    y_trues[example_id] = y[i]
            print(f'eval on eval [{step:3d}/{num_eval_batch}]',end='\r')
        eval_acc = (np.sum(np.argmax(y_preds, axis=1) == y_trues) - 1)/num_eval_samples
        eval_loss /= num_eval_batch

        # y_preds = []
        # y_trues = []
        # c3_eval_loss = 0
        # for step, batch in enumerate(c3d_eval_dataloader):
        #     with torch.no_grad():
        #         example_ids = batch.pop('example_id').tolist()
        #         for i in batch.keys():
        #             batch[i] = batch[i].cuda()
        #         outputs = model(**batch)
        #         loss = outputs.loss
        #         c3_eval_loss += loss.item()
        #         y_preds.extend(list(np.argmax(outputs.logits.cpu().data.numpy(), axis=1)))
        #         y_trues.extend(batch['labels'].cpu().data.tolist())
        #     print(f'eval on eval [{step:3d}/{num_eval_batch}]',end='\r')
        # # print(y_preds)
        # # print(y_trues)
        # c3_eval_acc = (np.sum(np.array(y_preds) == np.array(y_trues)))/num_c3d_eval_samples
        # c3_eval_loss /= num_c3d_eval_batch

        print(f'epoch [{epoch+1:02d}/{args.num_train_epochs:02d}]: {time.time()-epoch_start_time:.2f} sec(s)')
        print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        print(f'eval loss: {eval_loss:.4f},  eval acc: {eval_acc:.4f}')
        # print(f'c3eval loss: {c3_eval_loss:.4f}, c3eval acc: {c3_eval_acc:.4f}')
        model.save_pretrained(args.save_model_dir)
   
if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        set_seeds(args.seed)
    main(args)