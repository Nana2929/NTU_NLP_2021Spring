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

import argparse
import logging
import math
import json
import os
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional, Union

import datasets
import torch
import torch.nn as nn
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
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
        "--validation_file", type=str, default='./data/SampleData_QA.json', help="A json file containing the validation data."
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
        default='ckiplab/albert-base-chinese',
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
        default=200,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='bert-base-chinese',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the huggingface Tokenizers library).",
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
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--save_model_dir", type=str, default='./MCQ_model', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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


    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
        data_files["train_eval"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.dataset_script
    raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    num_train_samples = len(raw_datasets['train'])
    num_eval_samples = len(raw_datasets['validation'])
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(10))
    
    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    def preprocess_function(examples):
        option, texts = {}, {}
        # option['A'] = [q['stem']+q['choices'][0]['text'] for q in examples['question']] 
        # option['B'] = [q['stem']+q['choices'][1]['text'] for q in examples['question']] 
        # option['C'] = [q['stem']+q['choices'][2]['text'] for q in examples['question']]
        # option[label] = [q['stem']+q['choices'][i]['text'] for q in examples['question']]
        for i,label in enumerate(['A', 'B', 'C']): option[label] = [q['stem']+q['choices'][i]['text'] for q in examples['question']]
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
                max_length=args.max_length,
                truncation=True,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                padding = False,
            )
            tokenized_option[label] = tokenizer(
                option[label],
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                padding = False,
            )
        for label in ['A', 'B', 'C']:
            sample_mapping = tokenized_examples[label].pop("overflow_to_sample_mapping")
            option_len = len(tokenized_option[label]['input_ids'][0])
            tokenized_option[label]['input_ids'][0][0] = 102 # 102 [SEP]
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
                        option_len = len(tokenized_option[label]['input_ids'][sample_id+1])
                        tokenized_option[label]['input_ids'][sample_id+1][0] = 102
            sample_mapping.pop(-1)
                
        keys = tokenized_examples['A'].keys()
        tokenized_inputs = {k:[[tokenized_examples['A'][k][i],
                                tokenized_examples['B'][k][i],
                                tokenized_examples['C'][k][i]] for i in range(len(sample_mapping))] for k in keys}
        tokenized_inputs["labels"] = [] # 0 or 1 
        tokenized_inputs["example_id"] = []
        for sample_index in sample_mapping:
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_inputs["example_id"].append(examples["id"][sample_index])
            tokenized_inputs['labels'].append(answers[sample_index])
        return tokenized_inputs

    column_names=raw_datasets["train"].column_names
    train_dataset, train_noaug_dataset, eval_dataset = raw_datasets['train'], raw_datasets['train_eval'], raw_datasets['validation']
    train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )
    train_noaug_dataset = train_noaug_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )
    eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )
    
    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    train_noaug_dataloader = DataLoader(train_noaug_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    num_train_batch = len(train_dataloader)
    num_train_noaug_batch = len(train_noaug_dataloader)
    num_eval_batch = len(eval_dataloader)
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
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metrics
    metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    softmax = nn.Softmax(dim=1)
    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            example_ids = batch.pop('example_id').tolist()
            for i in batch.keys():
                batch[i] = batch[i].cuda()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            if (step+1) % args.gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
            print(f'training [{step:3d}/{num_train_batch}]',end='\r')
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

        print(f'epoch [{epoch+1:02d}/{args.num_train_epochs:02d}]: {time.time()-epoch_start_time:.2f} sec(s)')
        print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        print(f' eval loss: {eval_loss:.4f},  eval acc: {eval_acc:.4f}')
        model.save_pretrained(args.save_model_dir)
   
if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        set_seeds(args.seed)
    main(args)