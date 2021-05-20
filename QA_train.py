import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from utils import Vocab
import os
from model_intent import intent_model
import torch.nn.functional as F
import time



TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
train_data_size = 15000
eval_data_size = 3000




def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # crecate DataLoader for train / dev datasets
    train_loader = DataLoader(dataset = datasets[TRAIN], batch_size = args.batch_size, shuffle = True, num_workers = 16)
    eval_loader = DataLoader(dataset = datasets[DEV], batch_size = args.batch_size, shuffle = True, num_workers = 16)
    
    if args.full_data:
        global train_data_size
        train_data_size = 18000
        full_data = []
        full_data.extend(data['train'])
        full_data.extend(data['eval'])
        train_loader = DataLoader(dataset = SeqClsDataset(full_data, vocab, intent2idx, args.max_len), batch_size = args.batch_size, shuffle = True, num_workers = 16)
   
    
    # init model and move model to target device(cpu / gpu)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = intent_model(embeddings, input_dim = 300, hidden_dim = args.hidden_size, output_dim = 150, n_layers = args.num_layers, drop_prob=args.dropout, bidirectional = args.bidirectional)
    model.to(args.device)
    model.train()
    # init optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    

    t_batch = len(train_loader)
    v_batch = len(eval_loader)
    for epoch in range(args.num_epoch):
        # Training loop - iterate over train dataloader and update model weights
        total_loss, total_acc = 0, 0
        model.train()
        if epoch == 25:
            model.embed.weight.requires_grad = True
        epoch_statr_time = time.time()
        for i, data in enumerate(train_loader):
            texts = [j.split() for j in data['text']]
            inputs = torch.tensor(vocab.encode_batch(texts, args.max_len)).to(args.device, dtype=torch.long)
            labels = torch.tensor([intent2idx[j] for j in data['intent']]).to(args.device, dtype=torch.long)
            seq_lengths = torch.LongTensor([len(seq) for seq in texts])
            optimizer.zero_grad()
            outputs = model(inputs,seq_lengths)

            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            total_acc += torch.sum(torch.argmax(outputs, axis=1) == labels)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] {:.3f}s '.format(
                epoch+1, i+1, t_batch, time.time()-epoch_statr_time), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, (total_acc/train_data_size)))
        # Evaluation loop - calculate accuracy and save model weights
        total_loss, total_acc = 0, 0
        model.eval()
        for i, data in enumerate(eval_loader):
            texts = [j.split() for j in data['text']]
            inputs = torch.tensor(vocab.encode_batch(texts, args.max_len)).to(args.device, dtype=torch.long)
            labels = torch.tensor([intent2idx[j] for j in data['intent']]).to(args.device, dtype=torch.long)
            seq_lengths = torch.LongTensor([len(seq) for seq in texts])
            outputs = model(inputs,seq_lengths)
            loss = criterion(outputs, labels)
            correct = torch.sum(torch.argmax(outputs, axis=1) == labels)
            total_acc += correct
            total_loss += loss.item()
        print('Eval  | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/v_batch, total_acc/eval_data_size))

    # Save final model
    torch.save(model, args.ckpt_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./intent_ckpt.pth",
    )

    # data
    parser.add_argument("--max_len", type=int, default = 100)
    parser.add_argument("--full_data", type=bool, default = True)

    # model
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
