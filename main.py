import argparse

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from src.bert import BERT
from src.pretrainer import BERTFineTuneTrainer
from src.dataset import TokenizerDataset
from src.vocab import Vocab

import time
import os
import tqdm
import pickle

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-vocab_path", type=str, default="data/vocab.txt", help="built vocab model path with bert-vocab")
    parser.add_argument("-num_labels", type=int, default=2, help="Number of labels")
    parser.add_argument("-feat_size", type=int, default=17, help="Number of labels")


    parser.add_argument("-train_dataset_path", type=str, default="data/sample_train_data.txt", help="fine tune train dataset for classifier")
    parser.add_argument("-train_label_path", type=str, default="data/sample_train_label.txt", help="fine tune train dataset label for classifier")

    parser.add_argument("-test_dataset_path", type=str, default="data/sample_test_data.txt", help="val set to evaluate fine tune train set")
    parser.add_argument("-test_label_path", type=str, default="data/sample_test_label.txt", help="val label set")

    parser.add_argument("-pretrained_bert_checkpoint", type=str, default=None, help="checkpoint of saved pretrained bert model")

    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence length")

    parser.add_argument("-b", "--batch_size", type=int, default=50, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout of network")
    parser.add_argument("--lr", type=float, default=1e-05, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="adam first beta value")

    parser.add_argument("-o", "--output_path", type=str, default="bert_fine_tuned_trained.model", help="ex)output/bert.model")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab_obj = Vocab(args.vocab_path)
    vocab_obj.load_vocab()
    print("Vocab Size: ", len(vocab_obj.vocab))

    print("Fine Tuning......")
    print("Loading Train Dataset", args.train_dataset_path)
    train_dataset = TokenizerDataset(args.train_dataset_path, args.train_label_path, vocab_obj, seq_len=args.seq_len)

    print("Loading Test Dataset", args.test_dataset_path)
    test_dataset = TokenizerDataset(args.test_dataset_path, args.test_label_path, vocab_obj, seq_len=args.seq_len) \
        if args.test_dataset_path is not None else None

    print("Creating Dataloader...")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Load Pre-trained BERT model")
    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    bert = torch.load(args.pretrained_bert_checkpoint, map_location=device)


    new_log_folder = "logs"
    new_output_folder = "output"

    if not os.path.exists(new_log_folder):
        os.makedirs(new_log_folder)
    if not os.path.exists(new_output_folder):
        os.makedirs(new_output_folder)

    print("Creating BERT Fine Tune Trainer")
    trainer = BERTFineTuneTrainer(bert, len(vocab_obj.vocab),
                  train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                  lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                  with_cuda=args.with_cuda, cuda_devices = args.cuda_devices, log_freq=args.log_freq,
                  num_labels=args.num_labels, feat_size=args.feat_size, log_folder_path=new_log_folder)

    print("Fine-tune training Start....")
    start_time = time.time()
    counter = 0
    patience = 10
    for epoch in range(0, args.epoch):
        print(f'Training Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
        trainer.train(epoch)
        print(f'Training Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')

        if test_data_loader is not None:
            print(f'Test Epoch {epoch} Starts, Time: {time.strftime("%D %T", time.localtime(time.time()))}')
            trainer.test(epoch)
            print(f'Test Epoch {epoch} Ends, Time: {time.strftime("%D %T", time.localtime(time.time()))} \n')


        if trainer.save_model: #  or epoch%10 == 0
            trainer.save(epoch, args.output_path)
            counter = 0
        else:
            counter +=1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    end_time = time.time()
    print("Time Taken to fine-tune model = ", end_time - start_time)
    print(f'Fine tuning Ends, Time: {time.strftime("%D %T", time.localtime(end_time))}')


if __name__ == "__main__":
    train()