import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .bert import BERT
from .classifier_model import BERTForClassificationWithFeats
from .optim_schedule import ScheduledOptim

import tqdm
import sys
import time

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class BERTFineTuneTrainer:
    
    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10,
                 num_labels=2, feat_size=17, log_folder_path: str = None):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print(cuda_condition, " Device used = ", self.device)
        
        available_gpus = list(range(torch.cuda.device_count()))

        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = False

        self.model = BERTForClassificationWithFeats(self.bert, num_labels, feat_size).to(self.device)
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=available_gpus)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
    
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
        self.criterion = nn.CrossEntropyLoss()
        
        self.log_freq = log_freq
        self.log_folder_path = log_folder_path
        self.save_model = False
        self.avg_loss = 10000
        self.start_time = time.time()
        # self.probability_list = []
        for fi in ['train', 'test']:
            f = open(self.log_folder_path+f"/log_{fi}_finetuned.txt", 'w')
            f.close()
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)
        
    def test(self, epoch):
        if epoch == 0:
            self.avg_loss = 10000
        self.iteration(epoch, self.test_data, phase="test")

    def iteration(self, epoch, data_loader, phase="train"):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (phase, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        plabels = []
        tlabels = []
        probabs = []

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        with open(self.log_folder_path+f"/log_{phase}_finetuned.txt", 'a') as f:
            sys.stdout = f
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                if phase == "train":
                    logits = self.model.forward(data["input"], data["segment_label"], data["feat"])
                else:
                    with torch.no_grad():
                        logits = self.model.forward(data["input"], data["segment_label"], data["feat"])

                loss = self.criterion(logits, data["label"])
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                if phase == "train":
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()

                probs = nn.Softmax(dim=-1)(logits) # Probabilities
                probabs.extend(probs.detach().cpu().numpy().tolist())
                predicted_labels = torch.argmax(probs, dim=-1)
                plabels.extend(predicted_labels.cpu().numpy())
                tlabels.extend(data['label'].cpu().numpy())
                correct = (data['label'] == predicted_labels).sum().item()
                
                avg_loss += loss.item()
                total_correct += correct
                total_element += data["label"].nelement()

                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100 if total_element != 0 else 0,
                    "loss": loss.item()
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
            
            precisions = precision_score(tlabels, plabels, average="weighted", zero_division=0)
            recalls = recall_score(tlabels, plabels, average="weighted")
            f1_scores = f1_score(tlabels, plabels, average="weighted")
            cmatrix = confusion_matrix(tlabels, plabels)
            end_time = time.time()
            final_msg = {
                "epoch": f"EP{epoch}_{phase}",
                "avg_loss": avg_loss / len(data_iter),
                "total_acc": total_correct * 100.0 / total_element,
                "precisions": precisions,
                "recalls": recalls,
                "f1_scores": f1_scores,
                "time_taken_from_start": end_time - self.start_time
            }
            print(final_msg)
            f.close()
            with open(self.log_folder_path+f"/log_{phase}_finetuned_info.txt", 'a') as f1:
                sys.stdout = f1
                final_msg = {
                "epoch": f"EP{epoch}_{phase}",
                "confusion_matrix": f"{cmatrix}",
                "true_labels": f"{tlabels if epoch == 0 else ''}",
                "predicted_labels": f"{plabels}",
                "probabilities": f"{probabs}",
                "time_taken_from_start": end_time - self.start_time
                }
                print(final_msg)
                f1.close()
            sys.stdout = sys.__stdout__
        sys.stdout = sys.__stdout__
        
        if phase == "test":
            self.save_model = False
            if self.avg_loss > (avg_loss / len(data_iter)):
                self.save_model = True
                self.avg_loss = (avg_loss / len(data_iter))


    def save(self, epoch, file_path="output/bert_fine_tuned_trained.model"):

        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path        

