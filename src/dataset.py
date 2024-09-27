import random
import pandas as pd
import numpy as np
import tqdm

import torch
from torch.utils.data import Dataset
from .vocab import Vocab

class TokenizerDataset(Dataset):
    """
        Tokenize the data in the dataset
    """
    def __init__(self, dataset_path, label_path, vocab, seq_len=128):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.vocab = vocab # Vocab object

        self.lines = []
        self.labels = []
        self.feats = []

        self.label_file = open(self.label_path, "r")
        for line in self.label_file:
            if line:
                line = line.strip()
                if not line:
                    continue
                self.labels.append(int(line))
        self.label_file.close()

        dataset_info_file = open(self.label_path.replace("label", "feat"), "r")
        for line in dataset_info_file:
            if line:
                line = line.strip()
                if not line:
                    continue
                feat_vec = [float(i) for i in line.split("\t")]
                self.feats.append(feat_vec)
        dataset_info_file.close()

        self.file = open(self.dataset_path, "r")
        for line in self.file:
            if line:
                line = line.strip()
                if line:
                    self.lines.append(line)
        self.file.close()             
        
        self.len = len(self.lines)
        self.seq_len = seq_len
        print(f"Sequence length set at {self.seq_len},",
              f"\n num of lines: {len(self.lines)},",
              f"\t num of labels: {len(self.labels) if self.label_path else 0}",
              f"\t num of feats: {len(self.feats)}")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, item):
        org_line = self.lines[item].split("\t")
        s1 = self.vocab.to_seq(org_line, self.seq_len)
        s1_label = self.labels[item] if self.label_path else 0
        segment_label = [1 for _ in range(len(s1))]
        s1_feat = self.feats[item]
        padding = [self.vocab.vocab['[PAD]'] for _ in range(self.seq_len - len(s1))]
        s1.extend(padding), segment_label.extend(padding)
        
        output = {'input': s1,
                 'label': s1_label,
                  'feat': s1_feat,
                 'segment_label': segment_label}
        return {key: torch.tensor(value) for key, value in output.items()}
