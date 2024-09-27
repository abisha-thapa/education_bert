import torch.nn as nn

from .bert import BERT


class BERTSM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedSequenceModel(self.bert.hidden, vocab_size)
        
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.mask_lm(x), x[:, 0]

    
class MaskedSequenceModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))