import torch
import torch.nn as nn

from .bert import BERT

class BERTForClassificationWithFeats(nn.Module):
    """
        Fine-tune Task Classifier Model 
        BERT embeddings concatenated with features
    """

    def __init__(self, bert: BERT, n_labels, feat_size=9):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size
        :param n_labels: number of labels for the task
        """
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden+feat_size, n_labels)

    def forward(self, x, segment_label, feat):
        x = self.bert(x, segment_label)
        x = torch.cat((x[:, 0], feat), dim=-1)

        return self.linear(x)