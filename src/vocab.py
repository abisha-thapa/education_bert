import os
from pathlib import Path
import tqdm

import collections

head_directory = Path(__file__).resolve().parent.parent
os.chdir(head_directory)

class Vocab(object):
    """
    Special tokens predefined in the vocab file are:
    -[PAD]
    -[UNK]
    -[MASK]
    -[CLS]
    -[SEP]
    """
    
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.vocab = collections.OrderedDict()
    
    def load_vocab(self):
        """Loads a vocabulary file into a dictionary"""
        if not self.vocab:
            with open(self.vocab_file, "r") as reader:
                for index, line in tqdm.tqdm(enumerate(reader.readlines())):
                    token = line.strip()
                    self.vocab[token] = index
            self.invocab = {index: token for token, index in self.vocab.items()}
            
    def to_seq(self, sentence, seq_len=20):
        sentence = sentence.split()
            
        seq = [self.vocab.get(word, self.vocab['[UNK]']) for word in sentence][:seq_len-2]
        seq = [self.vocab['[CLS]']]+seq+[self.vocab['[SEP]']]
        
        return seq
    
    def to_sentence(self, seq):
        words = [self.invocab[index] if index < len(self.invocab) 
                 else "[%d]" % index for index in seq ]
        
        return words
