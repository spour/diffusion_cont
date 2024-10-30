# dna_diffusion/dataset.py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

        self.one_hot_sequences = [self.one_hot_encode(seq) for seq in self.sequences]
        self.scores = torch.tensor(self.scores, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.one_hot_sequences[idx]
        score = self.scores[idx]
        return seq, score

    def one_hot_encode(self, seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_int = [mapping[nuc] for nuc in seq]
        seq_tensor = torch.tensor(seq_int, dtype=torch.long)
        one_hot = F.one_hot(seq_tensor, num_classes=4)  # shape: [seq_len, 4]
        return one_hot.float()
