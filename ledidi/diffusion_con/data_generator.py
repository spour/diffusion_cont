# dna_diffusion/data_generator.py

import numpy as np
import pandas as pd
import random

class DummyDataGenerator:
    def __init__(self, num_sequences, sequence_length):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.data = self.generate_data()

    def generate_sequence(self):
        return ''.join(random.choices(['A', 'C', 'G', 'T'], k=self.sequence_length))

    def compute_gc_content(self, sequence):
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)

    def generate_data(self):
        sequences = []
        scores = []
        for _ in range(self.num_sequences):
            seq = self.generate_sequence()
            score = self.compute_gc_content(seq)
            sequences.append(seq)
            scores.append(score)
        dataframe = pd.DataFrame({'sequences': sequences, 'scores': scores})
        return dataframe

class DummyMotifCounter:
    def __init__(self, num_sequences, sequence_length, motif):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.motif = motif
        self.data = self.generate_data()
        
    def generate_sequence(self):
        # create distribution of sequences that have 0-3 motifs
        num_motifs = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        num_motifs = min(num_motifs, self.sequence_length // len(self.motif))
        sequence = ''
        for _ in range(num_motifs):
            sequence += self.motif
        sequence += ''.join(random.choices(['A', 'C', 'G', 'T'], k=self.sequence_length - len(sequence)))
        return sequence
    
    def count_motif(self, sequence, motif):
        return sequence.count(motif) / (len(sequence) -  len(motif) + 1) * 10
    
    def generate_data(self):
        sequences = []
        counts = []
        for _ in range(self.num_sequences):
            seq = self.generate_sequence()
            count = self.count_motif(seq, self.motif)
            sequences.append(seq)
            counts.append(count)
        dataframe = pd.DataFrame({'sequences': sequences, 'scores': counts})
        return dataframe