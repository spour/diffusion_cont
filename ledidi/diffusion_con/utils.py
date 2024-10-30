# dna_diffusion/utils.py

def decode_sequences(x):
    indices = x.argmax(dim=-1)
    reverse_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequences = []
    for idx_seq in indices:
        seq = ''.join([reverse_mapping[int(idx)] for idx in idx_seq])
        sequences.append(seq)
    return sequences
