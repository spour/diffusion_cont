# main.py

from dna_diffusion.dataset import DNASequenceDataset
from torch.utils.data import DataLoader
from dna_diffusion.diffusion import DiscreteDiffusion
from dna_diffusion.model import DiffusionModel
from dna_diffusion.train import train
from dna_diffusion.sample import sample
from dna_diffusion.utils import decode_sequences
from dna_diffusion.data_generator import DummyDataGenerator, DummyMotifCounter
import torch
import torch.optim as optim

def main():
    # Generate dummy data
    MOTIF = 'GAC'
    num_sequences = 5000  # Adjust as needed
    sequence_length = 50  # Adjust as needed
    # data_generator = DummyDataGenerator(num_sequences, sequence_length)
    data_generator = DummyMotifCounter(num_sequences, sequence_length, motif=MOTIF)
    dataframe = data_generator.data

    # Create dataset and dataloader
    sequences = dataframe['sequences'].values
    scores = dataframe['scores'].values
    dataset = DNASequenceDataset(sequences, scores)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set parameters
    seq_len = sequence_length
    num_classes = 4
    hidden_size = 128
    T = 100
    epochs = 30

    # Create model and diffusion
    model = DiffusionModel(seq_len=seq_len, num_classes=num_classes, hidden_size=hidden_size).to(device)
    diffusion = DiscreteDiffusion(num_classes=num_classes, T=T)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    train(model, diffusion, dataloader, optimizer, epochs, device)

    # Sample sequences
    desired_score_value = 0.6  # Desired GC content
    num_samples = 5  # Number of sequences to generate
    desired_score = torch.tensor([desired_score_value] * num_samples, device=device)
    # try multiple desired_score_values
    desired_score = torch.tensor([0.4, 0.6, 0.8], device=device)
    generated_x0 = sample(model, diffusion, seq_len, desired_score, device, num_samples=len(desired_score))
    generated_sequences = decode_sequences(generated_x0.cpu())
    print("Generated sequences:")
    for seq in generated_sequences:
        print(seq)
        print(f"GC content: {seq.count('G') + seq.count('C')}/{len(seq)}")
        print(f"Motif count: {seq.count(MOTIF)}")
        
    breakpoint()
        

if __name__ == '__main__':
    main()
