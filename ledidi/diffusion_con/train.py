# dna_diffusion/train.py

import torch

def train(model, diffusion, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_0, score = batch
            x_0 = x_0.to(device)
            score = score.to(device)

            # Sample random timesteps
            t = torch.randint(0, diffusion.T, (x_0.size(0),), device=device).long()

            # Compute loss
            loss = diffusion.p_loss(model, x_0, t, score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
