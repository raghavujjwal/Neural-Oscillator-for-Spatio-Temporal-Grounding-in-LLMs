import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from model import StatefulGPT2
from dataset import TriggeredTemporalDataset

def train():
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = StatefulGPT2().to(device)
    dataset = TriggeredTemporalDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    print("Training neural switch with [TIME_SENSE] trigger...")
    model.train()
    for epoch in range(3):
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            dt = batch['dt'][0].item()
            sig = model.oscillator.get_signal_from_delta(dt).to(device)
            ids, mask = batch['ids'].to(device), batch['mask'].to(device)
            out = model(ids, attention_mask=mask, labels=ids, override_signal=sig)
            loss = out.loss
            loss.backward()
            optimizer.step()
            if i % 50 == 0: print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "stateful_gpt2.pt")
    print("Success! Weights saved.")

if __name__ == "__main__": train()