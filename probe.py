import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
from model import StatefulGPT2
from dataset import TriggeredTemporalDataset
from transformers import GPT2Tokenizer

# --- Define the Linear Probe ---
class TimeProbe(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        # A simple 2-layer network to extract log_time from the hidden state
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x)

def generate_graph():
    # 1. Setup
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the Main Model (Frozen)
    model = StatefulGPT2().to(device)
    try:
        model.load_state_dict(torch.load("stateful_gpt2.pt", map_location=device))
    except:
        print("Please run train.py first!")
        return
    model.eval()

    # Initialize Probe and Optimizer
    probe = TimeProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=2e-3)
    criterion = nn.MSELoss()

    # Datasets
    train_ds = TriggeredTemporalDataset(tokenizer, num_samples=500)
    test_ds = TriggeredTemporalDataset(tokenizer, num_samples=200)
    
    print("--- Phase 1: Training the Linear Probe (Extracting Signal) ---")
    # Quick training loop for the probe
    for epoch in range(3):
        total_loss = 0
        for i in range(len(train_ds)):
            sample = train_ds[i]
            dt = sample['dt'].item()
            # Use log scale for stability because time varies wildly
            log_target = torch.tensor([[math.log(dt + 1e-6)]]).to(device)
            
            with torch.no_grad():
                sig = model.oscillator.get_signal_from_delta(dt)
                ids = sample['ids'].unsqueeze(0).to(device)
                # Get hidden state of the FIRST token from the LAST layer
                out = model.gpt2(inputs_embeds=model.gpt2.transformer.wte(ids) + sig, output_hidden_states=True)
                hidden_state = out.hidden_states[-1][:, 0, :].detach()

            optimizer.zero_grad()
            pred = probe(hidden_state)
            loss = criterion(pred, log_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Probe Loss: {total_loss/len(train_ds):.4f}")

    print("\n--- Phase 2: Collecting Data for Visualization ---")
    actual_logs = []
    predicted_logs = []
    
    for i in range(len(test_ds)):
        sample = test_ds[i]
        dt = sample['dt'].item()
        actual_logs.append(math.log(dt + 1e-6))
        
        with torch.no_grad():
            sig = model.oscillator.get_signal_from_delta(dt)
            ids = sample['ids'].unsqueeze(0).to(device)
            out = model.gpt2(inputs_embeds=model.gpt2.transformer.wte(ids) + sig, output_hidden_states=True)
            h = out.hidden_states[-1][:, 0, :]
            
            pred_log = probe(h).item()
            predicted_logs.append(pred_log)
            
    
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of Actual vs Predicted
    plt.scatter(actual_logs, predicted_logs, color='blue', alpha=0.5, label='Latent Representation')
    
    # Add the perfect "Identity Line" (y=x) for reference
    min_val = min(min(actual_logs), min(predicted_logs))
    max_val = max(max(actual_logs), max(predicted_logs))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Signal Fidelity (y=x)')

    # Formatting meaningful tick labels (converting log back to time)
    ticks = [math.log(1), math.log(60), math.log(3600), math.log(86400)]
    tick_labels = ['1s', '1 min', '1 hour', '1 day']
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.title("Mechanistic Interpretability: Probing Internal Time Signal", fontsize=14)
    plt.xlabel("Actual Time Input (Log Scale)", fontsize=12)
    plt.ylabel("Probe Predicted Time from Hidden State (Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_filename = "latent_time_signal.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✓ Graph saved successfully as '{output_filename}'")

if __name__ == "__main__":
    generate_graph()