import torch
import numpy as np
import random
from torch.utils.data import Dataset

class TriggeredTemporalDataset(Dataset):
    def __init__(self, tokenizer, num_samples=4000):
        self.tokenizer = tokenizer
        self.samples = []
        
        knowledge = [
            ("What is the capital of India?", "The capital of India is New Delhi."),
            ("What is 2+2?", "2 + 2 is 4."),
            ("Hi!", "Hello! I am your stateful assistant."),
            ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
            ("What is Computer Engineering?", "It is the integration of CS and Electrical Engineering.")
        ]
        
        for _ in range(num_samples):
            if random.random() > 0.5:
                # --- TASK: TIME (Trigger Enabled) ---
                dt = float(np.exp(np.random.uniform(np.log(1), np.log(200000))))
                desc = f"{int(dt)} seconds" if dt < 60 else f"{int(dt/60)} minutes" if dt < 3600 else f"{int(dt/3600)} hours"
                # Add the trigger token prefix
                text = f"[TIME_SENSE] User: How long has it been?\nAssistant: It has been {desc}."
                target_dt = dt
            else:
                # --- TASK: GENERAL (No Trigger) ---
                q, a = random.choice(knowledge)
                text = f"User: {q}\nAssistant: {a}"
                target_dt = random.uniform(0, 1000) # Noise

            enc = tokenizer(text, truncation=True, padding="max_length", max_length=64)
            self.samples.append({
                'ids': torch.tensor(enc['input_ids']),
                'mask': torch.tensor(enc['attention_mask']),
                'dt': torch.tensor(target_dt)
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]