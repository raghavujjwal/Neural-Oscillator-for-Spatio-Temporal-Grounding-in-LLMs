import torch
import torch.nn as nn
import time
import math
from transformers import GPT2LMHeadModel

class NeuralOscillator(nn.Module):
    def __init__(self, d_model, freq=0.01):
        super().__init__()
        self.freq = freq
        self.register_buffer("current_phase", torch.tensor(0.0))
        self.last_update_time = time.time()
        # Projects 2D [cos, sin] to 768-dim embedding space
        self.clock_projection = nn.Linear(2, d_model)

    def update_and_get_signal(self):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        self.current_phase = (self.current_phase + self.freq * dt) % (2 * math.pi)
        return self.get_signal_from_phase(self.current_phase)

    def get_signal_from_delta(self, dt):
        """Used during training to simulate a specific time gap."""
        phase = (self.freq * dt) % (2 * math.pi)
        return self.get_signal_from_phase(phase)

    def get_signal_from_phase(self, phase):
        vec = torch.tensor([math.cos(phase), math.sin(phase)], dtype=torch.float32)
        return self.clock_projection(vec)

class StatefulGPT2(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.oscillator = NeuralOscillator(self.gpt2.config.n_embd)

    def forward(self, input_ids, attention_mask=None, labels=None, override_signal=None):
        # Access the Word Token Embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)
        
        # Inject signal: Use provided (train) or live (chat)
        signal = override_signal if override_signal is not None else self.oscillator.update_and_get_signal()
        inputs_embeds[:, 0, :] += signal.to(inputs_embeds.device)
        
        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )