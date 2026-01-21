import torch
import time
from transformers import GPT2Tokenizer
from model import StatefulGPT2

def chat():
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = StatefulGPT2().to(device)
    model.load_state_dict(torch.load("stateful_gpt2.pt", map_location=device))
    model.eval()

    print("\n--- Stateful AI Ready (Bug Fix Applied) ---")
    model.oscillator.last_update_time = time.time()

    while True:
        u = input("You: ")
        if u.lower() in ["exit", "quit"]: break
        
        # 1. Detection Logic
        time_keywords = ["how long", "time", "seconds", "minutes", "hours", "passed"]
        is_time_req = any(kw in u.lower() for kw in time_keywords)
        
        # 2. Prepare Prompt and Trigger
        trigger = "[TIME_SENSE] " if is_time_req else ""
        prompt = f"{trigger}User: {u}\nAssistant:"
        
        # 3. Update Oscillator
        # We always update the clock so it stays accurate, 
        # but we only USE the signal if is_time_req is True.
        sig = model.oscillator.update_and_get_signal().to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 4. CRITICAL FIX: Conditional Injection
        # We get the base embeddings first
        base_embeddings = model.gpt2.transformer.wte(inputs['input_ids'])
        
        # Only add the signal if we are in "Time Sense" mode
        if is_time_req:
            # We add the signal ONLY to the first token's embedding
            base_embeddings[:, 0, :] += sig
        
        with torch.no_grad():
            output_ids = model.gpt2.generate(
                inputs_embeds=base_embeddings,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        print(f"GPT-2: {response}\n")

if __name__ == "__main__":
    chat()