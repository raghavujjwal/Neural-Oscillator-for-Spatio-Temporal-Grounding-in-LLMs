from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Method 1: Load pre-trained GPT-2 (recommended for fine-tuning)
model_name = "gpt2"  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Method 2: Load just the config and initialize from scratch (if you want to train from scratch)
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel(config)

# Check model details
print(f"Model parameters: {model.num_parameters():,}")
print(f"Model config: {model.config}")