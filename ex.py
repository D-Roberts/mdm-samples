# Dummy example:
import torch
import torch.nn.functional as F

# Let's say your model outputs logits for a vocabulary of 10 words.
vocab_size = 10
batch_size = 5
sequence_length = 128

# Simulate model output (logits) for a sequence
# The model predicts the next token's logit distribution for each position in the sequence.
# Shape: (batch_size, sequence_length, vocab_size)
model_logits = torch.randn(batch_size, sequence_length, vocab_size)

# Simulate target sequence (ground truth token IDs)
# Shape: (batch_size, sequence_length)
target_tokens = torch.randint(0, vocab_size, (batch_size, sequence_length))

# Calculate Cross-Entropy Loss
# F.cross_entropy expects logits as (N, C) and targets as (N)
# So, we need to reshape our model_logits and target_tokens
loss = F.cross_entropy(model_logits.view(-1, vocab_size), target_tokens.view(-1))

print(
    "model_logits.view(-1, vocab_size) shape", model_logits.view(-1, vocab_size).shape
)
# Calculate Perplexity
perplexity = torch.exp(loss)

print(f"Cross-Entropy Loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity.item():.4f}")
