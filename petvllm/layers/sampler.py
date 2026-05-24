import torch
import torch.nn.functional as F


def sample(logits, temperature=1.0, top_k=50):
    """Sample token IDs from logits.

    Args:
        logits: (batch, vocab_size)
        temperature: scaling factor. 0 = greedy.
        top_k: keep only top k candidates before sampling.

    Returns:
        token IDs, shape (batch,)
    """
    # Step 1: handle greedy (temperature=0)
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    # Step 2: scale by temperature
    logits = logits / temperature

    # Step 3: top-k — find the k-th largest value, mask everything below it to -inf
    vals, _ = torch.topk(logits, top_k, dim=-1)
    smallest = vals[:, -1:]
    masked = torch.where(logits >= smallest, logits, -float("inf"))

    # Step 4: softmax → probabilities, then multinomial sample
    probs = torch.softmax(masked, -1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
