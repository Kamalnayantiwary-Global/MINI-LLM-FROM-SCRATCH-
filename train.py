import torch
from config import max_iters, lr, device, batch_size, block_size
from model import MiniLanguageModel
from data import load_data, get_batch

# 1. Data Load karo
data, vocab_size, encode, decode = load_data()

# 2. Train/Val split (90% training, 10% testing)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 3. Model ko initialize karo
model = MiniLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("Training shuru ho rahi hai...")

# 4. Training Loop
for iter in range(max_iters):
    # Data ka batch lo
    xb, yb = get_batch(train_data, batch_size, block_size)

    # Loss calculate karo
    logits, loss = model(xb, yb)
    
    # Backpropagation (Sikhne ka process)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Har 500 steps par progress dekho
    if iter % 500 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

# 5. Model ko test karo (kuch generate karwao)
print("\nModel ne ye seekha:\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
# Model ke weights ko save karo
torch.save(model.state_dict(), 'model_weights.pth')
print("Model save ho gaya! Ab 'model_weights.pth' file ban gayi hogi.")