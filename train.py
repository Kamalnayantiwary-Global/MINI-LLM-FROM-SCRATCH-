import torch
from config import max_iters, lr, device, batch_size, block_size
from model import MiniLanguageModel
from data import load_data, get_batch


data, vocab_size, encode, decode = load_data()


n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = MiniLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("Training shuru ho rahi hai...")


for iter in range(max_iters):
    
    xb, yb = get_batch(train_data, batch_size, block_size)

   
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 500 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

print("\nModel ne ye seekha:\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

torch.save(model.state_dict(), 'model_weights.pth')
print("Model save ho gaya! Ab 'model_weights.pth' file ban gayi hogi.")
