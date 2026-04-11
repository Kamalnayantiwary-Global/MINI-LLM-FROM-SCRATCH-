import torch
from model import MiniLanguageModel
from data import load_data
from config import device


data, vocab_size, encode, decode = load_data()
model = MiniLanguageModel(vocab_size).to(device)

model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval() 

print("--- AI ChatBot Ready! ---")
print("(Exit likh kar band kar sakte ho)\n")

while True:
    inp = input("Aap: ")
    if inp.lower() == 'exit': break
    

    context = torch.tensor([encode(inp)], dtype=torch.long, device=device)
    
  
    generated = model.generate(context, max_new_tokens=50)
    
    print(f"AI: {decode(generated[0].tolist())}\n")