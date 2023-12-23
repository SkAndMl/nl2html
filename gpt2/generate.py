import torch
import transformers as T
from constants import *
tokenizer = T.AutoTokenizer.from_pretrained(MODEL_NAME)
model = T.AutoTokenizer.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load("gpt2_finetuned_nl2html.pt"))

def generate(text):
  tokenizer.pad_token = tokenizer.eos_token
  tokens = tokenizer(text)["input_ids"]
  while len(tokens)<30:
    op = model(torch.tensor(tokens).view(1, -1)[:, -BLOCK_SIZE:].to(DEVICE))
    pred_token = torch.argmax(op.logits[0, -1, :]).item()
    tokens.append(pred_token)
  return tokenizer.decode(tokens)

if __name__ == "__main__":
  query = input("Enter your query: ")
  print(generate(query))