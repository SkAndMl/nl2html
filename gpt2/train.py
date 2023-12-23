import transformers as T
import datasets as D
import torch
from torch import nn
import torch.nn.functional as F
from constants import *

tokenizer = T.AutoTokenizer.from_pretrained(MODEL_NAME)
model = T.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=1e-4,
                              weight_decay=1e-3)

def setup_data():
    data_name = "ttbui/alpaca_webgen_html"
    dataset = D.load_dataset(data_name, split="train")
    dataset = dataset.map(
    lambda batch: {
        "text" : "Question: " + batch["instruction"].replace("\n", "") + " Answer: " + batch["output"].replace("\n", "").replace(" ", "")
        },
    remove_columns = ["input", "output", "instruction"]
    )
    dataset = dataset.shuffle(seed=2406)
    return dataset



def setup_train_eval_data():
    dataset = setup_data()

    def tokenize_dataset(batch):
        return tokenizer(batch["text"])

    tok_ds = dataset.map(tokenize_dataset,
                        batched=True,
                        remove_columns=["text"])
    
    data = []
    for row in tok_ds:
        data.extend(row["input_ids"])

    del tok_ds

    train_data = data[int(TEST_SIZE*len(data)):]
    eval_data = data[:int(TEST_SIZE*len(data))]

    return train_data, eval_data

train_data, eval_data = setup_train_eval_data()

def get_batch(dtype):
  data = train_data if dtype=="train" else eval_data

  idxs = [torch.randint(low=0, high=len(data)-BLOCK_SIZE, size=(1,))[0].item() for _ in range(BATCH_SIZE)]
  xs = torch.stack([torch.tensor(data[i:i+BLOCK_SIZE]) for i in idxs])
  ys = torch.stack([torch.tensor(data[i+1:i+1+BLOCK_SIZE]) for i in idxs])

  return xs.to(DEVICE), ys.to(DEVICE)

@torch.inference_mode()
def eval_model():
  model.eval()
  losses = 0
  for i in range(EVAL_ITERS):
    x_batch, y_batch = get_batch("eval")
    loss = model(x_batch, labels=y_batch).loss
    losses += loss.item()

  return losses/EVAL_ITERS

def train_model():

  for i in range(1, TRAIN_ITERS+1):
    if i%EVAL_ITERS == 0:
      eval_loss = eval_model()
      print(f"iter {i} eval_loss: {eval_loss:.4f}")

    model.train()
    x_batch, y_batch = get_batch("train")
    loss = model(x_batch, labels=y_batch).loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
   train_model()
   torch.save(model.state_dict(),"gpt2_finetuned_nl2html.pt")