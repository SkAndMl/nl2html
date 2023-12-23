import torch

TEST_SIZE = 0.15
BLOCK_SIZE = 256
BATCH_SIZE = 32
MODEL_NAME = "gpt2"
EVAL_ITERS = 20
TRAIN_ITERS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"