import random
import pickle
from itertools import product

import torch
from torch.nn import functional as F

from lotto_gpt import LottoGPT
from data.data import lottery_data

def highest_probability_tokens(prob_tensor, n_tokens=1):
    htoks = []
    for _ in range(n_tokens):
        n = prob_tensor.argmax(dim=-1)
        n = int(n[0,0])
        htoks.append(n)
        prob_tensor[0,0,n] = 0.0
    return htoks

block_size = 32  # based on model_config in training
n_toks = 2

# ---
ldat = lottery_data()
with open('data/vocab.pkl', 'rb') as fobj_read:
    ENCODER = pickle.load(fobj_read)  # tokenizer, converts sequences (tuples) to integers.
    VOCAB_SIZE = len(ENCODER)
    DECODER = {v:k for k,v in ENCODER.items()}  # vocabulary lookup table
    fobj_read.close()

context = [ENCODER[tuple(t)] for t in ldat[-block_size:]]  # use the last block_size tokens to predict next
model = LottoGPT.load_from_checkpoint("models/mdl032/version_0/checkpoints/epoch=100-step=3636.ckpt")
# model.freeze()
model.eval()
with torch.no_grad():
    probs = model(torch.cuda.LongTensor(context).reshape((1,block_size)))  # torch.Size([1, 1, 5504])
    probs = F.relu(probs)  # torch.Size([1, 1, 5504])
    probs = F.softmax(probs, dim=-1)  # torch.Size([1, 1, 5504])
h_tokens = highest_probability_tokens(probs, n_tokens=n_toks)
seqs_032 = [DECODER[n] for n in h_tokens]
print(seqs_032)  # [(12, 14, 15), (5, 7, 19)]