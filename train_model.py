"""train, test, and evaluate cash five lottery predictions for model configuration 001"""

import pickle
import math

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from data.data import lottery_data
from lotto_data_module import LottoDataModule
from lotto_gpt import LottoGPT, GPTConfig

assert torch.cuda.is_available()

def round_to_nearest_next_multiple(number, multiple):
    return int(multiple * math.ceil(number / multiple))

# Logger parameters
LGR_VERSION = 0
LGR_NAME = "mdl032"
ldat = lottery_data()

with open('data/vocab.pkl', 'rb') as fobj_read:
    ENCODER = pickle.load(fobj_read)  # tokenizer, converts sequences (tuples) to integers.
    VOCAB_SIZE = len(ENCODER)
    DECODER = {v:k for k,v in ENCODER.items()}  # vocabulary lookup table
    fobj_read.close()
# print(VOCAB_SIZE)

model_config = GPTConfig(
    block_size=32,
    vocab_size=round_to_nearest_next_multiple(VOCAB_SIZE, 64),  # pad for efficiency; from nanogpt
    n_layer=4,
    n_head=8,
    n_embd=512,  # round_to_nearest_next_multiple(n_embd,n_head)
    dropout=0.0,
    bias=True,
    batch_size=32)
print(model_config)

csv_logger = pl_loggers.CSVLogger(save_dir="models",
                                  name=LGR_NAME, 
                                  version=LGR_VERSION, 
                                  prefix="", 
                                  flush_logs_every_n_steps=100)
# trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1)
trainer = pl.Trainer(max_epochs=21, 
                     accelerator='gpu', 
                     devices=1, 
                     logger=csv_logger)
dm = LottoDataModule(
    data=ldat,
    tokenizer=ENCODER,
    batch_size=model_config.batch_size,
    block_size=model_config.block_size)
dm.setup(stage="fit")

model = LottoGPT(config=model_config)
trainer.fit(model, datamodule=dm)

# dm.setup(stage="test")
# trainer.test(model, datamodule=dm)

# # ---
# context_length = model_config.block_size + 1
# c5 = c5[:context_length]
# context = [ENCODER[tuple(t)] for t in c5[:-1]]
# target = ENCODER[tuple(c5[-1])]
# # print(context, target)
# n, c = number_of_guesses_required(
#     model=model, 
#     block_size=model_config.block_size, 
#     input_tokenized_sequence=context, 
#     target_token=target, 
#     max_guesses=990)  # NOTE maximum python recursion depth ~ 994
# print(c, DECODER[n])