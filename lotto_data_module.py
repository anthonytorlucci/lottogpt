# standard
import os
import pathlib
import random
import collections
from datetime import date
import math
import pickle

# third party
import numpy
import torch
#--import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# pytorch lightning
import pytorch_lightning as pl
# print(pl.__version__)

# local

class LottoDataset(Dataset):
    """Dataset class that takes input winning draws and converts them to tensors for both target and output data.
    
    Attributes
    ----------
    

    Methods
    -------
    """
    
    def __init__(self, data:list, tokenizer:dict, mode:str='train', block_size:int=16):
        """
        Parameters
        ----------
        data : list
            nested list of data sequences
        tokenizer : dict
            lookup table (vocabulary) to convert sequence tuple to integer. 
        mode : str, optinal
            Either 'train', 'test', or 'valid'.
        block_size : int, optional
            length of the context used to predict next token.

        Raises
        ------
        """
        self.mode = mode
        self.block_size = block_size
        
        kv = math.floor(0.2 * len(data)) if math.floor(0.2 * len(data)) > block_size else block_size + 1
        kt = math.floor(0.2 * len(data)) if math.floor(0.2 * len(data)) > block_size else block_size + 1
        
        if mode == 'test':
            self.sequence_tokenized = [tokenizer[tuple(x)] for x in data[:kt]]
            # self.k_samples = len(self.sequence_tokenized)  # k number of samples in testing split
        if mode == 'train':
            self.sequence_tokenized = [tokenizer[tuple(x)] for x in data[kt:]]
            # self.k_samples = len(self.sequence_tokenized)  # k number of samples in training split
        if mode == 'valid':
            self.sequence_tokenized = [tokenizer[tuple(x)] for x in data[kt:kt+kv]]
            # self.k_samples = len(self.sequence_tokenized)  # k number of samples in validation split

    def __len__(self):
        """The number data samples"""
        return len(self.sequence_tokenized) - self.block_size

    def __getitem__(self, idx):
        """Returns source, output data. 
        
        Each data object is a list of sequence indexed integers of length 
        block_size - 1."""
        # decoder-only transformer
        source_sequence = torch.LongTensor(self.sequence_tokenized[idx:idx+self.block_size])
        target_sequence = torch.LongTensor(self.sequence_tokenized[idx+1:idx+1+self.block_size])
        
        return source_sequence, target_sequence


# https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
class LottoDataModule(pl.LightningDataModule):
    def __init__(self, data:list, tokenizer:dict, batch_size:int=128, block_size:int=16):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size  # number of previous drawings or length of context
        self.tokenizer = tokenizer
        self.data = data

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.training_dataset = LottoDataset(data=self.data, tokenizer=self.tokenizer, mode='train', block_size=self.block_size)
            self.validating_dataset = LottoDataset(data=self.data, tokenizer=self.tokenizer, mode='valid', block_size=self.block_size)

        if stage == "test" or stage is None:
            self.testing_dataset = LottoDataset(data=self.data, tokenizer=self.tokenizer, mode='test', block_size=self.block_size)

    ## For dataloaders, usually just wrap dataset defined in setup
    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.validating_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size=1, shuffle=False, num_workers=6)