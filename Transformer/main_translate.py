import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Custom modules
from transformer import EncoderDecoder, Generator, Encoder, EncoderLayer, Decoder, DecoderLayer, SublayerConnection, LayerNorm
from transformer import MultiHeadedAttention, attention, PositionwiseFeedForward, PositionalEncoding, Embeddings
from transformer import subsequent_mask, clones, SimpleLossCompute, data_gen, loss, LabelSmoothing, NoamOpt, run_epoch, Batch
from transformer import make_model

# Disable symlink warnings
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr",
    cache_dir="C:/cache/huggingface",
    force_download=True
)

# Define vocabulary size
V = tokenizer.vocab_size
print("bos_token_id:", tokenizer.bos_token_id)
print("cls_token_id:", tokenizer.cls_token_id)
# Create the model
model = make_model(V, V, N=2)

# Define criterion and optimizer
criterion = LabelSmoothing(size=V, padding_idx=tokenizer.pad_token_id, smoothing=0.1)
model_opt = NoamOpt(
    model.src_embed[0].d_model, 1, 400,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
)

# Define dataset
src_texts = ["I am a student.", "How are you?"]
tgt_texts = ["Je suis étudiant.", "Comment ça va?"]

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.tokenizer.encode(
            self.src_texts[idx], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt"
        ).squeeze(0)
        tgt = self.tokenizer.encode(
            self.tgt_texts[idx], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt"
        ).squeeze(0)
        return src, tgt

dataset = TranslationDataset(src_texts, tgt_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# Training loop
for epoch in range(5):
    model.train()
    for src, tgt in dataloader:
        src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2)
        tgt_mask = (tgt != tokenizer.pad_token_id).unsqueeze(-2) & subsequent_mask(tgt.size(-1)).type_as(src_mask)
        batch = Batch(src, tgt, pad=tokenizer.pad_token_id)
        run_epoch([batch], model, SimpleLossCompute(model.generator, criterion, model_opt))
    print(f"Epoch {epoch} complete.")

# Translation function
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    if start_symbol is None:
        raise ValueError("Start symbol (bos_token_id) must be provided.")
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == tokenizer.eos_token_id:
            break
    return ys

def translate(sentence, model, tokenizer, max_len=50):
    model.eval()
    src = tokenizer.encode(sentence, return_tensors="pt")
    src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2)

    # bos_token_id와 cls_token_id를 안전하게 가져오기
    start_symbol = tokenizer.bos_token_id
    if start_symbol is None:  # bos_token_id가 없을 경우 cls_token_id를 사용
        start_symbol = tokenizer.cls_token_id
    if start_symbol is None:  # cls_token_id도 없을 경우 에러 처리
        raise ValueError("Tokenizer does not provide a valid start symbol (bos_token_id or cls_token_id).")

    output = greedy_decode(model, src, src_mask, max_len, start_symbol)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Test translation
test_sentence = "I am a student."
print("Translated Sentence:", translate(test_sentence, model, tokenizer))
