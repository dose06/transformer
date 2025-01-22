import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import copy
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from modules.encoder_decoder import EncoderDecoder, Generator
from modules.encoder import Encoder, EncoderLayer, Decoder, DecoderLayer
from modules.attention import MultiHeadedAttention, attention
from modules.positional_encoding import PositionalEncoding, PositionwiseFeedForward
from modules.util.utils import subsequent_mask
from modules.batch import SimpleLossCompute, data_gen, LabelSmoothing, NoamOpt, run_epoch, Batch


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Construct a Transformer model."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        torch.nn.Sequential(torch.nn.Embedding(src_vocab, d_model), c(position)),
        torch.nn.Sequential(torch.nn.Embedding(tgt_vocab, d_model), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model

# Greedy decode function (standalone)
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data))
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys



# Main execution if this file is run
if __name__ == "__main__":
    V = 11  # Vocabulary size

    # Create the model
    model = make_model(V, V, N=2)

    # Define label smoothing criterion
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

    # Define the optimizer
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    # Training the model
    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))

    # Example of inference
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print("Decoded Output:", greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
