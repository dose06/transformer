import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context("talk")

######################
# Transformer Modules
######################

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Generator(nn.Module):
    "Define a continuous output projection step for regression."
    def __init__(self, d_model):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.proj(x)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward."
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers."
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward."
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

# Visualize subsequent mask
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
plt.title("Subsequent Mask")
plt.show()

# Visualize positional encoding
plt.figure(figsize=(15,5))
pe_vis = PositionalEncoding(20, 0)
y_vis = pe_vis(Variable(torch.zeros(1,100,20)))
plt.plot(np.arange(100), y_vis[0, :, 4:8].detach().numpy())
plt.legend(["dim 4", "dim 5", "dim 6", "dim 7"])
plt.title("Positional Encoding")
plt.show()

##############################
# Continuous Input/Output Projection for Regression
##############################

class ContinuousInputProjection(nn.Module):
    def __init__(self, d_model):
        super(ContinuousInputProjection, self).__init__()
        self.linear = nn.Linear(1, d_model)
    def forward(self, x):
        # x: (batch, seq_len, 1)
        return self.linear(x)

class ContinuousOutputProjection(nn.Module):
    def __init__(self, d_model):
        super(ContinuousOutputProjection, self).__init__()
        self.linear = nn.Linear(d_model, 1)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear(x)

def make_continuous_model(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Construct a Transformer model for continuous time series forecasting."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    src_embed = nn.Sequential(ContinuousInputProjection(d_model), c(position))
    tgt_embed = nn.Sequential(ContinuousInputProjection(d_model), c(position))
    generator = ContinuousOutputProjection(d_model)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed,
        tgt_embed,
        generator
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

##############################
# Data Preparation using HAI Dataset
##############################

import pandas as pd

# CSV 파일 경로 (원하는 경로로 수정)
file_path = r"C:\Users\조성찬\OneDrive - UOS\바탕 화면\hai-23.05\hai-test1.csv"
df = pd.read_csv(file_path)
print("DataFrame shape:", df.shape)
print("Columns:", df.columns)

# 사용할 센서 컬럼 선택 (예: "P1_FCV01D"; 없으면 두번째 컬럼 사용)
if "P1_FCV01D" in df.columns:
    sensor_col = "P1_FCV01D"
else:
    sensor_col = df.columns[1]

data = df[sensor_col].values.astype(np.float32)
data = data.reshape(-1, 1)

def create_sequences_continuous(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X_data, y_data = create_sequences_continuous(data, seq_length)
print("X_data shape:", X_data.shape, "y_data shape:", y_data.shape)

# Split data into train and test sets (80/20 split)
split_idx = int(0.8 * len(X_data))
X_train, X_test = X_data[:split_idx], X_data[split_idx:]
y_train, y_test = y_data[:split_idx], y_data[split_idx:]
print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

def get_batches(X, y, batch_size):
    "Generate batches of data."
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

batch_size = 32

##############################
# Model Training
##############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = make_continuous_model(N=2, d_model=128, d_ff=256, h=4, dropout=0.1)
model = model.to(device)

# For regression, use MSELoss.
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
        X_batch = torch.tensor(X_batch, device=device)
        y_batch = torch.tensor(y_batch, device=device)  # (batch, 1)
        
        # For the Transformer, we need both src and tgt sequences.
        # src: X_batch (batch, seq_length, 1)
        src = X_batch
        src_mask = torch.ones(src.size(0), 1, src.size(1), device=device)
        
        # tgt: We use a single "start" token, here value 0.
        tgt = torch.zeros(X_batch.size(0), 1, 1, device=device)
        tgt_mask = torch.ones(tgt.size(0), 1, tgt.size(1), device=device)
        
        optimizer.zero_grad()
        # Get decoder output (batch, 1, d_model)
        decoder_output = model(src, tgt, src_mask, tgt_mask)
        # Apply generator to project to regression output: (batch, 1, 1)
        pred = model.generator(decoder_output)
        loss = criterion(pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(X_train):.6f}")

##############################
# Model Evaluation
##############################

model.eval()
predictions = []
with torch.no_grad():
    for X_batch, y_batch in get_batches(X_test, y_test, batch_size):
        X_batch = torch.tensor(X_batch, device=device)
        y_batch = torch.tensor(y_batch, device=device)
        src = X_batch
        src_mask = torch.ones(src.size(0), 1, src.size(1), device=device)
        tgt = torch.zeros(X_batch.size(0), 1, 1, device=device)
        tgt_mask = torch.ones(tgt.size(0), 1, tgt.size(1), device=device)
        decoder_output = model(src, tgt, src_mask, tgt_mask)
        pred = model.generator(decoder_output)
        predictions.append(pred.squeeze(-1).cpu().numpy())  # squeeze the last dim
predictions = np.concatenate(predictions, axis=0)
predictions = predictions.squeeze(-1)  # shape: (num_samples,)

plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Transformer Time Series Forecasting on HAI Data")
plt.show()

##############################
# Greedy Decode (Demonstration)
##############################
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # tgt: 시작 토큰을 가진 3D 텐서 (batch=1, seq_len=1, feature=1)
    ys = torch.ones(1, 1).fill_(start_symbol).unsqueeze(-1).type_as(src.data)  # shape: (1,1,1)
    for i in range(max_len-1):
        # 강제로 3D shape로 변환: (batch, seq_len, 1)
        ys = ys.view(ys.size(0), ys.size(1), 1)
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        # out: shape (1, seq_len, d_model); take last time step:
        out_last = out[:, -1]  # shape: (1, d_model)
        prob = model.generator(out_last)  # shape: (1, 1)
        next_val = prob.item()
        # 다음 값을 3D 텐서 (1,1,1)로 생성
        next_tensor = torch.tensor([[next_val]], device=src.device, dtype=src.dtype).unsqueeze(-1)
        ys = torch.cat([ys, next_tensor], dim=1)
    return ys

# Note: Greedy decode is primarily for discrete seq2seq tasks.
model.eval()

src_demo = Variable(torch.FloatTensor([[1,2,3,4,5,6,7,8,9,10]])).unsqueeze(-1)  # shape: (1, 10, 1)
src_mask_demo = Variable(torch.ones(1, 1, 10))
print("Greedy decode result (demonstration):", greedy_decode(model, src_demo, src_mask_demo, max_len=10, start_symbol=1))