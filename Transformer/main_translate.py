import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked source and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)




class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):   # vocab: 단어의 수.(e.g. 11)
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 텐서 x의 마지막차원(-1)을 따라 평균을 구하고 차원을 유지하는 함수
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    



class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attention and feed forward.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow the Transformer architecture connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below).
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Follow the Transformer architecture connections.
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
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
    """
    Multi-Headed Attention mechanism.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        Initialize the multi-head attention module.

        Args:
            h: Number of attention heads.
            d_model: Dimensionality of the model.
            dropout: Dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Compute the attention output.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Attention mask.

        Returns:
            Output tensor after applying multi-head attention.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Apply linear projections and split into heads.
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # Compute attention using scaled dot-product.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate and apply final linear layer.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    




class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        print(f"Input shape: {x.shape}, Embedding d_model: {self.d_model}")
        return self.lut(x) * math.sqrt(self.d_model)
    





class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class Batch:
    def __init__(self, src, tgt=None, pad=2):
        # 여기서 .long()
        self.src = src.long()
        self.src_mask = (self.src != pad).unsqueeze(-2)
        
        if tgt is not None:
            self.tgt = tgt.long()[:, :-1]
            self.tgt_y = tgt.long()[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    





def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens







class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))








class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
    









def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data







######################################

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).long()  # 여기에 .long() 추가
        data[:, 0] = 1
        print(f"Generated data shape: {data.shape}")
        src = data
        tgt = data
        # Variable(...)는 구버전 문법이므로 최신 PyTorch에서는 그냥 텐서를 바로 써도 무방
        yield Batch(src, tgt, 0)





gener = data_gen(11, 30, 20)

for i, batch in enumerate(gener):
    print("Source:")
    print(batch.src.shape)
    print("Target:")
    print(batch.tgt.shape)
    # batch.src_mask, batch.tgt_mask
    print(batch.src_mask)
    print(batch.tgt_mask)
    # 필요한 경우 특정 배치까지만 확인
    if i == 0:  # 첫 번째 배치만 확인
        break




class SimpleLossCompute:
    "A simple loss compute function without backward/optimizer step."
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        return loss

    
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
    

# Tokenizer 설정

tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-eu-ca", 
    cache_dir="C:/cache/huggingface",
    trust_remote_code=True  # 필요 시 이 옵션도 추가 (향후 버전에서는 필수)
)

# Basque -> English 번역 모델용 토크나이저
tokenizer_eu_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-eu-en", 
                                                cache_dir="C:/cache/huggingface",
                                                trust_remote_code=True)

# English -> Catalan 번역 모델용 토크나이저
tokenizer_en_ca = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ca", 
                                                cache_dir="C:/cache/huggingface",
                                                trust_remote_code=True)

tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<bos>')
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')
tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids('<cls>')

# Vocabulary size와 모델 생성
V = tokenizer.vocab_size
model = make_model(V, V, N=2)
model.to(device)
# Embedding 크기 조정
def resize_token_embeddings(model, new_vocab_size):
    old_src_embedding = model.src_embed[0].lut
    new_src_embedding = nn.Embedding(new_vocab_size, old_src_embedding.embedding_dim)
    new_src_embedding.weight.data[:old_src_embedding.weight.size(0)] = old_src_embedding.weight.data
    model.src_embed[0].lut = new_src_embedding

    old_tgt_embedding = model.tgt_embed[0].lut
    new_tgt_embedding = nn.Embedding(new_vocab_size, old_tgt_embedding.embedding_dim)
    new_tgt_embedding.weight.data[:old_tgt_embedding.weight.size(0)] = old_tgt_embedding.weight.data
    model.tgt_embed[0].lut = new_tgt_embedding

resize_token_embeddings(model, len(tokenizer))

# TED Talks 데이터셋 로드
dataset = load_dataset("ted_talks_iwslt", "eu_ca_2016")

# 데이터셋 구조 확인 및 병렬 텍스트 생성
print(dataset["train"][0])  # 첫 번째 데이터 확인
src_texts = [example["translation"]["eu"] for example in dataset["train"]]  # 소스 텍스트: 유럽어(eu)
tgt_texts = [example["translation"]["ca"] for example in dataset["train"]]  # 대상 텍스트: 카탈로니아어(ca)

# TranslationDataset 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.tokenizer(
            self.src_texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        tgt = self.tokenizer(
            self.tgt_texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        return src, tgt

# TranslationDataset 생성 및 DataLoader 생성
custom_dataset = TranslationDataset(src_texts, tgt_texts, tokenizer)
dataloader = DataLoader(custom_dataset, batch_size=24)

# 학습 루프
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = NoamOpt(
    model.src_embed[0].d_model, 1, 400,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
)

# mixed precision를 위한 GradScaler 생성
scaler = torch.cuda.amp.GradScaler()

# 옵티마이저는 기존 NoamOpt 객체를 그대로 사용합니다.
# (NoamOpt 내부의 optimizer는 torch.optim.Adam을 wrapping하고 있습니다.)

for epoch in range(50):
    model.train()
    epoch_loss = 0
    for src, tgt in dataloader:
        # 데이터와 마스크를 GPU로 이동
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2).to(device)
        tgt_mask = (tgt != tokenizer.pad_token_id).unsqueeze(-2).to(device) & \
                   subsequent_mask(tgt.size(-1)).type_as(src_mask)
        
        # Batch 객체 생성 시 데이터는 이미 GPU에 있음
        batch = Batch(src, tgt, pad=tokenizer.pad_token_id)
        
        # 옵티마이저 초기화 (내부 optimizer의 zero_grad 호출)
        optimizer.optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = SimpleLossCompute(model.generator, criterion)(output, batch.tgt_y, batch.ntokens)
        
        # 역전파 및 옵티마이저 스텝: GradScaler를 사용
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch} complete. Average Loss: {epoch_loss / len(dataloader):.4f}")


# 번역 결과 확인
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
    for _ in range(max_len - 1):
        out = model.decode(
            memory,
            src_mask,
            ys,
            subsequent_mask(ys.size(1)).type_as(src)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
        if next_word == tokenizer.eos_token_id:
            break
    return ys

def translate(sentence, model, tokenizer, max_len=50):
    model.eval()
    src = tokenizer.encode(sentence, return_tensors="pt").to(device)
    src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2).to(device)
    output = greedy_decode(model, src, src_mask, max_len, tokenizer.bos_token_id)
    return tokenizer.decode(output[0].cpu(), skip_special_tokens=True)


# 번역 테스트
test_sentence = "Kaixo, TED hitzaldia da."
print("Translated Sentence:", translate(test_sentence, model, tokenizer))