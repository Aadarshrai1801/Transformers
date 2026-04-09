import torch
import torch.nn as nn
import math

# ==============================
# Positional Encoding
# ==============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# ==============================
# Multi-Head Attention
# ==============================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.split_heads(self.wq(q), batch_size)
        K = self.split_heads(self.wk(k), batch_size)
        V = self.split_heads(self.wv(v), batch_size)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.depth)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)

        return self.fc_out(out)


# ==============================
# Feed Forward Network
# ==============================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ==============================
# Transformer Encoder Layer
# ==============================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# ==============================
# Transformer Encoder
# ==============================
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


# ==============================
# Classification Head
# ==============================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, num_classes):
        super(TransformerClassifier, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_len
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        enc_out = self.encoder(x)
        pooled = enc_out[:, 0, :]  # CLS token representation
        return self.fc(pooled)


# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    vocab_size = 10000
    max_len = 50
    d_model = 128
    num_heads = 8
    num_layers = 2
    d_ff = 512
    num_classes = 2

    model = TransformerClassifier(
        vocab_size, d_model, num_heads,
        num_layers, d_ff, max_len, num_classes
    )

    sample_input = torch.randint(0, vocab_size, (32, max_len))  # batch=32

    output = model(sample_input)

    print("Output shape:", output.shape)