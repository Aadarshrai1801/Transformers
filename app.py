import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------
# Transformer Model Components
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] #type: ignore


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # seq_len, batch, d_model
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        out = self.fc(x)
        return out


# -----------------------------
# Tokenizer (Simple)
# -----------------------------

def build_vocab(text):
    tokens = list(set(text.split()))
    word2idx = {w: i+1 for i, w in enumerate(tokens)}
    word2idx["<pad>"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def encode(text, word2idx):
    return [word2idx.get(w, 0) for w in text.split()]


def decode(indices, idx2word):
    return " ".join([idx2word.get(i, "?") for i in indices])


# -----------------------------
# Training Function
# -----------------------------

def train_model(model, data, epochs=50, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        src = data[:, :-1]
        target = data[:, 1:]

        output = model(src)
        output = output.reshape(-1, output.shape[-1])
        target = target.reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return model


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Transformer Text Generator")

input_text = st.text_area("Enter training text:", "hello world hello transformer world")

if st.button("Train Model"):
    word2idx, idx2word = build_vocab(input_text)
    encoded = encode(input_text, word2idx)

    data = torch.tensor([encoded], dtype=torch.long)

    model = TransformerModel(vocab_size=len(word2idx))
    model = train_model(model, data)

    st.session_state.model = model
    st.session_state.word2idx = word2idx
    st.session_state.idx2word = idx2word

    st.success("Model trained successfully!")


# -----------------------------
# Text Generation
# -----------------------------

def generate_text(model, seed_text, word2idx, idx2word, max_len=10):
    model.eval()
    tokens = encode(seed_text, word2idx)

    for _ in range(max_len):
        inp = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            output = model(inp)

        next_token = torch.argmax(output[0, -1]).item()
        tokens.append(next_token)

    return decode(tokens, idx2word)


if "model" in st.session_state:
    seed = st.text_input("Enter seed text:", "hello")

    if st.button("Generate"):
        result = generate_text(
            st.session_state.model,
            seed,
            st.session_state.word2idx,
            st.session_state.idx2word
        )
        st.write("Generated Text:")
        st.success(result)
