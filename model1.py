import torch
from torch import nn
import math
import torch.nn.functional as F


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super().__init__()
        self.register_buffer('encoding', torch.zeros(max_len, embedding_dim))
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             (-math.log(10000.0) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return self.encoding[:, :x.size(1), :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, dropout):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_emb = PositionalEmbedding(embedding_dim, max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, x):
        tok_emb = self.tok_emb(x) * self.scale
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.split_heads(self.w_q(q))
        K = self.split_heads(self.w_k(k))
        V = self.split_heads(self.w_v(v))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN架构
        attn_out = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x), mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec, enc, t_mask, s_mask):
        # Self attention
        attn_out = self.self_attn(
            self.self_attn_norm(dec),
            self.self_attn_norm(dec),
            self.self_attn_norm(dec),
            s_mask
        )
        dec = dec + self.dropout(attn_out)

        # Cross attention
        cross_out = self.cross_attn(
            self.cross_attn_norm(dec),
            self.cross_attn_norm(enc),
            self.cross_attn_norm(enc),
            t_mask
        )
        dec = dec + self.dropout(cross_out)

        # FFN
        ffn_out = self.ffn(self.ffn_norm(dec))
        dec = dec + self.dropout(ffn_out)
        return dec


class Encoder(nn.Module):
    def __init__(self, voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(voc_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                                     for _ in range(n_layer)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(voc_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, dropout)
                                     for _ in range(n_layer)])
        self.fc = nn.Linear(d_model, voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        return self.fc(dec)


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size,
                 d_model=512, max_len=100, n_head=8, ffn_hidden=2048,
                 n_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, dropout)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_mask(self, src, trg):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        seq_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones(seq_len, seq_len, device=trg.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return src_mask, trg_mask

    def forward(self, src, trg):
        src_mask, trg_mask = self.make_mask(src, trg)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return dec_out