import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUGate(nn.Module):
    def __init__(self, d_model):
        super(GRUGate,self).__init__()

        self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_z = nn.Linear(d_model, d_model)
        self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)

    def forward(self, x, y):
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r*x))
        return (1.-z)*x + z*h_hat


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, dropout=0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k)  # queries
        self.w_ks = nn.Linear(d_model, n_head * d_k)  # keys
        self.w_vs = nn.Linear(d_model, n_head * d_k)  # values
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_model)
            
    def forward(self, x, adj):
        residual = x
        x = self.ln(x)

        d_k, n_head = self.d_k, self.n_head

        sz_b, len_x, _ = x.size()

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lv x dv

        adj = adj.unsqueeze(1).repeat(1, n_head, 1, 1).reshape(-1, len_x, len_x)
        output = self.attention(q, k, v, adj)
        output = output.view(n_head, sz_b, len_x, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_x, -1)  # b x lq x (n*dk)

        output = F.relu(self.dropout(self.fc(output)))
        output = self.gate(residual, output)
        return output



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, dhid, dropout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, dhid) # dhid (dimmension hidden layes should be bigger than in / out (
        self.w_2 = nn.Linear(dhid, d_in)
        self.ln = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_in)
            
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = F.relu(self.w_2(F.relu((self.w_1(x)))))
        return self.gate(residual, x)
        
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, adj):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = attn.masked_fill(adj == 0, -np.inf)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


class MyMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, dropout=0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        nn.init.normal_(self.qkv_layer.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.head_dim)))
        self.attention = ScaledDotProductAttention(temperature=np.power(self.head_dim, 0.5), attn_dropout=dropout)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_model)

    def forward(self, x, adj):
        residual = x
        batch_size, seq_length, _ = x.size()
        x = self.ln(x)
        qkv = self.qkv_layer(x).reshape(batch_size, seq_length, self.n_head, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3).contiguous().view(-1, seq_length,
                                                        3 * self.head_dim)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        adj = adj.unsqueeze(1).repeat(1, self.n_head, 1, 1).reshape(-1, seq_length, seq_length)
        output = self.attention(q, k, v, adj)

        output = output.view(self.n_head, batch_size, seq_length, self.head_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_length, -1)  # b x lq x (n*dk)
        output = F.relu(self.dropout(self.fc(output)))
        output = self.gate(residual, output)
        return output
