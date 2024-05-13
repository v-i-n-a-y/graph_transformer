"""
Vinay Williams
8th May 2024
Graph Transformer - Multihead Attention (only nodes)
"""

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    Multihead Attention
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        """
        Constructor

        Arguments

        in_dim - int
            Input Dimensions
        out_dim - int
            Output Dimensions
        num_heads - int
            Number of attention heads
        use_bias - bool
            Use bias flag
        """

        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.w_q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.w_k = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.w_v = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, x, edge_index=None):
        """
        Forward Call

        Arguments

        x - PyTorch Tensor
            Input
        edge_index - PyTorch Tensor
            Sparse Adjacency Matrix used to apply attention to only neighbours
        """
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        q, k, v = map(lambda t: t.view(-1, self.num_heads, self.out_dim).transpose(0, 1), (q, k, v))

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.out_dim)

        if edge_index is not None:
            temp = torch.zeros_like(attention_scores)
            temp[-1, edge_index[0], edge_index[1]] = attention_scores[-1, edge_index[0], edge_index[1]]
            attention_scores = temp

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = torch.matmul(attention_scores, v)

        attention_scores = attention_scores.transpose(0, 1).contiguous().view(-1, self.num_heads * self.out_dim)

        return attention_scores
