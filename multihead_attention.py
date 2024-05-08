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

    def forward(self, x):
        """
        Forward Call

        Arguments

        x - PyTorch Tensor
            Input
        """
        q = self.w_q(x).view(-1, self.num_heads, self.out_dim).transpose(0, 1)
        k = self.w_k(x).view(-1, self.num_heads, self.out_dim).transpose(0, 1)
        v = self.w_v(x).view(-1, self.num_heads, self.out_dim).transpose(0, 1)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.out_dim)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, v)

        attention_output = (
            attention_output.transpose(0, 1)
            .contiguous()
            .view(-1, self.num_heads * self.out_dim)
        )

        return attention_output
