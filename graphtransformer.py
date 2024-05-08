"""
Vinay Williams
8th May 2024
Graph Transformer - Graph Transformer
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

from multihead_attention import MultiheadAttention


class GraphTransformer(nn.Module):
    """
    Graph Transformer
    """

    def __init__(
        self,
        I,
        O,
        N,
        dropout=0.0,
        layer_norm=True,
        batch_norm=True,
        residual=True,
        use_bias=True,
    ):
        """
        Constructor

        Arguments

        I - int
            Input Dimensions
        O - int
            Output Dimensions
        N - int
            Number of Attention Heads
        dropoput - float
            Dropout rate
        layer_norm - bool
            Layer normalisation flag
        batch_norm - bool
            Batch Normalisation flag
        residual - bool
            Residual connections flag
        use_bias - bool
            Use bias flag
        """
        super().__init__()

        self.in_channels = I
        self.out_channels = O
        self.num_heads = N
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiheadAttention(
            I, O // N, N, use_bias
        )

        self.w_o = nn.Linear(O, O)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(O)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(O)

        self.w_1 = nn.Linear(O, O * 2)
        self.w_2 = nn.Linear(O * 2, O)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(O)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(O)

    def forward(self, x):
        """
        Forward Call

        Arguments

        x - PyTorch Tensor
            node features
        edge_index - PyTorch Tensor
            Edge Connectivity
        """
        h_in1 = x
        attn_out = self.attention(x)
        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
         
        h = self.w_o(h)

        if self.residual:
            h = h_in1 + h

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h

        h = self.w_1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.w_2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h


if __name__ == "__main__":
    num_nodes = 10
    edge_index = torch.randint(0, num_nodes, (2, 20))
    x = torch.randn(num_nodes, 32)

    data = Data(x=x, edge_index=edge_index)

    in_dim = 32
    out_dim = 32
    num_heads = 4
    layer = GraphTransformer(in_dim, out_dim, num_heads)

    out = layer(data.x)
    print("Input shape:", data.x.shape)
    print("Output shape:", out.shape)
