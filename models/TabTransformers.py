import torch
import torch.nn as nn
import torch.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is expected to have shape [batch_size, sequence_length, embed_size]
        attention = self.attention(x, x, x)[0]
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, num_heads, forward_expansion, dropout):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_size)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size,
            heads=num_heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )
        self.fc = nn.Linear(embed_size, output_dim)

    def forward(self, x):
        # Add sequence dimension
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_block(x)
        x = x.squeeze(1)
        out = self.fc(x)
        return out