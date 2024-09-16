import torch
import torch.nn as nn
import torch.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attn_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        
        self.query = nn.Linear(input_dim, attn_dim * num_heads)
        self.key = nn.Linear(input_dim, attn_dim * num_heads)
        self.value = nn.Linear(input_dim, attn_dim * num_heads)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(attn_dim * num_heads, input_dim)
        
    def forward(self, x):
        if len(x.size()) == 2:  # If input has 2 dimensions
            x = x.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, input_dim]

        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_layer(attn_output)

        return output

class IntersampleAttention(nn.Module):
    def __init__(self, input_dim, attn_dim, num_heads, dropout=0.1):
        super(IntersampleAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        
        self.query = nn.Linear(input_dim, attn_dim * num_heads)
        self.key = nn.Linear(input_dim, attn_dim * num_heads)
        self.value = nn.Linear(input_dim, attn_dim * num_heads)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(attn_dim * num_heads, input_dim)
        
    def forward(self, x):
        if len(x.size()) == 2:  # If input has 2 dimensions
            x = x.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, input_dim]

        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_layer(attn_output)

        return output

class SAINT(nn.Module):
    def __init__(self, input_dim, output_dim, attn_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(SAINT, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.self_attn_layers = nn.ModuleList([
            SelfAttention(hidden_dim, attn_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.intersample_attn_layers = nn.ModuleList([
            IntersampleAttention(hidden_dim, attn_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        for self_attn, intersample_attn in zip(self.self_attn_layers, self.intersample_attn_layers):
            x = self_attn(x)
            x = intersample_attn(x)
        
        x = torch.mean(x, dim=1)  # Global average pooling
        
        output = self.output_layer(x)
        
        return output