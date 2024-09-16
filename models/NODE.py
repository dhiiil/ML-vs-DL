import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepObliviousDecisionTreeLayer(nn.Module):
    def __init__(self, input_dim, num_trees, tree_depth, hidden_dim):
        super(DeepObliviousDecisionTreeLayer, self).__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.hidden_dim = hidden_dim
        
        # Define a deeper architecture with hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(tree_depth):
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim 
        
        self.output_layer = nn.Linear(hidden_dim, num_trees)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x)) 
        
        # Apply the output layer
        out = torch.sigmoid(self.output_layer(x))
        return out

class NODE(nn.Module):
    def __init__(self, input_dim, output_dim, num_trees, tree_depth, hidden_dim):
        super(NODE, self).__init__()
        self.tree_layer = DeepObliviousDecisionTreeLayer(input_dim, num_trees=num_trees, tree_depth=tree_depth, hidden_dim=hidden_dim)
        
        # Additional fully connected layers to make it deeper
        self.fc1 = nn.Linear(num_trees, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.tree_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x