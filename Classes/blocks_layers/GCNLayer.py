class GCNLayer(nn.Module):
    """
    Graph Convolutional Network (GCN) layer.

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        bias (bool): Whether to include a learnable bias.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GCN layer.
        
        Args:
            X (torch.Tensor): Node features of shape (num_nodes, in_features).
            A (torch.Tensor): Pre-normalized adjacency matrix of shape (num_nodes, num_nodes).
        
        Returns:
            torch.Tensor: Updated node features of shape (num_nodes, out_features).
        """
        support = self.linear(X)
        out = torch.matmul(A, support)
        return F.relu(out)
