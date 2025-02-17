class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, hidden_dim):
        """
        Encoder network that maps context (x, y) pairs into latent representation r.
        
        Args:
            x_dim (int): Dimensionality of x (input).
            y_dim (int): Dimensionality of y (output).
            r_dim (int): Dimensionality of the latent representation.
            hidden_dim (int): Number of units in hidden layers.
        """
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r_dim)
        )
    
    def forward(self, x, y):
        # x: (n_context, x_dim), y: (n_context, y_dim)
        input_pair = torch.cat([x, y], dim=-1)
        r = self.net(input_pair)  # (n_context, r_dim)
        return r
