class GCN(Model):
    def __init__(self, n_hidden, n_classes, dropout_rate):
        """
        Initializes a simple 2-layer GCN.
        
        Args:
            n_hidden (int): Number of hidden units.
            n_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        # First GCN layer: from input features to hidden representation.
        self.gcn1 = GCNConv(n_hidden, activation='relu')
        self.dropout = Dropout(dropout_rate)
        # Second GCN layer: from hidden representation to output classes.
        self.gcn2 = GCNConv(n_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        """
        Forward pass of the GCN.
        
        Args:
            inputs (tuple): A tuple (X, A) where X are node features and A is the adjacency matrix.
            training (bool): Whether in training mode (to apply dropout).
        
        Returns:
            Tensor: Node-level predictions.
        """
        x, a = inputs  # x: node features, a: adjacency matrix
        x = self.gcn1([x, a])
        if training:
            x = self.dropout(x, training=training)
        x = self.gcn2([x, a])
        return x
