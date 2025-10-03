import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_layer=1, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.price_feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.direction_feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.price_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        self.direction_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict price change
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Predict direction: up, down, or flat
        )


    def forward(self, x):
        # Process features separately
        price_features = self.price_feature_net(x)
        direction_features = self.direction_feature_net(x)
        
        # Shared GRU processing
        price_gru_out, _ = self.gru(price_features)
        direction_gru_out, _ = self.gru(direction_features)
        
        # Attention mechanisms
        price_attn_weights = torch.softmax(
            self.price_attention(price_gru_out), dim=1
        )
        price_context = torch.sum(price_attn_weights * price_gru_out, dim=1)
        
        direction_attn_weights = torch.softmax(
            self.direction_attention(direction_gru_out), dim=1
        )
        direction_context = torch.sum(direction_attn_weights * direction_gru_out, dim=1)
        
        # Output predictions
        price_change = self.price_head(price_context)
        direction_logit = self.direction_head(direction_context)
        
        return price_change, direction_logit