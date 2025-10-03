import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_layer=1, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.feature_net = nn.Sequential(
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
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, x):
        x = self.feature_net(x)
        gru_out, _ = self.gru(x)
        
        attn_weights = torch.softmax(
            self.attention(gru_out), dim=1
        )
        
        context = torch.sum(attn_weights * gru_out, dim=1)
        
        return self.output_net(context)