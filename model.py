import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_layer=1, dropout=0.2):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_output = gru_out[:, -1, :]
        prediction = self.fc(last_output)

        return prediction
