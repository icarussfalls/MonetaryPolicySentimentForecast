import torch
import torch.nn as nn

class FinBERTRegressor(nn.Module):
    def __init__(self, num_sectors, sector_embed_dim=32):
        super().__init__()
        self.sector_embedding = nn.Embedding(num_sectors, sector_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(768 + sector_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, report_emb, sector_id):
        sector_emb = self.sector_embedding(sector_id)
        x = torch.cat((report_emb, sector_emb), dim=1)
        return self.fc(x).squeeze(-1)
