import torch
from torch.utils.data import Dataset

class PolicyReturnDataset(torch.utils.data.Dataset):
    def __init__(self, embedding_dict, df, sector2id):
        self.embedding_dict = embedding_dict
        self.df = df.reset_index(drop=True)
        self.sector2id = sector2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fy = row["FY"]
        sector = row["IndexName"]
        y = row["Return_30d"]

        report_emb = torch.tensor(self.embedding_dict[fy], dtype=torch.float32)
        sector_id = torch.tensor(self.sector2id[sector], dtype=torch.long)

        return report_emb, sector_id, torch.tensor(y, dtype=torch.float32)
