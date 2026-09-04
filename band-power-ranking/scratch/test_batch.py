import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.backbone import build_backbone

class DummyDataset(Dataset):
    def __len__(self):
        return 2048
    def __getitem__(self, idx):
        return {'window': torch.randn(22, 1000)}

if __name__ == "__main__":
    batch_size = 512
    ds = DummyDataset()
    loader = DataLoader(ds, batch_size=batch_size)
    
    device = torch.device('cuda')
    model = build_backbone('mamba', num_channels=22, embed_dim=256).to(device)
    
    print(f"Starting fake epoch with batch size {batch_size}...", flush=True)
    t0 = time.time()
    
    for batch in loader:
        x = batch['window'].to(device)
        out = model(x)
        
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"Time for {len(loader)} batches ({len(ds)} windows): {t1-t0:.2f}s", flush=True)
    print(f"GPU Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB", flush=True)
