"""
Mamba State Space Model (SSM) CUDA Execution & Gradient Verification Test.
Tests PyTorch GPU acceleration on NVIDIA RTX A5000 with 22-channel EEG inputs.
"""

import sys
import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

def test_mamba_gpu():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 70)
    print("Mamba State Space Model (SSM) GPU Verification Test")
    print("=" * 70)

    # 1. Check PyTorch CUDA
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if not cuda_avail:
        print("ERROR: PyTorch CUDA is not available!")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Device:     {device_name}")

    # 2. Instantiate Mamba SSM Block
    d_model = 128  # Feature embedding dimension
    n_layers = 4   # Number of Mamba SSM layers
    config = MambaConfig(d_model=d_model, n_layers=n_layers)

    model = Mamba(config).to('cuda')
    print(f"Mamba Layers:   {n_layers}")
    print(f"Mamba Model D:  {d_model}")

    # Projection layer for 22 EEG channels to d_model
    in_proj = nn.Linear(22, d_model).to('cuda')
    classifier = nn.Linear(d_model, 6).to('cuda')  # 6 clinical event classes

    # 3. Simulate EEG Batch Input: (Batch=8, Sequence=500 time steps, 22 EEG Channels)
    batch_size = 8
    seq_len = 500
    x_eeg = torch.randn(batch_size, seq_len, 22, device='cuda')
    targets = torch.randint(0, 6, (batch_size,), device='cuda')

    print(f"\nInput EEG Batch Shape: {tuple(x_eeg.shape)}")

    # 4. Forward Pass
    h = in_proj(x_eeg)            # Shape: (8, 500, 128)
    mamba_out = model(h)          # Shape: (8, 500, 128)
    pooled = mamba_out.mean(dim=1) # Global sequence pooling -> Shape: (8, 128)
    logits = classifier(pooled)   # Shape: (8, 6)

    print(f"Mamba Output Shape:    {tuple(mamba_out.shape)}")
    print(f"Class Logits Shape:    {tuple(logits.shape)}")

    # 5. Backward Pass (Gradient Check)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    print(f"CrossEntropy Loss:     {loss.item():.4f}")
    print("=" * 70)
    print("SUCCESS: Mamba State Space Model is fully functional on CUDA GPU!")
    print("=" * 70)

if __name__ == '__main__':
    test_mamba_gpu()
