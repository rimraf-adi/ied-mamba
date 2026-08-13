"""
Official state-spaces/mamba (mamba_ssm v2.3.2) CUDA Execution & Gradient Verification Test.
Tests official state-spaces/mamba (Mamba2 & PyTorch Selective Scan) GPU acceleration on NVIDIA RTX A5000 with 22-channel EEG inputs.
"""

import sys
import torch
import torch.nn as nn

# Import official state-spaces/mamba packages
import causal_conv1d
import mamba_ssm
from mamba_ssm import Mamba2

def test_official_mamba_gpu():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 75)
    print("🚀 Official state-spaces/mamba (mamba_ssm v2.3.2) GPU Verification Test")
    print("=" * 75)

    # 1. Check PyTorch CUDA
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:       {cuda_avail}")
    if not cuda_avail:
        print("ERROR: PyTorch CUDA is not available!")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Device:           {device_name}")
    print(f"causal_conv1d ver:    {causal_conv1d.__version__}")
    print(f"mamba_ssm ver:        {mamba_ssm.__version__}")

    # 2. Instantiate Official state-spaces/mamba Mamba2 Block (State Space Duality)
    d_model = 128  # Feature embedding dimension
    d_state = 64   # SSM state expansion dimension

    mamba_block = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2).to('cuda')
    print(f"\nInitialized Official Mamba2 Block (d_model={d_model}, d_state={d_state})")

    # Projection layer for 22 EEG channels to d_model
    in_proj = nn.Linear(22, d_model).to('cuda')
    classifier = nn.Linear(d_model, 6).to('cuda')  # 6 clinical event classes

    # 3. Simulate EEG Batch Input: (Batch=8, Sequence=500 time steps, 22 EEG Channels)
    batch_size = 8
    seq_len = 500
    x_eeg = torch.randn(batch_size, seq_len, 22, device='cuda')
    targets = torch.randint(0, 6, (batch_size,), device='cuda')

    print(f"Input EEG Batch Shape: {tuple(x_eeg.shape)}")

    # 4. Forward Pass through official state-spaces/mamba
    h = in_proj(x_eeg)              # Shape: (8, 500, 128)
    mamba_out = mamba_block(h)      # Shape: (8, 500, 128)
    pooled = mamba_out.mean(dim=1)  # Global sequence pooling -> Shape: (8, 128)
    logits = classifier(pooled)     # Shape: (8, 6)

    print(f"Mamba2 Output Shape:   {tuple(mamba_out.shape)}")
    print(f"Class Logits Shape:    {tuple(logits.shape)}")

    # 5. Backward Pass (Gradient Check)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    print(f"CrossEntropy Loss:     {loss.item():.4f}")
    print("=" * 75)
    print("✅ SUCCESS: Official state-spaces/mamba (mamba_ssm Mamba2) is functional on CUDA!")
    print("=" * 75)

if __name__ == '__main__':
    test_official_mamba_gpu()
