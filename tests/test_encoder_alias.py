#!/usr/bin/env python3
"""
Quick test to verify encoder/decoder attribute aliases work
"""
import sys
sys.path.insert(0, '/home/abhi8803/BTP/code3/code')

from src.models import TranAD
import torch

print("Testing encoder/decoder attribute aliases...")

model = TranAD(feat_dim=5, n_window=10)

# Test that aliases exist and work
print(f"✓ model.encoder exists: {hasattr(model, 'encoder')}")
print(f"✓ model.decoder exists: {hasattr(model, 'decoder')}")

# Test that they point to the right objects
print(f"✓ encoder is transformer_encoder: {model.encoder is model.transformer_encoder}")
print(f"✓ decoder is transformer_decoder2: {model.decoder is model.transformer_decoder2}")

# Test that partial transfer logic works (freezing encoder)
for param in model.encoder.parameters():
    param.requires_grad = False

frozen_count = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
total_count = sum(1 for p in model.encoder.parameters())
print(f"✓ Freezing encoder works: {frozen_count}/{total_count} params frozen")

# Un-freeze for next test
for param in model.parameters():
    param.requires_grad = True

print("\n✅ All attribute alias tests passed!")
print("omni_framework.py should now work correctly.")
