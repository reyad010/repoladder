#!/usr/bin/env python3
# Ensures ViT-Base pretrained weights are present at
# shared/models/lst_vit/vit_base_patch16_224.pth, downloading via timm if
# missing. Required for gpu-tee mode (SGX enclaves have no network access).
# Idempotent — run.sh calls this on every invocation.

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(_HERE, 'models', 'lst_vit', 'vit_base_patch16_224.pth')

if os.path.exists(TARGET):
    print(f'[vit-weights] already present: {TARGET}')
    sys.exit(0)

print(f'[vit-weights] downloading via timm to {TARGET} ...')
import timm
import torch

m = timm.create_model('vit_base_patch16_224', pretrained=True)
os.makedirs(os.path.dirname(TARGET), exist_ok=True)
torch.save(m.state_dict(), TARGET)
print(f'[vit-weights] saved {os.path.getsize(TARGET) / (1 << 20):.1f} MB to {TARGET}')
