#!/bin/bash
# Test different alpha values for entropy

echo "Testing Entropy with alpha=0.5 (less aggressive)"
python -m scripts.full_comparison --datasets MSL --configs TranAD_Scratch Omni_Entropy --alpha 0.5

echo ""
echo "=========================================="
echo "Testing Entropy with alpha=1.0 (moderate)"
python -m scripts.full_comparison --datasets MSL --configs TranAD_Scratch Omni_Entropy --alpha 1.0

echo ""
echo "=========================================="
echo "Testing Entropy with alpha=2.0 (current default)"
python -m scripts.full_comparison --datasets MSL --configs TranAD_Scratch Omni_Entropy --alpha 2.0

echo ""
echo "=========================================="
echo "COMPARISON COMPLETE"
echo "Check results/csv/ for outputs"
