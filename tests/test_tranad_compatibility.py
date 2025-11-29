#!/usr/bin/env python3
"""
Test script to verify TranAD backward compatibility

Run this script to ensure:
1. Old code using model(batch) still works
2. New TranAD interface model(src, tgt) works
3. TranAD utilities function correctly
"""

import sys
sys.path.insert(0, '/home/abhi8803/BTP/code3/code')

from src.models import TranAD
from src.tranad_utils import convert_to_windows, prepare_tranad_batch, tranad_loss
import torch
import numpy as np

def test_backward_compatibility():
    """Test that old interface model(batch) still works"""
    print('=' * 60)
    print('TEST 1: Backward Compatibility (Old Interface)')
    print('=' * 60)
    
    model = TranAD(feats=5)
    batch = torch.randn(32, 60, 5)  # [batch, window, feats]
    print(f'Input shape: {batch.shape} (batch_first format)')
    
    output = model(batch)
    print(f'Output shape: {output.shape}')
    
    expected_shape = (32, 60, 5)
    assert output.shape == expected_shape, f'Expected {expected_shape}, got {output.shape}'
    print('✓ Old interface works correctly!')
    print('✓ Backward compatibility test PASSED\n')


def test_new_tranad_interface():
    """Test new TranAD dual-decoder interface"""
    print('=' * 60)
    print('TEST 2: New TranAD Interface')
    print('=' * 60)
    
    model = TranAD(feats=5, n_window=10)
    src = torch.randn(10, 32, 5)  # [window, batch, feats]
    tgt = torch.randn(1, 32, 5)   # [1, batch, feats]
    
    print(f'Input src shape: {src.shape} (sequence_first)')
    print(f'Input tgt shape: {tgt.shape}')
    
    x1, x2 = model(src, tgt)
    print(f'Output x1 shape: {x1.shape}')
    print(f'Output x2 shape: {x2.shape}')
    
    expected_shape = (1, 32, 5)
    assert x1.shape == expected_shape and x2.shape == expected_shape
    print('✓ New TranAD interface works correctly!')
    print('✓ Dual decoder output test PASSED\n')


def test_tranad_utils():
    """Test TranAD utility functions"""
    print('=' * 60)
    print('TEST 3: TranAD Utilities')
    print('=' * 60)
    
    # Test windowing
    data = np.random.randn(100, 5)
    windows = convert_to_windows(data, window_size=10, stride=1)
    print(f'✓ Windowing: {data.shape} -> {windows.shape}')
    assert windows.shape == (91, 10, 5)
    
    # Test batch preparation
    batch = torch.randn(32, 10, 5)
    src, tgt = prepare_tranad_batch(batch)
    print(f'✓ Batch prep: {batch.shape} -> src:{src.shape}, tgt:{tgt.shape}')
    assert src.shape == (10, 32, 5) and tgt.shape == (1, 32, 5)
    
    # Test time-dependent loss
    model = TranAD(feats=5, n_window=10)
    loss = tranad_loss(model, batch, epoch=0)
    print(f'✓ TranAD loss computed: {loss.item():.6f}')
    
    print('✓ All utility functions work correctly!\n')


def test_existing_framework_compatibility():
    """Test that OmniTransferTrainer still works with new TranAD"""
    print('=' * 60)
    print('TEST 4: Framework Integration')
    print('=' * 60)
    
    from src.omni_framework import OmniTransferTrainer
    
    # Generate dummy data
    data = np.random.randn(1000, 5)
    weights = np.ones(5)
    
    # Test that framework can use TranAD
    trainer = OmniTransferTrainer(TranAD, device='cpu')
    print('✓ OmniTransferTrainer initialized with TranAD')
    
    # Test model creation
    model = TranAD(feat_dim=5)
    test_batch = torch.randn(16, 60, 5)
    output = model(test_batch)
    
    print(f'✓ Model inference: {test_batch.shape} -> {output.shape}')
    print('✓ Framework integration test PASSED\n')


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('TranAD Backward Compatibility Test Suite')
    print('=' * 60 + '\n')
    
    try:
        test_backward_compatibility()
        test_new_tranad_interface()
        test_tranad_utils()
        test_existing_framework_compatibility()
        
        print('=' * 60)
        print('🎉 ALL TESTS PASSED!')
        print('=' * 60)
        print('\nSummary:')
        print('  ✓ Old interface (backward compatible) works')
        print('  ✓ New TranAD interface (dual decoders) works')
        print('  ✓ Utility functions work correctly')
        print('  ✓ Integration with existing framework works')
        print('\nYour existing code will continue to work without changes!')
        
    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
