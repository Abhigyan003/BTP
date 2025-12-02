"""
Quick verification test for JumpStarter integration.
Tests the adapter can be imported and basic functionality works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Test 1: Import test
print("Test 1: Importing JumpStarter adapter...")
try:
    from src.jumpstarter_adapter import JumpStarterAdapter
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check if JumpStarter path is correct
print("\nTest 2: Checking JumpStarter module availability...")
jumpstarter_path = os.path.join(os.path.dirname(__file__), '../..', 'JumpStarter')
sys.path.insert(0, jumpstarter_path)
try:
    from detector import CSAnomalyDetector
    print(f"✓ JumpStarter found at: {jumpstarter_path}")
except Exception as e:
    print(f"✗ JumpStarter not accessible: {e}")
    sys.exit(1)

# Test 3: Create adapter instance
print("\nTest 3: Creating adapter instance...")
try:
    adapter = JumpStarterAdapter(
        feat_dim=5,
        sample_rate=0.3,
        cluster_threshold=0.15,
        workers=1,
        window=10,
        windows_per_cycle=3
    )
    print("✓ Adapter instance created successfully")
    print(f"  Config: feat_dim=5, window=10, sample_rate=0.3")
except Exception as e:
    print(f"✗ Failed to create adapter: {e}")
    sys.exit(1)

# Test 4: Test with synthetic data (minimal)
print("\nTest 4: Testing with synthetic data...")
try:
    # Create small synthetic dataset
    np.random.seed(42)
    n_samples = 300  # Enough for window=10, windows_per_cycle=3
    train_data = np.random.randn(n_samples, 5) * 0.5 + 1.0
    test_data = np.random.randn(100, 5) * 0.5 + 1.0
    
    print(f"  Train shape: {train_data.shape}")
    print(f"  Test shape: {test_data.shape}")
    
    # Train (reconstruct)
    print("  Training (reconstructing)...")  
    reconstructed, retries = adapter.train(train_data)
    print(f"  ✓ Training complete. Retries: {retries}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    
    # Detect  
    print("  Detecting anomalies...")
    scores = adapter.detect(test_data)
    print(f"  ✓ Detection complete")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    print("\n✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("JumpStarter adapter is ready for use in full_comparison.py")
print("="*60)
