import numpy as np
import matplotlib.pyplot as plt
from src.clustering import WHAC_Clustering

def generate_sine_wave(phase_shift, n_samples=100, window_size=60):
    t = np.linspace(0, 4*np.pi, n_samples)
    data = np.sin(t + phase_shift).reshape(-1, 1)
    return data

def main():
    # Setup
    whac = WHAC_Clustering(periodic_weights=np.ones(1), window_size=60)
    
    # 1. Create Training Data (Phase 0)
    # Randomly sampled windows from a sine wave with Phase 0
    train_data = generate_sine_wave(phase_shift=0, n_samples=1000)
    train_segments = whac.segment_data(train_data, stride=5)
    
    # Align Training Data
    aligned_train = whac.align_phase_shifts(train_segments)
    train_pivot = np.mean(aligned_train, axis=0)
    
    # 2. Create Test Data (Phase PI/2)
    # Randomly sampled windows from a sine wave with Phase PI/2
    # The underlying shape is the same, just shifted globally.
    test_data = generate_sine_wave(phase_shift=np.pi/2, n_samples=1000)
    test_segments = whac.segment_data(test_data, stride=5)
    
    # Align Test Data (Using Train Pivot as Reference)
    # FIX: Pass the pivot explicitly!
    aligned_test = whac.align_phase_shifts(test_segments, pivot=train_pivot)
    test_pivot = np.mean(aligned_test, axis=0)
    
    # 3. Compare Pivots
    # With the fix, test_pivot should match train_pivot almost exactly.
    
    dist = np.linalg.norm(train_pivot - test_pivot)
    print(f"Distance between Train Pivot and Test Pivot: {dist:.4f}")
    
    # Check if they are just shifted versions
    min_dist = float('inf')
    for s in range(-10, 11):
        shifted_test = whac._safe_shift(test_pivot, s)
        d = np.linalg.norm(train_pivot - shifted_test)
        if d < min_dist: min_dist = d
        
    print(f"Best Shifted Distance: {min_dist:.4f}")
    
    if min_dist > 1.0: # Arbitrary threshold
        print("FAIL: Pivots converged to different canonical phases!")
    else:
        print("SUCCESS: Pivots are consistent.")

if __name__ == "__main__":
    main()
