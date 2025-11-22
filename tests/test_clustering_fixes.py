import unittest
import numpy as np
from src.clustering import WHAC_Clustering

class TestClusteringFixes(unittest.TestCase):
    def setUp(self):
        self.whac = WHAC_Clustering(periodic_weights=np.ones(10), window_size=10)

    def test_safe_shift_positive(self):
        arr = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        shifted = self.whac._safe_shift(arr, 2)
        # Expected: First 2 rows are [1, 1], rest shifted down
        expected = np.array([[1, 1], [1, 1], [1, 1], [2, 2], [3, 3]])
        np.testing.assert_array_equal(shifted, expected)

    def test_safe_shift_negative(self):
        arr = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        shifted = self.whac._safe_shift(arr, -2)
        # Expected: Last 2 rows are [5, 5], rest shifted up
        expected = np.array([[3, 3], [4, 4], [5, 5], [5, 5], [5, 5]])
        np.testing.assert_array_equal(shifted, expected)

    def test_align_phase_shifts_iterative(self):
        # Create 3 segments that are identical but shifted
        base = np.zeros((10, 10))
        base[5, :] = 1 # Spike at index 5
        
        seg1 = base.copy()
        seg2 = self.whac._safe_shift(base, 1) # Spike at 6
        seg3 = self.whac._safe_shift(base, -1) # Spike at 4
        
        segments = np.array([seg1, seg2, seg3])
        
        # Alignment should bring them to a common phase (likely the majority or mean)
        aligned = self.whac.align_phase_shifts(segments)
        
        # Check if they are more similar to each other than before
        # Variance of aligned should be lower than variance of raw
        raw_var = np.sum(np.var(segments, axis=0))
        aligned_var = np.sum(np.var(aligned, axis=0))
        
        print(f"Raw Variance: {raw_var}, Aligned Variance: {aligned_var}")
        self.assertLess(aligned_var, raw_var)

if __name__ == '__main__':
    unittest.main()
