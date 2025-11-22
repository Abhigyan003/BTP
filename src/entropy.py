import numpy as np
import math

class EntropicWeightCalculator:
    def __init__(self, alpha=1.0, order=3, delay=1):
        """
        alpha: Exponent for weighting (higher = harsher penalty for noise)
        order (D): Length of the patterns to check (3-5 is standard)
        delay: Gap between points (usually 1)
        """
        self.alpha = alpha
        self.order = order
        self.delay = delay

    def _permutation_entropy(self, time_series):
        """
        Calculates Normalized Permutation Entropy (0.0 to 1.0)
        0.0 = Perfectly Structured (Step, Ramp, Cycle)
        1.0 = Completely Random Noise
        """
        n = len(time_series)
        
        # Safety check for constant/dead signals
        if np.var(time_series) < 1e-6:
            # A flat line is "structured" (Entropy 0), BUT for anomaly detection,
            # a dead sensor is usually useless. We force weight to 0.
            return 1.0 # Entropy 1.0 -> Weight 0.0
            
        # 1. Embed the time series
        # We create a matrix of shape (N - (D-1)*tau, D)
        # Each row is a "window" of length D
        embed_size = n - (self.order - 1) * self.delay
        if embed_size < 1:
            return 1.0
            
        matrix = np.zeros((embed_size, self.order))
        for i in range(self.order):
            matrix[:, i] = time_series[i * self.delay : i * self.delay + embed_size]
            
        # 2. Rank Patterns (The "Permutation" step)
        # argsort returns indices that would sort the array
        # e.g., [10, 5, 8] -> [1, 2, 0] (index 1 is smallest, index 0 is largest)
        # We convert rows to tuples so we can count unique patterns
        patterns = [tuple(row) for row in np.argsort(matrix, axis=1)]
        
        # 3. Calculate Probabilities
        # Count frequency of each unique pattern
        _, counts = np.unique(patterns, return_counts=True, axis=0)
        probs = counts / len(patterns)
        
        # 4. Shannon Entropy
        # H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # 5. Normalize
        # Max entropy is log(factorial(Order))
        max_entropy = np.log(math.factorial(self.order))
        
        if max_entropy == 0: return 0.0
        
        norm_entropy = entropy / max_entropy
        return norm_entropy

    def compute_entropic_weights(self, data):
        """
        Input: (Time, Features) numpy array
        Output: (Features,) weight vector
        """
        M = data.shape[1]
        weights = np.zeros(M)
        
        # Downsample for speed if data is huge (Entropy is O(N), so it's fast, but let's be safe)
        # Using full length is better for accuracy, but 5000 points is plenty for distribution stats.
        if len(data) > 5000:
            # Take a slice from the middle
            mid = len(data) // 2
            data_slice = data[mid-2500 : mid+2500]
        else:
            data_slice = data

        for j in range(M):
            # Calculate H (Entropy)
            h = self._permutation_entropy(data_slice[:, j])
            
            # Calculate Information Score (Structure)
            # IS = 1 - H
            info_score = 1.0 - h
            
            # Apply Power Law (alpha)
            # Weight = (1 - H)^alpha
            weights[j] = np.power(max(0, info_score), self.alpha)
            
        return weights