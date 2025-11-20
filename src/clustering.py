# clustering.py
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

class WHAC_Clustering:
    def __init__(self, periodic_weights, window_size=60, n_clusters=5):
        self.pw = periodic_weights
        self.window_size = window_size
        self.n_clusters = n_clusters

    def segment_data(self, data, stride=None):
        """
        Updated to accept custom stride.
        Default (None) uses window_size // 2 for fast training.
        Set stride=1 for accurate testing.
        """
        N, M = data.shape
        segments = []
        
        # If no stride provided, use half-window (fast for training)
        if stride is None:
            actual_stride = self.window_size // 2
        else:
            actual_stride = stride
            
        for i in range(0, N - self.window_size + 1, actual_stride):
            seg = data[i : i + self.window_size, :]
            segments.append(seg)
        
        return np.array(segments)

    def weighted_euclidean(self, A, B):
        diff_sq = (A - B) ** 2
        return np.sum(diff_sq * self.pw)

    def align_phase_shifts(self, segments):
        # print("[Clustering] Aligning Phase Shifts...") # Commented out to reduce spam in detection
        aligned = []
        if len(segments) == 0: return np.array([])
        
        pivot = np.mean(segments, axis=0)
        max_shift = int(self.window_size * 0.1)
        
        for seg in segments:
            best_shift = 0
            min_dist = float('inf')
            for s in range(-max_shift, max_shift + 1):
                shifted = np.roll(seg, s, axis=0)
                dist = self.weighted_euclidean(shifted, pivot)
                if dist < min_dist:
                    min_dist = dist
                    best_shift = s
            aligned.append(np.roll(seg, best_shift, axis=0))
        return np.array(aligned)

    def fit_predict(self, segments):
        print(f"[Clustering] Running HAC on {len(segments)} segments...")
        if len(segments) > 2000:
            idx = np.random.choice(len(segments), 2000, replace=False)
            segments_fit = segments[idx]
        else:
            segments_fit = segments
            
        flat_data = segments_fit.reshape(len(segments_fit), -1)
        pw_flat = np.tile(np.sqrt(self.pw), self.window_size)
        weighted_data = flat_data * pw_flat
        
        Z = linkage(weighted_data, method='average', metric='euclidean')
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust')
        
        return labels, segments_fit