import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

class WHAC_Clustering:
    def __init__(self, periodic_weights, window_size=60, n_clusters=None):
        """
        n_clusters: If Int, forces that number. If None, auto-tunes.
        """
        self.pw = periodic_weights
        self.window_size = window_size
        self.n_clusters = n_clusters

    def segment_data(self, data, stride=None):
        N, M = data.shape
        segments = []
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

    def _safe_shift(self, segment, shift):
        """
        Shifts segment by 'shift' steps along axis 0.
        Pads with the edge value instead of wrapping (np.roll).
        """
        if shift == 0: return segment
        
        result = np.empty_like(segment)
        if shift > 0:
            result[:shift] = segment[0]
            result[shift:] = segment[:-shift]
        else:
            # shift < 0
            result[shift:] = segment[-1]
            result[:shift] = segment[-shift:]
        return result

    def align_phase_shifts(self, segments, pivot=None):
        if len(segments) == 0: return np.array([])
        
        # FIX 4: Iterative alignment to refine the pivot
        # If pivot is provided (e.g. from training), use it as fixed reference.
        if pivot is not None:
            aligned = []
            max_shift = int(self.window_size * 0.1)
            for seg in segments:
                best_shift = 0
                min_dist = float('inf')
                for s in range(-max_shift, max_shift + 1):
                    shifted = self._safe_shift(seg, s)
                    dist = self.weighted_euclidean(shifted, pivot)
                    if dist < min_dist:
                        min_dist = dist
                        best_shift = s
                aligned.append(self._safe_shift(seg, best_shift))
            return np.array(aligned)

        # Otherwise, self-align iteratively
        current_aligned = segments
        for _ in range(3):
            pivot = np.mean(current_aligned, axis=0)
            max_shift = int(self.window_size * 0.1)
            new_aligned = []
            
            for seg in segments:
                best_shift = 0
                min_dist = float('inf')
                for s in range(-max_shift, max_shift + 1):
                    shifted = self._safe_shift(seg, s)
                    dist = self.weighted_euclidean(shifted, pivot)
                    if dist < min_dist:
                        min_dist = dist
                        best_shift = s
                new_aligned.append(self._safe_shift(seg, best_shift))
            current_aligned = np.array(new_aligned)
            
        return current_aligned

    def fit_predict(self, segments):
        print(f"[Clustering] Running HAC on {len(segments)} segments...")
        
        # Downsample for speed (HAC is O(N^3))
        if len(segments) > 3000:
            idx = np.random.choice(len(segments), 3000, replace=False)
            segments_fit = segments[idx]
            is_subset = True
        else:
            segments_fit = segments
            is_subset = False
            
        # Flatten & Weight for Linkage
        flat_data = segments_fit.reshape(len(segments_fit), -1)
        pw_flat = np.tile(np.sqrt(self.pw), self.window_size)
        weighted_data = flat_data * pw_flat
        
        # 1. Perform Linkage
        Z = linkage(weighted_data, method='average', metric='euclidean')
        
        # 2. Determine K (Manual or Auto)
        if self.n_clusters is not None:
            final_k = self.n_clusters
        else:
            final_k = self._auto_find_k(weighted_data, Z)
            
        print(f"[Clustering] Selected optimal clusters: k={final_k}")
        
        # 3. Assign Labels (Using final_k)
        labels_fit = fcluster(Z, t=int(final_k), criterion='maxclust')
        
        if not is_subset:
            return labels_fit, segments
            
        # FIX 3: If we downsampled, assign ALL data to the nearest cluster centroid
        print(f"   > Assigning remaining {len(segments) - 3000} segments to clusters...")
        
        # Calculate centroids from the subset
        centroids = {}
        unique_labels = np.unique(labels_fit)
        for cid in unique_labels:
            cluster_data = segments_fit[labels_fit == cid]
            centroids[cid] = np.mean(cluster_data, axis=0)
            
        # Assign all segments
        final_labels = []
        for seg in segments:
            best_cid = -1
            min_dist = float('inf')
            for cid, centroid in centroids.items():
                dist = self.weighted_euclidean(seg, centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_cid = cid
            final_labels.append(best_cid)
            
        return np.array(final_labels), segments

    def _auto_find_k(self, data, Z):
        """
        Uses Silhouette Score to find best K between 2 and 10.
        """
        best_k = 2
        best_score = -1
        
        print("   > Auto-tuning K...", end=" ")
        # Try a range of clusters
        search_range = range(2, min(11, len(data)))
        
        for k in search_range:
            labels = fcluster(Z, t=k, criterion='maxclust')
            
            # Safety check: Need at least 2 clusters and distinct labels
            if len(np.unique(labels)) < 2: continue
            
            score = silhouette_score(data, labels, metric='euclidean')
            
            # Simple logic: maximize silhouette
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"Done. Best Score: {best_score:.3f}")
        return best_k