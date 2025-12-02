"""
JumpStarter Adapter for Comparison Framework

Wraps the JumpStarter CSAnomalyDetector to provide a consistent interface
with other models in the comparison framework.
"""

import numpy as np
import sys
import os

# Add JumpStarter to path
jumpstarter_path = os.path.join(os.path.dirname(__file__), '../../JumpStarter')
sys.path.insert(0, jumpstarter_path)

from detector import CSAnomalyDetector


class JumpStarterAdapter:
    """
    Adapter to wrap JumpStarter's CSAnomalyDetector for use in comparison framework.
    
    JumpStarter uses compressive sensing with:
    - Clustering-based feature grouping
    - Localized sampling based on confidence scores
    - CVXPY-based reconstruction
    - Window-based anomaly scoring
    """
    
    def __init__(self, feat_dim, device='cpu', 
                 sample_rate=0.3, 
                 cluster_threshold=0.15,
                 workers=1,
                 window=96,
                 windows_per_cycle=7,
                 rec_stride=1,
                 det_window=10,
                 det_stride=1,
                 latest_windows=96,
                 scale=5.0,
                 rho=0.1,
                 sigma=1/24,
                 random_state=42):
        """
        Initialize JumpStarter adapter.
        
        Args:
            feat_dim: Number of features (dimensions)
            device: Device to use (kept for API consistency, JumpStarter uses CPU/numpy)
            sample_rate: Sampling rate for compressive sensing (0.0-1.0)
            cluster_threshold: Threshold for feature clustering
            workers: Number of parallel workers for reconstruction
            window: Reconstruction window size (timesteps)
            windows_per_cycle: Number of windows per cycle
            rec_stride: Reconstruction stride
            det_window: Detection window size
            det_stride: Detection stride
            latest_windows: Historical windows for sampling confidence
            scale: Sampling parameter - expansion multiplier
            rho: Sampling parameter - center probability
            sigma: Sampling parameter - concentration
            random_state: Random seed for reproducibility
        """
        self.feat_dim = feat_dim
        self.device = device  # Not used by JumpStarter but kept for API consistency
        self.sample_rate = sample_rate
        self.cluster_threshold = cluster_threshold
        self.workers = workers
        self.window = window
        self.windows_per_cycle = windows_per_cycle
        self.rec_stride = rec_stride
        self.det_window = det_window
        self.det_stride = det_stride
        self.latest_windows = latest_windows
        self.scale = scale
        self.rho = rho
        self.sigma = sigma
        self.random_state = random_state
        
        # Normalization stats (computed per feature during training)
        self.means = None
        self.stds = None
        
    def _normalize(self, data):
        """
        Min-max normalization per feature (matches JumpStarter's utils.normalization).
        Normalizes to [0, 1] range.
        """
        n, d = data.shape
        normalized = np.zeros_like(data)
        for i in range(d):
            col = data[:, i]
            _range = np.max(col) - np.min(col)
            if _range == 0:
                normalized[:, i] = np.zeros_like(col) + 0.5
            else:
                normalized[:, i] = (col - np.min(col)) / _range
        return normalized
    
    def _sample_score_lesinn(self, incoming_data, historical_data):
        """
        Compute sampling confidence using LESINN.
        Returns normalized inverse distances.
        EXACTLY matches JumpStarter's lesinn_score function.
        """
        # Import here to avoid circular dependencies
        sys.path.insert(0, os.path.join(jumpstarter_path, 'algorithm'))
        from lesinn import online_lesinn
        
        distances = online_lesinn(
            incoming_data, historical_data,
            random_state=self.random_state,
            t=40,  # From detector-config.yml
            phi=20  # From detector-config.yml
        )
        
        # p_normalize from JumpStarter - EXACT implementation
        p_min = 0.05
        x = 1 / (distances + 1e-8)
        x_max, x_min = np.max(x), np.min(x)
        x_min *= (1 - p_min)
        return (x - x_min) / (x_max - x_min + 1e-8)
    
    def _anomaly_score(self, source, reconstructed):
        """
        Calculate anomaly score using distance-based method.
        EXACTLY matches JumpStarter's anomaly_score_example function.
        """
        n, d = source.shape
        d_dis = np.zeros((d,))
        
        # This is accessed from global config, hardcoded to match detector-config.yml
        anomaly_score_example_percentage = 90  # From config
        anomaly_distance_topn = 2  # From config (was 5, now corrected to 2!)
        
        for i in range(d):
            dis = np.abs(source[:, i] - reconstructed[:, i])
            dis = dis - np.mean(dis)
            d_dis[i] = np.percentile(dis, anomaly_score_example_percentage)
        
        # Use top-n dimensions with largest distances
        if d <= anomaly_distance_topn:
            return d / (np.sum(1 / (d_dis + 1e-8)))
        
        topn_vals = 1 / (d_dis[np.argsort(d_dis)][-anomaly_distance_topn:] + 1e-8)
        return anomaly_distance_topn / (np.sum(topn_vals))
    
    def train(self, data, epochs=None):
        """
        Train JumpStarter model (performs reconstruction).
        
        Args:
            data: Training data, shape (n_samples, n_features)
            epochs: Not used (kept for API consistency)
            
        Returns:
            Tuple of (reconstructed_data, retry_count)
        """
        # Normalize data
        data_normalized = self._normalize(data)
        
        # Create detector instance
        detector = CSAnomalyDetector(
            workers=self.workers,
            cluster_threshold=self.cluster_threshold,
            sample_rate=self.sample_rate,
            sample_score_method=self._sample_score_lesinn,
            distance=self._anomaly_score,
            scale=self.scale,
            rho=self.rho,
            sigma=self.sigma,
            random_state=self.random_state,
            retry_limit=10,
            without_grouping=None,
            without_localize_sampling=False,
            latest_windows=self.latest_windows
        )
        
        # Reconstruct
        print(f"  [JumpStarter] Reconstructing with window={self.window}, "
              f"windows_per_cycle={self.windows_per_cycle}...")
        reconstructed, retry_count = detector.reconstruct(
            data_normalized,
            window=self.window,
            windows_per_cycle=self.windows_per_cycle,
            stride=self.rec_stride
        )
        
        print(f"  [JumpStarter] Reconstruction complete. Retries: {retry_count}")
        
        # Store detector and reconstructed data
        self.detector = detector
        self.reconstructed = reconstructed
        
        return reconstructed, retry_count
    
    def detect(self, data, reconstructed=None):
        """
        Detect anomalies in test data.
        
        Args:
            data: Test data, shape (n_samples, n_features)
            reconstructed: Optional pre-computed reconstruction (if None, uses train reconstruction)
            
        Returns:
            Anomaly scores, shape (n_samples,)
        """
        if not hasattr(self, 'detector'):
            raise ValueError("Model not trained. Call train() first.")
        
        # Normalize test data using same normalization as training
        data_normalized = self._normalize(data)
        
        # If no reconstruction provided, use the one from training
        # (This assumes we're detecting on training data or we reconstruct test separately)
        if reconstructed is None:
            # For test data, we need to reconstruct it
            print(f"  [JumpStarter] Reconstructing test data...")
            reconstructed, _ = self.detector.reconstruct(
                data_normalized,
                window=self.window,
                windows_per_cycle=self.windows_per_cycle,
                stride=self.rec_stride
            )
        
        # Compute anomaly scores
        print(f"  [JumpStarter] Computing anomaly scores...")
        scores = self.detector.predict(
            data_normalized,
            reconstructed,
            window=self.det_window,
            stride=self.det_stride
        )
        
        return scores
