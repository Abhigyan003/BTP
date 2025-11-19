import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class UnifiedDataLoader:
    def __init__(self, root_path):
        self.root_path = root_path

    def load_dataset(self, dataset_name, entity_id=None):
        print(f"[Data] Loading {dataset_name}...")
        if dataset_name == 'SMD':
            return self._load_smd(entity_id)
        elif dataset_name in ['SMAP', 'MSL']:
            return self._load_satellite(dataset_name)
        else:
            raise ValueError("Unknown dataset. Supported: SMD, SMAP, MSL")

    def _load_smd(self, entity_id):
        # Logic for text/csv files
        path = os.path.join(self.root_path, 'SMD', 'train', f'{entity_id}.txt')
        if not os.path.exists(path):
            path = os.path.join(self.root_path, 'SMD', 'train', f'{entity_id}.csv')
        if not os.path.exists(path):
            # Fallback to dummy data if file missing for testing
            print(f"[Data] Warning: {path} not found. Generating dummy data.")
            return np.random.rand(2000, 38)
            
        try:
            return np.genfromtxt(path, delimiter=',')
        except:
            return np.genfromtxt(path, delimiter=None)

    def _load_satellite(self, name):
        # Logic for .npy files
        path = os.path.join(self.root_path, name, f'{name}_train.npy')
        if not os.path.exists(path):
             print(f"[Data] Warning: {path} not found. Generating dummy data.")
             return np.random.rand(2000, 25) # 25 dims for MSL/SMAP approx
        
        data = np.load(path)
        # Return first entity if 3D array
        if len(data.shape) == 3:
            return data[0, :, :]
        return data

class OmniPreprocess:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()

    def preprocess(self, mts_data):
        df = pd.DataFrame(mts_data)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(method='bfill').fillna(method='ffill')
        return self.scaler.fit_transform(df.values)

    def calculate_cmnd(self, metric_series, max_tau=1440):
        # Simplified CMND calculation
        if len(metric_series) < 50 or np.var(metric_series) < 1e-5:
            return 0.0 # Strong periodicity (constant)
            
        # Downsample for speed if needed
        if len(metric_series) > 5000: metric_series = metric_series[:5000]
        
        min_cmnd = float('inf')
        d_cumulative = 0
        
        # Check a few taus (sparse search for speed)
        taus = range(1, min(len(metric_series)//2, max_tau), 5)
        
        for tau in taus:
            u = metric_series[:-tau]
            v = metric_series[tau:]
            d_tau = np.sum((u - v)**2)
            d_cumulative += d_tau
            
            # Formula Eq(2) approximation
            avg_d = d_cumulative / len(range(1, tau+1, 5))
            if avg_d == 0: val = 1.0
            else: val = d_tau / avg_d
            
            if val < min_cmnd: min_cmnd = val
            
        return min_cmnd

    def compute_periodic_weights(self, mts_data):
        M = mts_data.shape[1]
        P = np.zeros(M)
        print(f"[Data] Calculating Periodicity Weights for {M} metrics...")
        for j in range(M):
            P[j] = self.calculate_cmnd(mts_data[:, j])
        return np.power(P + 1e-6, -self.alpha)