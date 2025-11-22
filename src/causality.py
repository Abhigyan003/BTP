import numpy as np
import pandas as pd
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    print("Warning: Tigramite not found. Causal weights will default to 0.")

class CausalWeightCalculator:
    def __init__(self, alpha_level=0.01, max_lag=1):
        self.alpha_level = alpha_level
        self.max_lag = max_lag

    def compute_causal_weights(self, data):
        """
        Input: (Time, Features) numpy array
        Output: (Features,) weight vector [0.0 to 1.0]
        """
        T, N = data.shape
        weights = np.zeros(N)

        if not TIGRAMITE_AVAILABLE or T < 50:
            return weights

        # OPTIMIZATION 1: Subsample Data
        # 1500 points is usually enough to detect statistical dependencies.
        limit = 1500
        if T > limit:
            start = (T - limit) // 2
            data_slice = data[start : start + limit]
        else:
            data_slice = data

        try:
            # 1. Setup Tigramite
            var_names = [f"V{i}" for i in range(N)]
            dataframe = pp.DataFrame(data_slice, var_names=var_names)
            
            # OPTIMIZATION 2: Use Analytic Significance
            parcorr = ParCorr(significance='analytic') 
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr)
            
            # 2. Run Discovery
            # FIX: Removed 'print_time_series' argument
            results = pcmci.run_pcmci(tau_max=self.max_lag, pc_alpha=0.01)
            
            val_matrix = results['val_matrix'] 
            p_matrix = results['p_matrix']     
            
            # 3. Calculate Centrality
            for i in range(N):     # Target
                for j in range(N): # Source
                    if i == j: continue
                    
                    for lag in range(self.max_lag + 1):
                        if p_matrix[j, i, lag] < self.alpha_level:
                            strength = abs(val_matrix[j, i, lag])
                            weights[j] += strength * 1.0 
                            weights[i] += strength * 0.5 
            
            # 4. Normalize
            if np.max(weights) > 0:
                weights = weights / np.max(weights)
                
        except Exception as e:
            print(f"   [Causality Skip] {e}. Using variance fallback.")
            # Fallback: Variance often correlates with 'information content'
            weights = np.var(data, axis=0)
            if np.max(weights) > 0:
                weights = weights / np.max(weights)
            
        return weights