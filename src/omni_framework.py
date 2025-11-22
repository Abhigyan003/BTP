import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.clustering import WHAC_Clustering
import copy

class OmniTransferTrainer:
    def __init__(self, model_class, device='cpu'):
        self.model_class = model_class
        self.device = device
        self.base_models = {} 
        self.cluster_centroids = {} 
        self.cluster_thresholds = {} # <--- NEW: Store auto-calculated Beta per cluster
        self.whac = None

    # ===========================================
    # PHASE 1: OFFLINE TRAINING (With Auto-Beta)
    # ===========================================
    def train_offline(self, mts_data, periodic_weights, n_clusters=3):
        # 1. Clustering
        self.whac = WHAC_Clustering(periodic_weights, window_size=60, n_clusters=n_clusters)
        segments = self.whac.segment_data(mts_data)
        aligned_segments = self.whac.align_phase_shifts(segments)
        labels, final_segments = self.whac.fit_predict(aligned_segments)
        
        unique_labels = np.unique(labels)
        print(f"[Framework] Formed {len(unique_labels)} Clusters. Training Base Models...")

        for cid in unique_labels:
            # Get data for this cluster
            cluster_data = final_segments[labels == cid]
            self.cluster_centroids[cid] = np.mean(cluster_data, axis=0)
            
            # Train Base Model
            feat_dim = mts_data.shape[1]
            model = self.model_class(feat_dim=feat_dim).to(self.device)
            
            print(f"   > Training Base Model {cid} ({len(cluster_data)} segments)...")
            self._train_loop(model, cluster_data, epochs=10)
            
            # --- NEW: AUTO-CALCULATE BETA ---
            # Determine what "Normal Error" looks like for this cluster
            baseline_error = self._calculate_batch_diff_score(model, cluster_data)
            
            # We set Beta to the 95th percentile of the Source Domain error.
            # Meaning: "If the target error is higher than 85% of the source data, it's a bad match."
            auto_beta = np.percentile(baseline_error, 95)
            self.cluster_thresholds[cid] = auto_beta
            
            print(f"     [Auto-Tuning] Calculated Beta for Cluster {cid}: {auto_beta:.6f}")
            
            self.base_models[cid] = model
            
        return self.base_models

    # ===========================================
    # PHASE 2: ONLINE TRANSFER (Dynamic Beta)
    # ===========================================
    def online_transfer(self, target_data, beta_threshold=None):
        """
        If beta_threshold is None, use the Auto-Calculated Beta from Offline Phase.
        """
        # 1. Prepare Target
        train_len = min(len(target_data), 2000) # Use limited data for transfer
        target_train = target_data[:train_len]
        segments = self.whac.segment_data(target_train)
        aligned = self.whac.align_phase_shifts(segments)
        
        if len(aligned) == 0: return None

        # 2. Match to Closest Cluster
        best_cid = -1
        min_dist = float('inf')
        target_shape = np.mean(aligned, axis=0)
        
        for cid, centroid in self.cluster_centroids.items():
            dist = self.whac.weighted_euclidean(target_shape, centroid)
            if dist < min_dist:
                min_dist = dist
                best_cid = cid
        
        # 3. Load Model & Threshold
        print(f"best cid {best_cid}")
        base_model = self.base_models[best_cid]
        target_model = copy.deepcopy(base_model)
        
        # Determine Threshold
        if beta_threshold is not None:
            # Use the one we calculated during training
            threshold_val = self.cluster_thresholds[best_cid]
        else:
            threshold_val = beta_threshold

        # 4. Calculate DiffScore (Current Fit)
        # We take the median error of the target segments to be robust against outliers
        target_errors = self._calculate_batch_diff_score(target_model, aligned)
        current_diff_score = np.mean(target_errors) # Mean or Median
        
        # print(f"   > Cluster {best_cid} | DiffScore: {current_diff_score:.6f} | Beta: {threshold_val:.6f}")
        
        # 5. Adaptive Transfer Strategy
        print(f"diff sore : {current_diff_score} | threshod : {threshold_val}")
        if current_diff_score < threshold_val:
            # The target looks just like the source -> Fine-tune everything
            print("     Strategy: FULL Parameter Transfer")
            self._train_loop(target_model, aligned, epochs=5)
        else:
            # The target is quite different -> Be careful, freeze encoder
            print("     Strategy: PARTIAL Parameter Transfer")
            for param in target_model.encoder.parameters():
                param.requires_grad = False
            self._train_loop(target_model, aligned, epochs=5)
            
        return target_model

    def detect(self, model, test_data):
        model.eval()
        # Use stride=1 for dense detection
        segments = self.whac.segment_data(test_data, stride=1)
        if len(segments) == 0: return np.array([])

        batch_size = 256
        scores = []
        with torch.no_grad():
            for i in range(0, len(segments), batch_size):
                batch_segs = segments[i : i + batch_size]
                tensor_x = torch.Tensor(batch_segs).to(self.device)
                rec = model(tensor_x)
                # MSE per window
                loss = torch.mean((tensor_x - rec) ** 2, dim=2)
                # Take last point error
                batch_scores = loss[:, -1].cpu().numpy()
                scores.extend(batch_scores)
        return np.array(scores)

    def _calculate_batch_diff_score(self, model, segments):
        """
        Helper to get raw MSE errors for a batch of segments
        """
        model.eval()
        batch_size = 256
        all_errors = []
        
        tensor_all = torch.Tensor(segments)
        # Process in chunks to avoid OOM
        with torch.no_grad():
            for i in range(0, len(tensor_all), batch_size):
                batch = tensor_all[i : i+batch_size].to(self.device)
                rec = model(batch)
                # Mean MSE per segment (over Time and Feats)
                # Shape: [Batch]
                mse = torch.mean((batch - rec)**2, dim=(1,2))
                all_errors.extend(mse.cpu().numpy())
                
        return np.array(all_errors)

    def _train_loop(self, model, data, epochs=5):
        tensor_x = torch.Tensor(data).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_x)
        # Increase batch size slightly for speed
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Filter frozen params
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()