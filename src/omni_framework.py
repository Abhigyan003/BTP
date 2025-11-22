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
        self.cluster_thresholds = {} 
        self.whac = None

    # ===========================================
    # PHASE 1: OFFLINE TRAINING
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
            # Sanitize ID to standard int to avoid numpy int64 hashing issues
            cid = int(cid)
            
            # Get data
            cluster_data = final_segments[labels == cid]
            self.cluster_centroids[cid] = np.mean(cluster_data, axis=0)
            
            # Train
            feat_dim = mts_data.shape[1]
            model = self.model_class(feat_dim=feat_dim).to(self.device)
            
            print(f"   > Training Base Model {cid} ({len(cluster_data)} segments)...")
            self._train_loop(model, cluster_data, epochs=10)
            
            # Auto-Calculate Beta (Threshold)
            baseline_error = self._calculate_batch_diff_score(model, cluster_data)
            # 95th percentile of source error
            auto_beta = float(np.percentile(baseline_error, 95))
            
            self.cluster_thresholds[cid] = auto_beta
            self.base_models[cid] = model
            
            print(f"     [Auto-Tuning] Cluster {cid} Beta: {auto_beta:.6f}")
            
        return self.base_models

    # ===========================================
    # PHASE 2: ONLINE TRANSFER
    # ===========================================
    def online_transfer(self, target_data, beta_threshold=None):
        # 1. Prepare Target
        train_len = min(len(target_data), 2000)
        target_train = target_data[:train_len]
        segments = self.whac.segment_data(target_train)
        aligned = self.whac.align_phase_shifts(segments)
        
        if len(aligned) == 0: return None

        # 2. Match to Closest Cluster
        best_cid = -1
        min_dist = float('inf')
        target_shape = np.mean(aligned, axis=0)
        
        # FIX: Z-Normalize for Shape-Based Matching
        # We want to match the *pattern*, not the amplitude/offset.
        target_std = np.std(target_shape) + 1e-8
        target_norm = (target_shape - np.mean(target_shape)) / target_std
        
        # print(f"   [Debug] Target Shape Mean: {np.mean(target_shape):.4f}, Var: {np.var(target_shape):.4f}")
        for cid, centroid in self.cluster_centroids.items():
            # Normalize centroid too
            cent_std = np.std(centroid) + 1e-8
            cent_norm = (centroid - np.mean(centroid)) / cent_std
            
            # Calculate distance on NORMALIZED shapes
            dist = self.whac.weighted_euclidean(target_norm, cent_norm)
            
            # print(f"   [Debug] Dist to Cluster {cid}: {dist:.4f} (Centroid Mean: {np.mean(centroid):.4f})")
            if dist < min_dist:
                min_dist = dist
                best_cid = int(cid)
        
        # 3. Load Base Model
        if best_cid not in self.base_models:
            print("   ! Warning: No matching cluster found. Initializing random model.")
            feat_dim = target_data.shape[1]
            return self.model_class(feat_dim).to(self.device)

        base_model = self.base_models[best_cid]
        target_model = copy.deepcopy(base_model)
        
        # 4. Determine Threshold (With Fallback)
        if beta_threshold is not None:
            threshold_val = beta_threshold
        else:
            # Safely get auto-beta
            threshold_val = self.cluster_thresholds.get(best_cid, 0.5) 
            
        # Safety check for None (fixes your specific error)
        if threshold_val is None: 
            threshold_val = 0.5

        # 5. Calculate DiffScore
        target_errors = self._calculate_batch_diff_score(target_model, aligned)
        current_diff_score = np.mean(target_errors)
        
        # 6. Adaptive Strategy
        print(f"diff score : {current_diff_score} | threshold : {threshold_val}")
        if current_diff_score < threshold_val:
            # FULL Transfer
            self._train_loop(target_model, aligned, epochs=5)
        else:
            # PARTIAL Transfer
            # print(f"   > Partial Transfer Triggered (Diff: {current_diff_score:.4f} > Beta: {threshold_val:.4f})")
            for param in target_model.encoder.parameters():
                param.requires_grad = False
            self._train_loop(target_model, aligned, epochs=5)
            
        # Store the reference shape (pivot) in the model for consistent alignment during detection
        target_model.reference_shape = target_shape
        
        return target_model

    def detect(self, model, test_data):
        model.eval()
        segments = self.whac.segment_data(test_data, stride=1)
        # FIX 1: Align phases before detection to match training distribution
        # FIX 5: Use stored reference shape for consistent alignment
        if hasattr(model, 'reference_shape'):
            segments = self.whac.align_phase_shifts(segments, pivot=model.reference_shape)
        else:
            segments = self.whac.align_phase_shifts(segments)

        if len(segments) == 0: return np.array([])

        batch_size = 256
        scores = []
        with torch.no_grad():
            for i in range(0, len(segments), batch_size):
                batch_segs = segments[i : i + batch_size]
                tensor_x = torch.Tensor(batch_segs).to(self.device)
                rec = model(tensor_x)
                loss = torch.mean((tensor_x - rec) ** 2, dim=2)
                batch_scores = loss[:, -1].cpu().numpy()
                scores.extend(batch_scores)
        return np.array(scores)

    def _calculate_batch_diff_score(self, model, segments):
        model.eval()
        batch_size = 256
        all_errors = []
        tensor_all = torch.Tensor(segments)
        with torch.no_grad():
            for i in range(0, len(tensor_all), batch_size):
                batch = tensor_all[i : i+batch_size].to(self.device)
                rec = model(batch)
                mse = torch.mean((batch - rec)**2, dim=(1,2))
                all_errors.extend(mse.cpu().numpy())
        return np.array(all_errors)

    def _train_loop(self, model, data, epochs=5):
        tensor_x = torch.Tensor(data).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
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