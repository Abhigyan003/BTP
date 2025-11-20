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
        self.cluster_centroids = {} # To store the "Shape" of each cluster
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
        print(f"[Framework] Formed {len(unique_labels)} Clusters. Starting Base Training...")

        for cid in unique_labels:
            # Get data for this cluster
            cluster_data = final_segments[labels == cid]
            
            # SAVE CENTROID (Mean of the cluster) - Needed for Matching later
            self.cluster_centroids[cid] = np.mean(cluster_data, axis=0)
            
            # Train Base Model
            feat_dim = mts_data.shape[1]
            model = self.model_class(feat_dim=feat_dim).to(self.device)
            
            print(f"   > Training Base Model {cid}...")
            self._train_loop(model, cluster_data, epochs=5) # 5 epochs for demo
            self.base_models[cid] = model
            
        return self.base_models

    # ===========================================
    # PHASE 2: ONLINE TRANSFER (The New Part)
    # ===========================================
    def online_transfer(self, target_data, beta_threshold=0.5):
        """
        Implements Section 4.4: Matching -> DiffScore -> Adaptive Transfer
        """
        print("\n[Framework] Starting Online Transfer for Target...")
        
        # 1. Prepare Target Segments
        # We only use a small portion (e.g., first 500 points) for transfer training
        # as per the "Few-Shot" benefit of OmniTransfer
        train_len = min(len(target_data), 1000)
        target_train = target_data[:train_len]
        
        segments = self.whac.segment_data(target_train)
        aligned = self.whac.align_phase_shifts(segments)
        
        # 2. Match to Closest Cluster (Distance Measurement)
        best_cid = -1
        min_dist = float('inf')
        
        # Average the target segments to get its "Shape"
        target_shape = np.mean(aligned, axis=0)
        
        for cid, centroid in self.cluster_centroids.items():
            # Use Weighted Euclidean (Eq 4)
            dist = self.whac.weighted_euclidean(target_shape, centroid)
            if dist < min_dist:
                min_dist = dist
                best_cid = cid
                
        print(f"   > Target Matched to Cluster {best_cid} (Dist: {min_dist:.4f})")
        
        # 3. Load Base Model
        base_model = self.base_models[best_cid]
        target_model = copy.deepcopy(base_model) # Clone it
        
        # 4. Calculate DiffScore (Eq 10)
        diff_score = self._calculate_diff_score(target_model, aligned)
        print(f"   > Calculated DiffScore: {diff_score:.4f} (Threshold Beta: {beta_threshold})")
        
        # 5. Adaptive Transfer Strategy
        if diff_score < beta_threshold:
            print("   > Strategy: FULL Parameter Transfer (Shapes are similar)")
            # Fine-tune everything
            self._train_loop(target_model, aligned, epochs=5)
        else:
            print("   > Strategy: PARTIAL Parameter Transfer (Shapes differ)")
            # Freeze Encoder, Fine-tune Decoder/Projection
            # TranAD specific: Freeze encoder layers
            for param in target_model.encoder.parameters():
                param.requires_grad = False
            
            self._train_loop(target_model, aligned, epochs=5)
            
        return target_model

    def _calculate_diff_score(self, model, segments):
        """
        Eq 10: DiffScore_E(H) = sum(AnomalyScore(H)) excluding top 5%
        """
        model.eval()
        tensor_x = torch.Tensor(segments).to(self.device)
        with torch.no_grad():
            reconstruction = model(tensor_x)
            # MSE per segment
            loss = torch.mean((tensor_x - reconstruction)**2, dim=(1,2)).cpu().numpy()
        
        # Remove top 5% extreme values (noise/anomalies)
        threshold = np.percentile(loss, 95)
        filtered_loss = loss[loss < threshold]
        
        # Normalize score for easier thresholding in this demo
        return np.mean(filtered_loss)

    # ===========================================
    # HELPERS
    # ===========================================
    def _train_loop(self, model, data, epochs=5):
        tensor_x = torch.Tensor(data).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Filter parameters that require grad (for Partial Transfer)
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
                
    def detect(self, model, test_data):
        """
        Runs the model on test data and returns Anomaly Scores (MSE).
        CRITICAL: Uses stride=1 to ensure alignment with labels.
        """
        model.eval()
        
        # Segment test data with STRIDE=1
        segments = self.whac.segment_data(test_data, stride=1)
        
        if len(segments) == 0:
            print("Error: Test data too short for window size.")
            return np.array([])

        # Process in batches to avoid GPU OOM on large test sets
        batch_size = 256
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(segments), batch_size):
                batch_segs = segments[i : i + batch_size]
                tensor_x = torch.Tensor(batch_segs).to(self.device)
                
                reconstruction = model(tensor_x)
                
                # MSE calculation
                loss = torch.mean((tensor_x - reconstruction) ** 2, dim=2)
                
                # Use the error of the last point in the window
                batch_scores = loss[:, -1].cpu().numpy()
                scores.extend(batch_scores)
            
        return np.array(scores)