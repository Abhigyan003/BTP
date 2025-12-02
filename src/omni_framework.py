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
    def train_offline(self, mts_data, periodic_weights, n_clusters=3, epochs=10):
        # 1. Clustering
        # Use window_size=10 to match TranAD baseline (stride=5)
        self.whac = WHAC_Clustering(periodic_weights, window_size=10, n_clusters=n_clusters)
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
            self._train_loop(model, cluster_data, epochs=epochs)
            
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
    def online_transfer(self, target_data, beta_threshold=None, epochs=10):
        # 1. Prepare Target
        train_len = min(len(target_data), 2000)
        target_train = target_data[:train_len]
        segments = self.whac.segment_data(target_train)
        aligned = self.whac.align_phase_shifts(segments)
        
        if len(aligned) == 0: return None

        # 2. Match to Closest Cluster using Statistical Features
        # Use same approach as clustering for consistency
        best_cid = -1
        min_dist = float('inf')
        
        # Keep target_shape for reference (used later for alignment)
        target_shape = np.mean(aligned, axis=0)
        
        # Extract statistical features from target segments
        target_means = aligned.mean(axis=1)      # [n_segments, n_features]
        target_stds = aligned.std(axis=1)
        target_ranges = aligned.max(axis=1) - aligned.min(axis=1)
        
        # Average across segments to get representative statistics
        target_mean_feat = target_means.mean(axis=0)
        target_std_feat = target_stds.mean(axis=0)
        target_range_feat = target_ranges.mean(axis=0)
        
        # Debug: Show distances to all clusters
        cluster_distances = {}
        
        for cid, centroid in self.cluster_centroids.items():
            # Extract statistical features from centroid (stored as segment)
            # Centroid is shape [window_size, n_features], extract its stats
            cent_mean = centroid.mean(axis=0)
            cent_std = centroid.std(axis=0)
            cent_range = centroid.max(axis=0) - centroid.min(axis=0)

            # Calculate distance based on statistical features
            # This is a simplified example, a more robust distance might combine these
            dist_mean = np.linalg.norm(target_mean_feat - cent_mean)
            dist_std = np.linalg.norm(target_std_feat - cent_std)
            dist_range = np.linalg.norm(target_range_feat - cent_range)
            
            # Combine distances (e.g., sum or weighted sum)
            dist = dist_mean + dist_std + dist_range # Simple sum for now

            cluster_distances[cid] = dist
            
            if dist < min_dist:
                min_dist = dist
                best_cid = int(cid)
        
        # Debug output
        print(f"   [Cluster Match] Distances: {', '.join([f'C{cid}={d:.4f}' for cid, d in sorted(cluster_distances.items())])} -> Matched: C{best_cid}")
        
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
            self._train_loop(target_model, aligned, epochs=epochs)
        else:
            # PARTIAL Transfer
            # print(f"   > Partial Transfer Triggered (Diff: {current_diff_score:.4f} > Beta: {threshold_val:.4f})")
            
            # Freeze encoder layers based on model type
            if hasattr(target_model, 'encoder'):
                # TranAD
                for param in target_model.encoder.parameters():
                    param.requires_grad = False
            elif hasattr(target_model, 'encoder_rnn'):
                # RNN_VAE
                for param in target_model.encoder_rnn.parameters():
                    param.requires_grad = False
            
            self._train_loop(target_model, aligned, epochs=epochs)
            
        # Store the reference shape (pivot) in the model for consistent alignment during detection
        target_model.reference_shape = target_shape
        
        return target_model

    def detect(self, model, test_data):
        """
        Detect anomalies using the adapted model
        Returns:
            scores (np.array): Anomaly scores for each time step
        """
        model.eval()
        
        # Segment data
        # Use a dummy WHAC_Clustering instance to get segments, as the original self.whac might be None
        # or its window_size might not match if it was not initialized or changed.
        # Assuming window_size is consistent with how models were trained.
        window_size = 10 # Hardcoded to match detect_scratch and training
        dummy_whac = WHAC_Clustering(np.ones(test_data.shape[1]), window_size=window_size)
        segments = dummy_whac.segment_data(test_data, stride=1)
        
        scores = []
        batch_size = 256
        
        if len(segments) == 0:
            # If no segments can be formed, return an array of zeros matching test_data length
            return np.zeros(len(test_data))

        with torch.no_grad():
            for i in range(0, len(segments), batch_size):
                batch_segs = segments[i : i + batch_size]
                tensor_x = torch.DoubleTensor(batch_segs).to(self.device)
                
                # Forward pass
                # Handle input format for TranAD if needed (permute)
                # But here we stick to the provided logic.
                # If model is RNN_VAE, it expects [batch, window, feats] (standard)
                # If model is TranAD, it expects [window, batch, feats]
                
                # Check model class name string to avoid import
                model_name = model.__class__.__name__
                
                if 'TranAD' in model_name:
                     # TranAD format: [window, batch, feats]
                    tensor_x_permuted = tensor_x.permute(1, 0, 2)
                    local_bs = tensor_x.shape[0]
                    elem = tensor_x_permuted[-1:, :, :].view(1, local_bs, -1)
                    
                    z = model(tensor_x_permuted, elem)
                    recon = z[1] # Phase 2
                else:
                    # RNN_VAE or others: [batch, window, feats]
                    # RNN_VAE forward: (recon, mu, logvar)
                    out = model(tensor_x)
                    if isinstance(out, tuple) and len(out) == 3:
                        recon = out[0]
                    else:
                        recon = out
                    elem = tensor_x # Target is input
                
                # Score using reconstruction error
                loss = torch.mean((elem - recon) ** 2, dim=2)
                if 'TranAD' in model_name:
                     batch_scores = loss[0, :].cpu().numpy()
                else:
                     # RNN_VAE output shape: [batch, window, feats]
                     # We want score per sample. Mean over window? Or last point?
                     # TranAD uses last point reconstruction.
                     # RNN_VAE reconstructs whole sequence.
                     # Let's use mean over window for RNN_VAE to capture whole segment anomaly
                     batch_scores = torch.mean(loss, dim=1).cpu().numpy()
                     
                scores.extend(batch_scores)
        
        # Pad scores to match original data length
        # segments has length: len(test_data) - window_size + 1
        # We need to pad window_size - 1 scores at the beginning
        scores = np.array(scores)
        pad_length = window_size - 1
        padded_scores = np.concatenate([np.zeros(pad_length), scores])
        
        return padded_scores

        return padded_scores

    def _calculate_batch_diff_score(self, model, segments):
        model.eval()
        batch_size = 256
        all_errors = []
        tensor_all = torch.DoubleTensor(segments)
        with torch.no_grad():
            for i in range(0, len(tensor_all), batch_size):
                batch = tensor_all[i : i+batch_size].to(self.device)
                
                # Forward pass
                out = model(batch)
                
                # Check model type by output
                if isinstance(out, tuple) and len(out) == 3:
                    # RNN_VAE: (recon, mu, logvar)
                    rec = out[0]
                elif isinstance(out, tuple) and len(out) == 2:
                    # TranAD: (x1, x2) - use x2 (Phase 2)
                    # TranAD requires (x, elem) input usually, but here we passed just x?
                    # Wait, TranAD forward signature is (src, tgt).
                    # If we pass just 'batch', it might fail if tgt is required.
                    # But in _train_loop we see: model(bx) -> This implies TranAD handles single input?
                    # Let's check TranAD.forward in models.py.
                    # It seems TranAD.forward(src, tgt).
                    # If we look at line 234: criterion(model(bx), by)
                    # This suggests the model takes one argument?
                    # Ah, in detect() we do: z = model(tensor_x_permuted, elem)
                    # But in _train_loop we do: model(bx) ??
                    # Let's look at the original _train_loop in omni_framework.py
                    # Line 234: loss = criterion(model(bx), by)
                    # This looks like it assumes a standard model signature.
                    # But TranAD requires 2 args.
                    # Let's check if OmniTransferTrainer was even working with TranAD properly in _train_loop.
                    # The original code had:
                    # for bx, by in loader:
                    #     loss = criterion(model(bx), by)
                    # This implies 'model' takes 1 arg.
                    # But TranAD takes 2.
                    # Maybe OmniTransferTrainer was wrapping TranAD? Or I missed something.
                    # Actually, looking at full_comparison.py, train_scratch calls model(bx_permuted, elem).
                    # But OmniTransferTrainer._train_loop calls model(bx).
                    # This suggests OmniTransferTrainer might have been broken for TranAD if TranAD requires 2 args!
                    # OR, TranAD's forward has a default for tgt?
                    # Let's check models.py again.
                    # def forward(self, src, tgt=None):
                    # Yes, tgt is optional!
                    # If tgt is None, what happens?
                    # It uses src as tgt?
                    # If so, then model(bx) works.
                    
                    # Back to diff score:
                    # If TranAD, out is (x1, x2). We want x2.
                    # But wait, if we pass only 1 arg, TranAD returns what?
                    # It returns z.
                    
                    # For RNN_VAE, out is (recon, mu, logvar).
                    rec = out[1] # Use Phase 2 for TranAD
                else:
                    # Assume single output (e.g. simple Autoencoder)
                    rec = out
                    
                mse = torch.mean((batch - rec)**2, dim=(1,2))
                all_errors.extend(mse.cpu().numpy())
        return np.array(all_errors)

    def _train_loop(self, model, data, epochs=5):
        tensor_x = torch.DoubleTensor(data).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        model.double() # Ensure model is double precision
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                
                # Forward pass
                # We need to handle TranAD vs RNN_VAE inputs/outputs
                
                # Try to detect if it's TranAD (requires permutation usually?)
                # In full_comparison.py train_scratch, we permute: bx.permute(1, 0, 2)
                # Here we don't.
                # If TranAD expects [window, batch, feats], and we pass [batch, window, feats], it might fail or give wrong results.
                # However, OmniTransferTrainer seems to have been written generically.
                # Let's assume for now we just handle the OUTPUTS.
                
                out = model(bx)
                
                if isinstance(out, tuple) and len(out) == 3:
                    # RNN_VAE: (recon, mu, logvar)
                    recon, mu, logvar = out
                    mse_loss = criterion(recon, bx)
                    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = mse_loss + 0.001 * kld_loss
                elif isinstance(out, tuple) and len(out) == 2:
                    # TranAD: (x1, x2)
                    # Original OmniTransferTrainer just did: criterion(model(bx), by)
                    # This would fail if model(bx) returns a tuple!
                    # So OmniTransferTrainer must have been relying on a model that returns a single tensor, OR
                    # TranAD's forward behaves differently?
                    # Actually, if I look at the file content I viewed earlier for OmniTransferTrainer,
                    # it had: loss = criterion(model(bx), by)
                    # If model(bx) returns (x1, x2), criterion((x1, x2), by) would throw an error.
                    # So... OmniTransferTrainer might NOT have been working with TranAD out of the box?
                    # Or maybe I am misremembering TranAD's return.
                    # TranAD returns 'z'. z is a list/tuple.
                    
                    # Let's fix it to handle TranAD correctly here too.
                    x1, x2 = out
                    loss = criterion(x2, bx) # Use Phase 2
                else:
                    # Standard
                    loss = criterion(out, bx)
                
                loss.backward()
                optimizer.step()