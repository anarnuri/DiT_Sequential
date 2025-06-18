import numpy as np
import torch
from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    def __init__(self, node_features_path, latent_features_path, edge_index_path, curves_path, max_nodes=10, shuffle=True):
        # Load all data sources
        self.node_features = np.load(node_features_path, allow_pickle=True)
        self.latent_features = np.load(latent_features_path, allow_pickle=True)
        self.edge_index = np.load(edge_index_path, allow_pickle=True)
        self.curves = np.load(curves_path, mmap_mode='r')
        self.max_nodes = max_nodes
        self.shuffle = shuffle

        # Verify alignment
        assert len(self.node_features) == len(self.latent_features), "Mismatched dataset lengths"
        
        if shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        indices = np.arange(len(self.node_features))
        np.random.shuffle(indices)
        self.node_features = self.node_features[indices]
        self.latent_features = self.latent_features[indices]
        self.edge_index = self.edge_index[indices]
        self.curves = self.curves[indices]

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        # Process original node features
        node_feats = self.node_features[idx][:, :2]  # [n, 2]
        node_attr = self.node_features[idx][:, 2]    # [n,]
        
        # Pad/truncate original features
        node_feats = self._pad_features(node_feats, (self.max_nodes, 2))
        node_attr = self._pad_features(node_attr, (self.max_nodes,))
        
        latent_feats = self.latent_features[idx]
        
        # Create adjacency matrix
        adj = self._create_adjacency(self.edge_index[idx], node_attr)
        
        # Get curve data
        curve = torch.tensor(self.curves[idx], dtype=torch.float32)  # [200, 2]

        return {
            # Original features
            'node_features': torch.tensor(node_feats, dtype=torch.float32),  # [max_nodes, 2]
            'node_attributes': torch.tensor(node_attr, dtype=torch.float32), # [max_nodes]
            'adjacency': torch.tensor(adj, dtype=torch.float32).unsqueeze(0), # [1, max_nodes, max_nodes]
            'curve_data': curve,  # [200, 2]
            
            # Latent representation
            'latent_features': torch.tensor(latent_feats, dtype=torch.float32),  # [1, latent_dim]
            
            # Additional useful info
            'original_length': torch.tensor(len(self.node_features[idx]), dtype=torch.long)  # Scalar
        }

    def _pad_features(self, arr, target_shape):
        """Pad or truncate features to target shape"""
        if len(arr) < target_shape[0]:
            pad = np.zeros(target_shape, dtype=arr.dtype)
            pad[:len(arr)] = arr
            return pad
        return arr[:target_shape[0]]

    def _create_adjacency(self, edge_index, node_attr):
        """Create adjacency matrix with node attributes on diagonal"""
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        
        # Set edges (symmetric)
        for i, j in zip(edge_index[0], edge_index[1]):
            if i < self.max_nodes and j < self.max_nodes:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        
        # Set node attributes on diagonal
        for i in range(len(node_attr)):
            adj[i, i] = node_attr[i]
            
        return adj