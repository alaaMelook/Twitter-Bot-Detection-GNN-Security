"""
FINAL PROJECT – Graph Security & Bot Detection on Twitter
Twitter Social Graph Security Analysis Using GCN & GraphSAGE

Team Members: Alaa Melook   2205214
              Manar Ahmed   2205119    
              Nadine Rasmy  2205203
              Yumna Medhat  2205231
Deadline: 17/12/2025

This notebook contains all project tasks:
1. Build the Graph
2. Compute Graph Metrics
3. Extract Graph Features
4. Baseline ML Classifier
5. GCN Model
6. GraphSAGE Model
7. Adversarial Attack 1 - Evasion (Bonus)
8. Adversarial Attack 2 - Poisoning (Bonus)
9. Security Analysis
"""

# ============================================================================
# SECTION 0: SETUP & IMPORTS
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

# Data Processing
import pandas as pd
import numpy as np
from collections import Counter

# Graph Libraries
import networkx as nx
from networkx.algorithms import community

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Deep Learning
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
import os
os.makedirs('outputs', exist_ok=True)

print(" All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# SECTION 1: LOAD & EXPLORE DATASET
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: DATA LOADING & EXPLORATION")
print("="*80)

# Load dataset
df = pd.read_csv('bot_detection_data.csv')

print(f"\n   Dataset Shape: {df.shape}")
print(f"\n First 5 rows:")
print(df.head())

print(f"\n    Dataset Info:")
print(df.info())

print(f"\n Bot Distribution:")
# Fix: Column name is 'Bot Label' not 'bot'
bot_column = 'Bot Label' if 'Bot Label' in df.columns else 'bot'
print(df[bot_column].value_counts())
print(f"\nBot Percentage: {df[bot_column].mean()*100:.2f}%")

# Check for missing values
print(f"\n Missing Values:")
print(df.isnull().sum())

# Handle missing values if any
df = df.fillna(0)

# Rename columns to match code expectations
column_mapping = {
    'Bot Label': 'bot',
    'Follower Count': 'followers_count',
    'Retweet Count': 'retweets_count',
    'Mention Count': 'mentions_count'
}

# Apply renaming for columns that exist
for old_name, new_name in column_mapping.items():
    if old_name in df.columns:
        df.rename(columns={old_name: new_name}, inplace=True)

# Select relevant features based on what's available
possible_features = [
    'followers_count', 'friends_count', 'statuses_count',
    'favourites_count', 'listed_count', 'retweets_count', 
    'mentions_count', 'Follower Count', 'Retweet Count', 'Mention Count'
]
feature_columns = [col for col in possible_features if col in df.columns]

# Make sure all needed columns exist
if len(feature_columns) == 0:
    print(" No feature columns found! Using available numeric columns...")
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['bot', 'User ID']]

available_features = [col for col in feature_columns if col in df.columns]
print(f"\n  Available features: {available_features}")

# ============================================================================
# SECTION 2: BUILD THE GRAPH (TASK 1)
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: GRAPH CONSTRUCTION")
print("="*80)

# Create empty graph
G = nx.Graph()

# Add nodes with attributes
print("\n Adding nodes to graph...")
for idx, row in df.iterrows():
    node_id = idx  # Use index as node ID
    G.add_node(node_id, 
               followers=row.get('followers_count', row.get('Follower Count', 0)),
               friends=row.get('friends_count', 0),
               bot=row['bot'])

print(f"  Added {G.number_of_nodes()} nodes")

# Create edges based on feature similarity using KNN
print("\n Creating edges based on user similarity...")

# Prepare features for similarity calculation
# Add more meaningful features for better graph construction
feature_data_for_graph = df[available_features].values

# Add derived features for better similarity
df['follower_friend_ratio'] = df['followers_count'] / (df['followers_count'] + df['retweets_count'] + 1)
df['activity_score'] = df['retweets_count'] + df['mentions_count']

# Use these for KNN
graph_features = df[['followers_count', 'retweets_count', 'mentions_count', 
                     'follower_friend_ratio', 'activity_score']].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(graph_features)

# Use KNN to find similar users
k_neighbors = min(10, len(df))  # Connect each node to 10 nearest neighbors
knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn.fit(features_scaled)

# Add edges
edge_count = 0
for idx in range(len(df)):
    distances, indices = knn.kneighbors([features_scaled[idx]])
    for neighbor_idx in indices[0][1:]:  # Skip self
        if not G.has_edge(idx, neighbor_idx):
            G.add_edge(idx, neighbor_idx)
            edge_count += 1

print(f"  Added {edge_count} edges")
print(f"\n Graph Statistics:")
print(f"   - Nodes: {G.number_of_nodes()}")
print(f"   - Edges: {G.number_of_edges()}")
print(f"   - Density: {nx.density(G):.6f}")
print(f"   - Is Connected: {nx.is_connected(G)}")

# ============================================================================
# SECTION 3: COMPUTE GRAPH METRICS (TASK 2)
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: GRAPH METRICS COMPUTATION")
print("="*80)

# 1. Degree Distribution
print("\n   Computing degree distribution...")
degrees = dict(G.degree())
degree_values = list(degrees.values())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(degree_values, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Degree', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Degree Distribution', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
degree_counts = Counter(degree_values)
degrees_sorted = sorted(degree_counts.items())
plt.loglog([d for d, c in degrees_sorted], [c for d, c in degrees_sorted], 'bo-')
plt.xlabel('Degree (log scale)', fontsize=12)
plt.ylabel('Frequency (log scale)', fontsize=12)
plt.title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/degree_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   - Average Degree: {np.mean(degree_values):.2f}")
print(f"   - Max Degree: {max(degree_values)}")
print(f"   - Min Degree: {min(degree_values)}")

# 2. Clustering Coefficient
print("\n Computing clustering coefficients...")
clustering = nx.clustering(G)
avg_clustering = np.mean(list(clustering.values()))
print(f"   - Average Clustering Coefficient: {avg_clustering:.4f}")

# 3. Centrality Measures
print("\n Computing centrality measures...")
degree_centrality = nx.degree_centrality(G)
print(f"   - Degree Centrality computed for {len(degree_centrality)} nodes")

# Betweenness centrality (sample for large graphs)
if G.number_of_nodes() < 1000:
    betweenness = nx.betweenness_centrality(G)
    print(f"   - Betweenness Centrality: {np.mean(list(betweenness.values())):.4f}")
else:
    print("   - Betweenness Centrality: Skipped (too large)")

# 4. Community Detection
print("\n👥 Detecting communities...")
communities_generator = community.greedy_modularity_communities(G)
communities_list = list(communities_generator)
print(f"   - Number of Communities: {len(communities_list)}")
print(f"   - Largest Community Size: {max(len(c) for c in communities_list)}")
print(f"   - Smallest Community Size: {min(len(c) for c in communities_list)}")

# Create community mapping
community_map = {}
for comm_id, comm_nodes in enumerate(communities_list):
    for node in comm_nodes:
        community_map[node] = comm_id

# ============================================================================
# SECTION 4: EXTRACT GRAPH FEATURES (TASK 3)
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: FEATURE EXTRACTION")
print("="*80)

print("\n Extracting features for each node...")

node_features_list = []
node_labels = []
node_ids = []

for node in G.nodes():
    # Original features from dataset
    user_data = df.iloc[node]
    
    # Graph-based features
    degree = G.degree(node)
    clustering_coef = clustering[node]
    degree_cent = degree_centrality[node]
    community_id = community_map.get(node, 0)
    
    # Combine all features - handle different column names
    features = [
        user_data.get('followers_count', user_data.get('Follower Count', 0)),
        user_data.get('friends_count', 0),
        user_data.get('statuses_count', 0),
        user_data.get('favourites_count', 0),
        user_data.get('listed_count', 0),
        user_data.get('retweets_count', user_data.get('Retweet Count', 0)),
        user_data.get('mentions_count', user_data.get('Mention Count', 0)),
        user_data.get('follower_friend_ratio', 0),
        user_data.get('activity_score', 0),
        int(user_data.get('Verified', False)),  # Add verified as feature
        degree,
        clustering_coef,
        degree_cent,
        community_id
    ]
    
    node_features_list.append(features)
    node_labels.append(user_data['bot'])
    node_ids.append(node)

# Convert to numpy arrays
X = np.array(node_features_list, dtype=np.float32)
y = np.array(node_labels, dtype=np.int64)

print(f"  Feature matrix shape: {X.shape}")
print(f"  Labels shape: {y.shape}")
print(f"   - Feature count: {X.shape[1]}")
print(f"   - Bot samples: {sum(y)}")
print(f"   - Human samples: {len(y) - sum(y)}")

# ============================================================================
# SECTION 5: BASELINE ML CLASSIFIER (TASK 4)
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: BASELINE ML CLASSIFIER (Random Forest)")
print("="*80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   Data Split:")
print(f"   - Training samples: {len(X_train)}")
print(f"   - Test samples: {len(X_test)}")

# Scale features
scaler_ml = StandardScaler()
X_train_scaled = scaler_ml.fit_transform(X_train)
X_test_scaled = scaler_ml.transform(X_test)

# Train Random Forest
print("\n Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluation
print("\n Random Forest Results:")
print(classification_report(y_test, y_pred_rf, target_names=['Human', 'Bot']))

baseline_accuracy = accuracy_score(y_test, y_pred_rf)
baseline_f1 = f1_score(y_test, y_pred_rf)
print(f"\n Baseline Metrics:")
print(f"   - Accuracy: {baseline_accuracy:.4f}")
print(f"   - F1-Score: {baseline_f1:.4f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
plt.title('Baseline RF - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('outputs/baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SECTION 6: PREPARE DATA FOR GNN (GCN & GraphSAGE)
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: PREPARING DATA FOR GNN")
print("="*80)

# Convert NetworkX graph to PyTorch Geometric format
print("\n🔄 Converting graph to PyTorch Geometric format...")

# Create edge index
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
# Make undirected (add reverse edges)
edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

print(f"   - Edge index shape: {edge_index.shape}")

# Node features and labels
x = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"   - Node features shape: {x.shape}")
print(f"   - Labels shape: {y_tensor.shape}")

# Create train/test masks
train_indices = np.random.choice(len(y), size=int(0.8*len(y)), replace=False)
test_indices = np.array([i for i in range(len(y)) if i not in train_indices])

train_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask = torch.zeros(len(y), dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True

# Create PyG Data object
data = Data(x=x, edge_index=edge_index, y=y_tensor, 
            train_mask=train_mask, test_mask=test_mask)

print(f"\n  PyTorch Geometric Data object created:")
print(f"   - Number of nodes: {data.num_nodes}")
print(f"   - Number of edges: {data.num_edges}")
print(f"   - Number of features: {data.num_node_features}")
print(f"   - Number of classes: {len(torch.unique(data.y))}")

# ============================================================================
# SECTION 7: GCN MODEL (TASK 5)
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: GCN MODEL")
print("="*80)

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn_model = GCN(data.num_node_features, hidden_channels=64, num_classes=2).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

print(f"\n GCN Model Architecture:")
print(gcn_model)
print(f"\n Training on: {device}")

# Training function
def train_gcn():
    gcn_model.train()
    optimizer.zero_grad()
    out = gcn_model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate_gcn(mask):
    gcn_model.eval()
    with torch.no_grad():
        out = gcn_model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc, pred[mask].cpu().numpy(), data.y[mask].cpu().numpy()

# Train GCN
print("\n🏋️ Training GCN...")
gcn_losses = []
for epoch in range(1, 201):
    loss = train_gcn()
    gcn_losses.append(loss)
    
    if epoch % 20 == 0:
        train_acc, _, _ = evaluate_gcn(data.train_mask)
        test_acc, _, _ = evaluate_gcn(data.test_mask)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Final evaluation
gcn_test_acc, gcn_pred, gcn_true = evaluate_gcn(data.test_mask)
gcn_f1 = f1_score(gcn_true, gcn_pred)

print(f"\n  GCN Final Results:")
print(f"   - Test Accuracy: {gcn_test_acc:.4f}")
print(f"   - Test F1-Score: {gcn_f1:.4f}")
print("\n" + classification_report(gcn_true, gcn_pred, target_names=['Human', 'Bot']))

# Confusion Matrix
cm_gcn = confusion_matrix(gcn_true, gcn_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gcn, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
plt.title('GCN - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('outputs/gcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SECTION 8: GraphSAGE MODEL (TASK 6)
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: GraphSAGE MODEL")
print("="*80)

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x

# Initialize model
sage_model = GraphSAGE(data.num_node_features, hidden_channels=64, num_classes=2).to(device)
optimizer_sage = torch.optim.Adam(sage_model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"\n GraphSAGE Model Architecture:")
print(sage_model)

# Training function
def train_sage():
    sage_model.train()
    optimizer_sage.zero_grad()
    out = sage_model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer_sage.step()
    return loss.item()

# Evaluation function
def evaluate_sage(mask):
    sage_model.eval()
    with torch.no_grad():
        out = sage_model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc, pred[mask].cpu().numpy(), data.y[mask].cpu().numpy()

# Train GraphSAGE
print("\n Training GraphSAGE...")
sage_losses = []
for epoch in range(1, 201):
    loss = train_sage()
    sage_losses.append(loss)
    
    if epoch % 20 == 0:
        train_acc, _, _ = evaluate_sage(data.train_mask)
        test_acc, _, _ = evaluate_sage(data.test_mask)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Final evaluation
sage_test_acc, sage_pred, sage_true = evaluate_sage(data.test_mask)
sage_f1 = f1_score(sage_true, sage_pred)

print(f"\n GraphSAGE Final Results:")
print(f"   - Test Accuracy: {sage_test_acc:.4f}")
print(f"   - Test F1-Score: {sage_f1:.4f}")
print("\n" + classification_report(sage_true, sage_pred, target_names=['Human', 'Bot']))

# Confusion Matrix
cm_sage = confusion_matrix(sage_true, sage_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_sage, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
plt.title('GraphSAGE - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('outputs/graphsage_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SECTION 9: EMBEDDINGS VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 9: EMBEDDINGS VISUALIZATION")
print("="*80)

# Get embeddings from GCN
gcn_model.eval()
with torch.no_grad():
    embeddings = gcn_model.conv2(
        F.relu(gcn_model.conv1(data.x, data.edge_index)), 
        data.edge_index
    ).cpu().numpy()

# PCA
print("\n Applying PCA...")
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

# t-SNE
print(" Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA plot
scatter1 = axes[0].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                           c=data.y.cpu(), cmap='coolwarm', alpha=0.6, s=10)
axes[0].set_title('Node Embeddings - PCA', fontsize=14, fontweight='bold')
axes[0].set_xlabel('PC1', fontsize=12)
axes[0].set_ylabel('PC2', fontsize=12)
axes[0].legend(*scatter1.legend_elements(), title="Classes", labels=['Human', 'Bot'])

# t-SNE plot
scatter2 = axes[1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                           c=data.y.cpu(), cmap='coolwarm', alpha=0.6, s=10)
axes[1].set_title('Node Embeddings - t-SNE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Dimension 1', fontsize=12)
axes[1].set_ylabel('Dimension 2', fontsize=12)
axes[1].legend(*scatter2.legend_elements(), title="Classes", labels=['Human', 'Bot'])

plt.tight_layout()
plt.savefig('outputs/embeddings_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("  Embeddings visualization saved!")

# ============================================================================
# SECTION 10: MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("SECTION 10: MODEL PERFORMANCE COMPARISON")
print("="*80)

# Create comparison table
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'GCN', 'GraphSAGE'],
    'Accuracy': [baseline_accuracy, gcn_test_acc, sage_test_acc],
    'F1-Score': [baseline_f1, gcn_f1, sage_f1]
})

print("\n   Performance Comparison:")
print(results_df.to_string(index=False))

# Visualize comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

results_df.plot(x='Model', y='Accuracy', kind='bar', ax=ax[0], color='steelblue', legend=False)
ax[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax[0].set_ylabel('Accuracy', fontsize=12)
ax[0].set_ylim([0, 1])
ax[0].grid(axis='y', alpha=0.3)

results_df.plot(x='Model', y='F1-Score', kind='bar', ax=ax[1], color='coral', legend=False)
ax[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
ax[1].set_ylabel('F1-Score', fontsize=12)
ax[1].set_ylim([0, 1])
ax[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SECTION 11: ADVERSARIAL ATTACK 1 - EVASION (BONUS)
# ============================================================================

print("\n" + "="*80)
print("SECTION 11: ADVERSARIAL ATTACK 1 - EVASION")
print("="*80)

print("\n Performing Evasion Attack...")
print("Strategy: Add edges from bot nodes to high-degree human accounts")

# Identify bot nodes in test set
test_bot_indices = [i for i in test_indices if y[i] == 1]
print(f"\n   Test set contains {len(test_bot_indices)} bot nodes")

# Find high-degree human nodes
human_degrees = [(node, degree) for node, degree in degrees.items() 
                 if y[node] == 0]
human_degrees.sort(key=lambda x: x[1], reverse=True)
top_humans = [node for node, _ in human_degrees[:50]]  # Top 50 humans

print(f" Target: Top {len(top_humans)} high-degree human accounts")

# Create attacked graph
G_attacked = G.copy()
attack_edges_added = 0

# Add edges from bots to high-degree humans
for bot_node in test_bot_indices[:50]:  # Attack first 50 bots
    for human_node in np.random.choice(top_humans, size=5, replace=False):
        if not G_attacked.has_edge(bot_node, human_node):
            G_attacked.add_edge(bot_node, human_node)
            attack_edges_added += 1

print(f"  Added {attack_edges_added} malicious edges")

# Convert attacked graph to PyG format
edge_index_attacked = torch.tensor(list(G_attacked.edges()), dtype=torch.long).t().contiguous()
edge_index_attacked = torch.cat([edge_index_attacked, edge_index_attacked[[1, 0]]], dim=1)

data_attacked = Data(x=data.x, edge_index=edge_index_attacked, y=data.y,
                     train_mask=data.train_mask, test_mask=data.test_mask).to(device)

print("\n Evaluating models on attacked graph (without retraining)...")

# Test GCN on attacked graph
gcn_model.eval()
with torch.no_grad():
    out_gcn_attack = gcn_model(data_attacked.x, data_attacked.edge_index)
    pred_gcn_attack = out_gcn_attack.argmax(dim=1)
    gcn_attack_acc = (pred_gcn_attack[data_attacked.test_mask] == data_attacked.y[data_attacked.test_mask]).sum().item() / data_attacked.test_mask.sum().item()
    
gcn_attack_pred = pred_gcn_attack[data_attacked.test_mask].cpu().numpy()
gcn_attack_true = data_attacked.y[data_attacked.test_mask].cpu().numpy()
gcn_attack_f1 = f1_score(gcn_attack_true, gcn_attack_pred)

print(f"\n   GCN Performance After Attack:")
print(f"   - Original Accuracy: {gcn_test_acc:.4f}")
print(f"   - After Attack: {gcn_attack_acc:.4f}")
print(f"   - Accuracy Drop: {(gcn_test_acc - gcn_attack_acc):.4f}")
print(f"   - Original F1: {gcn_f1:.4f}")
print(f"   - After Attack F1: {gcn_attack_f1:.4f}")

# Test GraphSAGE on attacked graph
sage_model.eval()
with torch.no_grad():
    out_sage_attack = sage_model(data_attacked.x, data_attacked.edge_index)
    pred_sage_attack = out_sage_attack.argmax(dim=1)
    sage_attack_acc = (pred_sage_attack[data_attacked.test_mask] == data_attacked.y[data_attacked.test_mask]).sum().item() / data_attacked.test_mask.sum().item()

sage_attack_pred = pred_sage_attack[data_attacked.test_mask].cpu().numpy()
sage_attack_true = data_attacked.y[data_attacked.test_mask].cpu().numpy()
sage_attack_f1 = f1_score(sage_attack_true, sage_attack_pred)

print(f"\n   GraphSAGE Performance After Attack:")
print(f"   - Original Accuracy: {sage_test_acc:.4f}")
print(f"   - After Attack: {sage_attack_acc:.4f}")
print(f"   - Accuracy Drop: {(sage_test_acc - sage_attack_acc):.4f}")
print(f"   - Original F1: {sage_f1:.4f}")
print(f"   - After Attack F1: {sage_attack_f1:.4f}")

# Calculate evasion rate (how many bots were misclassified as humans)
bot_mask_test = data_attacked.y[data_attacked.test_mask] == 1
gcn_evaded_bots = (pred_gcn_attack[data_attacked.test_mask][bot_mask_test] == 0).sum().item()
sage_evaded_bots = (pred_sage_attack[data_attacked.test_mask][bot_mask_test] == 0).sum().item()
total_test_bots = bot_mask_test.sum().item()

gcn_evasion_rate = gcn_evaded_bots / total_test_bots * 100
sage_evasion_rate = sage_evaded_bots / total_test_bots * 100

print(f"\n Bot Evasion Analysis:")
print(f"   - Total bots in test set: {total_test_bots}")
print(f"   - GCN: {gcn_evaded_bots} bots evaded ({gcn_evasion_rate:.2f}%)")
print(f"   - GraphSAGE: {sage_evaded_bots} bots evaded ({sage_evasion_rate:.2f}%)")

# Visualize attack impact
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# GCN Confusion Matrices
cm_gcn_attack = confusion_matrix(gcn_attack_true, gcn_attack_pred)
sns.heatmap(cm_gcn, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[0, 0].set_title('GCN - Before Attack', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('True Label')

sns.heatmap(cm_gcn_attack, annot=True, fmt='d', cmap='Reds', ax=axes[0, 1],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[0, 1].set_title('GCN - After Evasion Attack', fontsize=12, fontweight='bold')

# GraphSAGE Confusion Matrices
cm_sage_attack = confusion_matrix(sage_attack_true, sage_attack_pred)
sns.heatmap(cm_sage, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[1, 0].set_title('GraphSAGE - Before Attack', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

sns.heatmap(cm_sage_attack, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[1, 1].set_title('GraphSAGE - After Evasion Attack', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('outputs/evasion_attack_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SECTION 12: ADVERSARIAL ATTACK 2 - POISONING (BONUS)
# ============================================================================

print("\n" + "="*80)
print("SECTION 12: ADVERSARIAL ATTACK 2 - POISONING")
print("="*80)

print("\n Performing Poisoning Attack...")
print("Strategy: Flip 20% of bot labels to human in training data")

# Create poisoned training data
data_poisoned = data.clone()
train_bot_indices = torch.where((data.train_mask) & (data.y == 1))[0]
num_to_flip = int(0.2 * len(train_bot_indices))

# Randomly select 20% of training bots to flip
flip_indices = train_bot_indices[torch.randperm(len(train_bot_indices))[:num_to_flip]]
data_poisoned.y[flip_indices] = 0  # Flip bot labels to human

print(f"  Flipped {num_to_flip} bot labels to human in training set")
print(f"   - Original training bots: {len(train_bot_indices)}")
print(f"   - After poisoning: {len(train_bot_indices) - num_to_flip}")

# Retrain GCN on poisoned data
print("\n Retraining GCN on poisoned data...")
gcn_poisoned = GCN(data.num_node_features, hidden_channels=64, num_classes=2).to(device)
optimizer_poisoned = torch.optim.Adam(gcn_poisoned.parameters(), lr=0.01, weight_decay=5e-4)

def train_gcn_poisoned():
    gcn_poisoned.train()
    optimizer_poisoned.zero_grad()
    out = gcn_poisoned(data_poisoned.x, data_poisoned.edge_index)
    loss = criterion(out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask])
    loss.backward()
    optimizer_poisoned.step()
    return loss.item()

for epoch in range(1, 201):
    loss = train_gcn_poisoned()
    if epoch % 50 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

# Evaluate on CLEAN test set
print("\n Evaluating poisoned GCN on clean test data...")
gcn_poisoned.eval()
with torch.no_grad():
    out_poisoned = gcn_poisoned(data.x, data.edge_index)  # Use original clean data
    pred_poisoned = out_poisoned.argmax(dim=1)
    gcn_poisoned_acc = (pred_poisoned[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

gcn_poisoned_pred = pred_poisoned[data.test_mask].cpu().numpy()
gcn_poisoned_true = data.y[data.test_mask].cpu().numpy()
gcn_poisoned_f1 = f1_score(gcn_poisoned_true, gcn_poisoned_pred)

print(f"\n   GCN Performance After Poisoning:")
print(f"   - Original Accuracy: {gcn_test_acc:.4f}")
print(f"   - After Poisoning: {gcn_poisoned_acc:.4f}")
print(f"   - Performance Drop: {(gcn_test_acc - gcn_poisoned_acc):.4f}")
print(f"   - Original F1: {gcn_f1:.4f}")
print(f"   - After Poisoning F1: {gcn_poisoned_f1:.4f}")

# Retrain GraphSAGE on poisoned data
print("\n Retraining GraphSAGE on poisoned data...")
sage_poisoned = GraphSAGE(data.num_node_features, hidden_channels=64, num_classes=2).to(device)
optimizer_sage_poisoned = torch.optim.Adam(sage_poisoned.parameters(), lr=0.01, weight_decay=5e-4)

def train_sage_poisoned():
    sage_poisoned.train()
    optimizer_sage_poisoned.zero_grad()
    out = sage_poisoned(data_poisoned.x, data_poisoned.edge_index)
    loss = criterion(out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask])
    loss.backward()
    optimizer_sage_poisoned.step()
    return loss.item()

for epoch in range(1, 201):
    loss = train_sage_poisoned()
    if epoch % 50 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

# Evaluate on CLEAN test set
print("\n🔍 Evaluating poisoned GraphSAGE on clean test data...")
sage_poisoned.eval()
with torch.no_grad():
    out_sage_poisoned = sage_poisoned(data.x, data.edge_index)
    pred_sage_poisoned = out_sage_poisoned.argmax(dim=1)
    sage_poisoned_acc = (pred_sage_poisoned[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

sage_poisoned_pred = pred_sage_poisoned[data.test_mask].cpu().numpy()
sage_poisoned_true = data.y[data.test_mask].cpu().numpy()
sage_poisoned_f1 = f1_score(sage_poisoned_true, sage_poisoned_pred)

print(f"\n   GraphSAGE Performance After Poisoning:")
print(f"   - Original Accuracy: {sage_test_acc:.4f}")
print(f"   - After Poisoning: {sage_poisoned_acc:.4f}")
print(f"   - Performance Drop: {(sage_test_acc - sage_poisoned_acc):.4f}")
print(f"   - Original F1: {sage_f1:.4f}")
print(f"   - After Poisoning F1: {sage_poisoned_f1:.4f}")

# Visualize poisoning attack impact
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# GCN Confusion Matrices
cm_gcn_poisoned = confusion_matrix(gcn_poisoned_true, gcn_poisoned_pred)
sns.heatmap(cm_gcn, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[0, 0].set_title('GCN - Before Poisoning', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('True Label')

sns.heatmap(cm_gcn_poisoned, annot=True, fmt='d', cmap='Purples', ax=axes[0, 1],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[0, 1].set_title('GCN - After Poisoning Attack', fontsize=12, fontweight='bold')

# GraphSAGE Confusion Matrices
cm_sage_poisoned = confusion_matrix(sage_poisoned_true, sage_poisoned_pred)
sns.heatmap(cm_sage, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[1, 0].set_title('GraphSAGE - Before Poisoning', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

sns.heatmap(cm_sage_poisoned, annot=True, fmt='d', cmap='Purples', ax=axes[1, 1],
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
axes[1, 1].set_title('GraphSAGE - After Poisoning Attack', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('outputs/poisoning_attack_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SECTION 13: COMPREHENSIVE SECURITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 13: SECURITY ANALYSIS & RECOMMENDATIONS")
print("="*80)

print("\n SECURITY VULNERABILITY ANALYSIS")
print("="*60)

# Create comprehensive results table
security_results = pd.DataFrame({
    'Attack Type': ['None (Baseline)', 'Evasion Attack', 'Poisoning Attack'],
    'GCN Accuracy': [gcn_test_acc, gcn_attack_acc, gcn_poisoned_acc],
    'GCN F1': [gcn_f1, gcn_attack_f1, gcn_poisoned_f1],
    'GraphSAGE Accuracy': [sage_test_acc, sage_attack_acc, sage_poisoned_acc],
    'GraphSAGE F1': [sage_f1, sage_attack_f1, sage_poisoned_f1]
})

print("\n   Attack Impact Summary:")
print(security_results.to_string(index=False))

# Calculate vulnerability scores
gcn_evasion_vulnerability = (gcn_test_acc - gcn_attack_acc) / gcn_test_acc * 100
gcn_poisoning_vulnerability = (gcn_test_acc - gcn_poisoned_acc) / gcn_test_acc * 100
sage_evasion_vulnerability = (sage_test_acc - sage_attack_acc) / sage_test_acc * 100
sage_poisoning_vulnerability = (sage_test_acc - sage_poisoned_acc) / sage_test_acc * 100

print(f"\n Vulnerability Scores (% Performance Drop):")
print(f"   GCN:")
print(f"      - Evasion Vulnerability: {gcn_evasion_vulnerability:.2f}%")
print(f"      - Poisoning Vulnerability: {gcn_poisoning_vulnerability:.2f}%")
print(f"   GraphSAGE:")
print(f"      - Evasion Vulnerability: {sage_evasion_vulnerability:.2f}%")
print(f"      - Poisoning Vulnerability: {sage_poisoning_vulnerability:.2f}%")

# Detailed Security Analysis Report
security_analysis = f"""
{'='*80}
DETAILED SECURITY ANALYSIS REPORT
{'='*80}

1. IDENTIFIED VULNERABILITIES:
{'─'*80}

A) EVASION ATTACK VULNERABILITY:
   - Attack Strategy: Adding edges from bot accounts to high-degree legitimate users
   - Impact on GCN: {gcn_evasion_vulnerability:.2f}% accuracy drop
   - Impact on GraphSAGE: {sage_evasion_vulnerability:.2f}% accuracy drop
   - Bot Evasion Rate: {max(gcn_evasion_rate, sage_evasion_rate):.2f}% of bots successfully evaded detection
   
   Root Cause:
   - GNNs aggregate information from neighbors
   - Bots can "hide" by connecting to legitimate users
   - Models over-rely on graph structure without sufficient feature validation

B) POISONING ATTACK VULNERABILITY:
   - Attack Strategy: Flipping 20% of bot labels to human during training
   - Impact on GCN: {gcn_poisoning_vulnerability:.2f}% accuracy drop
   - Impact on GraphSAGE: {sage_poisoning_vulnerability:.2f}% accuracy drop
   
   Root Cause:
   - Models trust training data labels implicitly
   - No label validation or anomaly detection during training
   - Gradient-based learning can be misled by corrupted labels

2. SECURITY WEAKNESSES:
{'─'*80}

   a) Graph Structure Manipulation:
      - Attackers can modify edge connections at test time
      - No integrity verification for graph structure
      - Easy to add/remove edges without detection

   b) Training Data Integrity:
      - Lack of robust label validation mechanisms
      - No defense against label flipping attacks
      - Training process accepts poisoned data

   c) Feature Engineering Gaps:
      - Over-reliance on graph topology
      - Insufficient behavioral feature analysis
      - Missing temporal patterns and activity anomalies

   d) Model Robustness:
      - Gradient-based models vulnerable to adversarial examples
      - No adversarial training implemented
      - Missing uncertainty quantification

3. RECOMMENDED DEFENSE STRATEGIES:
{'─'*80}

A) IMMEDIATE COUNTERMEASURES:

   1. Graph Structure Validation:
      ✓ Implement edge anomaly detection
      ✓ Rate-limit connection requests
      ✓ Detect suspicious follower/following patterns
      ✓ Monitor sudden changes in user connections

   2. Feature Robustness:
      ✓ Add temporal behavioral features
      ✓ Include account age and activity patterns
      ✓ Monitor posting frequency and content similarity
      ✓ Implement multi-modal features (text, image, metadata)

   3. Training Data Protection:
      ✓ Use confident learning to detect label errors
      ✓ Implement cross-validation for label consistency
      ✓ Apply data sanitization techniques
      ✓ Use semi-supervised learning with unlabeled data

B) ADVANCED DEFENSE MECHANISMS:

   1. Adversarial Training:
      ✓ Train models on adversarially perturbed graphs
      ✓ Include attack samples in training data
      ✓ Use robust loss functions (e.g., focal loss)
      ✓ Implement gradient masking techniques

   2. Ensemble Methods:
      ✓ Combine GCN, GraphSAGE, and traditional ML
      ✓ Use voting mechanisms for final predictions
      ✓ Implement heterogeneous model architectures
      ✓ Add uncertainty-based rejection

   3. Graph Certification:
      ✓ Implement provably robust GNN architectures
      ✓ Use randomized smoothing for certification
      ✓ Apply Lipschitz constraints on model layers
      ✓ Verify predictions under perturbations

   4. Anomaly Detection Layer:
      ✓ Add autoencoder for normal behavior modeling
      ✓ Implement out-of-distribution detection
      ✓ Use Isolation Forest for graph anomalies
      ✓ Monitor prediction confidence scores

   5. Continuous Monitoring:
      ✓ Real-time graph structure monitoring
      ✓ Behavioral drift detection
      ✓ Model performance tracking
      ✓ Automated retraining pipelines

C) SYSTEM-LEVEL IMPROVEMENTS:

   1. Multi-Layer Security:
      ✓ Combine graph-based detection with rule-based systems
      ✓ Implement CAPTCHA for suspicious accounts
      ✓ Add human-in-the-loop verification
      ✓ Use rate limiting and IP tracking

   2. Explainability & Transparency:
      ✓ Implement GNNExplainer for decision interpretation
      ✓ Provide confidence scores with predictions
      ✓ Log suspicious activities for review
      ✓ Enable user appeal mechanisms

   3. Adaptive Learning:
      ✓ Implement online learning for model updates
      ✓ Use reinforcement learning for adversarial games
      ✓ Continuous feature engineering
      ✓ Automated hyperparameter tuning

4. IMPLEMENTATION PRIORITY:
{'─'*80}

   HIGH PRIORITY (Implement Immediately):
   1. Add temporal behavioral features
   2. Implement basic anomaly detection
   3. Add ensemble voting mechanism
   4. Enable real-time monitoring

   MEDIUM PRIORITY (Next Phase):
   1. Adversarial training pipeline
   2. Label validation system
   3. Explainability tools
   4. Advanced graph certification

   LOW PRIORITY (Future Research):
   1. Provably robust architectures
   2. Reinforcement learning defenses
   3. Zero-trust graph frameworks

5. CONCLUSION:
{'─'*80}

The analysis reveals that Graph Neural Networks, while powerful for bot detection,
are vulnerable to both evasion and poisoning attacks. The key insight is that
over-reliance on graph structure without sufficient validation creates exploitable
weaknesses. A defense-in-depth approach combining robust features, adversarial
training, ensemble methods, and continuous monitoring is essential for a secure
bot detection system.

Recommended Next Steps:
1. Implement immediate countermeasures (High Priority items)
2. Conduct red-team exercises to test defenses
3. Deploy monitoring infrastructure
4. Establish incident response procedures
5. Research advanced defense mechanisms
{'='*80}
"""

print(security_analysis)

# Save security analysis to file
with open('outputs/security_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(security_analysis)

print("\n  Security analysis report saved to 'outputs/security_analysis_report.txt'")

# ============================================================================
# SECTION 14: FINAL VISUALIZATIONS & SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SECTION 14: FINAL SUMMARY & VISUALIZATIONS")
print("="*80)

# Create comprehensive comparison visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Model Performance Comparison
ax1 = fig.add_subplot(gs[0, :])
models = ['RF', 'GCN', 'GraphSAGE']
accuracies = [baseline_accuracy, gcn_test_acc, sage_test_acc]
f1_scores = [baseline_f1, gcn_f1, sage_f1]

x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1-Score', color='coral')

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Model Performance Comparison (Clean Data)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# 2. Attack Impact on GCN
ax2 = fig.add_subplot(gs[1, 0])
scenarios = ['Clean', 'Evasion', 'Poisoning']
gcn_accs = [gcn_test_acc, gcn_attack_acc, gcn_poisoned_acc]
bars = ax2.bar(scenarios, gcn_accs, color=['green', 'orange', 'red'])
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('GCN Under Attack', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, gcn_accs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

# 3. Attack Impact on GraphSAGE
ax3 = fig.add_subplot(gs[1, 1])
sage_accs = [sage_test_acc, sage_attack_acc, sage_poisoned_acc]
bars = ax3.bar(scenarios, sage_accs, color=['green', 'orange', 'red'])
ax3.set_ylabel('Accuracy', fontsize=11)
ax3.set_title('GraphSAGE Under Attack', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, sage_accs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

# 4. Vulnerability Comparison
ax4 = fig.add_subplot(gs[1, 2])
vulnerabilities = {
    'GCN Evasion': gcn_evasion_vulnerability,
    'GCN Poisoning': gcn_poisoning_vulnerability,
    'SAGE Evasion': sage_evasion_vulnerability,
    'SAGE Poisoning': sage_poisoning_vulnerability
}
colors_vuln = ['#ff6b6b', '#ee5a6f', '#ffd93d', '#fcbf49']
bars = ax4.barh(list(vulnerabilities.keys()), list(vulnerabilities.values()), color=colors_vuln)
ax4.set_xlabel('Vulnerability (%)', fontsize=11)
ax4.set_title('Model Vulnerabilities', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for bar, vuln in zip(bars, vulnerabilities.values()):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
             f'{vuln:.1f}%', ha='left', va='center', fontsize=9)

# 5. Training Loss Curves
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(gcn_losses, label='GCN', color='green', linewidth=2)
ax5.set_xlabel('Epoch', fontsize=11)
ax5.set_ylabel('Loss', fontsize=11)
ax5.set_title('GCN Training Loss', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(sage_losses, label='GraphSAGE', color='orange', linewidth=2)
ax6.set_xlabel('Epoch', fontsize=11)
ax6.set_ylabel('Loss', fontsize=11)
ax6.set_title('GraphSAGE Training Loss', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 6. Summary Statistics Table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
summary_text = f"""
SUMMARY STATISTICS

Dataset:
• Total Nodes: {G.number_of_nodes()}
• Total Edges: {G.number_of_edges()}
• Bot Ratio: {y.mean()*100:.1f}%

Best Model: {"GCN" if gcn_test_acc > sage_test_acc else "GraphSAGE"}
• Accuracy: {max(gcn_test_acc, sage_test_acc):.4f}
• F1-Score: {max(gcn_f1, sage_f1):.4f}

Most Vulnerable: {"GCN" if max(gcn_evasion_vulnerability, gcn_poisoning_vulnerability) > max(sage_evasion_vulnerability, sage_poisoning_vulnerability) else "GraphSAGE"}
• Max Drop: {max(gcn_evasion_vulnerability, gcn_poisoning_vulnerability, sage_evasion_vulnerability, sage_poisoning_vulnerability):.1f}%

Attack Success:
• Evasion: {max(gcn_evasion_rate, sage_evasion_rate):.1f}% bots evaded
• Poisoning: {max(gcn_poisoning_vulnerability, sage_poisoning_vulnerability):.1f}% drop
"""
ax7.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('outputs/final_comprehensive_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n  All visualizations saved to 'outputs/' directory")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROJECT COMPLETION SUMMARY")
print("="*80)

final_summary = f"""
  ALL TASKS COMPLETED SUCCESSFULLY!

   Outputs Generated:
   1. degree_distribution.png
   2. baseline_confusion_matrix.png
   3. gcn_confusion_matrix.png
   4. graphsage_confusion_matrix.png
   5. embeddings_visualization.png
   6. performance_comparison.png
   7. evasion_attack_comparison.png
   8. poisoning_attack_comparison.png
   9. final_comprehensive_summary.png
   10. security_analysis_report.txt

🎯 Key Findings:
   • Best Clean Performance: {"GCN" if gcn_test_acc >= sage_test_acc else "GraphSAGE"} ({max(gcn_test_acc, sage_test_acc):.4f})
   • Baseline RF Accuracy: {baseline_accuracy:.4f}
   • GCN Accuracy: {gcn_test_acc:.4f}
   • GraphSAGE Accuracy: {sage_test_acc:.4f}
   • Most Vulnerable Model: {"GCN" if max(gcn_evasion_vulnerability, gcn_poisoning_vulnerability) > max(sage_evasion_vulnerability, sage_poisoning_vulnerability) else "GraphSAGE"}
   • Highest Evasion Rate: {max(gcn_evasion_rate, sage_evasion_rate):.2f}%

📋 Next Steps for Report:
   1. Introduction & Dataset Description
   2. Methodology (Graph Construction, Models)
   3. Results & Analysis (include all figures)
   4. Security Analysis (use generated report)
   5. Recommendations & Conclusion

💡 Defense Recommendations:
   • Implement adversarial training
   • Add ensemble methods
   • Enhance feature engineering
   • Deploy real-time monitoring
   • Use multi-layer security approach

{"="*80}
🎉 PROJECT COMPLETE! All code, figures, and analysis ready for report.
{"="*80}
"""

print(final_summary)

# Save all results to CSV for easy reference
results_summary = pd.DataFrame({
    'Model': ['Random Forest', 'GCN Clean', 'GraphSAGE Clean', 
              'GCN Evasion', 'GraphSAGE Evasion',
              'GCN Poisoning', 'GraphSAGE Poisoning'],
    'Accuracy': [baseline_accuracy, gcn_test_acc, sage_test_acc,
                 gcn_attack_acc, sage_attack_acc,
                 gcn_poisoned_acc, sage_poisoned_acc],
    'F1-Score': [baseline_f1, gcn_f1, sage_f1,
                 gcn_attack_f1, sage_attack_f1,
                 gcn_poisoned_f1, sage_poisoned_f1],
    'Attack_Type': ['None', 'None', 'None', 
                    'Evasion', 'Evasion',
                    'Poisoning', 'Poisoning']
})

results_summary.to_csv('outputs/all_results_summary.csv', index=False)
print("\n  Results summary saved to 'outputs/all_results_summary.csv'")

print("\n" + "="*80)
print(" Ready to write your report! All data and figures are in 'outputs/' folder")
print("="*80)