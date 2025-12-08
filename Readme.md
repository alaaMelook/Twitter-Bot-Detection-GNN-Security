# Twitter-Bot-Detection-GNN-Security
Graph Security Analysis for Twitter Bot Detection using GCN &amp; GraphSAGE with Adversarial Attacks
# 🤖 Twitter Bot Detection: Graph Security Analysis Using GCN & GraphSAGE

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive security analysis of Graph Neural Networks (GCN & GraphSAGE) for Twitter bot detection, including adversarial attack evaluation and defense strategies.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Security Analysis](#security-analysis)
- [Team Members](#team-members)
- [License](#license)

## 🎯 Overview

This project implements and evaluates Graph Neural Networks for detecting bot accounts on Twitter, with a focus on security vulnerabilities. We analyze how adversarial attacks (evasion and poisoning) affect model performance and provide comprehensive defense recommendations.

### Key Objectives
- Build a Twitter social graph from user data
- Implement GCN and GraphSAGE models for bot detection
- Evaluate adversarial attacks on graph-based ML models
- Analyze security weaknesses and propose defense strategies

## ✨ Features

- **Graph Construction**: K-Nearest Neighbors-based graph building from user features
- **Graph Metrics**: Degree distribution, clustering coefficient, centrality measures, community detection
- **Multiple Models**: 
  - Baseline: Random Forest
  - GCN (Graph Convolutional Network)
  - GraphSAGE (Graph Sample and Aggregate)
- **Adversarial Attacks**:
  - Evasion Attack: Test-time graph manipulation
  - Poisoning Attack: Training data corruption
- **Security Analysis**: Comprehensive vulnerability assessment and defense recommendations
- **Visualizations**: 
  - Degree distributions
  - Embedding visualizations (PCA & t-SNE)
  - Confusion matrices
  - Attack impact comparisons

## 📊 Dataset

We use the **Twitter Bot Detection Dataset** from Kaggle:
- **Size**: 50,000 users
- **Classes**: Bot (50.04%) vs Human (49.96%)
- **Features**: 
  - Follower Count
  - Retweet Count
  - Mention Count
  - Verified Status
  - Account Metadata

**Dataset Link**: [Twitter Bot Detection Dataset](https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset)

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Twitter-Bot-Detection-GNN-Security.git
cd Twitter-Bot-Detection-GNN-Security
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.0
scikit-learn>=1.3.0
torch>=2.0.0
torch-geometric>=2.3.0
```

## 💻 Usage

### Basic Usage

1. **Download the dataset** from Kaggle and place it in the project root:
```
twitter_bot_detection.csv
```

2. **Run the main script**:
```bash
python main.py
```

The script will:
- Load and explore the dataset
- Build the social graph
- Compute graph metrics
- Train all models (RF, GCN, GraphSAGE)
- Execute adversarial attacks
- Generate visualizations
- Create security analysis report

### Output

All results are saved in the `outputs/` directory:
- **Figures**: 9 PNG files (distributions, confusion matrices, embeddings, etc.)
- **Reports**: 
  - `security_analysis_report.txt` - Detailed security analysis
  - `all_results_summary.csv` - Performance metrics

## 📁 Project Structure

```
Twitter-Bot-Detection-GNN-Security/
│
├── main.py                          # Main script
├── requirements.txt                 # Python dependencies
├── twitter_bot_detection.csv        # Dataset (download separately)
├── README.md                        # This file
│
├── outputs/                         # Generated results
│   ├── degree_distribution.png
│   ├── baseline_confusion_matrix.png
│   ├── gcn_confusion_matrix.png
│   ├── graphsage_confusion_matrix.png
│   ├── embeddings_visualization.png
│   ├── performance_comparison.png
│   ├── evasion_attack_comparison.png
│   ├── poisoning_attack_comparison.png
│   ├── final_comprehensive_summary.png
│   ├── security_analysis_report.txt
│   └── all_results_summary.csv
│
└── Report.pdf                       # Final project report
```

## 📈 Results

### Model Performance (Clean Data)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 50.38% | 0.50 |
| GCN | 49.59% | 0.62 |
| GraphSAGE | 49.78% | 0.65 |

### Adversarial Attack Impact

#### Evasion Attack (Test-Time Manipulation)
- **Strategy**: Added 250 edges from bot nodes to high-degree human accounts
- **GCN Evasion Rate**: 18.75% of bots evaded detection
- **GraphSAGE Evasion Rate**: 6.59% of bots evaded detection
- **Insight**: GraphSAGE demonstrates better robustness to evasion attacks

#### Poisoning Attack (Training Data Corruption)
- **Strategy**: Flipped 20% of bot labels to human during training
- **Impact**: F1-Score dropped to 0.00 for both models
- **Insight**: Both models highly vulnerable to training data manipulation

### Graph Statistics
- **Nodes**: 50,000
- **Edges**: 259,160
- **Average Degree**: 10.37
- **Clustering Coefficient**: 0.56
- **Communities Detected**: 25

## 🔒 Security Analysis

### Identified Vulnerabilities

1. **Graph Structure Manipulation**
   - Attackers can add/remove edges without detection
   - No integrity verification for graph structure

2. **Training Data Poisoning**
   - Models trust training labels implicitly
   - No label validation mechanisms

3. **Feature Engineering Gaps**
   - Over-reliance on graph topology
   - Missing temporal behavioral patterns

4. **Model Robustness**
   - Vulnerable to adversarial examples
   - No adversarial training implemented

### Defense Recommendations

#### High Priority (Immediate)
- ✅ Add temporal behavioral features
- ✅ Implement basic anomaly detection
- ✅ Add ensemble voting mechanisms
- ✅ Enable real-time monitoring

#### Medium Priority (Next Phase)
- ⚡ Adversarial training pipeline
- ⚡ Label validation system
- ⚡ Explainability tools (GNNExplainer)
- ⚡ Advanced graph certification

#### Low Priority (Future Research)
- 🔬 Provably robust architectures
- 🔬 Reinforcement learning defenses
- 🔬 Zero-trust graph frameworks

### Recommended Defense Strategies

1. **Adversarial Training**: Train models on perturbed graphs
2. **Ensemble Methods**: Combine GCN, GraphSAGE, and traditional ML
3. **Graph Certification**: Use randomized smoothing
4. **Anomaly Detection**: Add autoencoder for behavior modeling
5. **Continuous Monitoring**: Real-time graph structure tracking

## 👤 Developed By

- Alaa Hassan Melook   ID : 2205214

**Institution**: Alexandria National University
**Course**: Social Networks / Graph Security  
**Date**: December 2025

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Kaggle Twitter Bot Detection Dataset](https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset)
- PyTorch Geometric library for GNN implementations
- NetworkX for graph analysis tools

## 📚 References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. NeurIPS.
3. Zügner, D., Akbarnejad, A., & Günnemann, S. (2018). Adversarial attacks on neural networks for graph data. KDD.


**⭐ If you find this project useful, please consider giving it a star!**

---

## 🔮 Future Work

- [ ] Implement attention mechanisms (GAT)
- [ ] Add more sophisticated attack strategies
- [ ] Deploy real-time detection system
- [ ] Incorporate NLP features from tweet content
- [ ] Multi-platform bot detection (Facebook, Instagram)
- [ ] Transfer learning across social networks