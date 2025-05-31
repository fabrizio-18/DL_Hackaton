# Graph Learning with Noisy Labels - DEEP Learning Hackathon

**Competition Entry for [Hackathon on Graph Learning with Noisy Labels](https://sites.google.com/view/hackathongraphnoisylabels/rules?authuser=0)**

Our solution tackles the challenge of building robust graph neural networks that can effectively learn from noisy node labels across four different datasets (A, B, C, D).

- - -

## 🚀 Solution Overview

**Core Architecture: Gated Graph Convolutional Network (GatedGCN)**

Our implementation leverages the GatedGCN architecture enhanced with GNNPlus, following the methodology described in *"*Unlocking the Potential of Classic GNNs for Graph-level Tasks: Simple Architectures Meet Excellence*"* by Luo et al., 2025 ([arXiv:2502.09263v1](https://arxiv.org/pdf/2502.09263v1)).

The key innovation lies in the gated message passing mechanism, which enables adaptive filtering and aggregation of neighbor information, making the model particularly resilient to label noise.

![Model Architecture](model.png)

Each dataset receives a tailored configuration designed to achieve optimal robustness against noisy labels:

- **Datasets A, B, C**: Powered by **GatedGCN** with loss function adaptation:
  - **Datasets A, B, C**: Feature Generalized Cross-Entropy (GCE) loss with distinct q settings
- **Dataset D**: Employs **GIN-Virtual** baseline architecture with GCE loss integration

- - -

## 📋 Experimental Configuration

Our approach involves dataset-specific hyperparameter tuning to optimize performance under noisy conditions. Each dataset receives a tailored configuration based on its unique characteristics and noise patterns.

**Common Training Settings:**

- **Datasets A, B, C**

    * Training Duration: 300 epochs across all datasets
    * Optimization: Adam with 0.0005 learning rate
    * Batch Processing: 32 samples per batch
    * Regularization: 0.5 dropout rate with edge dropping
    * Architecture: 3-layer GNN with residual connections and batch normalization

- **Dataset D**
    * Training Duration: 200 epochs
    * Optimization: Adam with 0.0005 learning rate
    * Batch Processing: 32 samples per batch
    * Regularization: 0.5 dropout rate with edge dropping
    * Architecture: 5-layer GNN with residual connections and batch normalization
- - -

## 🔧 Getting Started

**Setup Instructions:**

1. **Environment Setup**

``` bash
pip install -r requirements.txt
```

2. **Model Training**
Navigate to your project directory:

``` bash
cd path/to/project
```

Execute training for each dataset:

``` bash
python main.py --test_path <path_to_test_data> --train_path <path_to_train_data>
```

3. **Results**
Training outputs (checkpoints, logs, results) are automatically saved in their respective directories.

- - -

## 📝 Additional Information

**Development Team:** Fabrizio Italia & Giada Piacentini

**Repository Structure:**

* `/source/` \- Main implementation code
* `/model.png` \- Architecture visualization
* `/requirements.txt` \- Dependency specifications
* Pre-trained weights and configurations included