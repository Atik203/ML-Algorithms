# Machine Learning Algorithms Assignment

This repository contains production-quality, thoroughly documented, and pre-executed **Jupyter Notebooks** covering all **11 machine learning algorithm groups** specified in the assignment specification (`image.jpg`).

Each notebook implements the target algorithms using standard modern Python ML libraries (`scikit-learn`, `xgboost`, `catboost`, `PyTorch`, `transformers`, `minisom`, `hmmlearn`), complete mathematical background with LaTeX equations, exploratory data analysis, real-world datasets, hyperparameter sensitivity tuning, high-resolution visualizations, and quantitative evaluation benchmarks.

---

## Repository Structure

```
d:/ML/Assignment/
├── data/                                         # Datasets (CSV and preprocessed tables)
│   ├── mall_customers.csv                        # Mall Customers Segmentation dataset
│   ├── airline_passengers.csv                    # Monthly Airline Passengers Time Series
│   ├── heart_disease.csv                         # Heart Disease dataset (numerical & categorical)
│   └── market_regimes.csv                        # Financial asset returns and volatility regimes
├── notebooks/                                    # Executed Jupyter Notebooks
│   ├── 01_clustering_algorithms.ipynb            # K-Means, Bisecting K-Means, Hierarchical, FCM
│   ├── 02_density_based_learning.ipynb           # DBSCAN and HDBSCAN on non-convex manifolds
│   ├── 03_semi_supervised_learning.ipynb         # Self-Training Classifier (SVC & Random Forest)
│   ├── 04_ensemble_learning.ipynb                # RFR, RFC, XGBoost, AdaBoost, CatBoost
│   ├── 05_multilayer_perceptron.ipynb            # Scikit-Learn MLP & Custom PyTorch Deep MLP
│   ├── 06_recurrent_neural_network.ipynb         # PyTorch Vanilla RNN vs LSTM Sequence Forecaster
│   ├── 07_self_organizing_map.ipynb              # Kohonen SOM, U-Matrix, Win Map via MiniSom
│   ├── 08_hidden_markov_model.ipynb              # Gaussian HMM, Viterbi Regime Decoding via hmmlearn
│   ├── 09_support_vector_machine.ipynb           # SVC (Linear/Poly/RBF) & SVR with Epsilon Tube
│   ├── 10_large_language_model.ipynb             # Hugging Face Transformers, Inference & Fine-Tuning
│   └── 11_generalized_regression_neural_network.ipynb # Specht GRNN Architecture & Spread Optimization
├── image.jpg                                     # Original Assignment Prompt Image
└── README.md                                     # Project Documentation & Guide
```

---

## Detailed Summary of Notebooks

| # | Notebook | Algorithms Covered | Primary Libraries | Datasets Used | Key Visualizations & Outputs |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | [`01_clustering_algorithms.ipynb`](notebooks/01_clustering_algorithms.ipynb) | • K-Means<br>• Bisecting K-Means<br>• Hierarchical (Ward)<br>• Fuzzy C-Means (FCM) | `scikit-learn`<br>`scipy`<br>`numpy` | Mall Customers (`data/mall_customers.csv`) | Elbow Inertia curve, Silhouette across $K$, Dendrogram cut-offs, Soft membership heatmaps. |
| **02** | [`02_density_based_learning.ipynb`](notebooks/02_density_based_learning.ipynb) | • DBSCAN<br>• HDBSCAN | `scikit-learn` | Non-convex interlocking moons, concentric circles & noise | $k$-distance graph for $\epsilon$ elbow, Core vs. Border vs. Noise separation, HDBSCAN probability map. |
| **03** | [`03_semi_supervised_learning.ipynb`](notebooks/03_semi_supervised_learning.ipynb) | • Self-Training Classifier | `scikit-learn` | Breast Cancer Wisconsin (85% masked as unlabeled) | Confidence threshold ($\tau$) trade-off curves, Baseline (15%) vs. Semi-Supervised vs. Fully Supervised ceiling. |
| **04** | [`04_ensemble_learning.ipynb`](notebooks/04_ensemble_learning.ipynb) | • Random Forest Regression (RFR)<br>• Random Forest Classification (RFC)<br>• XGBoost<br>• AdaBoost<br>• CatBoost | `scikit-learn`<br>`xgboost`<br>`catboost` | California Housing (Regression) & Heart Disease (Classification) | OOB score, Predicted vs Actual scatter, Multi-model ROC curves, Feature importance bar charts. |
| **05** | [`05_multilayer_perceptron.ipynb`](notebooks/05_multilayer_perceptron.ipynb) | • Scikit-Learn MLPClassifier<br>• PyTorch Deep MLP | `scikit-learn`<br>`torch` | MNIST Handwritten Digits (8x8 grayscale images) | Cross-entropy loss vs. epochs, Train/Val accuracy curves, Confusion matrix heatmap, Sample digit predictions. |
| **06** | [`06_recurrent_neural_network.ipynb`](notebooks/06_recurrent_neural_network.ipynb) | • Vanilla RNN (`nn.RNN`)<br>• LSTM (`nn.LSTM`) | `torch` | Monthly Airline Passengers (`data/airline_passengers.csv`) | Sliding window sequences, Training loss comparison, 24-month multi-step forecast trajectory vs. Ground Truth. |
| **07** | [`07_self_organizing_map.ipynb`](notebooks/07_self_organizing_map.ipynb) | • Kohonen Self-Organizing Map | `minisom` | UCI Wine Recognition Dataset | U-Matrix distance map with projected wine cultivars, Win Map hit histogram, Component planes for chemical features. |
| **08** | [`08_hidden_markov_model.ipynb`](notebooks/08_hidden_markov_model.ipynb) | • Gaussian HMM (Baum-Welch EM & Viterbi Decoding) | `hmmlearn` | Asset Market Regimes (`data/market_regimes.csv`) | $3 \times 3$ Transition probability matrix heatmap, Gaussian emission densities, Asset price trajectory colored by latent regime. |
| **09** | [`09_support_vector_machine.ipynb`](notebooks/09_support_vector_machine.ipynb) | • Support Vector Classifier (SVC)<br>• Support Vector Regressor (SVR) | `scikit-learn` | Non-linear Moons, Breast Cancer & Sinusoidal Benchmark | Decision boundaries with highlighted support vectors across Linear/Poly/RBF kernels, $\epsilon$-insensitive tube plot, Grid search heatmap ($C$ vs $\gamma$). |
| **10** | [`10_large_language_model.ipynb`](notebooks/10_large_language_model.ipynb) | • Transformer Tokenizer<br>• Sentiment Pipeline<br>• Causal Text Generation<br>• Sequence Classifier Fine-Tuning | `transformers`<br>`torch` | DistilBERT SST-2, DistilGPT2 & Domain Support Inquiries | Tokenizer subword vocabulary ID mapping, Pipeline inference scores, Greedy vs. Nucleus ($p=0.9$) text generation, Mini fine-tuning loss curve. |
| **11** | [`11_generalized_regression_neural_network.ipynb`](notebooks/11_generalized_regression_neural_network.ipynb) | • Generalized Regression Neural Network (GRNN / Specht 1991) | Custom Vectorized Class + `scikit-learn` API | Multi-Modal Continuous Function & California Housing | 4-layer architecture analysis, Smoothing spread ($\sigma$) under/optimal/over smoothing plots, 5-Fold CV $\sigma$ optimization curve, SVR comparison. |

---

## Environment Setup and Running Locally

### 1. Environment & Dependencies
The project uses the dedicated Python 3.12 environment configured in `.venv`:
```bash
# Activate the virtual environment in PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Launching Jupyter Lab or Notebook
To explore or re-run the notebooks in your browser:
```bash
.venv\Scripts\jupyter lab
# OR
.venv\Scripts\jupyter notebook
```
In Jupyter or VS Code, select the registered kernel: **`Python (ML Assignment)`**.

### 3. Re-executing Notebooks via Command Line
To re-run any notebook headlessly and refresh all outputs:
```bash
.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks/01_clustering_algorithms.ipynb --inplace
```
