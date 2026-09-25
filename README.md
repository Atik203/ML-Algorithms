# Machine Learning Algorithms

Code and data repository behind the **Machine Learning Laboratory Manual — Completed Edition with Full Implementations and Results**: 11 pre-executed Jupyter notebooks (19 experiments + integrated mini project) covering classical, ensemble, neural, probabilistic, density-based, semi-supervised and generative machine learning on real datasets.

Repository: <https://github.com/Atik203/ML-Algorithms>

> **Note:** this repo tracks code, data and scripts only. The LaTeX manual source (`latex/`) and all built PDFs are kept local and listed in `.gitignore`.

---

## Repository structure

```
├── data/                       # Datasets (CSV)
│   ├── mall_customers.csv      # Mall Customers Segmentation dataset
│   ├── airline_passengers.csv  # Monthly Airline Passengers Time Series
│   ├── heart_disease.csv       # Heart Disease dataset (numerical & categorical)
│   ├── earthquakes.csv         # USGS global earthquake catalogue (2023)
│   ├── stock_index.csv         # DAX stock index daily prices 1991-1998
│   ├── geyser.csv              # Old Faithful geyser eruption data
│   ├── motorcycle.csv          # Motorcycle impact accelerometer data (MASS mcycle)
│   └── sms_spam.csv            # SMS Spam Collection (5,572 labeled messages)
├── notebooks/                  # Executed Jupyter Notebooks (Python 3.12, seeds fixed at 42)
│   ├── 01_clustering_algorithms.ipynb            # K-Means, Modified K-Means, Hierarchical, FCM
│   ├── 02_density_based_learning.ipynb           # DBSCAN and HDBSCAN
│   ├── 03_semi_supervised_learning.ipynb         # Self-Training Classifier
│   ├── 04_ensemble_learning.ipynb                # RFR, RFC, XGBoost, AdaBoost, CatBoost
│   ├── 05_multilayer_perceptron.ipynb            # Scikit-Learn MLP & PyTorch Deep MLP
│   ├── 06_recurrent_neural_network.ipynb         # Vanilla RNN vs LSTM forecaster
│   ├── 07_self_organizing_map.ipynb              # Kohonen SOM, U-Matrix via MiniSom
│   ├── 08_hidden_markov_model.ipynb              # Gaussian HMM, Viterbi decoding via hmmlearn
│   ├── 09_support_vector_machine.ipynb           # SVC (Linear/Poly/RBF) & SVR
│   ├── 10_large_language_model.ipynb             # Transformers inference & fine-tuning
│   └── 11_generalized_regression_neural_network.ipynb # GRNN & spread optimization
├── scripts/
│   └── export_pdf.py           # Rebuilds notebook PDFs (nbconvert -> HTML -> print)
├── README.md                   # This file
└── .gitignore                  # Excludes latex/, PDFs, docx and manual sources
```

---

## Detailed summary of notebooks

| # | Notebook | Algorithms Covered | Primary Libraries | Datasets Used |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [`01_clustering_algorithms.ipynb`](notebooks/01_clustering_algorithms.ipynb) | • K-Means<br>• Modified K-Means (outlier flag)<br>• Hierarchical (Ward)<br>• Fuzzy C-Means (FCM) | `scikit-learn`<br>`scipy`<br>`numpy` | Mall Customers (`data/mall_customers.csv`) |
| **02** | [`02_density_based_learning.ipynb`](notebooks/02_density_based_learning.ipynb) | • DBSCAN<br>• HDBSCAN | `scikit-learn` | USGS earthquakes 2023 (`data/earthquakes.csv`, 7,638 events, lat/lon + haversine) |
| **03** | [`03_semi_supervised_learning.ipynb`](notebooks/03_semi_supervised_learning.ipynb) | • Self-Training Classifier | `scikit-learn` | Breast Cancer Wisconsin (85% masked as unlabeled) |
| **04** | [`04_ensemble_learning.ipynb`](notebooks/04_ensemble_learning.ipynb) | • RFR<br>• RFC<br>• XGBoost<br>• AdaBoost<br>• CatBoost | `scikit-learn`<br>`xgboost`<br>`catboost` | California Housing (regression) & Heart Disease (classification) |
| **05** | [`05_multilayer_perceptron.ipynb`](notebooks/05_multilayer_perceptron.ipynb) | • Scikit-Learn MLPClassifier<br>• PyTorch Deep MLP | `scikit-learn`<br>`torch` | MNIST Handwritten Digits (8x8 grayscale images) |
| **06** | [`06_recurrent_neural_network.ipynb`](notebooks/06_recurrent_neural_network.ipynb) | • Vanilla RNN<br>• LSTM | `torch` | Airline Passengers + synthetic sequences |
| **07** | [`07_self_organizing_map.ipynb`](notebooks/07_self_organizing_map.ipynb) | • Kohonen Self-Organizing Map | `minisom` | UCI Wine Recognition Dataset |
| **08** | [`08_hidden_markov_model.ipynb`](notebooks/08_hidden_markov_model.ipynb) | • Gaussian HMM (Baum-Welch EM & Viterbi Decoding) | `hmmlearn` | DAX stock index 1991-1998 (`data/stock_index.csv`) |
| **09** | [`09_support_vector_machine.ipynb`](notebooks/09_support_vector_machine.ipynb) | • Support Vector Classifier (SVC)<br>• Support Vector Regressor (SVR) | `scikit-learn` | Breast Cancer (2 & 30 features) & Old Faithful geyser (`data/geyser.csv`) |
| **10** | [`10_large_language_model.ipynb`](notebooks/10_large_language_model.ipynb) | • Transformer Tokenizer<br>• Sentiment Pipeline<br>• Causal Text Generation<br>• Sequence Classifier Fine-Tuning | `transformers`<br>`torch` | DistilBERT SST-2, DistilGPT2 & SMS Spam Collection (`data/sms_spam.csv`) |
| **11** | [`11_generalized_regression_neural_network.ipynb`](notebooks/11_generalized_regression_neural_network.ipynb) | • Generalized Regression Neural Network (GRNN / Specht 1991) | Custom Vectorized Class + `scikit-learn` API | Real motorcycle accelerometer data (`data/motorcycle.csv`) |

---

## Environment setup and running locally

### 1. Environment & dependencies
The project uses the dedicated Python 3.12 environment configured in `.venv`:
```bash
# Activate the virtual environment in PowerShell
.venv\Scripts\Activate.ps1
```

> **GPU (optional):** the default `.venv` ships with CPU-only PyTorch. To run the PyTorch notebooks (05, 06, 10) on an NVIDIA GPU, install the CUDA wheel and let the notebooks auto-select it via `torch.cuda.is_available()`:
> ```bash
> uv pip install --python .venv\Scripts\python.exe --reinstall torch --index-url https://download.pytorch.org/whl/cu126
> ```
> The committed notebook outputs were produced on an NVIDIA RTX 3060 Laptop GPU.

### 2. Kernel registration (one-time setup)
Every notebook is pinned to the **`Python (ML Assignment)`** kernel, which launches the `.venv` interpreter. Register it once with:
```bash
.venv\Scripts\python -m ipykernel install --user --name ml_assignment --display-name "Python (ML Assignment)"
```

### 3. Launching Jupyter Lab or Notebook
```bash
.venv\Scripts\jupyter lab
# OR
.venv\Scripts\jupyter notebook
```
In Jupyter or VS Code, select the registered kernel: **`Python (ML Assignment)`**.

### 4. Re-executing notebooks via command line
```bash
.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks/01_clustering_algorithms.ipynb --inplace
```
All 11 notebooks in this repository are pre-executed and verified with this kernel (Python 3.12.13, zero cell errors).
