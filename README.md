# Machine Learning Algorithms Assignment

This repository contains production-quality, thoroughly documented, and pre-executed **Jupyter Notebooks** covering all **11 machine learning algorithm groups** specified in the assignment specification (`image.jpg`).

Each notebook implements the target algorithms using standard modern Python ML libraries (`scikit-learn`, `xgboost`, `catboost`, `PyTorch`, `transformers`, `minisom`, `hmmlearn`): a concise step-by-step algorithm walkthrough, commented code, a suitable dataset, and compact evaluation metrics and figures.

---

## Repository Structure

```
d:/ML/Assignment/
├── data/                                         # Datasets (CSV and preprocessed tables)
│   ├── mall_customers.csv                        # Mall Customers Segmentation dataset
│   ├── airline_passengers.csv                    # Monthly Airline Passengers Time Series
│   ├── heart_disease.csv                         # Heart Disease dataset (numerical & categorical)
│   ├── earthquakes.csv                           # USGS global earthquake catalogue (2023)
│   ├── stock_index.csv                           # DAX stock index daily prices 1991-1998 (EuStockMarkets)
│   ├── geyser.csv                                # Old Faithful geyser eruption data
│   ├── motorcycle.csv                            # Motorcycle impact accelerometer data (MASS mcycle)
│   └── sms_spam.csv                              # SMS Spam Collection (5,572 labeled messages)
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
├── pdf/                                          # Print-ready A4 PDFs with all outputs
│   ├── 00_combined_all_notebooks.pdf             # Every notebook in a single document
│   └── 01_..._11_*.pdf                           # One PDF per notebook
├── scripts/
│   └── export_pdf.py                             # Rebuilds pdf/ (nbconvert -> HTML -> Chromium print)
├── Machine_Learning_Lab_Manual.docx              # Faculty's laboratory manual (reference format)
├── Machine_Learning_Lab_Manual_Completed.docx    # Completed edition of the lab manual, 82 pages with full code and results
├── ML_Lab_Manual_Completed.md                    # Markdown source of the completed manual
├── image.jpg                                     # Original Assignment Prompt Image
└── README.md                                     # Project Documentation & Guide
```

---

## Detailed Summary of Notebooks

| # | Notebook | Algorithms Covered | Primary Libraries | Datasets Used | Key Visualizations & Outputs |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | [`01_clustering_algorithms.ipynb`](notebooks/01_clustering_algorithms.ipynb) | • K-Means<br>• Modified K-Means (outlier flag)<br>• Hierarchical (Ward)<br>• Fuzzy C-Means (FCM) | `scikit-learn`<br>`scipy`<br>`numpy` | Mall Customers (`data/mall_customers.csv`) | Elbow/silhouette plot, dendrogram, combined 2x2 partition grid, outlier flags, metrics table with FPC. |
| **02** | [`02_density_based_learning.ipynb`](notebooks/02_density_based_learning.ipynb) | • DBSCAN<br>• HDBSCAN | `scikit-learn` | USGS earthquakes 2023 (`data/earthquakes.csv`, 7,638 events, lat/lon + haversine) | k-distance graph, parameter-sensitivity table, DBSCAN/HDBSCAN world maps with noise, region check, summary table. |
| **03** | [`03_semi_supervised_learning.ipynb`](notebooks/03_semi_supervised_learning.ipynb) | • Self-Training Classifier | `scikit-learn` | Breast Cancer Wisconsin (85% masked as unlabeled) | Convergence log and comparison table: baseline (15%) vs semi-supervised vs fully supervised ceiling. |
| **04** | [`04_ensemble_learning.ipynb`](notebooks/04_ensemble_learning.ipynb) | • RFR<br>• RFC<br>• XGBoost<br>• AdaBoost<br>• CatBoost | `scikit-learn`<br>`xgboost`<br>`catboost` | California Housing (regression) & Heart Disease (classification) | Regression and classification comparison tables for all five ensembles. |
| **05** | [`05_multilayer_perceptron.ipynb`](notebooks/05_multilayer_perceptron.ipynb) | • Scikit-Learn MLPClassifier<br>• PyTorch Deep MLP | `scikit-learn`<br>`torch` | MNIST Handwritten Digits (8x8 grayscale images) | Scikit-Learn loss curve and classification report for both MLP implementations. |
| **06** | [`06_recurrent_neural_network.ipynb`](notebooks/06_recurrent_neural_network.ipynb) | • Vanilla RNN<br>• LSTM | `torch` | Airline Passengers + synthetic sequences | RMSE/MAE forecast table, forecast trajectory, sequence-classification accuracy (RNN vs LSTM). |
| **07** | [`07_self_organizing_map.ipynb`](notebooks/07_self_organizing_map.ipynb) | • Kohonen Self-Organizing Map | `minisom` | UCI Wine Recognition Dataset | U-Matrix with projected wine cultivars, quantization and topographic error. |
| **08** | [`08_hidden_markov_model.ipynb`](notebooks/08_hidden_markov_model.ipynb) | • Gaussian HMM (Baum-Welch EM & Viterbi Decoding) | `hmmlearn` | DAX stock index 1991-1998 (`data/stock_index.csv`) | Transition matrix heatmap and DAX price trajectory colored by Viterbi-decoded regime. |
| **09** | [`09_support_vector_machine.ipynb`](notebooks/09_support_vector_machine.ipynb) | • Support Vector Classifier (SVC)<br>• Support Vector Regressor (SVR) | `scikit-learn` | Breast Cancer (2 & 30 features) & Old Faithful geyser (`data/geyser.csv`) | SVC decision boundaries on real data across Linear/Poly/RBF kernels, $\epsilon$-insensitive tube plot, breast-cancer accuracy. |
| **10** | [`10_large_language_model.ipynb`](notebooks/10_large_language_model.ipynb) | • Transformer Tokenizer<br>• Sentiment Pipeline<br>• Causal Text Generation<br>• Sequence Classifier Fine-Tuning | `transformers`<br>`torch` | DistilBERT SST-2, DistilGPT2 & SMS Spam Collection (`data/sms_spam.csv`) | Tokenizer subword mapping, sentiment inference table, greedy vs. nucleus generation, SMS spam fine-tune (96.0% accuracy). |
| **11** | [`11_generalized_regression_neural_network.ipynb`](notebooks/11_generalized_regression_neural_network.ipynb) | • Generalized Regression Neural Network (GRNN / Specht 1991) | Custom Vectorized Class + `scikit-learn` API | Real motorcycle accelerometer data (`data/motorcycle.csv`) | Under/optimal/over-smoothing plots, 5-fold CV $\sigma$ curve, final GRNN metrics. |

---

## Environment Setup and Running Locally

### 1. Environment & Dependencies
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

### 2. Kernel Registration (One-Time Setup)
Every notebook is pinned to the **`Python (ML Assignment)`** kernel, which launches the `.venv` interpreter. Register it once with:
```bash
.venv\Scripts\python -m ipykernel install --user --name ml_assignment --display-name "Python (ML Assignment)"
```

### 3. Launching Jupyter Lab or Notebook
To explore or re-run the notebooks in your browser:
```bash
.venv\Scripts\jupyter lab
# OR
.venv\Scripts\jupyter notebook
```
In Jupyter or VS Code, select the registered kernel: **`Python (ML Assignment)`**.

### 4. Re-executing Notebooks via Command Line
To re-run any notebook headlessly and refresh all outputs:
```bash
.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks/01_clustering_algorithms.ipynb --inplace
```
All 11 notebooks in this repository are pre-executed and verified with this kernel (Python 3.12.13, zero cell errors).

### 5. Exporting to A4 PDF (Print / Submission)
The `pdf/` folder already contains print-ready A4 PDFs with all outputs (one per notebook + `00_combined_all_notebooks.pdf`). To regenerate them after editing a notebook:
```bash
.venv\Scripts\python scripts\export_pdf.py                                  # all notebooks + combined PDF
.venv\Scripts\python scripts\export_pdf.py 01_clustering_algorithms.ipynb   # a single notebook
.venv\Scripts\python scripts\export_pdf.py --combined-only                  # rebuild only the combined PDF
```
The exporter converts each notebook to HTML with nbconvert and prints it using headless Edge/Chrome with an A4 print stylesheet (minimal margins, enlarged code/output fonts, wrapped lines, MathJax equations). The first run needs internet access so MathJax can load. When printing choose **A4** paper and **100% / Actual size** scaling; the combined PDF prints as a single job.
