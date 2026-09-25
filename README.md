# Machine Learning Laboratory Manual

Official code repository for the **Machine Learning Laboratory Manual — Completed Edition with Full Implementations and Results**: a formal, print-ready academic lab manual (19 experiments + integrated mini project + appendices) with complete runnable Python code, real-dataset results, and result plots throughout.

Repository: <https://github.com/Atik203/ML-Algorithms>

---

## The manual (`latex/`)

Native LaTeX project (pdfLaTeX, no extra tooling). Each of the 12 parts opens with its own experiment index; every experiment follows the faculty 8-section format — Objective, Suggested Dataset, Required Libraries, Brief Theory, Procedure, Reference Implementation (full code listing), Evaluation / Expected Output, Viva / Discussion Questions.

```
latex/
├── main.tex                 # Master file: title, contents, lists, 12 parts
├── preamble.tex             # All styling: fonts, code boxes, tables, headings
├── chapters/
│   ├── 00_frontmatter.tex   # Preface, scope, guidelines, report format, metrics
│   ├── 01_clustering.tex    # Experiments 1-4:  K-means, Modified K-means,
│   │                        #   Hierarchical, Fuzzy C-means
│   ├── 02_density_based.tex # Experiments 5-6:  DBSCAN, HDBSCAN
│   ├── 03_semi_supervised.tex # Experiment 7:   Self-training
│   ├── 04_ensemble.tex      # Experiments 8-12: RFR, RFC, XGBoost, AdaBoost, CatBoost
│   ├── 05_mlp.tex           # Experiment 13:    Multilayer Perceptron
│   ├── 06_rnn.tex           # Experiment 14:    RNN vs LSTM
│   ├── 07_som.tex           # Experiment 15:    Self-Organizing Map
│   ├── 08_hmm.tex           # Experiment 16:    Hidden Markov Model
│   ├── 09_svm.tex           # Experiment 17:    SVC and SVR
│   ├── 10_llm.tex           # Experiment 18:    LLM fine-tuning (SMS spam)
│   ├── 11_grnn.tex          # Experiment 19:    GRNN
│   └── 12_mini_project_appendices.tex  # Comparative study + Appendices A-D
├── code/                    # Complete per-experiment listings (exp01-exp19)
└── figures/                 # Result plots referenced by the chapters
```

### Build the PDF

The document compiles with plain pdfLaTeX — no `latexmkrc`, no magic comments, no extra configuration:

```bash
cd latex
latexmk main.tex
```

- **VS Code (LaTeX Workshop):** open `latex/main.tex` and Build with the default `latexmk` recipe.
- **Overleaf:** upload the `latex/` folder (or a repo ZIP) and Recompile with the default compiler.
- Requires a full TeX Live installation (packages used: `libertinus`, `inconsolata`, `tcolorbox`, `titlesec`, `titletoc`, `microtype`, `placeins` — all standard).

Prepared by **Dr. Ohidujjaman Tuhin**, Associate Professor, Dept. of CSE, UIU — Department of Computer Science & Engineering, Machine Learning Lab.

---

## Notebooks, data and scripts

All manual code and results come from the executed notebooks below (Python 3.12, seeds fixed at 42). Notebook PDFs and the manual PDF are build outputs and are not committed — regenerate them locally.

| # | Notebook | Algorithms | Key libraries | Dataset |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [`01_clustering_algorithms.ipynb`](notebooks/01_clustering_algorithms.ipynb) | K-Means, Modified K-Means, Hierarchical (Ward), Fuzzy C-Means | `scikit-learn`, `scipy`, `numpy` | Mall Customers (`data/mall_customers.csv`) |
| **02** | [`02_density_based_learning.ipynb`](notebooks/02_density_based_learning.ipynb) | DBSCAN, HDBSCAN | `scikit-learn` | USGS earthquakes 2023 (`data/earthquakes.csv`, haversine) |
| **03** | [`03_semi_supervised_learning.ipynb`](notebooks/03_semi_supervised_learning.ipynb) | Self-Training Classifier | `scikit-learn` | Breast Cancer Wisconsin (85% labels masked) |
| **04** | [`04_ensemble_learning.ipynb`](notebooks/04_ensemble_learning.ipynb) | RFR, RFC, XGBoost, AdaBoost, CatBoost | `scikit-learn`, `xgboost`, `catboost` | California Housing & Heart Disease |
| **05** | [`05_multilayer_perceptron.ipynb`](notebooks/05_multilayer_perceptron.ipynb) | Scikit-Learn MLP, PyTorch Deep MLP | `scikit-learn`, `torch` | MNIST 8x8 digits |
| **06** | [`06_recurrent_neural_network.ipynb`](notebooks/06_recurrent_neural_network.ipynb) | Vanilla RNN vs LSTM | `torch` | Airline Passengers (`data/airline_passengers.csv`) |
| **07** | [`07_self_organizing_map.ipynb`](notebooks/07_self_organizing_map.ipynb) | Kohonen SOM, U-Matrix | `minisom` | UCI Wine |
| **08** | [`08_hidden_markov_model.ipynb`](notebooks/08_hidden_markov_model.ipynb) | Gaussian HMM, Viterbi decoding | `hmmlearn` | DAX 1991-1998 (`data/stock_index.csv`) |
| **09** | [`09_support_vector_machine.ipynb`](notebooks/09_support_vector_machine.ipynb) | SVC (Linear/Poly/RBF), SVR | `scikit-learn` | Breast Cancer & Old Faithful (`data/geyser.csv`) |
| **10** | [`10_large_language_model.ipynb`](notebooks/10_large_language_model.ipynb) | Tokenizer, pipelines, fine-tuning | `transformers`, `torch` | SMS Spam (`data/sms_spam.csv`) |
| **11** | [`11_generalized_regression_neural_network.ipynb`](notebooks/11_generalized_regression_neural_network.ipynb) | GRNN (Specht), spread CV | Custom class + `scikit-learn` | Motorcycle impact (`data/motorcycle.csv`) |

### Environment setup

```bash
# Activate the virtual environment in PowerShell
.venv\Scripts\Activate.ps1

# Register the notebook kernel once
.venv\Scripts\python -m ipykernel install --user --name ml_assignment --display-name "Python (ML Assignment)"

# Launch Jupyter
.venv\Scripts\jupyter lab
```

> **GPU (optional):** the default `.venv` ships with CPU-only PyTorch. For CUDA (the committed outputs used an NVIDIA RTX 3060 Laptop GPU):
> ```bash
> uv pip install --python .venv\Scripts\python.exe --reinstall torch --index-url https://download.pytorch.org/whl/cu126
> ```

### Exporting notebook PDFs

```bash
.venv\Scripts\python scripts\export_pdf.py                                  # all notebooks + combined PDF
.venv\Scripts\python scripts\export_pdf.py 01_clustering_algorithms.ipynb   # a single notebook
.venv\Scripts\python scripts\export_pdf.py --combined-only                  # combined PDF only
```
