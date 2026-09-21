# Viva / Presentation Preparation Guide

How to talk about each notebook: **dataset → preprocessing → algorithm → evaluation metric → value**.
All numbers below are the actual outputs of the executed notebooks in this repo.

**30-second answer pattern for any algorithm:**
"I used [library], trained [algorithm] with [key parameters] on [dataset], evaluated with [metric(s)] on a held-out test set, and got [value]."

---

## 01 - Clustering (K-Means, Modified K-Means, Hierarchical, Fuzzy C-Means)

- **Dataset:** Mall Customers (200 rows). Features used: Annual Income + Spending Score (2D so clusters can be plotted).
- **Preprocessing:** StandardScaler (mean 0, std 1) - distance-based algorithms are scale-sensitive.
- **Implementation:** `KMeans` (init `k-means++`, n_init 20), `BisectingKMeans` (biggest_inertia), `AgglomerativeClustering` (Ward), custom `FuzzyCMeans` class (NumPy, m=2.0, 150 max iterations, tol 1e-5).
- **How K was chosen:** elbow (inertia) + silhouette for K = 2..10 → **K = 5**.
- **Metrics & values:**
  - Silhouette: K-Means/FCM 0.5547, Hierarchical 0.5538, Bisecting 0.4807
  - Davies-Bouldin: ~0.57 (K-Means/FCM) - lower is better
- **Likely questions:**
  - *Why standardize?* Distance-based (centroid) methods break if one feature has a larger scale.
  - *What is "Modified K-Means" here?* K-Means++ smart initialization + Bisecting K-Means (divisive splitting of the largest-inertia cluster).
  - *K-Means vs FCM?* K-Means = hard assignment; FCM gives soft membership degrees u_ik (argmax gives hard labels).
  - *Why Ward linkage?* It merges the pair that increases within-cluster variance the least → compact clusters.

## 02 - Density-Based Learning (DBSCAN, HDBSCAN)

- **Dataset:** synthetic non-convex benchmark: 2 moons + 2 circles + 1 blob + uniform noise (600 points).
- **Preprocessing:** StandardScaler so ε has one comparable meaning.
- **Implementation:** `DBSCAN(eps=0.22, min_samples=6)`; `HDBSCAN(min_cluster_size=15, min_samples=6)`.
- **How ε was chosen:** k-distance graph (sorted distance to the 6th neighbor) → knee ≈ 0.22.
- **Metrics & values:** DBSCAN 3 clusters / 4.7% noise / ARI 0.4637; HDBSCAN 3 clusters / 3.7% noise / ARI 0.4578.
- **Likely questions:**
  - *Core / border / noise point?* Core: ≥ MinPts neighbors within ε. Border: within ε of a core point. Noise: neither (label -1).
  - *DBSCAN vs HDBSCAN?* DBSCAN needs one global ε and struggles with varying density; HDBSCAN builds a hierarchy over all density levels and keeps the most stable clusters (no ε).
  - *What is ARI?* Adjusted Rand Index - agreement with ground-truth labels, corrected for chance. ~0.46 = moderate (hard benchmark with noise).

## 03 - Semi-Supervised Learning (Self-Training)

- **Dataset:** Breast Cancer Wisconsin (569 samples, 30 features, binary).
- **Setup:** 70/30 stratified split. In training: only **15% labeled (61 samples)**, 85% masked as unlabeled (-1). Test = 171 samples.
- **Implementation:** `SelfTrainingClassifier(estimator=SVC(RBF, probability=True), threshold=0.80, max_iter=15)`.
- **Metrics & values:** baseline SVC (15% labels) 0.9298 acc / 0.9469 F1 → self-training **0.9415 / 0.9554** → fully supervised ceiling 0.9766 / 0.9813.
- **Likely questions:**
  - *What is self-training?* Train on labeled data, predict on unlabeled data, keep confident predictions (≥ τ) as pseudo-labels, repeat.
  - *Why did accuracy improve over the baseline?* Confident pseudo-labels add information about the data distribution; it captures part of the gap to the fully supervised model.
  - *Why is τ important?* Too low = wrong pseudo-labels (confirmation bias); too high = too few labels added.
  - *Why probability=True on SVC?* Self-training needs `predict_proba` for confidence; SVC needs Platt calibration for that.

## 04 - Ensemble Learning (RFR, RFC, XGBoost, AdaBoost, CatBoost)

- **Datasets:** California Housing (regression, 5,000-row sample) and Heart Disease (`data/heart_disease.csv`) for classification.
- **Implementation (regression):** RFR 150 trees/depth 12 with OOB; XGBoost 150 trees, lr 0.08, depth 6, subsample 0.8; AdaBoost 100 stumps, lr 0.1; CatBoost 200 iters, depth 6, L2=3.
- **Implementation (classification):** RFC 120 trees/depth 8 on one-hot-heart data (stratified 75/25).
- **Metrics & values (test):** RFR R² 0.7415 (RMSE 0.591); AdaBoost 0.5888 (0.745); XGBoost 0.8025 (0.517); CatBoost 0.8029 (0.516). RFC on heart: accuracy 0.7763, F1 0.8046, ROC-AUC 0.8704.
- **Likely questions:**
  - *Bagging vs boosting?* Bagging (Random Forest) trains trees in parallel on bootstrap samples and averages → reduces variance. Boosting trains sequentially, each model fixing previous errors → reduces bias.
  - *What is OOB score?* Out-of-bag: each tree is validated on the ~37% of samples it did not see; free validation estimate (0.7513 here).
  - *Why did XGBoost/CatBoost beat AdaBoost?* Gradient boosting with 2nd-order optimization + regularization handles this data better; AdaBoost's exponential loss is more sensitive to noisy samples.
  - *RFR vs RFC?* Regression averages tree outputs; classification uses majority vote.

## 05 - Multilayer Perceptron (MLP)

- **Dataset:** Handwritten digits (1,797 samples, 8x8 = 64 features, 10 classes). Split 70/15/15.
- **Implementation 1:** `MLPClassifier(hidden_layer_sizes=(128,64), relu, adam, alpha=0.001, early_stopping=True)` → test accuracy **0.9630** (converged in 15 iterations).
- **Implementation 2 (PyTorch):** 64→128→64→10 with BatchNorm + Dropout (0.25/0.20), `CrossEntropyLoss`, Adam (lr 0.003, weight decay 1e-4), 50 epochs, batch 32, ReduceLROnPlateau → test accuracy **0.9889**.
- **Likely questions:**
  - *Why ReLU?* Cheap, avoids sigmoid saturation/vanishing gradient in hidden layers.
  - *Why Dropout / BatchNorm?* Regularization against overfitting; BatchNorm stabilizes and speeds training.
  - *What loss?* Categorical cross-entropy (softmax output + NLL), the standard for multi-class classification.
  - *Why is the PyTorch model better?* BatchNorm + Dropout + learning-rate scheduling and more epochs; the sklearn model stops early.
  - *What is early stopping?* Stop when validation score stops improving - prevents overfitting.

## 06 - Recurrent Neural Network (RNN vs LSTM)

- **Dataset:** Monthly Airline Passengers (144 months). Sliding windows: 12 months → predict next month. Chronological 80/20 split (105 train / 27 test sequences).
- **Preprocessing:** MinMaxScaler to [0,1]; sequences are NOT shuffled (time order matters).
- **Implementation:** 2-layer `nn.RNN` and `nn.LSTM`, hidden size 64, output Linear(64,1); Adam lr 0.005, 120 epochs, MSE loss.
- **Metrics & values (test, original passenger units):** Vanilla RNN RMSE 70.83 / MAE 63.16; **LSTM RMSE 42.17** / MAE 34.88.
- **Likely questions:**
  - *Why LSTM better?* Gated cell state (additive) preserves long-term memory; vanilla RNN gradients vanish over long sequences.
  - *What is BPTT?* Backpropagation through time - gradients flow backwards through all time steps.
  - *Why sliding window 12?* One full annual seasonal cycle as input history.
  - *Why temporal split without shuffle?* Random shuffling would leak future information into training.

## 07 - Self-Organizing Map (SOM)

- **Dataset:** UCI Wine (178 samples, 13 features). Labels used only for visualization - SOM training is unsupervised.
- **Preprocessing:** MinMaxScaler to [0,1].
- **Implementation:** MiniSom 12x12 grid (144 neurons), Gaussian neighborhood, sigma 1.5, learning rate 0.5, PCA weight init, 5,000 iterations.
- **Metrics & values:** quantization error 0.4447 → **0.1965** after training; topographic error **0.0393** (~4% foldings).
- **Likely questions:**
  - *What is a BMU?* Best Matching Unit - the neuron whose weight vector is closest to the input.
  - *What does the U-Matrix show?* Average distance to neighboring neurons: dark valleys = clusters, bright ridges = boundaries.
  - *Quantization vs topographic error?* QE = how well neurons represent data (lower better); TE = how well the 2D topology is preserved (lower is better, 0 = perfect).
  - *Why PCA initialization?* Starts the map already spread along the data's main directions → faster, more faithful convergence.

## 08 - Hidden Markov Model (HMM)

- **Dataset:** Market returns & regimes (`data/market_regimes.csv`, 1,000 trading days). Observations = daily returns.
- **Implementation:** `GaussianHMM(n_components=3, covariance_type="full", n_iter=200)` trained with Baum-Welch (EM); decoded with Viterbi. States reordered by volatility → Bull / Sideways / Bear.
- **Metrics & values:** EM converged = True; model log-likelihood **3018.45** (higher/less negative better).
- **Likely questions:**
  - *What is hidden vs observed?* Hidden = market regime (latent state); observed = daily returns emitted from a Gaussian per state.
  - *What does the transition matrix tell you?* A_ij = probability of switching from regime i to j; large diagonal = regime persistence.
  - *Baum-Welch vs Viterbi?* Baum-Welch (EM) learns parameters; Viterbi finds the single most likely state sequence (decoding).
  - *Why sort states by volatility?* So "Regime 0/1/2" always mean low/medium/high volatility - stable interpretation across runs.

## 09 - Support Vector Machines (SVC, SVR)

- **Datasets:** two-moons synthetic (2D), Breast Cancer (569x30), 1D sine+trend+noise for SVR.
- **Implementation:** `SVC(kernel = linear/poly/rbf, C=1.0)` for the 2D demo; RBF SVC on Breast Cancer; `SVR(kernel='rbf', C=20, epsilon=0.18, gamma=0.5)`.
- **Metrics & values:** SVC (RBF) test accuracy on Breast Cancer **0.9790**, using 96 of 426 training samples as support vectors. SVR: R² **0.9864**, RMSE 0.1157, 18/120 points (15%) are support vectors.
- **Likely questions:**
  - *What is a support vector?* Training points on/inside the margin that define the decision boundary; removing others doesn't change the model.
  - *Role of C?* Penalty for margin violations - large C = strict (small margin, may overfit), small C = wider margin, more violations.
  - *Role of gamma?* RBF width: large gamma = local, wiggly boundary; small gamma = smoother/almost linear.
  - *Kernel trick?* Compute inner products in a high-dimensional space without explicitly mapping points there.
  - *What is the ε-tube in SVR?* Errors within ±ε cost nothing; only points outside the tube become support vectors.

## 10 - Large Language Model (LLM)

- **Models:** DistilBERT SST-2 (tokenization + sentiment) and DistilGPT2 (generation) via Hugging Face; PyTorch for a mini fine-tune.
- **What is shown:**
  1. Tokenization: text → subword tokens + IDs + attention mask (padding) - e.g. `[CLS] ...[SEP]` then `[PAD]`.
  2. Sentiment inference pipeline on 4 sentences (label + confidence).
  3. Generation: greedy vs top-p (T=0.7, p=0.9) from the prompt "Artificial Intelligence will fundamentally transform".
  4. Fine-tune: 8 domain queries (Technical vs Billing), 3 epochs, lr 1e-4, AdamW → correct predictions (Billing 88.3%, Technical 84.0%).
- **Likely questions:**
  - *Why subword tokenization?* Handles any word (including unseen) with a fixed vocabulary.
  - *What is the attention mask for?* Marks real tokens (1) vs padding (0) so padding doesn't affect attention.
  - *Greedy vs top-p?* Greedy = always most likely token (deterministic, can loop); top-p samples from the smallest token set with cumulative probability ≥ p → more diverse.
  - *What did fine-tuning change?* Replaced the classification head (2 labels) and adapted weights to the support-domain data with a small learning rate.

## 11 - Generalized Regression Neural Network (GRNN)

- **Dataset:** synthetic noisy 1D function y = sin(2x) + cos(0.5x²) + noise (100 points).
- **Implementation:** custom `GRNN` class (NumPy, scikit-learn API). Prediction = S(x)/D(x) where S = Σ y_i·exp(-‖x-x_i‖²/2σ²), D = Σ exp(...). One-pass training (memorizes the data, no backprop).
- **σ selection:** 5-fold CV over σ = 0.05..1.0 → **σ = 0.069** (CV RMSE 0.2137).
- **Metrics & values:** final held-out test R² **0.9473**, RMSE 0.1718; training R² 0.9829.
- **Likely questions:**
  - *How is GRNN trained?* One pass: store the training set. Only hyperparameter is the smoothing spread σ.
  - *What does σ control?* Small σ = memorizes noise (high variance); large σ = over-smooth (high bias).
  - *Why is it called a neural network?* Four layers (input → pattern → summation S/D → output); pattern layer has one Gaussian neuron per training point.
  - *What is it mathematically?* Nadaraya-Watson kernel regression = estimate of E[y|x].

---

## Evaluation metrics cheat sheet

| Metric | Used in | What it measures | Range | Good value (rule of thumb) |
|---|---|---|---|---|
| Silhouette | 01 | cohesion vs separation | -1..1 | >0.5 good; <0.25 weak |
| Davies-Bouldin | 01 | avg cluster similarity | ≥0 | lower better; <1 good |
| Calinski-Harabasz | 01 | between/within dispersion | ≥0 | higher better; compare models |
| ARI | 02 | agreement with ground truth, chance-adjusted | -0.5..1 | 0 = random; >0.5 strong |
| Accuracy | 03,04,05,09 | % correct | 0..1 | task-dependent; >0.9 strong here |
| Precision / Recall / F1 | 03,04,05 | F1 = harmonic mean of P and R | 0..1 | >0.9 excellent |
| ROC-AUC | 04 | ranking quality across thresholds | 0..1 | 0.5 random; >0.8 good; >0.9 excellent |
| R² | 04,06(implicit),11 | variance explained | -∞..1 | >0.7 good for noisy tabular data |
| RMSE / MAE | 04,06,09,11 | average error in target units | ≥0 | lower better; compare models only |
| MSE loss | 06 | squared error, training objective | ≥0 | decreasing/converging curve |
| Cross-entropy loss | 05,10 | classification training objective | ≥0 | decreasing/converging curve |
| Quantization error | 07 | avg distance sample → BMU | ≥0 | lower better; dropped 0.44→0.20 |
| Topographic error | 07 | topology preservation | 0..1 | <0.05 good; 0 = perfect |
| Log-likelihood | 08 | model fit of the HMM | -∞..+∞ | higher (less negative) better |

## Validation / CV used - where and why

| Notebook | Validation scheme | Why |
|---|---|---|
| 01 | elbow + silhouette over K=2..10 | unsupervised: pick K by internal quality |
| 02 | k-distance graph (knee) | heuristic to pick ε without labels |
| 03 | stratified holdout (70/30) + 15% labeling | simulate label scarcity, test once |
| 04 | holdout + OOB (RFR) | fast validation; OOB is free for bagging |
| 05 | early-stopping validation split (15%) | stop training when validation stops improving |
| 06 | chronological 80/20 split | time series must not shuffle |
| 07 | none (internal QE/TE) | unsupervised; QE/TE are self-assessment metrics |
| 08 | none (EM log-likelihood) | unsupervised; convergence via monitor |
| 09 | holdout test set | simple unbiased estimate for SVC/SVR |
| 10 | small fine-tune set + unseen queries | check transfer to held-out domain queries |
| 11 | 5-fold CV over σ | select the only hyperparameter reliably |

## Datasets used

| Data | Notebooks | Size | Target |
|---|---|---|---|
| Mall Customers (`data/mall_customers.csv`) | 01 | 200 x 5 | 2 features clustered (income, spending) |
| Synthetic moons/circles/blob/noise | 02 | 600 x 2 | 3 shapes + noise |
| Breast Cancer (sklearn) | 03, 09 | 569 x 30 | malignant / benign |
| California Housing (sklearn, 5k sample) | 04 | 5,000 x 8 | median house value |
| Heart Disease (`data/heart_disease.csv`) | 04 | 303 x 13 | disease present 0/1 |
| Digits (sklearn) | 05 | 1,797 x 64 | digit 0-9 |
| Airline Passengers (`data/airline_passengers.csv`) | 06 | 144 months | next-month passengers |
| Wine (sklearn) | 07 | 178 x 13 | 3 cultivars (visualization only) |
| Market Regimes (`data/market_regimes.csv`) | 08 | 1,000 days | hidden regime (unsupervised) |
| Synthetic moons + sinusoid | 09 | 250 / 120 | 2 classes / continuous |
| DistilBERT + DistilGPT2 (HF Hub) | 10 | pre-trained | sentiment / generation / 2-class domain |
| Synthetic 1D function | 11 | 100 | continuous y |

## Other common questions

- **Why StandardScaler for most algorithms and MinMaxScaler for RNN/SOM?** Standardization centers data for distance/gradient methods; MinMax [0,1] matches sigmoid/tanh ranges and SOM weight range.
- **Why fixed `random_state=42` everywhere?** Reproducibility - same numbers every run.
- **Train/test leakage - how avoided?** Scalers are fit on training data only; time series split is chronological; stratified splits keep class ratios.
- **Which metric would you pick and why?** Classification with imbalance → F1/ROC-AUC over accuracy; regression → R² for interpretability + RMSE for real error size; clustering without labels → silhouette/DB.
- **How was overfitting checked?** Separate test sets in every supervised notebook (e.g. MLP 98.89% test, RFR OOB 0.7513 vs test 0.7415).
