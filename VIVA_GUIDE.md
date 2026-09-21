# Viva / Presentation Preparation Guide

**How to use this guide (2-hour plan):**
1. Read the **Index** (2 min) to see the whole map.
2. Read each notebook section in order: **Quick facts → Step-by-step → Key concepts → Results → Likely questions** (≈ 8-10 min each, ~100 min total).
3. Skim the **Metrics cheat sheet** and **Validation/CV** appendices (10 min) - these answer cross-cutting questions.
4. Finish with the **Rapid-fire 30-second answers** (5 min) the night before / morning of the viva.

Everything below is based on the actual executed notebooks in this repository; every number is a real output.

---

# Index

| # | Notebook | Algorithms | Dataset | One-line result |
|---|---|---|---|---|
| [01](#01--clustering) | Clustering | K-Means, Modified K-Means, Hierarchical, Fuzzy C-Means | Mall Customers | K=5 clusters, silhouette 0.55 |
| [02](#02--density-based-learning) | Density-based | DBSCAN, HDBSCAN | Synthetic moons/circles | 3 clusters, ARI ≈ 0.46 |
| [03](#03--semi-supervised-learning) | Semi-supervised | Self-Training (SVC) | Breast Cancer (15% labels) | 0.9298 → 0.9415 accuracy |
| [04](#04--ensemble-learning) | Ensembles | RFR, RFC, XGBoost, AdaBoost, CatBoost | California Housing + Heart Disease | XGB/CatBoost R² ≈ 0.80 |
| [05](#05--multilayer-perceptron-mlp) | Neural net | MLP (sklearn + PyTorch) | Digits (8x8) | 0.9889 test accuracy |
| [06](#06--recurrent-neural-network-rnn) | Sequence | Vanilla RNN vs LSTM | Airline Passengers | LSTM RMSE 42.2 vs RNN 70.8 |
| [07](#07--self-organizing-map-som) | Topological | Kohonen SOM | Wine (178x13) | QE 0.1965, TE 0.0393 |
| [08](#08--hidden-markov-model-hmm) | Probabilistic | Gaussian HMM + Viterbi | Market Regimes | log-lik 3018.45, EM converged |
| [09](#09--support-vector-machines-svm) | Margin-based | SVC (3 kernels), SVR | Moons, Breast Cancer, 1D sine | SVC 0.9790, SVR R² 0.9864 |
| [10](#10--large-language-models-llm) | Transformer | DistilBERT/DistilGPT2 tokenize, infer, generate, fine-tune | SST-2 + custom domain | correct domain classification |
| [11](#11--generalized-regression-neural-network-grnn) | Kernel regression | GRNN (from scratch) | Noisy 1D function | σ=0.069, test R² 0.9473 |
| [A](#appendix-a--metrics-cheat-sheet) | **Appendix A** | Metrics cheat sheet | - | what/good values |
| [B](#appendix-b--validation--cross-validation-used) | **Appendix B** | Validation & CV used | - | where and why |
| [C](#appendix-c--datasets-used) | **Appendix C** | Datasets used | - | sizes and targets |
| [D](#appendix-d--cross-cutting-questions) | **Appendix D** | Common cross-cutting questions | - | scaling, seeds, leakage |
| [E](#appendix-e--rapid-fire-30-second-answers) | **Appendix E** | Rapid-fire 30-second answers | - | final revision |

---

# 01 – Clustering

### Quick facts
| | |
|---|---|
| **Algorithms** | K-Means (Lloyd), Modified K-Means (K-Means++ init + Bisecting), Hierarchical (Agglomerative/Ward), Fuzzy C-Means (from scratch) |
| **Dataset** | Mall Customers (`data/mall_customers.csv`), 200 rows |
| **Features used** | Annual Income (k$) + Spending Score (1-100) - 2 features so clusters can be plotted |
| **Key parameters** | K=5, K-Means++ init, n_init=20; Ward linkage; FCM m=2.0, tol=1e-5, 150 max iterations |
| **Result** | Silhouette: K-Means 0.5547, Hierarchical 0.5538, Bisecting 0.4807; Davies-Bouldin ≈ 0.57 |

### Step-by-step (what the code does)
1. **Import** clustering models, `StandardScaler`, three clustering metrics, scipy `linkage`/`dendrogram`.
2. **Load** the CSV and print shape/head - exploratory check.
3. **Preprocess:** keep the two features → `StandardScaler` (mean 0, std 1) so both dimensions contribute equally to Euclidean distance.
4. **Choose K:** loop K = 2…10, fit K-Means, record inertia (WCSS) and silhouette → plot elbow + silhouette curves → elbow/peak at **K = 5**.
5. **Final K-Means:** fit with K=5, `init='k-means++'`, `n_init=20`; print silhouette and Davies-Bouldin.
6. **Modified K-Means:** K-Means++ is already used in the standard model; additionally fit `BisectingKMeans` which repeatedly splits the largest-inertia cluster with 2-means until 5 clusters exist; print metrics.
7. **Hierarchical:** build the Ward linkage matrix → plot the dendrogram (horizontal cut ⇒ 5 clusters) → fit `AgglomerativeClustering(n_clusters=5, linkage='ward')`; print metrics.
8. **FCM:** define the `FuzzyCMeans` class (random membership matrix U → alternate centroid update v_k = Σu_ik^m x_i / Σu_ik^m and membership update u_ik ∝ 1/d_ik^(2/(m-1)) until convergence) → fit → hard labels = argmax membership.
9. **Visual summary:** one 2x2 figure showing all four partitions.
10. **Evaluate:** table with Silhouette / Davies-Bouldin / Calinski-Harabasz for all four.

### Key concepts
- **K-Means objective:** minimize within-cluster sum of squares (inertia).
- **K-Means++:** spreads initial centroids (probability ∝ squared distance to nearest centroid) → more stable, faster convergence.
- **Bisecting K-Means:** divisive hierarchical variant; splits the worst cluster repeatedly - good, balanced cluster sizes.
- **Ward linkage:** merge the pair of clusters whose merge increases within-cluster variance the least.
- **FCM:** soft assignment; each point has membership degrees over all clusters that sum to 1; fuzzifier m controls fuzziness (m→1 ≈ hard K-Means).
- **Elbow method:** inertia always drops with K - pick the "knee" where the drop slows.
- **Silhouette:** (b-a)/max(a,b) per point; a = mean intra-cluster distance, b = mean nearest-cluster distance.

### How to read the results
- Silhouette 0.55 = reasonably well-separated clusters (0.5+ is good on real data).
- Davies-Bouldin 0.57 (< 1 is good) - clusters are compact and separated.
- FCM matches K-Means here (same silhouette) because the clusters are fairly separated; soft memberships matter most at boundaries.
- Bisecting is slightly worse on this data (0.48) but gives more uniform cluster sizes.

### Likely questions
- **Why standardize?** Distance-based algorithms: income (0-137) would dominate spending (1-99) without scaling.
- **How did you pick K=5?** Elbow (inertia) + silhouette peak over K=2..10 both point to 5.
- **Difference K-Means vs Fuzzy C-Means?** Hard vs soft membership; FCM tells you how ambiguous each point is.
- **What exactly is "Modified K-Means"?** K-Means++ initialization and the Bisecting (divisive) variant - both fix weaknesses of plain random-init K-Means.
- **Why did you show the dendrogram?** It visualizes the full merge hierarchy and justifies the horizontal cut that yields K=5.
- **What does Calinski-Harabasz say?** Ratio of between-cluster to within-cluster dispersion; higher is better; K-Means/FCM scored highest (~249 vs ~151 for Bisecting).

### One-line summary
"Four clustering algorithms compared on the same 2-feature scaled customer data; K=5 chosen by elbow+silhouette; K-Means/FCM give the best separation (silhouette 0.55)."

---

# 02 – Density-Based Learning

### Quick facts
| | |
|---|---|
| **Algorithms** | DBSCAN and HDBSCAN |
| **Dataset** | Synthetic: 2 interleaving moons + 2 concentric circles + 1 Gaussian blob + uniform noise = 600 points, 2D |
| **Key parameters** | DBSCAN: eps=0.22, min_samples=6. HDBSCAN: min_cluster_size=15, min_samples=6 |
| **Result** | DBSCAN 3 clusters / 4.7% noise / ARI 0.4637; HDBSCAN 3 clusters / 3.7% noise / ARI 0.4578 |

### Step-by-step (what the code does)
1. **Generate data** with `make_moons`, `make_circles`, `make_blobs` plus uniform random noise → stack into one matrix `X_raw`; build ground-truth labels (noise = -1).
2. **Preprocess:** `StandardScaler` so ε is a comparable distance in both axes.
3. **Choose ε:** for every point compute the distance to its 6th nearest neighbor; sort ascending and plot the **k-distance graph**; the "knee" (~0.22) marks where dense clusters transition to sparse noise.
4. **DBSCAN:** fit with `eps=0.22, min_samples=6`; identify core points via `core_sample_indices_`; count clusters and noise; print ARI vs ground truth; plot core/border/noise points.
5. **HDBSCAN:** fit with `min_cluster_size=15, min_samples=6` (no ε needed); count clusters/noise; print ARI; plot clusters + noise.
6. **Evaluate:** summary table with clusters found, noise %, ARI, silhouette computed only on non-noise points.

### Key concepts
- **ε-neighborhood:** all points within distance ε of a point.
- **Core point:** has ≥ MinPts neighbors inside its ε-neighborhood. **Border:** within ε of a core point. **Noise:** neither (label -1).
- **Density-reachability:** chains of core points connect into one cluster.
- **DBSCAN limitation:** one global ε struggles when clusters have very different densities.
- **HDBSCAN:** builds a hierarchy over all density scales (core distance → mutual reachability → minimum spanning tree → condensed tree), then keeps the most *stable* clusters. Outputs membership probabilities.
- **ARI (Adjusted Rand Index):** agreement with ground truth corrected for chance.

### How to read the results
- ARI ≈ 0.46: moderate agreement - expected because the benchmark is deliberately hard (3 shape families + uniform noise).
- Noise detection: DBSCAN 4.7%, HDBSCAN 3.7% - close to the true 50/600 ≈ 8.3% noise, both conservative.
- Silhouette (non-noise) ≈ 0.43 - meaningful separation for non-convex shapes.
- HDBSCAN finds the same 3 groups without any ε tuning.

### Likely questions
- **Why does K-Means fail on this data?** It assumes spherical clusters and splits the moons/circles with straight boundaries; also it cannot label noise.
- **How is ε selected?** k-distance elbow heuristic with k = MinPts.
- **What does MinPts control?** Minimum density for a region to be considered a cluster; larger = more noise, fewer clusters.
- **DBSCAN vs HDBSCAN?** Fixed radius vs hierarchy over density levels; HDBSCAN removes the ε parameter and handles multi-density data.
- **Why is silhouette computed without noise points?** Noise has no cluster, so it would distort the metric.

### One-line summary
"Density-based methods find arbitrary-shaped clusters that K-Means cannot; ε from the k-distance knee; both algorithms recover the 3 shape families and flag outliers as noise."

---

# 03 – Semi-Supervised Learning

### Quick facts
| | |
|---|---|
| **Algorithm** | Self-Training Classifier with an RBF-kernel SVC base estimator |
| **Dataset** | Breast Cancer Wisconsin (569 samples, 30 features, binary) |
| **Setup** | 70/30 stratified split. Training set 398 samples: only **15% (61) labeled**, 85% (337) masked -1 (unlabeled). Test: 171 samples |
| **Key parameters** | threshold τ=0.80, max_iter=15, SVC(C=1, RBF, probability=True) |
| **Result** | baseline 0.9298 / F1 0.9469 → self-training **0.9415 / 0.9554** → fully supervised ceiling 0.9766 / 0.9813 |

### Step-by-step (what the code does)
1. **Import** self-training wrapper, SVC, metrics; load Breast Cancer.
2. **Split** train/test with `stratify=y` to keep the class ratio (this matters - the test set must mirror the real distribution).
3. **Scale** features: `StandardScaler` fit on training data only (no test leakage).
4. **Hide labels:** generate a random mask; 85% of training labels set to `-1` (scikit-learn's "unlabeled" convention). Count and print labeled/unlabeled.
5. **Baseline:** SVC trained only on the 61 labeled samples → test accuracy/F1.
6. **Self-training:** wrap the SVC in `SelfTrainingClassifier(threshold=0.80, max_iter=15)` and fit on **all 398 training samples** (labeled + unlabeled). The wrapper iteratively: predicts probabilities → adds predictions with confidence ≥ 0.80 as pseudo-labels → refits. It printed 4 iterations, then stopped (`no_change`); 321 pseudo-labels were added.
7. **Ceiling:** train the same SVC on all 398 true labels for an upper reference.
8. **Summary table** comparing the three.

### Key concepts
- **Semi-supervised learning:** use a small labeled set + a large unlabeled set.
- **Pseudo-labeling:** treat confident predictions as if they were true labels.
- **Confidence threshold τ:** controls the purity/quantity trade-off; too low → wrong labels pollute training ("confirmation bias"); too high → too few labels added.
- **Self-training loop:** train → predict → select confident → augment → repeat.
- **Why `probability=True`?** The wrapper needs `predict_proba`; SVC normally outputs only decision values, so Platt scaling is enabled.

### How to read the results
- Self-training adds **+1.2 accuracy points** over the 15%-labeled baseline using *no extra human labels*.
- It stays below the 100%-labeled ceiling (0.977) - expected, pseudo-labels are noisier than true labels.
- 382/398 samples ended up labeled → almost the whole unlabeled pool was used.
- Convergence behavior: labels added per iteration dropped 292 → 22 → 5 → 2 then stopped - a healthy sign (no runaway confirmation bias).

### Likely questions
- **Why did accuracy improve?** Confident pseudo-labels expose the classifier to the structure of the unlabeled data.
- **What if τ was 0.60 or 0.99?** Lower τ risks wrong pseudo-labels; higher τ accepts almost nothing and behaves like the baseline.
- **How do you detect confirmation bias?** Watch iteration counts and validation performance; here the additions decayed quickly.
- **Why is the test set untouched during self-training?** It simulates real deployment - the model never sees test labels.

### One-line summary
"With only 15% of labels, self-training on the unlabeled pool recovered about 1.2 accuracy points without any new human annotations."

---

# 04 – Ensemble Learning

### Quick facts
| | |
|---|---|
| **Algorithms** | Random Forest Regression (RFR), XGBoost, AdaBoost, CatBoost (regression) + Random Forest Classification (RFC) |
| **Datasets** | California Housing (5,000-row sample, 8 features) for regression; Heart Disease (`data/heart_disease.csv`, 303 rows) for classification |
| **Key parameters** | RFR 150 trees/depth 12; XGB 150 trees, lr 0.08, depth 6, subsample 0.8; AdaBoost 100, lr 0.1; CatBoost 200 iters, depth 6, L2 3; RFC 120 trees/depth 8 |
| **Result** | XGBoost R² 0.8025 / CatBoost 0.8029 / RFR 0.7415 / AdaBoost 0.5888. RFC: acc 0.7763, F1 0.8046, ROC-AUC 0.8704 |

### Step-by-step (what the code does)
1. **Load & sample** California Housing (5,000 rows for speed), 80/20 train/test split.
2. **RFR:** bag 150 trees on bootstrap samples; each split considers a random feature subset; average predictions; report OOB R², test R², RMSE, MAE.
3. **XGBoost:** gradient-boosted trees; each round fits a tree to the 1st/2nd-order gradients (Taylor expansion) of the squared loss with L1/L2 regularization and subsampling; report metrics.
4. **AdaBoost:** sequentially re-weight samples - mispredicted points get bigger weights; final prediction = weighted median; report metrics.
5. **CatBoost:** ordered boosting (avoids target leakage) with oblivious (symmetric) trees; report metrics.
6. **Comparison table** of the four regressors (highlight best R², lowest errors).
7. **Classification (RFC):** load heart CSV, one-hot encode categoricals, stratified 75/25 split, fit 120-tree forest; report accuracy/F1/ROC-AUC.

### Key concepts
- **Bagging (Random Forest):** parallel trees on bootstrap samples + random feature subsets → reduces **variance**; the average is more stable than a single tree.
- **OOB score:** each tree validates on the ~37% of samples it never saw (bootstrap out-of-bag) → free internal validation.
- **Boosting:** sequential models, each correcting the previous ensemble's errors → reduces **bias**.
- **XGBoost:** 2nd-order (Newton) boosting with explicit regularization Ω(f) = γT + ½λΣw².
- **AdaBoost:** exponential loss, sample re-weighting; sensitive to noisy/outlier points.
- **CatBoost:** ordered boosting + symmetric trees + native categorical handling (target statistics on permutations).
- **Why RFC for classification?** Majority vote of trees; `predict_proba` gives probability of the positive class for ROC-AUC.

### How to read the results
- XGBoost and CatBoost essentially tie at the top (R² ≈ 0.80, RMSE ≈ 0.516) - both are gradient boosting with regularization.
- RFR 0.7415 is respectable for out-of-the-box settings; OOB 0.7513 ≈ test 0.7415 → no overfit-signs.
- AdaBoost 0.5888 is clearly weaker - exponential loss is sensitive to outliers in housing prices.
- RMSE/MAE are in $100k units (target MedHouseVal): RMSE 0.516 ≈ $51.6k average error.
- RFC on heart: AUC 0.87 = good discriminative power for a small dataset; accuracy 0.776 with class-imbalance-aware F1 0.80.

### Likely questions
- **Bagging vs boosting?** Parallel variance reduction vs sequential bias reduction.
- **Why does XGBoost beat AdaBoost here?** Regularized 2nd-order optimization vs sensitive exponential loss on noisy data.
- **What does CatBoost add?** Better categorical handling and reduced target leakage - but here all data is numeric, so it ties XGBoost.
- **Is RFR overfitting?** No: OOB ≈ test R², and more trees always reduce variance without overfitting.
- **What is ROC-AUC?** Probability that a random positive is ranked above a random negative; 0.5 = random, > 0.8 = good.

### One-line summary
"Five ensembles benchmarked: gradient-boosted XGBoost/CatBoost lead regression (R² 0.80), Random Forest gives a strong baseline plus OOB validation, and RFC handles the heart-disease classification at AUC 0.87."

---

# 05 – Multilayer Perceptron (MLP)

### Quick facts
| | |
|---|---|
| **Algorithms** | MLP with scikit-learn (`MLPClassifier`) and a custom PyTorch deep MLP |
| **Dataset** | Handwritten digits, 1,797 samples, 8x8 = 64 features, 10 classes |
| **Split** | 70% train / 15% validation / 15% test (stratified) |
| **Key parameters** | sklearn: (128,64) ReLU, Adam, alpha=0.001, early stopping. PyTorch: 64→128→64→10, BatchNorm + Dropout(0.25/0.20), CrossEntropyLoss, Adam lr 0.003 + weight decay 1e-4, batch 32, 50 epochs, ReduceLROnPlateau |
| **Result** | sklearn 0.9630 test accuracy (15 iterations); PyTorch **0.9889** test accuracy |

### Step-by-step (what the code does)
1. **Load digits**, print shape; split train/val/test with stratification; `StandardScaler` fit on train only.
2. **sklearn MLP:** build `MLPClassifier(hidden_layer_sizes=(128,64), activation='relu', solver='adam', alpha=0.001, batch_size=64, early_stopping=True, validation_fraction=0.15)`; fit; test accuracy; plot training loss curve (and validation error).
3. **PyTorch MLP:** wrap arrays in Tensors/DataLoaders (shuffle train only, batch 32); define `DeepMLP` = Linear→BatchNorm→ReLU→Dropout ×2 then Linear(10).
4. **Training loop (50 epochs):** for each batch - forward pass, `CrossEntropyLoss`, `backward()`, Adam step; after each epoch - validation pass under `torch.no_grad()`; scheduler reduces LR when validation loss plateaus; track loss/accuracy history.
5. **Final evaluation:** predict the test set and print a per-class classification report.

### Key concepts
- **Forward pass:** a⁽ˡ⁾ = σ(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾); output layer = 10 logits.
- **Cross-entropy loss:** −Σ y·ln(ŷ) after softmax; standard for multi-class.
- **Backpropagation:** chain rule to get gradients; optimizer updates weights.
- **ReLU:** cheap non-linearity avoiding sigmoid saturation.
- **BatchNorm:** normalizes layer inputs → stable/faster training.
- **Dropout:** randomly zeroes neurons → regularization against overfitting.
- **Early stopping:** stop when validation stops improving (sklearn model used only 15 iterations).
- **Why a validation set?** Tune/stop without touching the test set.

### How to read the results
- sklearn: 96.3% after only 15 iterations (early stopping kicked in).
- PyTorch: **98.89%** - BatchNorm + Dropout + more epochs + LR scheduling helped.
- Loss curve: train loss 1.20 → 0.01; validation loss tracked it without exploding → no severe overfitting.
- Per-class report: almost all digits ≈ 1.00 precision/recall; a few confusions between visually similar digits (e.g., 8/1, 9/4) are typical.

### Likely questions
- **Why is the PyTorch model better?** More regularization and a longer, scheduled training (LR reduction on plateau); the sklearn model stops at iteration 15.
- **What loss and why?** Categorical cross-entropy - it directly penalizes wrong class probabilities.
- **What is Dropout doing at test time?** Nothing - it's disabled (model.eval()); BatchNorm switches to running statistics.
- **What would you change to improve further?** Data augmentation, deeper/wider layers, hyperparameter search, CNN for image data.

### One-line summary
"Two MLP implementations on 8x8 digits; the custom PyTorch network with BatchNorm + Dropout reaches 98.9% test accuracy vs 96.3% for the early-stopped scikit-learn baseline."

---

# 06 – Recurrent Neural Network (RNN)

### Quick facts
| | |
|---|---|
| **Algorithms** | Vanilla RNN (Elman) and LSTM (both PyTorch) |
| **Dataset** | Monthly Airline Passengers (`data/airline_passengers.csv`), 144 months |
| **Preprocessing** | MinMaxScaler to [0,1]; sliding windows of 12 months → next month |
| **Split** | Chronological 80/20: 105 train / 27 test sequences (no shuffling) |
| **Key parameters** | 2 layers, hidden 64, Linear(64,1), MSE loss, Adam lr 0.005, 120 epochs |
| **Result** | Vanilla RNN RMSE 70.83 / MAE 63.16 · **LSTM RMSE 42.17 / MAE 34.88** (passengers) |

### Step-by-step (what the code does)
1. **Load** the CSV; keep the `Passengers` column as float32.
2. **Normalize** to [0,1] with MinMaxScaler - RNNs train poorly on large raw values.
3. **Build sequences:** for each t, input = months t-12…t-1, target = month t (`create_sequences`).
4. **Split chronologically** (first 80% train, last 20% test) and wrap in a DataLoader (shuffle train only, batch 16).
5. **Define models:** `VanillaRNN` = `nn.RNN(input=1, hidden=64, layers=2)` + Linear; `LSTMModel` = `nn.LSTM(...)` + Linear; both take the **last time step's** hidden state.
6. **Train** each model with the same function: MSE loss, Adam, 120 epochs, BPTT (loss.backward through time), track epoch loss.
7. **Evaluate:** predict the test windows, invert scaling to real passenger counts, compute RMSE/MAE, print the comparison table.
8. **Plot:** full history + test ground truth + both forecasts, with a vertical train/test cutoff line.

### Key concepts
- **Recurrent cell:** h_t = tanh(W_ih x_t + b + W_hh h_{t-1} + b_hh) - carries a memory of past inputs.
- **BPTT:** backpropagation through time - gradients flow through every time step.
- **Vanishing gradient:** repeated multiplications by W_hh shrink gradients exponentially → vanilla RNN forgets long-range patterns.
- **LSTM gates:** forget f_t (drop old memory), input i_t + candidate c̃_t (write new memory), output o_t; cell state c_t = f_t⊙c_{t-1} + i_t⊙c̃_t is an additive "conveyor belt" that preserves gradients.
- **Why 12-month lookback?** Captures the annual seasonality cycle.
- **Why no shuffle?** Shuffling sequences would leak future info into training.

### How to read the results
- LSTM RMSE 42.2 vs RNN 70.8 passengers - the gap directly shows the benefit of gating on 12-step dependencies.
- MAE ~35 passengers for a series averaging ~280/month ≈ 12% error; respects both trend and seasonality visually.
- Loss curves both decrease; LSTM converges lower.

### Likely questions
- **Why use LSTM over vanilla RNN?** Long-term memory through additive cell state; avoids vanishing gradients.
- **What is the input shape?** (batch, 12, 1) - 12 time steps, one feature.
- **Why MinMax and not StandardScaler?** Keeps inputs in the same [0,1] range used by the network's tanh/sigmoid activations; also makes inverting predictions trivial.
- **How would you forecast further ahead?** Feed predictions back as inputs (autoregressive rollout); errors accumulate - a known limitation.

### One-line summary
"Same training recipe for RNN and LSTM on 12-month windows: LSTM cuts RMSE from 70.8 to 42.2 passengers, demonstrating gating's advantage on seasonal long-range dependencies."

---

# 07 – Self-Organizing Map (SOM)

### Quick facts
| | |
|---|---|
| **Algorithm** | Kohonen Self-Organizing Map (MiniSom) |
| **Dataset** | UCI Wine, 178 samples, 13 chemical features; 3 cultivars used only for visualization |
| **Key parameters** | 12x12 grid (144 neurons), Gaussian neighborhood, sigma 1.5, learning rate 0.5, PCA init, 5,000 iterations |
| **Result** | Quantization error 0.4447 → **0.1965**; topographic error **0.0393** |

### Step-by-step (what the code does)
1. **Load wine**, print samples/features/classes; **MinMax scale** to [0,1] (SOM weights live in input range).
2. **Create SOM:** 12x12 grid, each neuron has a 13-dim weight vector; Gaussian neighborhood function; sigma/learning rate decay during training.
3. **Initialize** weights with `pca_weights_init` - neurons start spread along the data's principal directions → faster, topologically faithful convergence.
4. **Train:** `som.train_random(X, 5000)` - repeatedly pick a random sample, find its BMU, pull the BMU and neighbors toward it.
5. **Measure:** quantization error (avg distance sample→BMU) and topographic error (fraction of samples whose two best neurons are not adjacent).
6. **Visualize U-Matrix:** average distance between each neuron and its neighbors (dark valleys = clusters, bright ridges = boundaries); overlay each sample on its BMU, marker shape/color = true cultivar.

### Key concepts
- **Competitive learning:** only the winning neuron (BMU) and its neighborhood update.
- **BMU:** argmin_j‖x − w_j‖.
- **Adaptation:** w_j ← w_j + α(t)·h_cj(t)·(x − w_j), with Gaussian kernel h and decaying α, σ.
- **U-Matrix:** unified distance matrix - visualizes cluster boundaries.
- **Quantization error:** how well neurons represent the data (lower = better).
- **Topographic error:** how well the 2D grid preserves neighborhood structure (0 = perfect, <0.05 good).
- **Why PCA initialization?** Random init can create topological folds; PCA starts the map unfolded.

### How to read the results
- QE halved from 0.44 → 0.196: the map learned a much better representation.
- TE 0.039: only ~4% of samples land on non-adjacent top-2 neurons → topology well preserved.
- On the U-matrix, the three cultivars occupy distinct regions even though labels were never used in training - evidence that SOM discovered structure.

### Likely questions
- **Is SOM supervised?** No - labels were only used to color the map for interpretation.
- **Why 12x12?** ~144 neurons for 178 samples - enough resolution without over-fragmenting.
- **What is the difference from K-Means?** K-Means finds centroids only; SOM additionally arranges them on a topology-preserving grid.
- **What do the two errors trade off?** More neurons → lower quantization error but potentially higher topographic error (folds).

### One-line summary
"An unsupervised Kohonen map (12x12) organizes the wine chemistry: quantization error halved to 0.196, topographic error 0.039, and the three cultivars separate on the U-matrix without labels."

---

# 08 – Hidden Markov Model (HMM)

### Quick facts
| | |
|---|---|
| **Algorithm** | Gaussian HMM trained with Baum-Welch (EM), decoded with Viterbi (hmmlearn) |
| **Dataset** | Market returns & regimes (`data/market_regimes.csv`), 1,000 trading days |
| **Observations** | Daily returns (1D) |
| **Key parameters** | 3 states, full covariance, 200 EM iterations, random_state=42 |
| **Result** | EM converged (True); model log-likelihood **3018.45** |

### Step-by-step (what the code does)
1. **Load** the CSV; print head; take `Daily_Return` as the observation sequence and `Close` for plotting.
2. **Model:** define `GaussianHMM(n_components=3, covariance_type="full")` → 3 hidden regimes, each emitting returns from its own Gaussian.
3. **Train (Baum-Welch/EM):** `fit(returns)` - E-step computes state posteriors with forward-backward; M-step re-estimates start probabilities, transition matrix A, and per-state (μ, σ).
4. **Decode (Viterbi):** `predict(returns)` returns the most likely state sequence.
5. **Stabilize labels:** sort states by volatility (σ) so Regime 0/1/2 always mean Bull/Sideways/Bear; reorder the transition matrix to match.
6. **Visualize:** transition-matrix heatmap and a price chart where each daily segment is colored by its decoded regime, plus the state sequence strip.

### Key concepts
- **Hidden vs observed:** hidden = market regime; observed = daily returns.
- **Markov property:** next state depends only on the current state.
- **Transition matrix A:** A_ij = P(state j at t+1 | state i at t); large diagonal = regime persistence.
- **Emission:** each state has a Gaussian distribution over returns, P(x_t | z_t = s_k) = N(μ_k, σ_k²).
- **Forward-backward:** computes P(state = k at time t | all observations) - used in EM.
- **Baum-Welch:** EM for HMMs; iterates until the log-likelihood converges.
- **Viterbi:** dynamic programming for the single most likely state path (δ_t(j) = max_i[δ_{t-1}(i)A_ij]·b_j(x_t)).

### How to read the results
- Log-likelihood 3018.45: higher (less negative) = better fit; meaningful for comparing HMM configurations.
- EM converged before the 200-iteration limit - stable solution.
- Sorting by volatility makes the regimes interpretable: low-vol (bull), medium (sideways), high-vol (bear/crisis).
- The colored price chart shows the model switching to the high-volatility state during drawdowns - volatility clustering.

### Likely questions
- **How do you know 3 states is right?** Domain choice (bull/sideways/bear); could be compared by log-likelihood or BIC across 2-5 states.
- **What does the transition matrix tell you?** Regime persistence (diagonal) and switching probabilities (off-diagonal).
- **Supervised or unsupervised?** Unsupervised - states are latent, learned only from returns.
- **Why full covariance?** Only 1 feature, so full is effectively per-state variance; kept for generality.

### One-line summary
"A 3-state Gaussian HMM learns latent market regimes via Baum-Welch from daily returns, Viterbi decodes the regime path, and the sorted states map cleanly to bull/sideways/bear behavior."

---

# 09 – Support Vector Machines (SVM)

### Quick facts
| | |
|---|---|
| **Algorithms** | SVC (linear, poly, RBF kernels) and SVR (RBF, ε-insensitive tube) |
| **Datasets** | Two-moons synthetic (250 points, 2D); Breast Cancer (569x30); 1D sinusoid + trend + noise (120 points) |
| **Key parameters** | SVC C=1.0, gamma='scale', degree=3; SVR C=20, epsilon=0.18, gamma=0.5 |
| **Result** | SVC (RBF) test accuracy **0.9790** (96/426 SVs); SVR R² **0.9864**, RMSE 0.1157 (18/120 SVs) |

### Step-by-step (what the code does)
1. **Two-moons demo:** generate data, scale; train an SVC for each kernel (linear/poly/rbf); draw the decision regions, boundary (Z=0), margins (Z=±1) and highlight support vectors in a 3-panel figure.
2. **Real-data SVC:** load Breast Cancer, stratified 75/25 split, scale; fit RBF SVC; print test accuracy and how many training points ended up as support vectors.
3. **SVR setup:** generate a 1D non-linear target (sin + trend + noise); fit `SVR(kernel='rbf', C=20, epsilon=0.18, gamma=0.5)`.
4. **SVR visualization:** dense grid prediction line, ±ε tube boundaries and shaded tube, highlight support vectors (points on/outside the tube).
5. **SVR metrics:** R², RMSE, and the support-vector fraction.

### Key concepts
- **Maximum margin:** SVC finds the hyperplane that maximizes the margin between classes.
- **Soft margin & C:** slack variables ξ allow violations; C is the penalty. Large C = narrow margin, may overfit; small C = wider margin, more violations.
- **Support vectors:** training points with α_i > 0 - on/inside the margin; they alone define the boundary.
- **Kernel trick:** K(x_i, x_j) computes inner products in a high-dimensional space without mapping explicitly.
- **γ in RBF:** width of the Gaussian; large γ = very local/wiggly boundary, small γ = smooth/linear-like.
- **ε-insensitive tube (SVR):** residuals within ±ε cost nothing; only outside points become support vectors.
- **Dual formulation:** solution depends only on inner products K(x_i, x_j) - enables kernels.

### How to read the results
- Moons figure: linear kernel underfits (straight boundary); poly and RBF separate the moons with curved boundaries; gold circles show how few points define each boundary.
- Breast Cancer 0.9790 accuracy = strong; 96/426 SVs (22.5%) means most training points are ignorable - a sparse solution.
- SVR R² 0.9864 (explains 98.6% of variance); 18/120 SVs (15%) - the curve is defined by a small subset.
- ε tube (0.18) visually contains most points; only points outside drive the fit.

### Likely questions
- **Why does the RBF kernel work on moons?** It maps 2D data into an infinite-dimensional feature space where the classes become linearly separable.
- **What happens if C → ∞?** No margin violations allowed → hard margin, overfitting and sensitivity to outliers.
- **What if γ is too large?** Each point influences only a tiny region → wiggly boundary and overfitting.
- **Why is the SVM solution sparse?** Only support vectors have non-zero Lagrange multipliers.
- **Difference SVC vs SVR?** Classification finds a separating hyperplane; SVR fits a function with an ε-tube tolerance.

### One-line summary
"SVC with three kernels shows the kernel trick on non-linear moons, reaches 0.979 on Breast Cancer with only 22% support vectors, and SVR fits a noisy sinusoid (R² 0.986) with an explicit ε-tube."

---

# 10 – Large Language Models (LLM)

### Quick facts
| | |
|---|---|
| **Models** | DistilBERT (SST-2 fine-tuned) for tokenization + sentiment; DistilGPT2 for generation; PyTorch for mini fine-tuning |
| **Libraries** | Hugging Face `transformers`, `torch` |
| **Tasks** | (1) tokenization, (2) zero-shot sentiment inference, (3) greedy vs top-p generation, (4) 2-class domain fine-tune |
| **Key parameters** | Generation: max_new_tokens=40, T=0.7, p=0.9. Fine-tune: 8 samples, 3 epochs, lr 1e-4, AdamW |
| **Result** | Both held-out domain queries classified correctly (Billing 88.3%, Technical 84.0%) |

### Step-by-step (what the code does)
1. **Load a tokenizer** (DistilBERT) and tokenize a sample sentence with padding/truncation to length 20; print the subword tokens, input IDs and attention mask (1 = real token, 0 = padding).
2. **Sentiment pipeline:** `pipeline("sentiment-analysis")` loads the pretrained model; classify 4 sentences; display label + confidence in a table.
3. **Text generation:** load DistilGPT2; generate from the prompt "Artificial Intelligence will fundamentally transform" using:
   - **greedy search** (`do_sample=False`) → deterministic, most likely tokens,
   - **nucleus sampling** (`do_sample=True, temperature=0.7, top_p=0.9`) → samples from the smallest token set whose cumulative probability ≥ 0.9.
4. **Fine-tuning:** define 8 domain queries labeled Technical (0) / Billing (1); tokenize; wrap in a PyTorch Dataset/DataLoader; load DistilBERT with a fresh 2-class head (`ignore_mismatched_sizes=True`); train 3 epochs with AdamW (lr 1e-4) - forward pass computes the loss internally, `loss.backward()`, optimizer step; print loss per epoch.
5. **Evaluate** on two unseen queries: tokenize, forward pass, softmax logits, argmax → predicted class + confidence.

### Key concepts
- **Subword tokenization (WordPiece/BPE):** splits text into vocabulary pieces; can represent any word; special tokens [CLS]/[SEP].
- **Attention mask:** zeros out padding so it doesn't affect attention.
- **Transformer self-attention:** Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)V.
- **Autoregressive generation:** predict one token at a time, append, repeat.
- **Temperature:** scales logits before softmax; <1 sharpens, >1 flattens.
- **Top-p (nucleus):** restricts sampling to the smallest token set with cumulative probability ≥ p.
- **Transfer learning/fine-tuning:** reuse pretrained weights, replace the head, train briefly on domain data.

### How to read the results
- Tokenizer: 13 real tokens + 7 padding; [CLS]/[SEP] frame the sentence; IDs map to the vocabulary.
- Sentiment: confidence scores in the 90-99% range on clear positive/negative sentences.
- Generation: greedy gives the "safe" continuation; nucleus sampling gives a different, more varied sentence - same core meaning.
- Fine-tune losses: 2.663 → 0.041 → 0.069 - the model rapidly fits the 8 examples (this is a demo of the mechanics, not a production model).
- Held-out queries both correct with 84-88% confidence → limited but successful transfer.

### Likely questions
- **Why is fine-tuning needed if the model is pretrained?** Pretraining is generic; a new task/domain needs a task head and adaptation.
- **Why `ignore_mismatched_sizes=True`?** The original SST-2 head had 2 classes but different mapping/weights; we replace it with a fresh head.
- **What does temperature do?** Controls randomness of sampling (lower = more deterministic).
- **Greedy vs top-p?** Greedy is deterministic but can loop; top-p balances coherence and diversity.
- **Any risks?** Hallucinations; tiny fine-tune sets can overfit - here it is a mechanics demo.

### One-line summary
"LLM notebook demonstrates the full pipeline: subword tokenization, transformer inference, controllable generation, and a 3-epoch fine-tune that correctly classifies held-out technical vs billing queries."

---

# 11 – Generalized Regression Neural Network (GRNN)

### Quick facts
| | |
|---|---|
| **Algorithm** | Specht's GRNN = Nadaraya-Watson kernel regression, implemented from scratch (NumPy) |
| **Dataset** | Noisy 1D function y = sin(2x) + cos(0.5x²) + N(0,0.18), 100 points |
| **Key parameters** | Only σ (smoothing spread); selected by 5-fold CV over 0.05…1.0 |
| **Result** | σ = 0.069 (CV RMSE 0.2137); final test R² **0.9473**, RMSE 0.1718; train R² 0.9829 |

### Step-by-step (what the code does)
1. **Define `GRNN(sigma)`** with the scikit-learn API:
   - `fit` stores the training data (one-pass "training" - no gradients),
   - `predict` computes squared distances from each query to all training points (via the ‖a−b‖² expansion), applies the Gaussian kernel exp(−d²/2σ²), then returns ŷ = Σ y_i·k_i / Σ k_i.
2. **Generate data:** noisy 1D function, 100 points; dense grid for plotting.
3. **σ demo:** fit three GRNNs with σ = 0.05 (under-smoothed), 0.35, 1.50 (over-smoothed); plot each fit with the true curve and training R².
4. **σ selection:** 5-fold CV (`KFold(shuffle=True, random_state=42)`) over 50 candidate σ values; for each σ: fit on 4 folds, RMSE on the held-out fold, average → pick argmin → σ = 0.069.
5. **Final model:** refit with σ = 0.069 on a 75% train split and evaluate on the 25% test split: R² and RMSE.

### Key concepts
- **GRNN architecture:** input → pattern layer (one Gaussian neuron per training point) → summation layer (S = Σy_i·p_i, D = Σp_i) → output S/D.
- **One-pass learning:** no iterative optimization; the training set *is* the model.
- **σ (smoothing spread):** the only hyperparameter. Small σ → memorizes noise (high variance); large σ → over-smooth (high bias).
- **Nadaraya-Watson estimator:** ŷ(x) = Σ K(x,x_i)·y_i / Σ K(x,x_i) - an estimate of E[y|x].
- **Why 5-fold CV?** Reliable selection of the single hyperparameter without touching the test set.

### How to read the results
- σ=0.05: curve wiggles through every point - overfitting the noise.
- σ=1.50: almost flat - underfitting.
- σ=0.069 (from CV): smooth curve that follows the true signal → CV chose a small-but-not-tiny σ for this dense 1D data.
- Test R² 0.9473 = explains ~95% of held-out variance; RMSE 0.1718 vs noise std 0.18 → essentially at the noise floor (cannot do much better).
- Train R² 0.9829 higher than test - normal, small generalization gap.

### Likely questions
- **How is it "neural" if there is no backprop?** Four-layer network structure (input/pattern/summation/output); the pattern layer is radial-basis neurons; learning is memory-based.
- **Why is training instant?** Only stores data; prediction cost is O(N·d) per query.
- **What happens for very large data?** Prediction becomes slow (kernel with every training point) - a known GRNN limitation.
- **Why did CV pick a small σ?** The 100 points are dense in [-3,3]; small σ still has enough neighbors to smooth, keeping the fit detailed.
- **How would you make it better?** Weighted/adaptive σ per dimension, or sparse kernel approximations.

### One-line summary
"GRNN implemented from scratch needs no gradient training: 5-fold CV picks σ=0.069, giving a smooth fit with test R² 0.9473 - essentially at the noise limit of the data."

---

# Appendix A – Metrics cheat sheet

| Metric | Used in | What it measures | Range | How to judge |
|---|---|---|---|---|
| **Silhouette** | 01, 02 | (b−a)/max(a,b): cohesion vs separation | −1…1 | >0.5 good, 0.25-0.5 weak, <0 overlapping |
| **Davies-Bouldin** | 01 | average similarity between each cluster and its most similar one | ≥0 | lower better; <1 good |
| **Calinski-Harabasz** | 01 | ratio of between- to within-cluster dispersion | ≥0 | higher better; only compare models |
| **ARI** | 02 | agreement with ground truth, corrected for chance | −0.5…1 | 0 = random, >0.5 strong, 1 perfect |
| **Accuracy** | 03, 04, 05, 09 | fraction correctly classified | 0…1 | >0.9 strong on these datasets |
| **Precision** | 05 | TP/(TP+FP) - how clean positive predictions are | 0…1 | high when false positives are costly |
| **Recall** | 05 | TP/(TP+FN) - how many positives are found | 0…1 | high when misses are costly |
| **F1** | 03, 04, 05 | harmonic mean of precision and recall | 0…1 | >0.9 excellent; robust to imbalance |
| **ROC-AUC** | 04 | probability a random positive scores above a random negative | 0…1 | 0.5 random, 0.7-0.8 acceptable, >0.8 good, >0.9 excellent |
| **R²** | 04, 11 | fraction of target variance explained | −∞…1 | 1 perfect; >0.7 good for noisy data; can be negative |
| **RMSE** | 04, 06, 09, 11 | √mean squared error (same units as target) | ≥0 | lower better; compare same-target models |
| **MAE** | 04, 06 | mean absolute error | ≥0 | lower better; robust to outliers |
| **MSE loss** | 06 | squared error used as the training objective | ≥0 | decreasing, stable curve |
| **Cross-entropy** | 05, 10 | −Σy·ln(ŷ), classification objective | ≥0 | decreasing, stable curve |
| **Quantization error** | 07 | average distance sample → BMU | ≥0 | lower better; relative improvement matters |
| **Topographic error** | 07 | fraction of samples whose top-2 BMUs are not neighbors | 0…1 | <0.05 good; 0 perfect |
| **Log-likelihood** | 08 | how well the HMM explains the observed sequence | −∞…+∞ | higher (less negative) better; compare models |

**Choosing a metric (common exam question):** imbalanced classification → F1 or ROC-AUC (not accuracy); regression → R² for interpretation + RMSE for real error size; clustering without labels → silhouette / Davies-Bouldin; density clustering → ARI if ground truth exists.

# Appendix B – Validation / cross-validation used

| Notebook | Scheme | Why that scheme |
|---|---|---|
| 01 | Elbow + silhouette over K=2..10 | No labels: choose K by internal cluster quality |
| 02 | k-distance graph (knee) | Heuristic to set ε without labeled data |
| 03 | Stratified holdout 70/30 + 15% labels | Simulates scarce labeling; evaluate once on clean test |
| 04 | Holdout + OOB for RFR | OOB is free internal validation for bagging; holdout for all |
| 05 | Early-stopping validation split (15%) | Stop training when validation stops improving |
| 06 | Chronological 80/20 split | Time series: shuffling would leak future data |
| 07 | None (uses QE/TE) | Unsupervised self-assessment metrics |
| 08 | None (EM log-likelihood + convergence) | Unsupervised; monitor assures convergence |
| 09 | Holdout test set | Simple unbiased estimate for SVC/SVR |
| 10 | Unseen domain queries | Checks transfer beyond the 8 fine-tune samples |
| 11 | **5-fold CV** over σ grid | Tune the only hyperparameter reliably; test set kept clean |

**Note:** only notebooks 11 (KFold, shuffle=True) uses explicit k-fold CV; 05 uses a single validation split; 04 uses OOB. Everything else is a holdout or heuristic.

# Appendix C – Datasets used

| Data | Notebook | Size | Target / use |
|---|---|---|---|
| Mall Customers (`data/mall_customers.csv`) | 01 | 200 x 5 | Cluster income + spending |
| Synthetic moons/circles/blob + noise | 02 | 600 x 2 | Non-convex clustering + noise detection |
| Breast Cancer (sklearn) | 03, 09 | 569 x 30 | Malignant/benign classification |
| California Housing (sklearn, 5k sample) | 04 | 5,000 x 8 | Median house value (regression) |
| Heart Disease (`data/heart_disease.csv`) | 04 | 303 x 13 | Disease 0/1 (classification) |
| Digits (sklearn) | 05 | 1,797 x 64 | Digit 0-9 (10 classes) |
| Airline Passengers (`data/airline_passengers.csv`) | 06 | 144 months | Next-month passenger count |
| Wine (sklearn) | 07 | 178 x 13 | 3 cultivars (visualization only) |
| Market Regimes (`data/market_regimes.csv`) | 08 | 1,000 days | Latent market regime (unsupervised) |
| Synthetic moons / sinusoid | 09 | 250 / 120 | 2-class demo / 1D regression |
| DistilBERT, DistilGPT2 (HF Hub) | 10 | pretrained | Sentiment, generation, domain classification |
| Synthetic 1D function | 11 | 100 | Noisy continuous regression |

# Appendix D – Cross-cutting questions

- **Why scale features?** Distance-based models (K-Means, SVM, KNN), gradient-based models (MLP), and RNNs/SOMs all need comparable input ranges; without scaling one feature dominates.
  - `StandardScaler` (mean 0, std 1) for clustering/SVM/MLP; `MinMaxScaler` [0,1] for SOM and RNN (matches activation/weight ranges).
- **Why `random_state=42` everywhere?** Reproducibility: the same notebook produces the same results on any machine.
- **How is data leakage avoided?**
  - Scalers are fit on the training set only.
  - The RNN uses a chronological split (no shuffling).
  - Stratified splits preserve class ratios.
  - CV is done only on training data (notebook 11).
- **How do you know a model is not overfitting?** Compare train vs test metrics: e.g., RFR OOB 0.7513 vs test 0.7415; GRNN train R² 0.983 vs test 0.947; MLP val acc ≈ test acc. Large gaps would signal overfitting.
- **Why different metrics per notebook?** The metric must match the task: regression → R²/RMSE; classification → accuracy/F1/AUC; clustering → silhouette/DB/ARI; density/SOM/HMM → task-specific errors/likelihood.
- **What is a hyperparameter vs a parameter?** Parameters are learned from data (weights, centroids); hyperparameters are chosen by us (K, ε, σ, C, γ, threshold, layers).
- **Which model performed best overall?** Task-dependent: XGBoost/CatBoost for tabular regression, PyTorch MLP for digits, LSTM for sequences, GRNN for smooth 1D regression, SVC for small high-dimensional classification.

# Appendix E – Rapid-fire 30-second answers

1. **01** – "Four clustering algorithms on scaled income/spending data; K=5 by elbow+silhouette; K-Means/FCM best (silhouette 0.55); Bisecting gives balanced sizes."
2. **02** – "DBSCAN and HDBSCAN on moons/circles/noise; ε=0.22 from the k-distance knee; both find 3 clusters, ARI ≈ 0.46, and label outliers as noise."
3. **03** – "With just 15% labels, self-training pseudo-labeling raised SVC accuracy 0.9298 → 0.9415, closing part of the gap to the 100%-labeled ceiling (0.9766)."
4. **04** – "Four boosters/baggers on housing (XGBoost/CatBoost R² ≈ 0.80) and Random Forest classification on heart disease (AUC 0.87)."
5. **05** – "MLP on 8x8 digits: sklearn 96.3% with early stopping; custom PyTorch net with BatchNorm+Dropout 98.9% test accuracy."
6. **06** – "Vanilla RNN vs LSTM on 12-month windows of airline data: LSTM RMSE 42.2 vs 70.8 passengers - gating beats vanishing gradients."
7. **07** – "12x12 Kohonen SOM on wine: quantization error halved to 0.196, topographic error 0.039; cultivars separate on the U-matrix without labels."
8. **08** – "3-state Gaussian HMM on 1,000 days of returns: Baum-Welch converged (log-lik 3018), Viterbi-decoded bull/sideways/bear regimes."
9. **09** – "SVC kernels on moons, RBF SVC 0.979 on breast cancer with 22% support vectors; SVR fits noisy sine with an ε=0.18 tube (R² 0.986)."
10. **10** – "DistilBERT tokenization + sentiment, DistilGPT2 greedy vs top-p generation, and a 3-epoch fine-tune that correctly classifies held-out technical/billing queries."
11. **11** – "GRNN from scratch: one-pass training, σ=0.069 by 5-fold CV, test R² 0.947 - at the noise floor of the 1D data."
