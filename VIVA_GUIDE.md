# Viva / Presentation Preparation Guide (Deep Edition)

**How to use this guide (within 2 hours):**
1. Skim the **Index** (2 min).
2. Read each notebook section in order (≈ 9-11 min each, ~110 min total). Each section is structured the same way:
   **0** 60-second explanation → **1** Quick facts → **2** How it works → **3** Parameters → **4** Step-by-step + key code → **5** Results → **6** Limitations → **7** Viva Q&A → **8** Common mistakes → **9** One-line summary.
3. Finish with **Appendices A-E** (10 min): metrics, validation, datasets, cross-cutting questions, rapid-fire answers.

All numbers are real outputs of the executed notebooks in this repository. Code excerpts are copied from the notebooks (shortened).

---

# Index

| # | Notebook | Algorithms | Dataset | One-line result |
|---|---|---|---|---|
| [01](#01--clustering) | Clustering | K-Means, Modified K-Means, Hierarchical, Fuzzy C-Means | Mall Customers | K=5, silhouette 0.55 |
| [02](#02--density-based-learning) | Density-based | DBSCAN, HDBSCAN | USGS earthquakes 2023 | 55 clusters, 13.5% noise |
| [03](#03--semi-supervised-learning) | Semi-supervised | Self-Training (SVC) | Breast Cancer (15% labels) | 0.9298 → 0.9415 accuracy |
| [04](#04--ensemble-learning) | Ensembles | RFR, RFC, XGBoost, AdaBoost, CatBoost | California Housing + Heart Disease | XGB/CatBoost R² ≈ 0.80 |
| [05](#05--multilayer-perceptron-mlp) | Neural net | MLP (sklearn + PyTorch) | Digits (8x8) | 0.9889 test accuracy |
| [06](#06--recurrent-neural-network-rnn) | Sequence | Vanilla RNN vs LSTM | Airline Passengers | LSTM RMSE 42.2 vs RNN 70.8 |
| [07](#07--self-organizing-map-som) | Topological | Kohonen SOM | Wine (178x13) | QE 0.1965, TE 0.0393 |
| [08](#08--hidden-markov-model-hmm) | Probabilistic | Gaussian HMM + Viterbi | Market Regimes | log-lik 3018.45, EM converged |
| [09](#09--support-vector-machines-svm) | Margin-based | SVC (3 kernels), SVR | Moons, Breast Cancer, 1D sine | SVC 0.9790, SVR R² 0.9864 |
| [10](#10--large-language-models-llm) | Transformer | DistilBERT/DistilGPT2 | SST-2 + custom domain | correct domain classification |
| [11](#11--generalized-regression-neural-network-grnn) | Kernel regression | GRNN (from scratch) | Noisy 1D function | σ=0.069, test R² 0.9473 |
| [A](#appendix-a--metrics-cheat-sheet) | Appendix A | Metrics cheat sheet | - | definitions + good values |
| [B](#appendix-b--validation--cross-validation-used) | Appendix B | Validation & CV | - | where and why |
| [C](#appendix-c--datasets-used) | Appendix C | Datasets | - | sizes, targets |
| [D](#appendix-d--cross-cutting-questions) | Appendix D | Cross-cutting Q&A | - | scaling, leakage, overfitting |
| [E](#appendix-e--rapid-fire-30-second-answers) | Appendix E | Rapid-fire answers | - | last-minute revision |

---

# 01 – Clustering

## 0. In one paragraph
Four clustering algorithms - K-Means (Lloyd's), Modified K-Means (K-Means++ init + Bisecting), Hierarchical (Ward) and Fuzzy C-Means (implemented from scratch) - partition 200 Mall Customers into K=5 groups using Annual Income and Spending Score. K is chosen by the elbow + silhouette curves; all four partitions are then compared with three internal metrics.

## 1. Quick facts
| | |
|---|---|
| **Algorithms** | K-Means, Modified K-Means (K-Means++ / Bisecting), Hierarchical (Agglomerative/Ward), Fuzzy C-Means (custom NumPy class) |
| **Dataset** | `data/mall_customers.csv`, 200 rows; features: Annual Income (k$), Spending Score (1-100) |
| **Preprocessing** | StandardScaler (mean 0, std 1) |
| **Key parameters** | K=5; `init='k-means++'`, `n_init=20`; Ward linkage; FCM m=2.0, tol=1e-5, 150 iterations |
| **Results** | Silhouette: K-Means 0.5547, Hierarchical 0.5538, Bisecting 0.4807, FCM 0.5547; DB index ≈ 0.57; Calinski-Harabasz ≈ 249 (K-Means) |

## 2. How the algorithm works
**K-Means (Lloyd's):** minimize the within-cluster sum of squares
J = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖².
1) Initialize K centroids; 2) assign each point to the nearest centroid; 3) set each centroid to the mean of its assigned points; 4) repeat until assignments stop changing. Each step never increases J, so it converges - but only to a **local** minimum, which is why initialization matters.

**Modified variants:**
- *K-Means++* picks the first centroid at random, then each next centroid with probability proportional to D(x)² = squared distance to the nearest chosen centroid. This spreads centroids out and avoids bad local minima.
- *Bisecting K-Means* is divisive: start with one cluster, repeatedly split the cluster with the largest inertia using 2-means until K clusters exist. Gives more balanced sizes.

**Hierarchical (Ward):** start with N single-point clusters; repeatedly merge the pair (A,B) that increases within-cluster variance the least: ΔW(A,B) = (n_A n_B)/(n_A+n_B)·‖μ_A−μ_B‖². The dendrogram shows all merges; cutting it at a height gives the partition.

**Fuzzy C-Means:** soft version of K-Means. Each point x_i has membership u_ik ∈ [0,1] over all clusters with Σₖ u_ik = 1. Alternates:
- centroid update: v_k = Σᵢ u_ik^m x_i / Σᵢ u_ik^m
- membership update: u_ik = 1 / Σⱼ (d_ik/d_ij)^(2/(m−1))
until memberships stop changing. Hard labels = argmax_k u_ik.

## 3. Parameters & tuning
| Parameter | Value | What it does | If increased | If decreased |
|---|---|---|---|---|
| `n_clusters` K | 5 | number of clusters | finer clusters; lower inertia | coarser; higher inertia |
| `init` | `k-means++` | centroid seeding | — | `random` → unstable inertia across seeds |
| `n_init` | 20 | restarts (best kept) | more stable, slower | faster, less stable |
| `bisecting_strategy` | `biggest_inertia` | which cluster to split | — | `largest_cluster` splits by size |
| FCM `m` | 2.0 | fuzziness | softer memberships | →1 hard K-Means |
| FCM `tol` | 1e-5 | convergence tolerance | fewer iterations | more precise convergence |

## 4. Step-by-step implementation (with key code)
1. **Load + inspect** the CSV (shape, head).
2. **Select 2 features + scale** - clustering is distance-based, so feature scales must match:
```python
X_mall = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_mall_scaled = scaler.fit_transform(X_mall)
```
3. **Choose K** by looping K=2…10 and recording inertia + silhouette:
```python
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_mall_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_mall_scaled, kmeans.labels_))
```
Elbow + silhouette peak both point to K=5.
4. **Final K-Means** (K=5):
```python
kmeans_model = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X_mall_scaled)
```
5. **Bisecting variant:**
```python
bisecting_km = BisectingKMeans(n_clusters=5, random_state=42, bisecting_strategy='biggest_inertia')
bisecting_labels = bisecting_km.fit_predict(X_mall_scaled)
```
6. **Hierarchical:** Ward linkage matrix → dendrogram → final fit:
```python
linkage_matrix = linkage(X_mall_scaled, method='ward')
agg_model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
agg_labels = agg_model.fit_predict(X_mall_scaled)
```
7. **FCM:** the custom class alternates the two update equations; the core loop:
```python
Um = U ** self.m
centers = np.dot(Um.T, X) / Um.sum(axis=0)[:, np.newaxis]
dist = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
U = inv_dist_p / inv_dist_p.sum(axis=1, keepdims=True)   # membership update
```
8. **Visual summary:** one 2x2 figure with all four partitions.
9. **Evaluate:** silhouette, Davies-Bouldin, Calinski-Harabasz for each partition.

## 5. Results & interpretation
| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| Standard K-Means | 0.5547 | 0.5722 | 248.65 |
| Bisecting K-Means | 0.4807 | 0.6788 | 150.94 |
| Hierarchical (Ward) | 0.5538 | 0.5779 | 244.41 |
| Fuzzy C-Means | 0.5547 | 0.5722 | 248.65 |

- Silhouette ≈ 0.55 means compact, well separated clusters (0.5+ is good on real data).
- FCM converging to the same partition as K-Means shows the clusters are mostly unambiguous; soft memberships would matter more with overlapping groups.
- Bisecting is slightly weaker here but produces more uniform cluster sizes - its advantage is robustness, not raw separation.

## 6. Limitations & how to improve
- K-Means assumes spherical, similar-sized clusters → fails on moons/circles (demonstrated in notebook 02).
- Means are not robust: outliers pull centroids → K-Medoids or trimming would help.
- K must be supplied externally → elbow/silhouette (used here), gap statistic, or BIC for GMMs.
- FCM is O(N·K·d) per iteration and sensitive to initialization; multiple restarts recommended.

## 7. Viva questions
- **Why standardize?** Income (0-137) would dominate spending (1-99) in Euclidean distance.
- **How was K chosen?** Elbow (inertia) + silhouette over K=2..10 → K=5.
- **K-Means vs Fuzzy C-Means?** Hard vs soft membership; FCM quantifies ambiguity at boundaries.
- **What exactly is "Modified K-Means"?** K-Means++ initialization and the Bisecting divisive variant.
- **Why K-Means++ better than random init?** Probabilistic spread of seeds → lower and more consistent inertia.
- **What does the dendrogram tell us?** The merge order and distances; the cut height determines K.
- **Which metric is best for clustering?** No single one - silhouette + DB + CH together; ARI only if labels exist.
- **Why is inertia not comparable across K?** It always decreases with K, so it cannot be a quality score.

## 8. Common mistakes
- Forgetting to scale features.
- Treating inertia like accuracy (always decreases when K grows).
- Reporting a single metric; clustering quality is multi-faceted.

## 9. One-line summary
"Four clustering algorithms on the same scaled customer data; K=5 chosen by elbow+silhouette; K-Means/FCM achieve the best separation (silhouette 0.55, DB 0.57)."

---

# 02 – Density-Based Learning

## 0. In one paragraph
DBSCAN and HDBSCAN cluster **real earthquake epicenters** from the 2023 USGS catalogue: 16,190 global events at magnitude ≥ 4.0, filtered to 7,638 events at magnitude ≥ 4.5 for a well-recorded subset. Coordinates are converted to radians and clustered with the **haversine** (great-circle) metric because lat/lon degrees are not Euclidean. DBSCAN (ε = 0.03 rad ≈ 191 km, MinPts = 10) finds 55 clusters with 13.5% noise; HDBSCAN (min_cluster_size = 50) finds 40 clusters with 21.5% noise. The largest clusters correspond to real seismic zones (Philippines, Papua New Guinea, Tonga, Japan, Turkey) - a real-world validation with no labels needed.

## 1. Quick facts
| | |
|---|---|
| **Algorithms** | DBSCAN, HDBSCAN (scikit-learn, haversine metric) |
| **Dataset** | `data/earthquakes.csv` - USGS global catalogue 2023; 16,190 events (M ≥ 4.0), **7,638 used (M ≥ 4.5)** |
| **Features** | Epicenter latitude/longitude (converted to radians); magnitude/depth available but not clustered on |
| **Key parameters** | DBSCAN: eps = 0.03 rad (≈ 191 km), min_samples = 10. HDBSCAN: min_cluster_size = 50, min_samples = 10 |
| **Results** | DBSCAN: 55 clusters, 13.5% noise, silhouette 0.326 (3k sample). HDBSCAN: 40 clusters, 21.5% noise, silhouette **0.631** |

## 2. How the algorithm works
**DBSCAN** defines clusters as connected regions of density:
- **ε-neighborhood:** N_ε(p) = {q : haversine(p,q) ≤ ε}.
- **Core point:** |N_ε(p)| ≥ MinPts. **Border:** within ε of a core. **Noise:** neither (label −1).
- Clusters grow by chaining density-reachable core points; border points join a neighboring core's cluster.
- One global ε means varying-density data is hard to handle.

**HDBSCAN** removes ε:
1. core distance d_core(p) = distance to its k-th nearest neighbor;
2. mutual reachability d_mreach(a,b) = max{d_core(a), d_core(b), d(a,b)};
3. minimum spanning tree over mutual reachability;
4. hierarchy → condensed tree (cluster births/deaths);
5. keep clusters with the best stability S(C) = Σ (λ_p − λ_birth(C)).
It also outputs membership probabilities and handles varying density.

**Why haversine?** Latitude/longitude form a sphere, not a plane; haversine gives the true great-circle distance. In scikit-learn the metric expects **radian** coordinates, and eps is in radians (0.03 rad × 6371 km ≈ 191 km).

## 3. Parameters & tuning
| Parameter | Value | What it does | If increased | If decreased |
|---|---|---|---|---|
| DBSCAN `eps` | 0.03 rad (~191 km) | neighborhood radius | fewer/larger clusters, less noise | more noise, clusters fragment |
| DBSCAN `min_samples` | 10 | MinPts density threshold | stricter, more noise | merges clusters |
| HDBSCAN `min_cluster_size` | 50 | smallest allowed cluster | fewer/larger clusters | many small clusters |
| HDBSCAN `min_samples` | 10 | conservativeness (k for core distance) | more noise, more conservative | more clusters |
| Distance metric | haversine | great-circle on the sphere | — | euclidean on degrees = wrong |

## 4. Step-by-step implementation (with key code)
1. **Load + filter** the USGS catalogue; print shape and preview.
2. **Convert to radians** for the spherical metric:
```python
quakes = df[df['mag'] >= 4.5].reset_index(drop=True)
X_rad = np.radians(quakes[['latitude', 'longitude']].values)   # [lat, lon] in radians
```
3. **Choose ε** from the k-distance graph (haversine):
```python
nbrs = NearestNeighbors(n_neighbors=10, metric='haversine').fit(X_rad)
distances, _ = nbrs.kneighbors(X_rad)
k_distances = np.sort(distances[:, -1])          # knee ~0.03 rad
```
4. **DBSCAN fit + core points:**
```python
dbscan = DBSCAN(eps=0.03, min_samples=10, metric='haversine')
db_labels = dbscan.fit_predict(X_rad)
core_samples_mask[dbscan.core_sample_indices_] = True
```
5. **Plot epicenters**: clustered events in color, noise as black crosses (world map layout).
6. **Real-world check**: for the six largest clusters print size, centroid and the dominant region from the USGS `place` column (Philippines, Papua New Guinea, Tonga, Japan, South Sandwich Islands, Turkey).
7. **HDBSCAN fit:**
```python
hdb = HDBSCAN(min_cluster_size=50, min_samples=10, metric='haversine', copy=False)
hdb_labels = hdb.fit_predict(X_rad)
```
8. **Summary table** with cluster count, noise %, largest cluster, silhouette (3,000-event sample for tractability).

## 5. Results & interpretation
| Algorithm | Clusters | Noise | Largest cluster | Silhouette (3k sample) ↑ |
|---|---|---|---|---|
| DBSCAN | 55 | 13.5% | 1,629 (Philippines) | 0.326 |
| HDBSCAN | 40 | 21.5% | 560 | **0.631** |

- **Real-world validation:** the largest clusters are the Philippines (1,629 events), Papua New Guinea (1,068), Tonga (947), Japan (895), South Sandwich Islands (253) and Turkey (207) - exactly the tectonically active boundaries (Ring of Fire, Alpine belt).
- **Noise = isolated seismicity:** ~1,000-1,600 events far from any plate boundary (intraplate events) are labeled −1.
- **HDBSCAN silhouette is roughly double DBSCAN's:** DBSCAN's fixed ε chains elongated arc clusters along an entire boundary into one large, elongated group; HDBSCAN's adaptive density splits them into compact regions.
- **No ARI here:** the catalogue has no ground-truth cluster labels, so quality is judged by noise share, silhouette and the region sanity-check.

## 6. Limitations & how to improve
- Distance-based density estimation on a sphere is fine for 2D but the method degrades in high dimensions (curse of dimensionality).
- Cluster shapes are elongated (fault lines) - compactness metrics like silhouette penalize them; use region validation as done here.
- `min_cluster_size`/`min_samples` still need judgment; HDBSCAN reduces but does not eliminate tuning.
- Improvements: cluster in 3D Cartesian coordinates (unit sphere) for convenience, add depth/magnitude as features, or use a grid-based method (DBSCAN on projected UTM tiles).

## 7. Viva questions
- **Why haversine instead of Euclidean?** Lat/lon are spherical coordinates; haversine gives true great-circle distances (eps in radians).
- **Why filter to M ≥ 4.5?** Catalogue completeness and recording quality improve with magnitude; also keeps computation tractable.
- **How was ε chosen?** k-distance graph knee (k = 10) → 0.03 rad ≈ 191 km.
- **What is noise here physically?** Isolated earthquakes far from plate boundaries (intraplate seismicity).
- **Define core/border/noise.** Core: ≥ MinPts within ε; border: in ε of a core; noise: neither.
- **Why is HDBSCAN's silhouette higher?** Variable-density clusters split arc chains into compact groups instead of one elongated DBSCAN cluster.
- **Why no ARI?** No ground-truth labels exist for real seismicity clustering.
- **What do the big clusters represent?** Known seismic zones - Philippines, Papua New Guinea, Tonga, Japan, Turkey.

## 8. Common mistakes
- Passing degrees to a haversine metric (scikit-learn silently gives wrong distances).
- Expecting all plate-boundary events in one cluster - fault zones are elongated, so several clusters appear.
- Comparing DBSCAN/HDBSCAN cluster counts directly: their noise definitions and density rules differ.

## 9. One-line summary
"DBSCAN/HDBSCAN on 7,638 real 2023 earthquakes: ε = 0.03 rad (~191 km) from the k-distance knee; 55/40 clusters with 13.5%/21.5% noise, and the largest clusters match the Philippines, Papua New Guinea, Tonga, Japan and Turkey - real seismic zones discovered without labels."

# 03 – Semi-Supervised Learning

## 0. In one paragraph
With only 15% of the training labels available (61 of 398) on Breast Cancer, a self-training classifier wraps an RBF SVC and iteratively converts its most confident predictions on the unlabeled pool into pseudo-labels. This lifts test accuracy from 0.9298 (baseline) to 0.9415, partially closing the gap to the 100%-label ceiling (0.9766).

## 1. Quick facts
| | |
|---|---|
| **Algorithm** | Self-Training Classifier (`SelfTrainingClassifier`) with SVC (RBF, probability=True) |
| **Dataset** | Breast Cancer Wisconsin: 569 x 30, binary; 70/30 stratified split |
| **Semi-supervised setup** | Training = 398 samples: 61 labeled (15%), 337 masked −1 (unlabeled); test = 171 |
| **Key parameters** | threshold τ=0.80, criterion='threshold', max_iter=15, SVC(C=1.0) |
| **Results** | Baseline 0.9298 / F1 0.9469 → Self-training **0.9415 / 0.9554** → Ceiling 0.9766 / 0.9813; 321 pseudo-labels added over 4 iterations, stop reason `no_change` |

## 2. How the algorithm works
Given labeled set L and unlabeled pool U:
1. train classifier f on L;
2. compute f's class probabilities on U;
3. select samples with max P(y|x) ≥ τ and add (x, argmax) to L as **pseudo-labels**;
4. remove them from U;
5. repeat until no new labels pass the threshold or max_iter is reached.
The model never sees the test set. The threshold τ controls the purity/quantity trade-off: low τ pollutes training with mistakes (**confirmation bias**), high τ accepts almost nothing.

## 3. Parameters & tuning
| Parameter | Value | What it does | If increased | If decreased |
|---|---|---|---|---|
| `threshold` τ | 0.80 | confidence needed for a pseudo-label | fewer but purer pseudo-labels | more labels, more errors |
| `criterion` | `threshold` | selection rule | — | `k_best` picks fixed count instead |
| `max_iter` | 15 | max self-training rounds | more chances to label, slower | may stop too early |
| SVC `probability` | True | enables `predict_proba` | — | no probabilities → wrapper unusable |
| SVC `C` | 1.0 | margin penalty | stricter fit | softer margin |

## 4. Step-by-step implementation (with key code)
1. **Load + split + scale** (fit scaler on train only, stratify to preserve class ratio).
2. **Mask labels to simulate scarcity:**
```python
random_unlabeled_points = rng.rand(len(y_train_full)) < 0.85
y_train_semi = np.copy(y_train_full)
y_train_semi[random_unlabeled_points] = -1        # -1 = unlabeled convention
```
3. **Baseline:** SVC trained only on `y_train_semi != -1` (61 samples).
4. **Self-training:**
```python
self_training_svc = SelfTrainingClassifier(
    estimator=SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
    threshold=0.80, criterion='threshold', max_iter=15, verbose=True)
self_training_svc.fit(X_train_full_scaled, y_train_semi)   # labeled + unlabeled
```
The wrapper logs: iter1 +292 labels, iter2 +22, iter3 +5, iter4 +2, then `no_change`.
5. **Ceiling:** same SVC trained on all 398 true labels.
6. **Summary table** comparing accuracy/F1 for the three settings.

## 5. Results & interpretation
| Setting | Accuracy | F1 | Training samples |
|---|---|---|---|
| Baseline (15% labeled) | 0.9298 | 0.9469 | 61 |
| Self-training | **0.9415** | **0.9554** | 382 |
| Fully supervised ceiling | 0.9766 | 0.9813 | 398 |

- Self-training gains +1.2 accuracy points over the baseline with no extra human labeling.
- 382/398 samples eventually received labels; the decaying label additions (292→22→5→2) indicate stable convergence rather than runaway error accumulation.
- The remaining gap to the ceiling is the price of noisy pseudo-labels.

## 6. Limitations & how to improve
- Confirmation bias: wrong confident predictions reinforce themselves → use curriculum thresholds (start low, raise), or co-training with two views.
- Assumes the unlabeled data comes from the same distribution as the labeled data.
- Base estimator must expose probabilities → SVC needs Platt calibration (`probability=True`).
- Improve: label propagation, contrastive/self-supervised pretraining, or active learning for the most uncertain points.

## 7. Viva questions
- **What is self-training?** Iterative pseudo-labeling of confident unlabeled predictions.
- **Why did accuracy improve?** Pseudo-labels expose the classifier to the unlabeled distribution.
- **What is confirmation bias?** The model reinforcing its own mistakes when pseudo-labels are wrong.
- **How do you choose τ?** Trade-off: high confidence = purity but fewer labels; 0.80 balanced here (converged in 4 iterations).
- **Why `probability=True`?** The wrapper needs `predict_proba` to compute confidence.
- **Why a stratified split?** To keep the malignant/benign ratio identical in train and test.
- **Could you use unlabeled test data?** No - transductive use of the test set would break the evaluation.
- **What's the ceiling?** Fully supervised accuracy with all labels - the upper bound self-training approaches.

## 8. Common mistakes
- Letting the test set participate in pseudo-labeling (leakage).
- Using too low a threshold and reporting the inflated training accuracy.
- Forgetting that pseudo-labels are not ground truth in the error analysis.

## 9. One-line summary
"Self-training with 85% hidden labels recovers +1.2 accuracy points on Breast Cancer (0.9298 → 0.9415) by iteratively adding high-confidence pseudo-labels, while staying below the fully supervised ceiling."

---

# 04 – Ensemble Learning

## 0. In one paragraph
Five ensemble methods are implemented: Random Forest Regression and Classification (bagging) plus XGBoost, AdaBoost and CatBoost (boosting). Regression runs on a 5,000-row California Housing sample and classification on Heart Disease. Gradient boosting wins the regression benchmark (XGBoost R² 0.8025, CatBoost 0.8029), Random Forest gives a robust baseline with free OOB validation (0.7415), and RFC classifies heart disease at ROC-AUC 0.87.

## 1. Quick facts
| | |
|---|---|
| **Algorithms** | RFR, RFC, XGBoost, AdaBoost, CatBoost |
| **Datasets** | California Housing (5,000 x 8, regression); Heart Disease (303 x 13, classification) |
| **Key parameters** | RFR 150 trees/depth 12; XGB 150 trees, lr 0.08, depth 6, subsample 0.8; AdaBoost 100, lr 0.1; CatBoost 200 iters, depth 6; RFC 120 trees/depth 8 |
| **Results** | R²: XGB 0.8025, Cat 0.8029, RFR 0.7415 (OOB 0.7513), Ada 0.5888. RFC: acc 0.7763, F1 0.8046, AUC 0.8704 |

## 2. How the algorithm works
**Bagging (Random Forest):** train B trees on bootstrap samples, each split considers a random feature subset; regression averages outputs, classification takes a majority vote. Averaging decorrelated trees reduces **variance**. OOB score: each tree is validated on the ~37% of samples it did not receive.

**Boosting:** sequential models where each new learner focuses on the current ensemble's errors → reduces **bias**.
- **AdaBoost:** increase weights of mispredicted samples; learner weight α_m = ½ ln((1−ε_m)/ε_m); final = weighted vote (classification) or weighted median (regression).
- **XGBoost:** 2nd-order Taylor expansion of the loss: L ≈ Σ[g_i f_t(x_i) + ½ h_i f_t²(x_i)] + Ω(f), with Ω(f) = γT + ½λΣw_j². Gradients/Hessians drive tree growth; subsampling + regularization fight overfitting.
- **CatBoost:** ordered boosting (compute leaf values on permutations to avoid target leakage/prediction shift) + oblivious (symmetric) trees; strong native handling of categorical features.

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| `n_estimators` | 150-200 | number of trees/rounds | better fit, slower, risk of overfit (boosting) | underfit |
| `learning_rate` (boosting) | 0.08-0.1 | step size per round | faster fit, less stable | slower, usually better generalization |
| `max_depth` | 4-12 | tree complexity | captures interactions, overfits | underfits |
| `subsample`/`colsample_bytree` | 0.8 | row/column sampling | (higher) less regularization | more stochastic, regularized |
| `reg_lambda`/`reg_alpha` | 1.0 / 0.1 | L2/L1 penalties | stronger regularization | weaker |
| `l2_leaf_reg` (CatBoost) | 3.0 | leaf penalty | smoother leaves | sharper leaves |

## 4. Step-by-step implementation (with key code)
1. **Load + sample** California Housing; 80/20 split.
2. **RFR with OOB:**
```python
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=12, oob_score=True, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg, y_train_reg)
print(rf_reg.oob_score_, r2_score(y_test_reg, rf_reg.predict(X_test_reg)))
```
3. **XGBoost:**
```python
xgb_reg = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=1.0, random_state=42)
```
4. **AdaBoost:** `AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)`.
5. **CatBoost:** `CatBoostRegressor(iterations=200, learning_rate=0.08, depth=6, l2_leaf_reg=3.0, verbose=False)`.
6. **Comparison table** with R², RMSE, MAE.
7. **RFC:** one-hot encode heart data, stratified split, fit:
```python
rfc = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
rfc.fit(X_train_clf, y_train_clf)
```
then accuracy / F1 / ROC-AUC on the test set.

## 5. Results & interpretation
| Model | R² ↑ | RMSE ↓ | MAE ↓ |
|---|---|---|---|
| Random Forest Regressor | 0.7415 (OOB 0.7513) | 0.5908 | 0.3999 |
| AdaBoost Regressor | 0.5888 | 0.7452 | 0.5842 |
| XGBoost Regressor | 0.8025 | 0.5165 | 0.3467 |
| CatBoost Regressor | 0.8029 | 0.5160 | 0.3535 |

- XGBoost/CatBoost tie at the top (R² ≈ 0.80): both are regularized gradient boosting.
- RFR's OOB (0.7513) ≈ test R² (0.7415) → no overfitting, and the OOB estimate is free.
- AdaBoost is weakest: exponential loss is sensitive to noisy/outlier house prices.
- RMSE is in $100k units (target MedHouseVal): 0.516 ≈ $51.6k average error.
- RFC on heart: accuracy 0.7763, F1 0.8046, ROC-AUC 0.8704 → good ranking quality on a small dataset.

## 6. Limitations & how to improve
- Hyperparameters were fixed, not tuned → grid search/random search would gain a few points.
- California sample (5k) loses some geographic diversity; full data would help.
- AdaBoost assumptions (low noise) are violated by housing data.
- Improvements: hyperparameter search, early stopping for boosting rounds, feature engineering (distance to coast), target transforms.

## 7. Viva questions
- **Bagging vs boosting?** Parallel variance reduction vs sequential bias reduction.
- **What is OOB and why is it useful?** Free validation from bootstrap out-of-bag samples; no extra split needed.
- **Why does XGBoost beat AdaBoost here?** Regularized 2nd-order optimization vs sensitive exponential loss.
- **What makes CatBoost special?** Ordered boosting + symmetric trees + native categorical support.
- **RFR vs RFC?** Regression averages tree values; classification votes/is prob-averages.
- **What is ROC-AUC?** Probability a random positive is ranked above a random negative (0.5 = random).
- **Why subsample rows/columns?** Stochasticity decorrelates trees/rounds and regularizes.
- **How would you detect overfitting in boosting?** Validation curve turning up while training loss falls; fix with lower depth/lr or early stopping.

## 8. Common mistakes
- Comparing RMSE across different target units/transforms.
- Reading OOB as test performance for models other than bagging (boosting has no OOB).
- Forgetting `stratify` for the imbalanced heart target.

## 9. One-line summary
"Five ensembles on two tasks: gradient boosting (XGBoost/CatBoost) leads housing regression at R² ≈ 0.80, Random Forest provides a variance-reduced baseline with free OOB validation, and RFC classifies heart disease at AUC 0.87."

---

# 05 – Multilayer Perceptron (MLP)

## 0. In one paragraph
Two MLP implementations classify 8x8 handwritten digits: a scikit-learn `MLPClassifier` (128-64 ReLU, early stopping) reaching 0.9630 test accuracy, and a custom PyTorch network (64→128→64→10 with BatchNorm + Dropout, Adam + LR scheduling) reaching **0.9889**. The notebook covers the full training loop, validation tracking, and per-class evaluation.

## 1. Quick facts
| | |
|---|---|
| **Algorithms** | Scikit-Learn MLPClassifier; custom PyTorch `DeepMLP` |
| **Dataset** | Digits: 1,797 samples, 64 features (8x8), 10 classes |
| **Split** | 70% train / 15% validation / 15% test, stratified |
| **Key parameters** | sklearn: (128,64), ReLU, Adam, alpha=0.001, early stopping. PyTorch: BatchNorm + Dropout(0.25/0.20), CrossEntropyLoss, Adam lr 0.003, weight decay 1e-4, batch 32, 50 epochs, ReduceLROnPlateau |
| **Results** | sklearn 0.9630 (15 iterations); PyTorch **0.9889** test accuracy |

## 2. How the algorithm works
An MLP stacks fully connected layers with non-linear activations:
a⁽¹⁾ = σ(W⁽¹⁾x + b⁽¹⁾), …, ŷ = softmax(W⁽ᴸ⁺¹⁾a⁽ᴸ⁾ + b⁽ᴸ⁺¹⁾).
Training minimizes cross-entropy L = −Σᵢ Σₖ y_ik ln ŷ_ik (+ L2 penalty) with backpropagation:
1. forward pass → predictions; 2. loss; 3. backward pass (chain rule) → gradients; 4. optimizer update. Repeat over mini-batches/epochs.

Regularization used here:
- **BatchNorm:** normalizes each layer's activations → stable, faster training.
- **Dropout:** randomly zeros neurons during training → prevents co-adaptation/overfitting.
- **Early stopping / LR scheduling:** stop or shrink the step when validation stalls.

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| Hidden layers | (128, 64) / (128, 64) | capacity | more capacity, overfit risk | underfit |
| Activation | ReLU | non-linearity | — | sigmoid/tanh saturate (vanishing gradients) |
| `alpha` / `weight_decay` | 0.001 / 1e-4 | L2 regularization | smoother weights, underfit | overfit |
| Dropout | 0.25 / 0.20 | neuron dropout | stronger regularization | weaker |
| `learning_rate` | 0.005 / 0.003 | step size | unstable/diverges | very slow |
| `batch_size` | 64 / 32 | gradient noise | smoother but slower updates | noisier, regularizing |

## 4. Step-by-step implementation (with key code)
1. **Load + split + scale** (scaler fit on train only).
2. **sklearn MLP:**
```python
mlp_sklearn = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                            alpha=0.001, batch_size=64, learning_rate_init=0.005,
                            max_iter=150, early_stopping=True, validation_fraction=0.15)
mlp_sklearn.fit(X_train_scaled, y_train)
```
stopped after 15 iterations with 0.9630 test accuracy.
3. **PyTorch model:**
```python
self.net = nn.Sequential(
    nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
    nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.20),
    nn.Linear(64, 10))
```
4. **Training loop** (50 epochs):
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# each batch: optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
# each epoch: validation pass under torch.no_grad(); scheduler.step(val_loss)
```
5. **Final evaluation:** accuracy + per-class classification report on the test set.

## 5. Results & interpretation
- sklearn: 0.9630 test accuracy, converged in 15 iterations (early stopping prevented overfitting).
- PyTorch: **0.9889** (270 test images, 3 mistakes). Train loss fell 1.20 → 0.01; validation loss stayed in the same range → no severe overfitting.
- Per-class report: nearly all digits have precision/recall ≈ 1.00; residual errors are visually similar digits.
- Validation accuracy (0.9889 at epoch 50) matches test accuracy → the validation split was representative.

## 6. Limitations & how to improve
- MLPs ignore spatial structure of images (a CNN is the right tool for digits).
- Fixed architecture - no systematic hyperparameter search.
- Small dataset (1,797) → augmentation/regularization matter.
- Improvements: CNN, data augmentation, dropout tuning, learning-rate warmup, ensembling.

## 7. Viva questions
- **What loss and why?** Categorical cross-entropy; directly penalizes wrong class probabilities.
- **Why ReLU?** Non-saturating gradient, fast to compute.
- **What do BatchNorm and Dropout do?** Stabilize training / regularize to prevent overfitting.
- **Difference between parameters and hyperparameters?** Weights/biases are learned; layers, LR, dropout are chosen.
- **Why did you need a validation set?** For early stopping and LR scheduling without touching the test set.
- **What is backpropagation?** Reverse-mode chain rule computing gradients layer by layer.
- **Why is PyTorch better than sklearn here?** BatchNorm + Dropout + longer scheduled training; sklearn stopped at iteration 15.
- **How do you detect overfitting?** Train loss much lower than validation loss while validation worsens.

## 8. Common mistakes
- Scaling the test set with statistics from the test set (leakage).
- Forgetting `model.eval()` before evaluation (Dropout/BatchNorm behave differently).
- Comparing accuracies without checking the same split/seed.

## 9. One-line summary
"Two MLP implementations on 8x8 digits: scikit-learn reaches 96.3% with early stopping, while the custom PyTorch network with BatchNorm + Dropout and LR scheduling reaches 98.9% test accuracy."

---

# 06 – Recurrent Neural Network (RNN)

## 0. In one paragraph
A vanilla Elman RNN and an LSTM forecast monthly airline passenger counts from the previous 12 months. After a chronological 80/20 split, both train identically (2 layers, hidden 64, MSE, Adam, 120 epochs). The LSTM achieves RMSE 42.17 vs 70.83 passengers for the vanilla RNN - a direct demonstration that gated memory handles long-range seasonality better.

## 1. Quick facts
| | |
|---|---|
| **Algorithms** | Vanilla RNN (`nn.RNN`), LSTM (`nn.LSTM`), both PyTorch |
| **Dataset** | `data/airline_passengers.csv`, 144 monthly values (1949-1960) |
| **Preprocessing** | MinMaxScaler → [0,1]; sliding windows: 12 months → next month |
| **Split** | Chronological 80/20 → 105 train / 27 test sequences; no shuffling |
| **Key parameters** | 2 layers, hidden 64, Linear(64,1), MSELoss, Adam lr 0.005, 120 epochs, batch 16 |
| **Results** | Vanilla RNN RMSE 70.83 / MAE 63.16; **LSTM RMSE 42.17 / MAE 34.88** |

## 2. How the algorithm works
**Vanilla RNN:** h_t = tanh(W_ih x_t + b_ih + W_hh h_{t−1} + b_hh); output from the last step ŷ = W_ho h_T + b_o. Trained with backpropagation through time (BPTT). Problem: repeated multiplication by W_hh makes gradients vanish (or explode) over long sequences.

**LSTM:** adds a cell state c_t with gates:
- forget: f_t = σ(W_f x_t + U_f h_{t−1} + b_f),
- input: i_t = σ(...), candidate: c̃_t = tanh(...),
- cell update: c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t (additive → gradients flow),
- output: o_t = σ(...), h_t = o_t ⊙ tanh(c_t).

**Why sliding windows?** The 12-month lookback supplies one full annual cycle; the model learns seasonality + trend from position in the window.

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| `seq_length` | 12 | lookback window | more history, harder to train | less context |
| `hidden_size` | 64 | memory capacity | more capacity, slower | underfit |
| `num_layers` | 2 | depth of recurrence | more abstraction, overfit risk | shallow |
| `lr` | 0.005 | Adam step | diverges | slow |
| `epochs` | 120 | training length | overfit possible | underfit |
| `batch_size` | 16 | gradient noise | smoother | noisier, regularizing |

## 4. Step-by-step implementation (with key code)
1. **Load + normalize** the passenger series to [0,1].
2. **Build sequences:**
```python
def create_sequences(data, seq_length=12):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])   # 12 past months
        ys.append(data[i + seq_length])       # next month
    return np.array(xs), np.array(ys)
```
3. **Chronological split** (no shuffle) + DataLoader (shuffle train only).
4. **Models:**
```python
self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
self.fc  = nn.Linear(64, 1)          # Vanilla RNN

self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
self.fc   = nn.Linear(64, 1)         # LSTM
```
5. **Train** (same function for both): MSE loss, Adam lr 0.005, 120 epochs:
```python
optimizer.zero_grad(); pred = model(x_b); loss = criterion(pred, y_b)
loss.backward(); optimizer.step()    # BPTT through the 12 steps
```
6. **Evaluate:** predict test windows, invert MinMax scaling, compute RMSE/MAE.
7. **Plot** history + test truth + both forecasts with a train/test cutoff line.

## 5. Results & interpretation
| Model | RMSE (passengers) ↓ | MAE ↓ |
|---|---|---|
| Vanilla RNN | 70.83 | 63.16 |
| LSTM | **42.17** | **34.88** |

- LSTM error is 40% lower than the vanilla RNN - gating preserves the seasonal signal across 12 steps.
- MAE 35 passengers against a series averaging ~280/month ≈ 12% relative error.
- Forecast plot: LSTM follows both the upward trend and the yearly oscillation; RNN lags.
- Training ran ~19 s for both models on CPU (small dataset).

## 6. Limitations & how to improve
- Multi-step forecasting currently feeds true past values; a real deployment must roll predictions forward (error accumulation).
- Only 144 data points - a small dataset; more history or covariates (holidays, GDP) would help.
- Vanilla RNN could be improved with gradient clipping, but LSTM/GRU is the standard fix.
- Improvements: GRU, attention/Transformer for sequences, probabilistic forecasting (quantiles).

## 7. Viva questions
- **Why does LSTM beat vanilla RNN?** Additive cell state prevents vanishing gradients over 12 steps.
- **What is BPTT?** Backpropagation through time - chain rule across all time steps.
- **Why 12-month windows?** One full seasonal cycle.
- **Why no shuffling?** Shuffling would leak future values into training.
- **Why MinMax scaling?** Matches activation ranges and makes inversion trivial.
- **What is the input shape?** (batch, 12, 1).
- **What would happen with a longer lookback?** More context, but vanilla RNN degrades faster; LSTM tolerates longer windows.
- **Why MSE loss?** Regression target; penalizes larger errors more, smooth gradients.

## 8. Common mistakes
- Random train/test split on time series (leakage).
- Forgetting to invert scaling before computing RMSE (units become meaningless).
- Comparing RNN/LSTM with different epochs/hidden sizes (unfair benchmark).

## 9. One-line summary
"On 12-month airline windows, both models train identically; the LSTM's gated memory cuts test RMSE from 70.8 to 42.2 passengers, showing why gating matters for long-range seasonality."

---

# 07 – Self-Organizing Map (SOM)

## 0. In one paragraph
A 12x12 Kohonen Self-Organizing Map learns the topology of 178 wine samples (13 chemical features) without using any labels. Training improves the quantization error from 0.445 to 0.196 and leaves a topographic error of 0.039; projecting the three true cultivars onto the U-matrix shows they occupy distinct regions - evidence of unsupervised structure discovery.

## 1. Quick facts
| | |
|---|---|
| **Algorithm** | Kohonen Self-Organizing Map (MiniSom) |
| **Dataset** | UCI Wine: 178 x 13; 3 cultivars used only for visualization |
| **Preprocessing** | MinMaxScaler to [0,1] |
| **Key parameters** | 12x12 grid (144 neurons), Gaussian neighborhood, sigma=1.5, learning rate=0.5, PCA init, 5,000 iterations |
| **Results** | Quantization error 0.4447 → **0.1965**; topographic error **0.0393** |

## 2. How the algorithm works
An MxN grid of neurons, each with a weight vector w_j ∈ R^d. For each training sample:
1. **Competition:** find the Best Matching Unit c = argmin_j ‖x − w_j‖.
2. **Cooperation:** compute the Gaussian neighborhood h_cj = exp(−‖r_c − r_j‖²/(2σ(t)²)) on grid coordinates r.
3. **Adaptation:** w_j ← w_j + α(t)·h_cj·(x − w_j).
Learning rate α(t) and radius σ(t) decay over time: early on large neighborhoods organize the global layout; later small updates refine details. Result: a topology-preserving 2D projection where nearby grid neurons have similar weight vectors.

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| Grid size | 12x12 | resolution | finer map, longer training | coarse map, less detail |
| `sigma` | 1.5 | initial neighborhood radius | smoother/global organization | local, fragmented map |
| `learning_rate` | 0.5 | initial step size | faster but unstable | slow convergence |
| `num_iteration` | 5000 | training length | better convergence | under-trained map |
| `neighborhood_function` | gaussian | kernel shape | — | bubble/mexican-hat alternative |
| Init | PCA | initial weights | — | random init → possible folds |

## 4. Step-by-step implementation (with key code)
1. **Load wine + MinMax scale.**
2. **Create and initialize the map:**
```python
som = MiniSom(x=12, y=12, input_len=13, sigma=1.5, learning_rate=0.5,
              neighborhood_function='gaussian', random_seed=42)
som.pca_weights_init(X)                 # spread weights along principal directions
```
3. **Train:**
```python
som.train_random(data=X, num_iteration=5000)   # random samples; BMU + neighborhood update
```
4. **Quality metrics:**
```python
qe = som.quantization_error(X)    # avg distance from samples to their BMU
te = som.topographic_error(X)     # fraction whose top-2 BMUs are not neighbors
```
5. **U-Matrix:** `som.distance_map()` = average distance to neighboring neurons; plot heatmap and overlay each sample at its BMU with a marker colored by cultivar.

## 5. Results & interpretation
- QE halved (0.445 → 0.196): neurons now sit close to the data they represent.
- TE = 0.039 → only ~4% of samples are mapped to non-adjacent top-2 neurons; topology preserved.
- On the U-matrix, the three cultivars occupy contiguous regions separated by bright ridges - the SOM discovered group structure without labels.
- PCA initialization gave a well-unfolded map from the start (initial QE 0.445 instead of a random map's higher error).

## 6. Limitations & how to improve
- Grid size/σ/LR must be chosen manually; too large a grid fragments clusters.
- The map is a fixed-resolution projection - distances are not exact (topology, not geometry).
- No probabilistic interpretation; a GMM/UMAP+t-SNE gives complementary views.
- Improvements: tune neighborhood function/iterations, add hit-map analysis, cluster the neuron weights (e.g., K-Means on codebook vectors) for automatic grouping.

## 7. Viva questions
- **What is a BMU?** The neuron whose weight vector is closest to the input.
- **Is SOM supervised?** No - labels were used only for visualization.
- **What does the U-Matrix show?** Average neighbor distance: dark valleys = clusters, bright ridges = boundaries.
- **Quantization vs topographic error?** Representation quality vs topology preservation.
- **Why does the learning rate decay?** Early coarse organization, later fine-tuning.
- **Why 12x12?** Enough resolution for 178 samples without over-fragmenting.
- **SOM vs K-Means?** Both compress data, but SOM adds a topology-preserving grid.
- **What is the advantage of PCA init?** Avoids folded maps and speeds convergence.

## 8. Common mistakes
- Not scaling inputs (weights live in input space).
- Using too few iterations and reading an under-trained map.
- Interpreting U-matrix distances as true data distances.

## 9. One-line summary
"A 12x12 Kohonen map organizes wine chemistry without labels: quantization error halved to 0.196, topographic error 0.039, and the three cultivars separate cleanly on the U-matrix."

---

# 08 – Hidden Markov Model (HMM)

## 0. In one paragraph
A 3-state Gaussian Hidden Markov Model is trained with the Baum-Welch (EM) algorithm on 1,000 days of financial returns to uncover latent market regimes. The model converges (log-likelihood 3018.45) and Viterbi decoding assigns each day to Bull, Sideways or Bear after states are sorted by volatility - revealing persistence and volatility clustering on the price chart.

## 1. Quick facts
| | |
|---|---|
| **Algorithm** | Gaussian HMM (hmmlearn): Baum-Welch EM training + Viterbi decoding |
| **Dataset** | `data/market_regimes.csv`, 1,000 trading days; observations = daily returns |
| **Key parameters** | n_components=3, covariance_type="full", n_iter=200, random_state=42 |
| **Results** | EM converged (True); model log-likelihood 3018.45 |

## 2. How the algorithm works
An HMM is a doubly stochastic process:
- hidden state sequence z_t (1st-order Markov): P(z_t | z_{t−1}, …, z_1) = P(z_t | z_{t−1}), summarized by the transition matrix A with A_ij = P(z_{t+1}=s_j | z_t=s_i);
- observations x_t emitted from the active state: x_t | z_t = s_k ~ N(μ_k, σ_k²).

Three classic problems:
1. **Evaluation:** forward algorithm gives P(X | λ) (log-likelihood).
2. **Decoding:** Viterbi dynamic programming finds the most likely state path: δ_t(j) = max_i[δ_{t−1}(i)·A_ij]·b_j(x_t).
3. **Learning:** Baum-Welch (EM) - E-step computes posteriors with forward-backward; M-step re-estimates π, A, and Gaussian parameters; iterate until log-likelihood converges.

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| `n_components` | 3 | number of regimes | finer regimes, overfit risk | coarse regimes |
| `covariance_type` | full | emission covariance | more flexible (full matrix) | diag/spherical = simpler |
| `n_iter` | 200 | EM iterations | better convergence, slower | may stop early |
| `random_state` | 42 | reproducibility | — | runs may find different local optima |

## 4. Step-by-step implementation (with key code)
1. **Load** the CSV; `returns = df_market['Daily_Return'].values.reshape(-1, 1)`.
2. **Fit the HMM:**
```python
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42)
hmm_model.fit(returns)
print(hmm_model.monitor_.converged, hmm_model.score(returns))
```
3. **Decode the regime path:**
```python
hidden_states = hmm_model.predict(returns)     # Viterbi
```
4. **Stabilize labels** by sorting states by volatility:
```python
volatilities = [np.sqrt(hmm_model.covars_[i][0][0]) for i in range(3)]
state_order = np.argsort(volatilities)         # Bull(0) < Sideways(1) < Bear(2)
```
and reorder the transition matrix accordingly.
5. **Visualize:** transition-matrix heatmap + price series colored by daily regime (Viterbi path).

## 5. Results & interpretation
- EM converged; log-likelihood 3018.45 (higher is better) - the 3-state model explains the return sequence well.
- Sorting states by σ makes labels stable and interpretable: low-vol (bull), medium (sideways), high-vol (bear).
- The price chart shows the model switching into the high-volatility state during drawdowns - the classic volatility-clustering behavior.
- The transition matrix diagonal (regime persistence) is visibly larger than off-diagonal values → regimes are sticky, as expected in markets.

## 6. Limitations & how to improve
- Only returns are modeled; adding volatility/volume features (multivariate emissions) would enrich states.
- Number of states is a modeling choice - compare 2-5 states via log-likelihood/BIC.
- Gaussian emissions underestimate fat tails of returns (t-distributions are more realistic).
- Improvements: hidden semi-Markov models for duration modeling, HMM + GARCH hybrids.

## 7. Viva questions
- **What is hidden vs observed?** Regimes are latent; returns are observed.
- **What does the transition matrix tell you?** Probability of staying/switching regimes (persistence).
- **Baum-Welch vs Viterbi?** Learning parameters vs decoding the most likely state path.
- **Why sort states by volatility?** Deterministic labels across runs (Bull/Sideways/Bear).
- **How do you know it converged?** `monitor_.converged` is True and log-likelihood stabilized.
- **Is this supervised?** No - the states are learned unsupervised from returns.
- **What are the three classic HMM problems?** Evaluation (forward), decoding (Viterbi), learning (Baum-Welch).
- **Why is diagonal dominance expected?** Markets rarely switch regimes daily; regimes persist.

## 8. Common mistakes
- Re-training with different random seeds and comparing regimes with inconsistent labels (sorting fixes this).
- Using raw prices instead of returns (non-stationary emissions).
- Reading log-likelihood across different datasets - it is only comparable on the same data.

## 9. One-line summary
"A 3-state Gaussian HMM converges (log-likelihood 3018) on 1,000 daily returns; Viterbi decoding plus volatility-sorted states cleanly separates bull, sideways, and bear regimes used for the price overlay."

---

# 09 – Support Vector Machines (SVM)

## 0. In one paragraph
SVC is demonstrated with linear, polynomial and RBF kernels on two interleaving moons, then applied to Breast Cancer (RBF, 0.9790 test accuracy with only 96/426 support vectors). SVR with an ε-insensitive tube fits a noisy 1D sinusoid (R² 0.9864), and the plotted tube shows how only 18/120 points actually constrain the regression curve.

## 1. Quick facts
| | |
|---|---|
| **Algorithms** | SVC (linear/poly/RBF kernels), SVR (RBF, ε-insensitive tube) |
| **Datasets** | Two-moons (250 x 2); Breast Cancer (569 x 30); 1D sinusoid + trend + noise (120) |
| **Key parameters** | SVC C=1.0, gamma='scale', degree=3; SVR C=20, epsilon=0.18, gamma=0.5 |
| **Results** | SVC (RBF) cancer accuracy 0.9790, 96/426 SVs; SVR R² 0.9864, RMSE 0.1157, 18/120 SVs |

## 2. How the algorithm works
**SVC (soft margin):** minimize ½‖w‖² + C·Σξᵢ subject to yᵢ(wᵀφ(xᵢ)+b) ≥ 1−ξᵢ, ξᵢ ≥ 0. The dual problem depends only on inner products K(xᵢ,xⱼ) - this enables the **kernel trick** (compute similarity in a high-dimensional space without mapping data there). The solution is sparse: only points with αᵢ > 0 (**support vectors**) define the boundary.

**Kernels:** linear K = xᵀz; polynomial K = (γxᵀz + r)^d; RBF K = exp(−γ‖x−z‖²).

**SVR:** same machinery for regression with an **ε-insensitive tube** - residuals within ±ε cost nothing; only points outside the tube get αᵢ > 0. Objective: ½‖w‖² + C·Σ(ξᵢ + ξᵢ*).

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| `C` | 1.0 (SVC), 20 (SVR) | violation penalty | narrow margin, overfit | wide margin, underfit |
| `gamma` (RBF) | 'scale' / 0.5 | kernel width | local, wiggly boundary | smooth, linear-like |
| `kernel` | linear/poly/rbf | feature mapping | — | linear = no mapping |
| `degree` (poly) | 3 | polynomial order | more complex | simpler |
| `epsilon` (SVR) | 0.18 | tube half-width | fewer SVs, coarser fit | more SVs, tighter fit |

## 4. Step-by-step implementation (with key code)
1. **Moons experiment** - one SVC per kernel:
```python
clf = SVC(kernel=k_name, C=1.0, gamma='scale', degree=3, random_state=42)
clf.fit(X_2d_scaled, y_2d)
Z = clf.decision_function(...)                 # draw boundary Z=0 and margins Z=±1
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], ...)  # highlight SVs
```
2. **Breast Cancer (real data):**
```python
svc_rbf = SVC(kernel='rbf', C=1.0, random_state=42)
svc_rbf.fit(X_train_c_s, y_train_c)
print(svc_rbf.score(X_test_c_s, y_test_c), len(svc_rbf.support_))
```
3. **SVR on the 1D signal:**
```python
svr_model = SVR(kernel='rbf', C=20.0, epsilon=0.18, gamma=0.5)
svr_model.fit(X_svr, y_svr)
# plot prediction, ±ε tube, and support vectors (svr_model.support_)
```
4. Report R², RMSE, support-vector percentage.

## 5. Results & interpretation
- Moons: the linear kernel cannot separate the classes; poly and RBF curve the boundary; SV counts (gold circles) show the sparse solution.
- Breast Cancer: 0.9790 test accuracy; 96/426 training points are support vectors (22.5%) → 77% of the data could be discarded without changing the model.
- SVR: R² 0.9864 (98.6% variance explained), RMSE 0.1157; 18/120 SVs (15%) anchor the curve; the ε=0.18 tube contains most residuals.

## 6. Limitations & how to improve
- Does not scale well to very large n (kernel matrix is O(n²)); use LinearSVC/SGD or Nyström approximation.
- Requires feature scaling and careful C/γ tuning (grid search).
- No native probability output (Platt scaling needed; used in notebook 03).
- Improvements: kernel selection by CV, class weights for imbalance, ν-SVC for automatic margin control.

## 7. Viva questions
- **What is a support vector?** Training point on/inside the margin with α > 0; defines the boundary.
- **Role of C?** Penalty for margin violations - high C = strict, low C = tolerant.
- **Role of γ?** RBF width: large γ = very local; small γ = smooth.
- **What is the kernel trick?** Inner products in feature space without explicit mapping.
- **Why is the solution sparse?** Only support vectors have non-zero dual coefficients.
- **What is the ε-tube?** Zero-loss zone in SVR; only outside points become support vectors.
- **Why haversine distance for earthquakes?** Latitude/longitude are spherical coordinates; haversine gives true great-circle distances (radians input; 0.03 rad ≈ 191 km).
- **Why scale features?** The kernel uses Euclidean distance; unscaled features distort it.
- **SVC vs SVR?** Classification (separating hyperplane) vs regression (tube around a function).

## 8. Common mistakes
- Not scaling → RBF behaves poorly.
- Setting γ too large and reporting overfit boundaries.
- Judging SVR with accuracy (it is regression: use R²/RMSE).

## 9. One-line summary
"SVC's three kernels demonstrate the kernel trick on moons; the RBF model reaches 0.979 on Breast Cancer using only 22% support vectors, and SVR fits a noisy sinusoid (R² 0.986) with an explicit ε-tube containing most residuals."

---

# 10 – Large Language Models (LLM)

## 0. In one paragraph
Three LLM capabilities are demonstrated with Hugging Face transformers: subword tokenization (DistilBERT), zero-shot sentiment inference (SST-2 pipeline), and controllable generation (DistilGPT2 greedy vs nucleus sampling). Finally, the pretrained classifier is fine-tuned for 3 epochs on 8 domain queries (Technical vs Billing), after which both held-out queries are classified correctly (88.3% / 84.0% confidence).

## 1. Quick facts
| | |
|---|---|
| **Models** | `distilbert-base-uncased-finetuned-sst-2-english`, `distilgpt2` |
| **Libraries** | `transformers`, `torch` |
| **Tasks** | Tokenization, sentiment inference, text generation, 2-class fine-tuning |
| **Key parameters** | Generation: max_new_tokens=40, T=0.7, top_p=0.9. Fine-tune: 3 epochs, lr 1e-4, AdamW, batch 4 |
| **Results** | Held-out queries: Billing 88.3%, Technical 84.0%; fine-tune loss 2.663 → 0.041 → 0.069 |

## 2. How the algorithm works
**Tokenization:** text → subword tokens (WordPiece/BPE) → integer IDs; special tokens [CLS]/[SEP]; padding with an attention mask (1 = real, 0 = pad) so padding does not affect attention.

**Transformer self-attention:** Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)·V - every token attends to every other token; stacked blocks build contextual representations.

**Causal generation:** predict the next token from all previous ones: P(t_n | t_1…t_{n−1}) = softmax(z_n / T).
- Greedy: always argmax → deterministic, can loop.
- Temperature T: <1 sharpens, >1 flattens the distribution.
- Top-p (nucleus): keep the smallest set of tokens with cumulative probability ≥ p, sample inside it.

**Fine-tuning:** load pretrained weights, replace the classification head (2 labels), train briefly on task data with a small learning rate (1e-4) so pretrained knowledge is preserved.

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| `max_new_tokens` | 40 | generation length | longer text, more compute | shorter output |
| `temperature` | 0.7 | sampling randomness | more creative/chaotic (>1) | more deterministic (<1) |
| `top_p` | 0.9 | nucleus size | more diverse | more conservative |
| `lr` (fine-tune) | 1e-4 | adaptation step | faster but forgetting/overfit | too slow to learn |
| `epochs` | 3 | fine-tune passes | overfit on small data | underfit |
| `batch_size` | 4 | samples per step | smoother gradients | noisy, regularizing |

## 4. Step-by-step implementation (with key code)
1. **Tokenize** with padding/truncation and inspect tokens/IDs/mask:
```python
encoded = tokenizer(text, padding="max_length", max_length=20, truncation=True, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
```
2. **Sentiment inference** with a pipeline:
```python
classifier = pipeline("sentiment-analysis", model=model_id, device=-1)
predictions = classifier(test_sentences)      # label + confidence
```
3. **Generation** with explicit configs:
```python
generator = pipeline("text-generation", model="distilbert/distilgpt2",
                     device=-1, clean_up_tokenization_spaces=False)
greedy  = generator(prompt, generation_config=GenerationConfig(max_new_tokens=40, do_sample=False))
sampled = generator(prompt, generation_config=GenerationConfig(
            max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9))
```
4. **Fine-tune** the classifier on 8 labeled queries:
```python
ft_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, ignore_mismatched_sizes=True)
optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1e-4)
outputs = ft_model(input_ids=ids, attention_mask=mask, labels=labels)   # loss inside
outputs.loss.backward(); optimizer.step()                               # 3 epochs
```
5. **Evaluate** on unseen queries with softmax probabilities.

## 5. Results & interpretation
- Tokenizer: 20 slots → 13 real tokens + [CLS]/[SEP] + padding; the attention mask marks the real part.
- Sentiment: confident scores on clearly positive/negative sentences; the model's pretrained knowledge transfers without any training.
- Generation: greedy produced the deterministic "the way we think about the world"; nucleus sampling produced a different but coherent continuation - the effect of T/top-p.
- Fine-tuning: loss 2.663 → 0.041 → 0.069 over 3 epochs; both held-out queries correctly classified (Billing 88.3%, Technical 84.0%).
- The tiny dataset (8 examples) makes this a **mechanics** demo, not a production classifier.

## 6. Limitations & how to improve
- 8 training examples → overfitting risk; use hundreds of examples for real tasks.
- Small models (DistilBERT/DistilGPT2) hallucinate and have limited reasoning.
- Generation quality depends on decoding settings; production systems use better checkpoints + repetition penalties.
- Improvements: LoRA/adapters for cheaper fine-tuning, larger datasets, evaluation on a proper test split.

## 7. Viva questions
- **Why subword tokenization?** Fixed vocabulary can represent any word, including unseen ones.
- **What does the attention mask do?** Excludes padding tokens from attention.
- **Greedy vs top-p?** Deterministic argmax vs sampling from the nucleus; top-p adds diversity.
- **What is temperature?** Logit scaling before softmax; controls randomness.
- **What changed during fine-tuning?** The classification head was replaced and all weights adapted slightly (lr=1e-4).
- **Why `ignore_mismatched_sizes=True`?** The pretrained head's size/labels differ; a fresh 2-class head is initialized.
- **Why so few epochs?** Small dataset; more epochs would overfit (and 3 sufficed to separate the classes).
- **Is this training from scratch?** No - transfer learning from pretrained weights.

## 8. Common mistakes
- Assuming the pipeline's label IDs match your label mapping (always verify).
- Fine-tuning with a large learning rate → catastrophic forgetting.
- Reporting training accuracy on the 8 fine-tune samples instead of held-out queries.

## 9. One-line summary
"LLM pipeline demo: tokenization + attention mask, sentiment inference, greedy vs top-p generation, and a 3-epoch fine-tune on 8 examples that correctly classifies both held-out domain queries."

---

# 11 – Generalized Regression Neural Network (GRNN)

## 0. In one paragraph
A GRNN (Specht, 1991 - equivalent to Nadaraya-Watson kernel regression) is implemented from scratch in NumPy. It has no gradient training: it stores the data and predicts a Gaussian-weighted average of targets. The only hyperparameter σ is selected by 5-fold CV (σ=0.069), giving a smooth fit of a noisy 1D function with test R² 0.9473 - essentially at the noise floor.

## 1. Quick facts
| | |
|---|---|
| **Algorithm** | Generalized Regression Neural Network (from scratch, scikit-learn-style API) |
| **Dataset** | Synthetic 1D: y = sin(2x) + cos(0.5x²) + N(0, 0.18²), 100 points in [−3, 3] |
| **Key parameters** | σ selected by 5-fold CV over 0.05…1.0 → σ=0.069 (CV RMSE 0.2137) |
| **Results** | Final 75/25 split: train R² 0.9829, **test R² 0.9473**, test RMSE 0.1718 |

## 2. How the algorithm works
Four layers:
1. **Input:** query vector x.
2. **Pattern layer:** one Gaussian unit per training point: p_i(x) = exp(−‖x−x_i‖²/(2σ²)).
3. **Summation layer:** S(x) = Σᵢ yᵢ·p_i(x) (numerator), D(x) = Σᵢ p_i(x) (denominator).
4. **Output:** ŷ(x) = S(x)/D(x) - the Nadaraya-Watson conditional-mean estimate.

Training is one-pass: just store the data. Prediction cost is O(N·d) per query. σ controls the bias-variance trade-off: small σ memorizes noise (high variance), large σ over-smooths (high bias).

## 3. Parameters & tuning
| Parameter | Value | Effect | If increased | If decreased |
|---|---|---|---|---|
| `sigma` | 0.069 (CV) | smoothing spread | smoother, higher bias | wiggly, higher variance |
| CV folds | 5 | reliability of σ estimate | more stable, slower | noisier estimate |
| σ grid | 0.05…1.0 (50 pts) | search resolution | finer optimum | coarser |

## 4. Step-by-step implementation (with key code)
1. **Define the class:**
```python
class GRNN(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.X_train_, self.y_train_ = np.asarray(X), np.asarray(y).ravel()   # memorize
        return self
    def predict(self, X):
        dist_sq = ...                          # ||x - x_i||^2 via the expansion trick
        kernels = np.exp(-dist_sq / (2.0 * self.sigma ** 2))    # pattern layer
        D = np.sum(kernels, axis=1)            # denominator
        S = np.dot(kernels, self.y_train_)     # numerator
        return S / np.where(D < 1e-12, 1e-12, D)
```
2. **σ behavior demo:** fit σ = 0.05 / 0.35 / 1.50 and plot against the true curve - shows under/over-smoothing.
3. **σ selection:**
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for sig in sigma_candidates:               # 50 values
    fold_rmses = [root_mean_squared_error(y_va, GRNN(sigma=sig).fit(X_tr, y_tr).predict(X_va))
                  for train_idx, val_idx in kf.split(X_syn)]
    cv_scores.append(np.mean(fold_rmses))
best_sigma = sigma_candidates[np.argmin(cv_scores)]     # 0.069
```
4. **Final fit + holdout evaluation** (75/25): train R², test R², test RMSE.

## 5. Results & interpretation
- CV curve: clear minimum at σ = 0.069 (CV RMSE 0.2137) - small but not tiny, because the 100 points are dense.
- σ=0.05 fit wiggles through noise (overfit); σ=1.50 nearly flat (underfit); σ=0.069 follows the true function.
- Test R² 0.9473 = ~95% of held-out variance explained; RMSE 0.1718 ≈ noise std 0.18 → at the noise limit.
- Train R² 0.9829 > test R² - small, expected generalization gap.

## 6. Limitations & how to improve
- Prediction is O(N) per query; large datasets need approximations (kernels, k-d trees).
- A single global σ performs poorly when dimensions/features have different scales - normalize features or use adaptive σ per dimension.
- Curse of dimensionality: kernel methods degrade in high dimensions.
- Improvements: feature scaling, anisotropic (per-feature) σ, sparse/approximate kernels, or combine with a linear term.

## 7. Viva questions
- **How is GRNN trained?** One pass - it stores the data; no backpropagation.
- **Why is it called a neural network?** Four-layer radial-basis structure (input/pattern/summation/output).
- **What does σ control?** Smoothing: small = variance/overfit, large = bias/underfit.
- **What is it mathematically?** Nadaraya-Watson kernel regression estimating E[y|x].
- **Why 5-fold CV?** To select the single hyperparameter without touching the test set.
- **Why does the final test R² stop at ~0.95?** The data has irreducible noise (σ_noise=0.18); RMSE matches it.
- **Is it parametric?** No - it is non-parametric/memory-based.
- **Why did CV choose 0.069 and not smaller?** With 100 dense points, smaller σ would start fitting noise in validation folds.

## 8. Common mistakes
- Confusing GRNN with a standard MLP (no weights/gradients; kernel memory).
- Selecting σ on the test set instead of via CV.
- Forgetting feature scaling before computing Euclidean distances.

## 9. One-line summary
"GRNN implemented from scratch: one-pass training, σ=0.069 chosen by 5-fold CV, and a smooth fit at the noise floor of the data (test R² 0.9473)."

---

# Appendix A – Metrics cheat sheet

| Metric | Used in | What it measures | Range | How to judge |
|---|---|---|---|---|
| **Silhouette** | 01, 02 | (b−a)/max(a,b): cohesion vs separation | −1…1 | >0.5 good, 0.25-0.5 weak, <0 overlapping |
| **Davies-Bouldin** | 01 | average similarity between each cluster and its most similar one | ≥0 | lower better; <1 good |
| **Calinski-Harabasz** | 01 | between-cluster vs within-cluster dispersion | ≥0 | higher better; compare models only |
| **ARI** | concept only (no labeled clustering in this repo) | chance-corrected agreement with ground truth | −0.5…1 | 0 = random, >0.5 strong, 1 perfect |
| **Accuracy** | 03, 04, 05, 09 | fraction correct | 0…1 | >0.9 strong on these datasets |
| **Precision** | 05 | TP/(TP+FP) | 0…1 | high when false positives are costly |
| **Recall** | 05 | TP/(TP+FN) | 0…1 | high when misses are costly |
| **F1** | 03, 04, 05 | harmonic mean of precision & recall | 0…1 | >0.9 excellent; robust to imbalance |
| **ROC-AUC** | 04 | P(random positive ranked above random negative) | 0…1 | 0.5 random, 0.7-0.8 acceptable, >0.8 good |
| **R²** | 04, 11 | fraction of target variance explained | −∞…1 | >0.7 good for noisy data; can be negative |
| **RMSE** | 04, 06, 09, 11 | √MSE in target units | ≥0 | lower better; same-target comparisons only |
| **MAE** | 04, 06 | mean absolute error | ≥0 | lower better; robust to outliers |
| **MSE loss** | 06 | squared error objective | ≥0 | decreasing/stable curve |
| **Cross-entropy** | 05, 10 | −Σy·ln(ŷ) objective | ≥0 | decreasing/stable curve |
| **Quantization error** | 07 | avg distance sample → BMU | ≥0 | lower better; relative improvement matters |
| **Topographic error** | 07 | fraction of non-adjacent top-2 BMUs | 0…1 | <0.05 good, 0 perfect |
| **Log-likelihood** | 08 | how well the HMM explains the sequence | −∞…+∞ | higher (less negative) better; same data only |

**Choosing a metric:** imbalanced classification → F1 or ROC-AUC (not accuracy); regression → R² for interpretation + RMSE for real error size; clustering without labels → silhouette / Davies-Bouldin; density clustering with labels → ARI.

# Appendix B – Validation / cross-validation used

| Notebook | Scheme | Why |
|---|---|---|
| 01 | Elbow + silhouette over K=2..10 | No labels: choose K by internal quality |
| 02 | k-distance graph (knee) | Heuristic to set ε without labels |
| 03 | Stratified holdout 70/30 + 15% labels | Simulate label scarcity; single clean test |
| 04 | Holdout + OOB (RFR) | OOB is free validation for bagging |
| 05 | Early-stopping validation split (15%) | Stop training when validation stalls |
| 06 | Chronological 80/20 | Time series must not shuffle |
| 07 | None (QE/TE internal metrics) | Unsupervised self-assessment |
| 08 | None (EM convergence + log-likelihood) | Unsupervised; monitor assures convergence |
| 09 | Holdout test set | Simple unbiased estimate |
| 10 | Held-out domain queries | Checks transfer beyond the 8 fine-tune samples |
| 11 | **5-fold CV** over the σ grid | Tune the only hyperparameter reliably |

**Summary:** only notebook 11 uses explicit k-fold CV; 05 uses a single validation split; 04 uses OOB; the rest use holdouts or internal heuristics.

# Appendix C – Datasets used

| Data | Notebook | Size | Target / use |
|---|---|---|---|
| Mall Customers (`data/mall_customers.csv`) | 01 | 200 x 5 | Cluster income + spending |
| USGS earthquakes 2023 (`data/earthquakes.csv`) | 02 | 16,190 (7,638 with M ≥ 4.5 used) | Lat/lon epicenters -> density clustering |
| Breast Cancer (sklearn) | 03, 09 | 569 x 30 | Malignant/benign |
| California Housing (sklearn, 5k sample) | 04 | 5,000 x 8 | Median house value |
| Heart Disease (`data/heart_disease.csv`) | 04 | 303 x 13 | Disease 0/1 |
| Digits (sklearn) | 05 | 1,797 x 64 | Digit 0-9 |
| Airline Passengers (`data/airline_passengers.csv`) | 06 | 144 months | Next-month passengers |
| Wine (sklearn) | 07 | 178 x 13 | 3 cultivars (visualization) |
| Market Regimes (`data/market_regimes.csv`) | 08 | 1,000 days | Latent regime |
| Synthetic moons / sinusoid | 09 | 250 / 120 | Classification / regression |
| DistilBERT, DistilGPT2 (HF Hub) | 10 | pretrained | Sentiment, generation, domain |
| Synthetic 1D function | 11 | 100 | Continuous regression |

# Appendix D – Cross-cutting questions

- **Why scale features?** Distance/gradient-based models need comparable ranges: StandardScaler for clustering/SVM/MLP; MinMaxScaler [0,1] for SOM/RNN (matches activation and weight ranges).
- **Why `random_state=42`?** Reproducibility - identical results on every machine.
- **How is leakage avoided?** Scalers fit on train only; chronological split for time series; stratified splits; CV only on training data.
- **How is overfitting checked?** Train vs validation/test gaps: RFR OOB 0.7513 vs test 0.7415; GRNN train 0.983 vs test 0.947; MLP validation ≈ test.
- **Parameter vs hyperparameter?** Parameters are learned (weights, centroids); hyperparameters are chosen (K, ε, σ, C, γ, τ, layers).
- **Which model was best?** Task-dependent: XGBoost/CatBoost for tabular regression; PyTorch MLP for digits; LSTM for sequences; RBF SVC for small high-dimensional classification; GRNN for smooth low-dimensional regression.
- **Why multiple metrics?** Each captures a different failure mode (e.g., accuracy hides class imbalance; RMSE weights outliers more than MAE).

# Appendix E – Rapid-fire 30-second answers

1. **01** - "Four clustering algorithms on scaled income/spending; K=5 by elbow+silhouette; K-Means/FCM best (silhouette 0.55), Bisecting gives balanced sizes."
2. **02** - "DBSCAN/HDBSCAN on 7,638 real USGS earthquakes (2023, M≥4.5): ε=0.03 rad (~191 km) from the k-distance knee; 55 clusters/13.5% noise (DBSCAN) and 40/21.5% (HDBSCAN); largest clusters match Philippines, Papua New Guinea, Tonga, Japan, Turkey."
3. **03** - "With 15% labels, self-training lifted SVC accuracy 0.9298 → 0.9415, below the fully supervised 0.9766 ceiling."
4. **04** - "RFR/XGBoost/AdaBoost/CatBoost on housing (XGB/Cat R² ≈ 0.80) plus RFC on heart disease (accuracy 0.78, AUC 0.87)."
5. **05** - "MLP on 8x8 digits: sklearn 96.3% with early stopping; custom PyTorch with BatchNorm+Dropout 98.9% test accuracy."
6. **06** - "RNN vs LSTM on 12-month airline windows: LSTM RMSE 42.2 vs 70.8 passengers - gates beat vanishing gradients."
7. **07** - "12x12 SOM on wine: quantization error 0.196, topographic error 0.039; cultivars separate on the U-matrix without labels."
8. **08** - "3-state Gaussian HMM on 1,000 daily returns; Baum-Welch converged (log-lik 3018); Viterbi decodes bull/sideways/bear regimes."
9. **09** - "SVC kernels on moons; RBF SVC 0.979 on breast cancer with 22% support vectors; SVR fits a noisy sine (R² 0.986) with an ε=0.18 tube."
10. **10** - "DistilBERT tokenization and sentiment, DistilGPT2 greedy vs top-p generation, 3-epoch fine-tune correctly classifies held-out technical/billing queries."
11. **11** - "GRNN from scratch: one-pass training, σ=0.069 by 5-fold CV, test R² 0.947 - at the noise floor of the data."
