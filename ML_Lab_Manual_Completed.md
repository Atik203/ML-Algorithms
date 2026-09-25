---
title: "Machine Learning Laboratory Manual — Completed Report"
---

# Machine Learning Laboratory Manual — Completed Report

**Course:** Machine Learning Laboratory
**Department of Computer Science & Engineering**

| Field | Entry |
|---|---|
| Student Name | ____________________________ |
| Student ID | ____________________________ |
| Section | ____________________________ |
| Date | ____________________________ |

*Completed following the laboratory manual format of Dr. Ohidujjaman Tuhin, Dept. of CSE, UIU.*

**Environment.** Python 3.12 (`.venv`), NumPy/Pandas, scikit-learn, XGBoost, CatBoost, PyTorch 2.14 (CUDA on RTX 3060 Laptop GPU), Hugging Face `transformers`, MiniSom, hmmlearn. Every experiment is fully implemented in the executed Jupyter notebooks (`notebooks/01`-`11`); results below are the actual outputs. All random seeds are fixed (`random_state=42` / `random_seed=42` / `torch.manual_seed(42)`).

**Note on datasets.** All experiments use real, public datasets committed under `data/`: Mall Customers, USGS earthquakes 2023, Breast Cancer, California Housing, Heart Disease, Digits, Airline Passengers, Wine, DAX index (EuStockMarkets), Old Faithful geyser, Motorcycle accelerometer (MASS mcycle), SMS Spam Collection.

---

# Experiment 1: K-means Clustering

**Category:** Clustering

## 1. Objective
Partition unlabeled observations into K clusters by iteratively assigning points to the nearest centroid and updating centroids; select K using the elbow and silhouette diagnostics.

## 2. Dataset Used
Mall Customers (`data/mall_customers.csv`), 200 customers, two features: Annual Income (k$) and Spending Score (1-100). Features are standardized (`StandardScaler`).

## 3. Required Libraries
numpy, pandas, matplotlib, scikit-learn

## 4. Brief Theory
K-means minimizes the within-cluster sum of squares J = Σ_k Σ_{x∈C_k} ||x − μ_k||². It alternates assignment and centroid update until convergence (a local optimum). Euclidean distance is scale-sensitive, so features are standardized. K is chosen externally, e.g. by the elbow (inertia) and silhouette (cohesion vs separation).

## 5. Procedure
1. Load the dataset and keep the two numerical features.
2. Standardize the features.
3. Try K = 2…10; record inertia (WCSS) and silhouette for each K.
4. Choose K at the elbow / silhouette peak and fit the final model with `init='k-means++'`, `n_init=20`, `random_state=42`.
5. Visualize clusters and centroids.
6. Report silhouette, Davies-Bouldin and Calinski-Harabasz.

## 6. Reference Implementation
```python
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_mall_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_mall_scaled, kmeans.labels_))
# final model
kmeans_model = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X_mall_scaled)
```

## 7. Results
| Metric | Value |
|---|---|
| Selected K (elbow + silhouette peak) | 5 |
| Silhouette Score | 0.5547 |
| Davies-Bouldin Index | 0.5722 |
| Calinski-Harabasz Score | 248.65 |

The elbow curve flattens after K = 5 and the silhouette curve peaks at K = 5, so K = 5 is selected. Silhouette ≈ 0.55 indicates compact, reasonably separated clusters on real data.

## 8. Discussion and Conclusion
K-means recovers a clear 5-segment customer structure. The result is stable because K-Means++ initialization avoids poor local minima and `n_init=20` keeps the best run. Standardization matters: without it, income (0-137) dominates spending (1-99). K-means assumes roughly spherical clusters of similar size — a limitation addressed by the density-based and fuzzy methods later.

## 9. Answers to Viva Questions
- **Why does K-means depend on feature scaling?** It minimizes Euclidean distance; unscaled features with larger ranges dominate the distance and therefore the assignments.
- **What is inertia?** The within-cluster sum of squared distances to centroids (WCSS); lower is tighter, but it always decreases with K, so it is not a standalone quality score.
- **Why can a high silhouette score still be misleading?** A high score can come from a simple/trivial partition (e.g., one big dense cluster), and silhouette favors compact convex clusters — it can misjudge elongated or density-varying structures.
- **What happens when clusters are non-spherical?** K-means splits them with straight Voronoi boundaries and merges elongated clusters incorrectly (demonstrated with real seismic data in Experiment 5).

---

# Experiment 2: Modified K-means

**Category:** Clustering

## 1. Objective
Implement the prescribed transparent modification of K-means: K-Means++ initialization plus an explicit outlier-distance check with a documented percentile threshold before final cluster interpretation.

## 2. Dataset Used
Mall Customers (200 × 2, standardized). The modification is applied on top of the same features.

## 3. Required Libraries
numpy, matplotlib, scikit-learn

## 4. Brief Theory
"Modified K-means" is not a single standard algorithm; the manual defines it precisely for reproducibility: initialize with K-Means++, fit K-means, compute each sample's distance to its assigned centroid, take a percentile threshold (97th percentile) and flag more distant samples as candidate outliers. It is an experimental variant that adds outlier awareness, not a replacement for robust clustering.

## 5. Procedure
1. Fit K-means with `init='k-means++'`, K = 5.
2. Compute d_i = ||x_i − μ_{c_i}|| for every sample.
3. Set threshold = 97th percentile of the training distances.
4. Flag samples with d_i > threshold as candidate outliers.
5. Plot centroids/clusters with flagged points highlighted.
6. Compare with the standard labels.

## 6. Reference Implementation
```python
km_mod = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
mod_labels = km_mod.fit_predict(X_mall_scaled)
dist_to_centroid = np.linalg.norm(X_mall_scaled - km_mod.cluster_centers_[mod_labels], axis=1)
threshold = np.percentile(dist_to_centroid, 97)
outlier_flag = dist_to_centroid > threshold
```

## 7. Results
| Item | Value |
|---|---|
| Distance threshold (97th percentile) | 1.072 (standardized units) |
| Flagged candidate outliers | 6 of 200 (3%) |
| Cluster labels | identical to standard K-means (same partition) |

The flagged customers are the most weakly attached points of their clusters — typically boundary customers with unusual income/spending combinations. The cluster structure itself is unchanged, which is expected: the modification adds an outlier interpretation layer.

## 8. Discussion and Conclusion
The variant is useful when the analyst needs to know which observations are poorly represented by any centroid (e.g., unusual customers). Because the threshold is a documented percentile, the flag rate is controlled and reproducible (3% here). The limitation is that distance-to-centroid is biased for elongated or density-varying clusters, and the flagged points are not automatically errors.

## 9. Answers to Viva Questions
- **Why is it important to define a modification precisely?** Without a formal definition, "modified K-means" is ambiguous; a precise version (init method, threshold rule, percentile) is reproducible and testable.
- **How does K-Means++ differ from random initialization?** K-Means++ selects initial centroids with probability proportional to squared distance to the nearest chosen centroid, spreading seeds out and reducing convergence to poor local optima.
- **Why might distance-to-centroid fail for elongated clusters?** Centroid distance is isotropic; a point can be close to the centroid along the long axis yet far along the short axis, so elongated clusters can flag legitimate points and missing real outliers.

---

# Experiment 3: Hierarchical Clustering

**Category:** Clustering

## 1. Objective
Perform agglomerative hierarchical clustering, interpret the dendrogram, and compare Ward linkage with the other algorithms.

## 2. Dataset Used
Mall Customers (200 × 2, standardized).

## 3. Required Libraries
scipy, scikit-learn, matplotlib

## 4. Brief Theory
Agglomerative clustering starts with one cluster per point and repeatedly merges the closest pair. Ward linkage merges the pair that increases total within-cluster variance the least: ΔW(A,B) = (n_A n_B)/(n_A+n_B) ||μ_A − μ_B||². The dendrogram shows all merges; a horizontal cut determines the number of clusters.

## 5. Procedure
1. Standardize the features.
2. Compute the Ward linkage matrix.
3. Plot the dendrogram (truncated for readability) and choose a cut level.
4. Fit `AgglomerativeClustering(n_clusters=5, linkage='ward')`.
5. Evaluate with silhouette, Davies-Bouldin and Calinski-Harabasz.
6. Discuss linkage choice.

## 6. Reference Implementation
```python
linkage_matrix = linkage(X_mall_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=25, leaf_rotation=45, show_contracted=True)
agg_model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
agg_labels = agg_model.fit_predict(X_mall_scaled)
```

## 7. Results
| Metric | Value |
|---|---|
| Silhouette Score | 0.5538 |
| Davies-Bouldin Index | 0.5779 |
| Calinski-Harabasz Score | 244.41 |

The dendrogram shows a clear large merge gap that supports cutting at five clusters. The partition is nearly identical in quality to K-means (silhouette difference < 0.001).

## 8. Discussion and Conclusion
Ward linkage produces compact, balanced clusters and requires no initialization, at the cost of O(n²) memory and the need to choose a cut. Single linkage would chain across the space and complete linkage would fragment; Ward is the appropriate default for compact clusters of similar size.

## 9. Answers to Viva Questions
- **What does a dendrogram show?** The full merge hierarchy: leaf order, merge heights (distances) and the structure of nested clusters; cutting at a height yields a flat partition.
- **What is linkage?** The rule that defines the distance between two clusters (single = closest pair, complete = farthest pair, average = mean pair, Ward = variance increase).
- **When is Ward linkage inappropriate?** With non-spherical/elongated clusters, unequal cluster sizes or outliers — it assumes compact, similar-variance clusters.

---

# Experiment 4: Fuzzy C-means

**Category:** Clustering

## 1. Objective
Cluster data with soft memberships so each observation can belong to multiple clusters with different degrees; report the fuzzy partition coefficient (FPC).

## 2. Dataset Used
Mall Customers (200 × 2, standardized), K = 5, fuzzifier m = 2.0.

## 3. Required Libraries
numpy, pandas, matplotlib (the c-means equations are implemented from scratch; `scikit-fuzzy` is an equivalent library implementation)

## 4. Brief Theory
Fuzzy C-means minimizes J_m = Σ_i Σ_k u_ik^m ||x_i − v_k||² with memberships u_ik ∈ [0,1], Σ_k u_ik = 1. It alternates:
v_k = Σ_i u_ik^m x_i / Σ_i u_ik^m and u_ik = 1 / Σ_j (d_ik/d_ij)^(2/(m−1)).
The fuzzifier m controls softness (m → 1 becomes K-means). FPC = (1/N) Σ_i Σ_k u_ik² measures fuzziness (1 = hard, 1/K = maximally fuzzy).

## 5. Procedure
1. Standardize the data.
2. Initialize the membership matrix randomly (rows sum to 1).
3. Alternate centroid and membership updates until max|ΔU| < 1e-5.
4. Derive hard labels via argmax membership.
5. Report metrics and FPC; visualize memberships/hard partition.
6. Discuss ambiguous samples and m.

## 6. Reference Implementation
```python
Um = U ** self.m
centers = np.dot(Um.T, X) / Um.sum(axis=0)[:, np.newaxis]
dist = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2) + 1e-10
U = (1.0 / dist) ** (2.0 / (self.m - 1.0))
U = U / U.sum(axis=1, keepdims=True)
labels = np.argmax(U, axis=1)
```

## 7. Results
| Metric | Value |
|---|---|
| Silhouette Score (hard labels) | 0.5547 |
| Davies-Bouldin Index | 0.5722 |
| Fuzzy Partition Coefficient (FPC) | 0.6711 |
| Iterations to convergence | < 50 |

FPC = 0.67 (between 1/K = 0.2 and 1.0) shows the partition is mostly crisp but retains soft boundary memberships. Customers with membership vectors close to (0.5, 0.5, …) sit between clusters and are the genuinely ambiguous ones.

## 8. Discussion and Conclusion
FCM converges to the same hard partition as K-means here (silhouette 0.5547) because the Mall Customers clusters are well separated; the added value is the membership matrix, which quantifies uncertainty at boundaries. Larger m makes memberships softer (lower FPC) without changing the hard structure much; smaller m (~1.1) approaches K-means.

## 9. Answers to Viva Questions
- **How is Fuzzy C-means different from K-means?** It assigns membership degrees instead of hard labels, so each point belongs to all clusters with weights that sum to 1; K-means is the limiting hard case.
- **What does m control?** The fuzziness exponent: larger m spreads membership across clusters (softer), m approaching 1 gives crisp K-means-like assignments.
- **What does a membership vector represent?** The degree to which one observation belongs to each cluster; near-uniform values indicate an ambiguous point on a cluster boundary.

---

# Experiment 5: DBSCAN

**Category:** Density-based learning

## 1. Objective
Detect dense groups and noise with DBSCAN without specifying the number of clusters, on real spatial data.

## 2. Dataset Used
Real USGS earthquake catalogue 2023 (`data/earthquakes.csv`): 16,190 global events with magnitude ≥ 4.0; the experiment uses the 7,638 events with magnitude ≥ 4.5. Coordinates are converted to radians and clustered with the haversine (great-circle) metric.

## 3. Required Libraries
scikit-learn, numpy, pandas, matplotlib

## 4. Brief Theory
DBSCAN uses a radius ε and minimum points MinPts. Points with ≥ MinPts neighbors within ε are core points; points within ε of a core are border points; the rest are noise (label −1). Clusters are chains of density-reachable core points, so arbitrary shapes can be discovered. A single global ε struggles when densities vary.

## 5. Procedure
1. Load the catalogue and filter to magnitude ≥ 4.5 (well-recorded events).
2. Convert latitude/longitude to radians for the haversine metric.
3. Choose ε from the k-distance graph (k = 10) — the knee at 0.03 rad ≈ 191 km.
4. Fit DBSCAN and count clusters/noise.
5. Show a parameter-sensitivity table for at least two settings.
6. Visualize epicenters and check the largest clusters against known seismic regions.

## 6. Reference Implementation
```python
X_rad = np.radians(quakes[['latitude', 'longitude']].values)
nbrs = NearestNeighbors(n_neighbors=10, metric='haversine').fit(X_rad)
k_distances = np.sort(nbrs.kneighbors(X_rad)[0][:, -1])       # knee ~0.03 rad
dbscan = DBSCAN(eps=0.03, min_samples=10, metric='haversine')
db_labels = dbscan.fit_predict(X_rad)
```

## 7. Results
Parameter sensitivity (min_samples = 10):

| eps (rad) | approx km | clusters | noise % |
|---|---|---|---|
| 0.02 | 127 | 70 | 21.2% |
| **0.03 (selected)** | **191** | **55** | **13.5%** |
| 0.05 | 319 | 50 | 5.7% |

Largest clusters (real-world validation): Philippines (1,629 events), Papua New Guinea (1,068), Tonga (947), Japan (895), South Sandwich Islands (253), Turkey (207). Silhouette on non-noise (3,000-event sample) = 0.326.

## 8. Discussion and Conclusion
DBSCAN recovers the global seismic structure without labels: dense chains along tectonic plate boundaries form clusters, while ~1,000 isolated intraplate events are noise. Smaller ε fragments clusters and labels more events as noise; larger ε merges boundaries and reduces noise. The single global ε is the main limitation, which motivates HDBSCAN.

## 9. Answers to Viva Questions
- **Why does DBSCAN not require K?** Clusters are grown from density-connected core points, so their number and shape emerge from the data instead of being fixed in advance.
- **What does label −1 mean?** The point is neither a core nor a border point — it lies in a sparse region and is treated as noise/outlier (here: isolated earthquakes).
- **Why can DBSCAN struggle with varying densities?** A single ε that is small enough for a dense cluster fragments a sparse one, while one large enough for the sparse cluster merges the dense ones — no global setting fits both.

---

# Experiment 6: HDBSCAN

**Category:** Density-based learning

## 1. Objective
Apply hierarchical density-based clustering to the same spatial data to discover clusters of different densities and identify noise without selecting ε.

## 2. Dataset Used
USGS earthquakes 2023, magnitude ≥ 4.5 (7,638 events), radians + haversine metric.

## 3. Required Libraries
scikit-learn (`HDBSCAN`), numpy, pandas, matplotlib

## 4. Brief Theory
HDBSCAN builds a hierarchy over all density levels: core distance → mutual reachability distance d_mreach(a,b) = max{d_core(a), d_core(b), d(a,b)} → minimum spanning tree → condensed cluster tree, then keeps the most stable clusters S(C) = Σ (λ_p − λ_birth(C)). It also returns membership probabilities, so no global ε is needed.

## 5. Procedure
1. Reuse the prepared radian coordinates.
2. Fit HDBSCAN with `min_cluster_size=50`, `min_samples=10`, `metric='haversine'`.
3. Count clusters and noise; inspect membership probabilities.
4. Visualize the partition.
5. Compare with DBSCAN.
6. Discuss the effect of `min_cluster_size`.

## 6. Reference Implementation
```python
hdb = HDBSCAN(min_cluster_size=50, min_samples=10, metric='haversine', copy=False)
hdb_labels = hdb.fit_predict(X_rad)
```

## 7. Results
| Algorithm | Clusters | Noise events | Largest cluster | Silhouette (3k sample) |
|---|---|---|---|---|
| DBSCAN (ε=0.03) | 55 | 13.5% | 1,629 | 0.326 |
| HDBSCAN | 40 | 21.5% | 560 | **0.631** |

Membership probabilities are high (> 0.9) for events deep inside seismic zones and lower near cluster edges. HDBSCAN's silhouette is roughly double DBSCAN's because adaptive density splits DBSCAN's long arc-shaped clusters (e.g., the whole Japan-Kuril chain) into compact, well-separated groups.

## 8. Discussion and Conclusion
HDBSCAN needs no ε and handles the varying density along plate boundaries better than DBSCAN; it is more conservative (21.5% vs 13.5% noise) because clusters must be stable across density scales. Increasing `min_cluster_size` reduces the number of small clusters; decreasing it fragments. The cost is a more complex algorithm and less transparent parameter meaning.

## 9. Answers to Viva Questions
- **What is hierarchical about HDBSCAN?** It constructs a full hierarchy of density-based clusters across all ε values, then condenses it by tracking cluster birth/death and keeping the most stable clusters.
- **What does membership probability mean?** A per-point confidence that it belongs to its assigned cluster, derived from the point's persistence in the cluster as density thresholds change (low values indicate borderline points).
- **Why is HDBSCAN useful for variable-density data?** Because clusters are selected by stability across density levels rather than by one global ε, so dense and sparse clusters can coexist.

---

# Experiment 7: Self-training for Semi-supervised Learning

**Category:** Semi-supervised learning

## 1. Objective
Train a classifier with a small labeled subset and a larger unlabeled subset by iteratively adding high-confidence pseudo-labels; compare with a supervised baseline and the fully supervised ceiling.

## 2. Dataset Used
Breast Cancer Wisconsin (569 × 30, binary). Stratified 70/30 split: 398 training / 171 test. Only 15% of training labels (61 samples) are kept; 337 samples are masked as unlabeled (−1).

## 3. Required Libraries
scikit-learn, numpy

## 4. Brief Theory
Self-training fits a base classifier on labeled data, predicts class probabilities for the unlabeled pool, converts sufficiently confident predictions (max probability ≥ threshold) to pseudo-labels, adds them to the labeled set and repeats. Errors can propagate (confirmation bias), so the confidence threshold and convergence monitoring matter.

## 5. Procedure
1. Split the data stratified; scale features (fit on training data only).
2. Mask 85% of training labels to −1.
3. Train a supervised baseline SVC on the labeled 61 samples.
4. Wrap an RBF SVC in `SelfTrainingClassifier(threshold=0.80)`, fit on labeled + unlabeled, monitor iterations.
5. Train a fully supervised SVC on all 398 labels as the ceiling.
6. Compare accuracy/F1 on the untouched test set.

## 6. Reference Implementation
```python
self_training_svc = SelfTrainingClassifier(
    estimator=SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
    threshold=0.80, criterion='threshold', max_iter=15, verbose=True)
self_training_svc.fit(X_train_full_scaled, y_train_semi)   # labels + unlabeled (-1)
```

## 7. Results
| Setting | Accuracy | F1 | Training samples |
|---|---|---|---|
| Supervised baseline (15% labeled) | 0.9298 | 0.9469 | 61 |
| Self-training (semi-supervised) | **0.9415** | **0.9554** | 382 |
| Fully supervised ceiling (100%) | 0.9766 | 0.9813 | 398 |

Pseudo-labels added per iteration: +292, +22, +5, +2, then termination `no_change` (321 new labels total). The decaying additions indicate stable convergence rather than runaway error accumulation.

## 8. Discussion and Conclusion
Using no extra human labels, self-training improves accuracy by +1.2 points over the baseline, closing part of the gap to the fully supervised model. The remaining gap is the cost of noisy pseudo-labels. Convergence was quick (4 useful iterations) because the threshold 0.80 admitted mostly correct predictions.

## 9. Answers to Viva Questions
- **What is a pseudo-label?** A predicted label for an unlabeled sample that the model accepts as training target because its confidence exceeds the threshold.
- **What is confirmation bias in self-training?** The model reinforces its own mistakes: wrong confident predictions enter the training set, make the model more confident about similar errors, and can degrade performance. High thresholds and validation monitoring mitigate this.
- **Why must the test labels remain hidden during training?** Otherwise the evaluation is contaminated — test data would influence the model, and the reported accuracy would no longer estimate generalization.

---

# Experiment 8: Random Forest Regression (RFR)

**Category:** Ensemble learning

## 1. Objective
Predict a continuous target using an ensemble of randomized decision trees; evaluate with MAE, RMSE and R² and inspect out-of-bag validation.

## 2. Dataset Used
California Housing (`fetch_california_housing`), 5,000-row random sample, 8 features, target = median house value. 80/20 train/test split (4,000/1,000).

## 3. Required Libraries
scikit-learn, numpy, pandas

## 4. Brief Theory
Random Forest Regression averages predictions of B trees trained on bootstrap samples, each split considering a random feature subset. Averaging decorrelated trees reduces variance. Out-of-bag samples (not drawn in a bootstrap) give a free internal validation estimate.

## 5. Procedure
1. Load and sample the dataset; split train/test.
2. Train `RandomForestRegressor(n_estimators=150, max_depth=12, oob_score=True)`.
3. Predict the test set; compute R², RMSE, MAE.
4. Report the OOB R² and compare with the test R² to check overfitting.

## 6. Reference Implementation
```python
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=12,
                               oob_score=True, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg, y_train_reg)
print(rf_reg.oob_score_, r2_score(y_test_reg, rf_reg.predict(X_test_reg)))
```

## 7. Results
| Metric | Value |
|---|---|
| OOB R² | 0.7513 |
| Test R² | 0.7415 |
| Test RMSE | 0.5908 ($100k units ≈ $59k) |
| Test MAE | 0.3999 ($100k units ≈ $40k) |

OOB R² (0.7513) is very close to the test R² (0.7415), so the model is not overfitting. Feature importances are dominated by median income and location (latitude/longitude), consistent with housing economics.

## 8. Discussion and Conclusion
RFR provides a strong, low-maintenance baseline for tabular regression: no feature scaling is required, it captures non-linearities and interactions, and OOB gives free validation. Gradient boosting (Experiments 10-12) reaches higher R² on the same data at the cost of more tuning.

## 9. Answers to Viva Questions
- **Why does a forest reduce variance?** Averaging B approximately uncorrelated trees gives Var ≈ ρσ² + (1−ρ)σ²/B; as B grows the second term vanishes, so predictions are more stable than a single tree's.
- **What is bagging?** Bootstrap aggregating: train each model on a bootstrap sample of the data and combine predictions (average for regression, vote for classification).
- **Why can feature importance be misleading?** Impurity-based importance is biased toward high-cardinality/continuous features, is computed on training data (no causality), and correlated features share/dilute importance arbitrarily.

---

# Experiment 9: Random Forest Classification (RFC)

**Category:** Ensemble learning

## 1. Objective
Classify observations with an ensemble of randomized decision trees using a stratified split and report accuracy, precision, recall, F1 and the confusion matrix.

## 2. Dataset Used
Heart Disease (`data/heart_disease.csv`, 303 rows, 13 features plus target). Categorical variables are one-hot encoded for the forest; stratified 75/25 split (227/76).

## 3. Required Libraries
scikit-learn, numpy, pandas

## 4. Brief Theory
Random Forest Classification aggregates tree votes: Ĉ(x) = argmax_k Σ_b I(T_b(x) = k). Bootstrap sampling plus feature subsampling decorrelates trees, reducing variance relative to a single decision tree. The class-probability output (fraction of votes) enables ROC-AUC.

## 5. Procedure
1. Load the heart data; one-hot encode categorical features.
2. Use a stratified 75/25 split to preserve the disease ratio.
3. Train `RandomForestClassifier(n_estimators=120, max_depth=8)`.
4. Predict labels and probabilities on the test set.
5. Compute accuracy, F1 and ROC-AUC.

## 6. Reference Implementation
```python
rfc = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
rfc.fit(X_train_clf, y_train_clf)
y_prob_rfc = rfc.predict_proba(X_test_clf)[:, 1]
```

## 7. Results
| Metric | Value |
|---|---|
| Accuracy | 0.7763 |
| Precision / Recall / F1 | 0.8046 (F1) |
| ROC-AUC | 0.8704 |
| Confusion matrix (test, 76 samples) | TN/FP/FN/TP reported in notebook |

ROC-AUC 0.87 indicates good ranking quality on a small dataset; accuracy 0.776 is typical for heart-disease prediction with these features and reflects the problem's noise, not a modelling failure.

## 8. Discussion and Conclusion
RFC gives a robust classifier with minimal preprocessing and interpretable feature importances. On 227 training rows the variance-reduction of bagging is important; class imbalance (about 45% positive here) is mild, but F1 and AUC are safer than accuracy alone.

## 9. Answers to Viva Questions
- **RFR vs RFC?** Regression averages the trees' numeric outputs; classification takes a majority vote (and averages class probabilities for scoring).
- **What is majority voting?** Each tree votes for one class and the most frequent class becomes the ensemble prediction (soft voting averages probabilities instead).
- **How can class imbalance affect accuracy?** With a rare positive class, a model can predict the majority class everywhere and still show high accuracy; precision/recall/F1 and AUC expose the failure.

---

# Experiment 10: XGBoost

**Category:** Ensemble learning / Gradient boosting

## 1. Objective
Train a gradient-boosted tree classifier and analyze the effect of boosting hyperparameters.

## 2. Dataset Used
Heart Disease (`data/heart_disease.csv`), same stratified 75/25 split as RFC for a fair comparison.

## 3. Required Libraries
xgboost, scikit-learn, numpy, pandas

## 4. Brief Theory
XGBoost builds trees sequentially, each fitting the 1st/2nd-order gradients (g_i, h_i) of the loss: L ≈ Σ[g_i f_t(x_i) + ½ h_i f_t²(x_i)] + Ω(f_t), with Ω(f) = γT + ½λΣw_j². Controls: `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.

## 5. Procedure
1. Prepare the one-hot encoded heart data and the same stratified split.
2. Fit `XGBClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')`.
3. Predict labels and probabilities.
4. Report accuracy, F1 and ROC-AUC.
5. Discuss the complexity/generalization trade-off.

## 6. Reference Implementation
```python
xgb_clf = xgb.XGBClassifier(n_estimators=120, learning_rate=0.08, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, eval_metric='logloss')
xgb_clf.fit(X_train_clf, y_train_clf)
y_prob_xgb_c = xgb_clf.predict_proba(X_test_clf)[:, 1]
```

## 7. Results
| Metric | Value |
|---|---|
| Accuracy | 0.7763 |
| F1 | 0.8132 |
| ROC-AUC | 0.8509 |

XGBoost matches RFC on accuracy with a slightly higher F1 but a slightly lower AUC on this small test set (76 samples). Differences of ~0.01-0.02 are within noise for this dataset size.

## 8. Discussion and Conclusion
Gradient boosting is the strongest family for tabular data in general (see regression results: XGBoost R² 0.8025 vs RFR 0.7415), but on this small heart dataset with fixed hyperparameters it does not clearly beat simpler ensembles. Lower learning rates with more trees usually improve generalization at higher training cost.

## 9. Answers to Viva Questions
- **Bagging vs boosting?** Bagging trains models in parallel on bootstrap samples and combines them to reduce variance; boosting trains models sequentially so each corrects the previous ensemble's errors, mainly reducing bias.
- **What does learning rate do?** It shrinks each tree's contribution. Smaller values need more trees but usually generalize better; larger values fit faster and can overfit.
- **Why can boosting overfit?** Later trees keep fitting remaining errors, including label noise, and the ensemble can become too complex; regularization (depth, λ, subsample, early stopping) controls this.

---

# Experiment 11: AdaBoost

**Category:** Ensemble learning

## 1. Objective
Build an adaptive boosting classifier and study how weak learners combine into a stronger ensemble.

## 2. Dataset Used
Heart Disease (`data/heart_disease.csv`), same stratified 75/25 split.

## 3. Required Libraries
scikit-learn, numpy, pandas

## 4. Brief Theory
AdaBoost trains weak learners sequentially on re-weighted data: misclassified samples receive larger weights. At round m the learner weight is α_m = ½ ln((1 − ε_m)/ε_m), and sample weights update as w_i ← w_i exp(−α_m y_i h_m(x_i)). The final prediction is a weighted vote. Exponential loss makes it sensitive to noisy points.

## 5. Procedure
1. Prepare the heart data and stratified split.
2. Fit `AdaBoostClassifier(n_estimators=100, learning_rate=0.1)` (stumps by default).
3. Predict labels and probabilities on the test set.
4. Report accuracy, F1 and ROC-AUC; compare with RFC and XGBoost.

## 6. Reference Implementation
```python
ada_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
ada_clf.fit(X_train_clf, y_train_clf)
y_prob_ada_c = ada_clf.predict_proba(X_test_clf)[:, 1]
```

## 7. Results
| Metric | Value |
|---|---|
| Accuracy | 0.7895 (best of the four classifiers) |
| F1 | 0.8182 (best) |
| ROC-AUC | 0.8714 (best) |

AdaBoost with shallow stumps slightly outperforms the deeper models on this small dataset, which is a common outcome when the signal is mostly additive and the sample size is limited.

## 8. Discussion and Conclusion
AdaBoost remains a strong, cheap baseline: stump ensembles are fast, need little tuning, and here achieve top accuracy/F1/AUC. Its weakness is sensitivity to label noise and outliers (weights grow exponentially for hard points); gradient boosting with regularization is more robust on noisier data.

## 9. Answers to Viva Questions
- **Why use weak learners?** Individually they are simple and low-variance; boosting combines many of them so the ensemble can fit complex boundaries while keeping each step's variance low.
- **How does AdaBoost focus on difficult examples?** After each round, weights of misclassified samples increase, so the next learner optimizes a distribution that emphasizes current errors.
- **What is the effect of noisy labels?** Mislabeled points look "hard", receive ever-larger weights, and can dominate training, degrading accuracy — AdaBoost is known to be noise-sensitive.

---

# Experiment 12: CatBoost

**Category:** Ensemble learning

## 1. Objective
Train a gradient-boosted model with native handling of categorical features and inspect correct categorical declaration.

## 2. Dataset Used
Heart Disease (`data/heart_disease.csv`): continuous features (age, trestbps, chol, thalach, oldpeak) plus categorical features (sex, cp, fbs, restecg, exang, slope, ca, thal). The raw (unencoded) table is used for CatBoost.

## 3. Required Libraries
catboost, pandas, scikit-learn

## 4. Brief Theory
CatBoost uses ordered boosting (leaf values computed on random permutations) to avoid target leakage/prediction shift, and builds oblivious (symmetric) trees. Categorical features are encoded internally with ordered target statistics, so `cat_features` must be declared correctly and the data split must not leak the target.

## 5. Procedure
1. Identify categorical columns in the raw heart table and cast them to string.
2. Use the same stratified 75/25 split on the raw features.
3. Fit `CatBoostClassifier(iterations=150, depth=5, cat_features=categorical_cols)`.
4. Predict labels/probabilities; report accuracy, F1, ROC-AUC.
5. Explain why target-statistic encoding can beat one-hot encoding.

## 6. Reference Implementation
```python
categorical_cols = X_clf_raw.select_dtypes(include=['object', 'category']).columns.tolist()
cb_clf = cb.CatBoostClassifier(iterations=150, learning_rate=0.08, depth=5,
                               cat_features=categorical_cols, random_seed=42, verbose=False)
cb_clf.fit(X_tr_cb, y_tr_cb)
```

## 7. Results
| Metric | Value |
|---|---|
| Accuracy | 0.7632 |
| F1 | 0.7955 |
| ROC-AUC | 0.8551 |

CatBoost performs at a similar level to the other ensembles without any manual one-hot encoding — its native categorical handling did the encoding internally. On this tiny dataset, the advanced target statistics cannot show their full advantage (they matter most with high-cardinality categories, e.g. thousands of city/user IDs).

## 8. Discussion and Conclusion
CatBoost is the ensemble of choice when tables mix numeric and categorical columns: it removes manual encoding, reduces target leakage, and remains strong out-of-the-box. Here its metrics sit within noise of RFC/XGBoost/AdaBoost because the categorical cardinalities are small (2-4 values).

## 9. Answers to Viva Questions
- **How does CatBoost treat categorical features?** It computes ordered target statistics (mean target per category on random permutations with prior smoothing) rather than one-hot vectors.
- **Why should categories not be encoded using target leakage?** Plain target encoding uses each row's own target when encoding that row, which leaks label information and inflates training performance; ordered statistics avoid this.
- **CatBoost vs ordinary one-hot encoding?** One-hot explodes dimensionality for high-cardinality features and loses category statistics; CatBoost encodes compactly and exploits target-category relationships with leakage control.

---

# Experiment 13: Multilayer Perceptron (MLP)

**Category:** Neural networks

## 1. Objective
Implement feed-forward neural networks for handwritten-digit classification and study architecture and regularization.

## 2. Dataset Used
Digits (scikit-learn): 1,797 samples, 8×8 grayscale images (64 features), 10 classes. Stratified 70/15/15 train/validation/test split; standardized (scaler fit on train only).

## 3. Required Libraries
scikit-learn, numpy, pandas, matplotlib, PyTorch (torch)

## 4. Brief Theory
An MLP stacks affine transformations with non-linear activations: a⁽ˡ⁾ = σ(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾), trained by backpropagation to minimize cross-entropy. BatchNorm stabilizes activations, Dropout regularizes, and an LR scheduler adapts the step size.

## 5. Procedure
1. Load digits; split 70/15/15; standardize.
2. Train `MLPClassifier(hidden_layer_sizes=(128,64), ReLU, Adam, alpha=0.001, early_stopping=True)`.
3. Build a custom PyTorch MLP: 64→128→64→10 with BatchNorm + Dropout(0.25/0.20), CrossEntropyLoss, Adam (lr 0.003, weight decay 1e-4), 50 epochs, batch 32, ReduceLROnPlateau.
4. Train with the validation loop; plot the loss curve.
5. Evaluate the test set; report the classification report.

## 6. Reference Implementation
```python
mlp_sklearn = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                            alpha=0.001, batch_size=64, learning_rate_init=0.005,
                            max_iter=150, early_stopping=True, validation_fraction=0.15)
# PyTorch
self.net = nn.Sequential(
    nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
    nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.20),
    nn.Linear(64, 10))
```

## 7. Results
| Model | Test accuracy | Notes |
|---|---|---|
| Scikit-Learn MLP (128-64) | 0.9630 | early stopping after 15 iterations |
| PyTorch Deep MLP | **0.9815** | GPU (CUDA), 50 epochs, val acc 0.9889 |

Training loss fell from 1.20 to ~0.02 while validation loss tracked it closely (no severe overfitting). Per-class precision/recall are near 1.00 for almost all digits.

## 8. Discussion and Conclusion
The custom PyTorch model beats the sklearn baseline by ~2 points thanks to BatchNorm, Dropout and scheduled training over more epochs. With only 1,797 images, regularization is the key; a CNN would be the natural next step because MLPs ignore spatial structure.

## 9. Answers to Viva Questions
- **Why is scaling important for MLP?** Gradient descent converges poorly when input features have very different scales; standardization equalizes their influence and stabilizes the loss surface.
- **What is backpropagation?** Reverse-mode application of the chain rule that computes the gradient of the loss with respect to every weight, layer by layer.
- **What causes vanishing/exploding gradients?** Repeated multiplication of small (or large) Jacobians through many layers; ReLU, BatchNorm, careful initialization and residual connections mitigate it.

---

# Experiment 14: Recurrent Neural Network (RNN)

**Category:** Sequence modeling

## 1. Objective
Build recurrent networks for sequential data: (a) sequence classification and (b) time-series forecasting, comparing vanilla RNN with LSTM.

## 2. Datasets Used
(a) Synthetic sequence benchmark: 1,200 sequences of length 20, label = 1 if the sum is positive (1,000 train / 200 test). (b) Monthly Airline Passengers (`data/airline_passengers.csv`): 144 months, 12-month sliding windows, chronological 80/20 split.

## 3. Required Libraries
PyTorch, numpy, pandas, matplotlib (the manual's TensorFlow/Keras example is implemented equivalently in PyTorch, the framework installed in this environment)

## 4. Brief Theory
An RNN maintains a hidden state h_t = tanh(W x_t + U h_{t−1} + b) so the output depends on the sequence history; training uses backpropagation through time. Vanilla RNNs suffer from vanishing gradients over long sequences; LSTM gates (forget/input/output + additive cell state) preserve long-range information.

## 5. Procedure
1. Generate the synthetic sequence classification data and split 1,000/200.
2. Build RNN and LSTM classifiers (hidden 32, linear sigmoid output); train 10 epochs with Adam and BCEWithLogitsLoss.
3. Report test accuracy for both.
4. For forecasting: normalize the passenger series, build 12-month windows, split chronologically.
5. Train RNN and LSTM forecasters (hidden 64, 2 layers, MSE, Adam, 120 epochs).
6. Report RMSE/MAE and plot the forecasts.

## 6. Reference Implementation
```python
# classification
self.rnn = nn.RNN(1, hidden, batch_first=True) if cell == 'rnn' else nn.LSTM(1, hidden, batch_first=True)
self.fc = nn.Linear(hidden, 1)
# forecasting
self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
```

## 7. Results
**(a) Sequence classification (synthetic, manual experiment):**

| Model | Test accuracy |
|---|---|
| Vanilla RNN | 0.9300 |
| LSTM | **0.9450** |

**(b) Airline passenger forecasting:**

| Model | RMSE (passengers) | MAE |
|---|---|---|
| Vanilla RNN | 69.93 | 62.36 |
| LSTM | **44.12** | **36.78** |

## 8. Discussion and Conclusion
The LSTM outperforms the vanilla RNN on both tasks: +1.5 accuracy points on classification and ~37% lower RMSE on forecasting. The gap is larger on forecasting because the task needs 12-step seasonal memory, exactly where gating prevents vanishing gradients. On the easy synthetic benchmark both models do well because the decision rule (sum > 0) is nearly linear.

## 9. Answers to Viva Questions
- **Why does an RNN have memory?** Its hidden state is carried across time steps, so each output depends on previous inputs, not only the current one.
- **What is a hidden state?** The internal vector summarizing the sequence seen so far; it is updated at every step and used to produce outputs.
- **Why are LSTM/GRU often preferred for long dependencies?** Their gates control what is stored/forgotten and the additive cell state lets gradients flow across many steps, avoiding the exponential decay of vanilla RNN gradients.

---

# Experiment 15: Self-Organizing Map (SOM)

**Category:** Unsupervised neural learning

## 1. Objective
Map high-dimensional observations onto a low-dimensional grid while preserving neighborhood structure; evaluate with quantization and topographic error.

## 2. Dataset Used
UCI Wine: 178 samples, 13 chemical features, 3 cultivars (labels used only for visualization). MinMax-scaled to [0,1].

## 3. Required Libraries
minisom, numpy, pandas, matplotlib

## 4. Brief Theory
A SOM has a grid of neurons with weight vectors. For each input the Best Matching Unit (BMU) is found (minimum distance), then the BMU and its Gaussian neighborhood are moved toward the input: w_j ← w_j + α(t) h_cj(t) (x − w_j), with decaying α(t) and σ(t). The result is a topology-preserving 2D projection; the U-Matrix visualizes cluster boundaries.

## 5. Procedure
1. Load wine; MinMax scale.
2. Create a 12×12 SOM with Gaussian neighborhood (σ=1.5, lr=0.5); initialize weights with PCA.
3. Train for 5,000 random samples.
4. Compute quantization error (average sample→BMU distance) and topographic error (fraction of non-adjacent top-2 BMUs).
5. Plot the U-Matrix with each sample projected onto its BMU (marker = true cultivar).
6. Discuss grid size and σ.

## 6. Reference Implementation
```python
som = MiniSom(x=12, y=12, input_len=13, sigma=1.5, learning_rate=0.5,
              neighborhood_function='gaussian', random_seed=42)
som.pca_weights_init(X)
som.train_random(data=X, num_iteration=5000)
qe = som.quantization_error(X); te = som.topographic_error(X)
```

## 7. Results
| Metric | Value |
|---|---|
| Initial quantization error | 0.4447 |
| Final quantization error | **0.1965** |
| Topographic error | **0.0393** (~4% foldings) |

On the U-Matrix, the three wine cultivars occupy contiguous regions separated by bright ridges, even though the SOM never saw the labels — evidence of unsupervised structure discovery.

## 8. Discussion and Conclusion
The map halved its quantization error during training and preserves topology well (topographic error < 0.05). PCA initialization gave a well-unfolded map from the start. A larger grid reduces quantization error but risks fragmenting clusters; σ/lr must decay for stable convergence.

## 9. Answers to Viva Questions
- **SOM vs K-means?** Both compress data by prototypes, but K-means only finds centroids while SOM arranges them on a grid so that neighboring neurons represent similar inputs (topology preservation).
- **What is a BMU?** The neuron whose weight vector is closest to the input, i.e. the winner of the competition.
- **Why is SOM called a topology-preserving map?** Because the update neighborhood ensures nearby grid positions end up with similar weight vectors, so high-dimensional neighborhoods are approximately preserved in 2D.

---

# Experiment 16: Hidden Markov Model (HMM)

**Category:** Probabilistic sequence modeling

## 1. Objective
Model a sequence with hidden states, transition probabilities and emission distributions; decode the most likely hidden-state path with Viterbi.

## 2. Dataset Used
Real DAX stock index daily closes 1991-1998 (`data/stock_index.csv`, R `EuStockMarkets`), 1,859 trading days. The observation sequence is the daily return; the close price is used only for plotting. This is a realistic alternative to a toy discrete sequence.

## 3. Required Libraries
hmmlearn, numpy, pandas, matplotlib

## 4. Brief Theory
An HMM assumes a hidden Markov state sequence z_t (P(z_t | z_{t−1})) that emits observations x_t | z_t = s_k ~ N(μ_k, σ_k²). The transition matrix A_ij = P(z_{t+1}=j | z_t=i). Baum-Welch (EM) learns parameters (forward-backward E-step, re-estimation M-step); Viterbi decoding finds the most likely state path by dynamic programming; the forward algorithm gives the log-likelihood.

## 5. Procedure
1. Load the DAX series; build the daily-return observation vector.
2. Fit `GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42)`.
3. Check convergence and log-likelihood.
4. Decode the state path with Viterbi.
5. Sort states by volatility so they map to Bull/Sideways/Bear; reorder the transition matrix.
6. Plot the transition heatmap and the price colored by regime.

## 6. Reference Implementation
```python
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42)
hmm_model.fit(returns)
hidden_states = hmm_model.predict(returns)                 # Viterbi
state_order = np.argsort([np.sqrt(hmm_model.covars_[i][0][0]) for i in range(3)])
```

## 7. Results
| Item | Value |
|---|---|
| EM converged | True |
| Model log-likelihood | 6019.79 (1,859 days) |
| Regimes | Bull (low vol), Sideways (medium), Bear (high vol) after sorting |

The colored DAX chart shows the model tracking real episodes: the 1990s bull phases, corrections such as 1994, and the high-volatility cluster around the 1998 crisis. The transition-matrix diagonal is dominant, confirming regime persistence.

## 8. Discussion and Conclusion
The 3-state Gaussian HMM extracts interpretable, persistent market regimes from real data without labels. Its main assumptions are the Markov property (next state depends only on the current one) and Gaussian emissions; return fat tails and regime duration dependence motivate extensions (t-emissions, semi-Markov models).

## 9. Answers to Viva Questions
- **What is hidden in an HMM?** The state sequence: we observe emissions (returns) but not the underlying regime that generated them.
- **Viterbi vs forward algorithm?** Viterbi returns the single most likely state path (max over paths); the forward algorithm computes the total probability P(X | λ) by summing over all paths.
- **What is the Markov assumption?** The future state depends on the past only through the present state: P(z_t | z_{1:t−1}) = P(z_t | z_{t−1}).

---

# Experiment 17: Support Vector Machine (SVM)

**Category:** Supervised / margin-based learning

## 1. Objective
Train SVM classifiers and investigate kernels, C and gamma; also demonstrate Support Vector Regression with an ε-insensitive tube.

## 2. Datasets Used
(a) Breast Cancer (569 × 30): two features (mean radius, mean texture) for the 2D kernel view and all 30 features for the accuracy benchmark. (b) Old Faithful geyser (`data/geyser.csv`, 272 eruptions) for SVR.

## 3. Required Libraries
scikit-learn, numpy, pandas, matplotlib

## 4. Brief Theory
SVC maximizes the margin while penalizing violations: min ½||w||² + C Σξ_i. The dual depends only on inner products K(x_i, x_j) (kernel trick): linear, polynomial, RBF. C controls the violation penalty; γ controls the RBF influence radius. SVR ignores residuals within ±ε and uses only outside points (support vectors).

## 5. Procedure
1. Scale features.
2. Train SVC with linear/poly/RBF kernels on the 2D cancer data; plot boundaries, margins and support vectors.
3. Train the RBF SVC on all 30 features; report accuracy and support-vector count.
4. Fit SVR (RBF, C=10, ε=0.3) to waiting time → eruption duration; plot the fitted curve and ε-tube.
5. Report R², RMSE and support-vector share.

## 6. Reference Implementation
```python
clf = SVC(kernel=k_name, C=1.0, gamma='scale', degree=3, random_state=42).fit(X_2d_scaled, y_2d)
svc_rbf = SVC(kernel='rbf', C=1.0, random_state=42).fit(X_train_c_s, y_train_c)
svr_model = SVR(kernel='rbf', C=10.0, epsilon=0.3, gamma='scale').fit(X_svr_scaled, y_svr)
```

## 7. Results
| Experiment | Metric | Value |
|---|---|---|
| SVC (RBF, 30 features) | Test accuracy | 0.9790 |
| | Support vectors | 96 of 426 (22.5%) |
| SVR (Old Faithful) | R² | 0.8968 |
| | RMSE | 0.366 minutes |
| | Support vectors | 103 of 272 (37.9%) |

The 2D kernel figure shows the linear kernel underfitting the real diagnostic data while poly/RBF produce curved boundaries. The geyser scatter is bimodal; the RBF SVR smoothly bridges the two modes and the ε=0.3 tube leaves most residuals unpenalized.

## 8. Discussion and Conclusion
SVMs give accurate, sparse solutions: 77% of the cancer training points could be discarded without changing the classifier. They require scaling and tuning (C, γ), and kernel methods scale as O(n²), which limits them on very large datasets.

## 9. Answers to Viva Questions
- **What is a support vector?** A training point on or inside the margin (α > 0) that alone defines the decision boundary.
- **Effect of C?** Large C penalizes violations strongly → narrow margin, possible overfitting; small C allows a wider margin and more violations.
- **Effect of gamma?** Large γ makes each point's influence local → wiggly boundary; small γ makes the RBF smoother, approaching a linear model.
- **Why does SVM require careful scaling?** Margins and kernels use Euclidean distances; unscaled features with larger ranges dominate them and degrade the fit.

---

# Experiment 18: Large Language Model (LLM) Experiment

**Category:** Generative / foundation models

## 1. Objective
Demonstrate a reproducible LLM workflow: tokenization, inference, controlled generation, and fine-tuning on a real public classification dataset.

## 2. Dataset Used
DistilBERT (SST-2) for tokenization/sentiment, DistilGPT2 for generation, and the SMS Spam Collection (`data/sms_spam.csv`, 5,572 labeled messages) for fine-tuning. A balanced subset (300 ham + 300 spam) is split 400 train / 200 stratified test.

## 3. Required Libraries
transformers, torch (CUDA), pandas

## 4. Brief Theory
LLMs process text as subword tokens (BPE/WordPiece) with an attention mask; self-attention computes Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)V. Causal generation predicts one token at a time; temperature scales logits and top-p restricts sampling to the nucleus. Fine-tuning reuses pretrained weights and replaces the classification head.

## 5. Procedure
1. Tokenize a sentence; inspect tokens, IDs and attention mask.
2. Run the sentiment pipeline on fixed test sentences.
3. Generate text with greedy decoding and with temperature/top-p sampling.
4. Load SMS Spam; build the balanced subset and stratified split.
5. Fine-tune DistilBERT (fresh 2-class head, AdamW lr 5e-5, 2 epochs, batch 16).
6. Evaluate accuracy/F1 on the untouched test messages and inspect examples.

## 6. Reference Implementation
```python
ft_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, ignore_mismatched_sizes=True)
optimizer = torch.optim.AdamW(ft_model.parameters(), lr=5e-5)
outputs = ft_model(input_ids=ids, attention_mask=mask, labels=labels)
outputs.loss.backward(); optimizer.step()
```

## 7. Results
| Item | Value |
|---|---|
| Fine-tune loss | 0.705 → 0.079 (2 epochs) |
| Held-out accuracy | 0.9600 |
| Held-out F1 (spam) | 0.9592 |
| Examples | spam 99.4%, ham 99.5%, spam 89.7% confidence (all correct) |

Generation: greedy produced a safe deterministic continuation; nucleus sampling (T=0.7, p=0.9) produced a different but coherent sentence. Sentiment scores were in the 90-99% range on clear sentences.

## 8. Discussion and Conclusion
The experiment shows the complete practical LLM pipeline and that fine-tuning a small pretrained model on 400 real messages already gives 96% spam detection. Limitations: the balanced subset ignores the natural 13% spam prior; the models are small and can hallucinate; fine-tuning data and prompts must be reproducible. No private data is used.

## 9. Answers to Viva Questions
- **What is tokenization?** Splitting text into subword units that map to integer vocabulary IDs, letting a fixed vocabulary represent any word; special tokens and attention masks handle sequence framing/padding.
- **Pretraining vs fine-tuning vs prompting?** Pretraining learns general language representation from massive corpora; fine-tuning adapts those weights to a task with labeled data; prompting steers a frozen model through instructions/examples.
- **Why is LLM output not automatically ground truth?** The model generates the most plausible continuation, not verified facts; it can hallucinate, and confidence scores are not calibrated correctness probabilities.
- **How can prompt wording affect results?** Meaning, tone, format and label wording all change the conditional distribution the model samples from; small prompt changes can flip outputs, so prompts must be fixed and reported.

---

# Experiment 19: Generalized Regression Neural Network (GRNN)

**Category:** Non-parametric regression

## 1. Objective
Implement GRNN-style kernel regression from scratch and study the effect of the smoothing parameter σ with cross-validation.

## 2. Dataset Used
Real motorcycle impact data (`data/motorcycle.csv`, R `MASS::mcycle`): 133 observations of time after impact (ms) versus head acceleration (g). Time is standardized before computing kernel distances.

## 3. Required Libraries
numpy, pandas, matplotlib, scikit-learn (API base classes, KFold, metrics)

## 4. Brief Theory
GRNN predicts the Nadaraya-Watson conditional mean: ŷ(x) = Σ_i y_i exp(−||x−x_i||²/2σ²) / Σ_i exp(−||x−x_i||²/2σ²). Layer structure: input → pattern (one Gaussian per training point) → summation (S = Σy_i p_i, D = Σp_i) → output S/D. Training is one-pass (store the data); σ controls the bias-variance trade-off.

## 5. Procedure
1. Load mcycle; standardize time.
2. Implement the GRNN class (`fit` stores data; `predict` computes kernel weights).
3. Visualize σ = 0.05 / 0.25 / 1.0 (under/over-smoothing).
4. Select σ by 5-fold CV over 0.02…1.0.
5. Refit on a 75% split and evaluate on the 25% held-out set (R², RMSE).

## 6. Reference Implementation
```python
kernels = np.exp(-dist_sq / (2.0 * self.sigma ** 2))   # pattern layer
D = np.sum(kernels, axis=1)                            # denominator
S = np.dot(kernels, self.y_train_)                     # numerator
return S / np.where(D < 1e-12, 1e-12, D)
```

## 7. Results
| Item | Value |
|---|---|
| CV-selected σ | 0.080 (CV RMSE 25.00 g) |
| Train R² | 0.8073 |
| Test R² | 0.7255 |
| Test RMSE | 22.78 g (acceleration range −134…+75 g) |

σ = 0.05 shows local wiggles (variance), σ = 0.25-1.0 underfits the sharp deceleration peak. The 5-fold CV curve has a clear minimum at σ = 0.08.

## 8. Discussion and Conclusion
GRNN is instant to train and needs only one hyperparameter. On the smooth 1D sine benchmark it reaches R² ≈ 0.95, but on real motorcycle data the test R² is 0.73 — a single global σ cannot be simultaneously small at the impact discontinuity and large elsewhere. This is the classic bias-variance limitation of global-bandwidth kernel regression; adaptive bandwidths or local polynomial regression would improve it.

## 9. Answers to Viva Questions
- **Why is GRNN called non-parametric?** It does not learn a fixed set of parameters; the training data itself forms the model (memory-based kernel regression).
- **What does σ control?** The smoothing bandwidth: small σ fits noise (high variance), large σ over-smooths (high bias).
- **How is GRNN related to kernel regression?** It is exactly the Nadaraya-Watson Gaussian kernel regression estimator of E[y | x], with one kernel per training observation.

---

# Integrated Mini Project: Comparative Machine Learning Study

**Problem:** Compare algorithms from different families under identical data, splits and seeds, and explain the differences rather than picking one score.

## Classification comparison — Heart Disease (same stratified 75/25 split, seed 42)

| Model | Key Hyperparameters | Test Accuracy | F1 | ROC-AUC | Observation |
|---|---|---|---|---|---|
| Baseline (majority class) | — | ~0.55 | — | 0.50 | naive reference |
| Random Forest (RFC) | 120 trees, depth 8 | 0.7763 | 0.8046 | 0.8704 | robust, minimal tuning |
| XGBoost | 120 trees, lr 0.08, depth 4 | 0.7763 | 0.8132 | 0.8509 | strong F1, needs tuning |
| AdaBoost | 100 stumps, lr 0.1 | **0.7895** | **0.8182** | **0.8714** | best on this small data |
| CatBoost | 150 iters, depth 5, native cats | 0.7632 | 0.7955 | 0.8551 | no manual encoding needed |
| SVC (RBF) | C=1.0, γ='scale' | 0.9790 (Breast Cancer) | — | — | separate 30-feature benchmark |

## Regression comparison — California Housing (5,000 sample, same 80/20 split)

| Model | Test R² | RMSE | MAE | Observation |
|---|---|---|---|---|
| Random Forest Regressor | 0.7415 | 0.5908 | 0.3999 | OOB 0.7513 ≈ test, no overfit |
| XGBoost Regressor | 0.8025 | 0.5165 | 0.3467 | best trade-off |
| AdaBoost Regressor | 0.5888 | 0.7452 | 0.5842 | noise-sensitive |
| CatBoost Regressor | **0.8029** | **0.5160** | 0.3535 | ties XGBoost |

## Clustering comparison — Mall Customers (K=5, standardized)

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| K-Means | 0.5547 | 0.5722 | 248.65 |
| Modified K-Means (97th pct outlier flag) | 0.5547 | 0.5722 | 248.65 (6 outliers flagged) |
| Hierarchical (Ward) | 0.5538 | 0.5779 | 244.41 |
| Fuzzy C-Means | 0.5547 | 0.5722 | 248.65 (FPC 0.6711) |

## Density-based comparison — 7,638 real earthquakes (M ≥ 4.5, 2023)

| Algorithm | Clusters | Noise | Silhouette | Observation |
|---|---|---|---|---|
| DBSCAN (ε=0.03 rad) | 55 | 13.5% | 0.326 | needs ε; elongated arc clusters |
| HDBSCAN | 40 | 21.5% | **0.631** | adaptive density, no ε |

**Conclusions.** (1) On small tabular classification, simple boosting (AdaBoost) or bagging (RFC) match heavier models; the differences are within noise. (2) On larger regression data, gradient boosting clearly wins (R² ≈ 0.80 vs 0.74). (3) Clustering algorithms agree on well-separated data; density methods are the only option for arbitrary shapes and noise. (4) Fair comparison requires identical splits, fixed seeds, untouched test sets, and reporting at least two metrics per task.

**Common mistakes avoided.** Scaling fitted only on training data; no repeated test-set peeking; same partitions for all models; F1/AUC reported alongside accuracy; cluster labels never treated as ground truth; one hyperparameter changed at a time; prompts and model versions recorded; no private data sent to external models.

---

# Appendix A: Quick Reference Formulas

| Concept | Formula / Definition |
|---|---|
| Euclidean distance | d(x,y) = sqrt(Σ_j (x_j − y_j)²) |
| K-means objective | J = Σ_i ||x_i − μ_{c_i}||² |
| Regression MAE | MAE = (1/n) Σ_i \|y_i − ŷ_i\| |
| Regression RMSE | RMSE = sqrt[(1/n) Σ_i (y_i − ŷ_i)²] |
| R² | R² = 1 − SS_res / SS_tot |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 × Precision × Recall / (Precision + Recall) |
| Sigmoid | σ(z) = 1 / (1 + e^(−z)) |
| SVM margin idea | Maximize margin while penalizing violations via C |
| GRNN kernel | w_i(x) = exp(−\|\|x−x_i\|\|² / (2σ²)) |

# Appendix B: Reproducibility Checklist

- Python 3.12 (`.venv`) and all package versions recorded (`pip list` in the repository documentation).
- Random seeds fixed everywhere (`random_state=42`, `random_seed=42`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`).
- Datasets committed under `data/` with documented sources (USGS, UCI, scikit-learn, Rdatasets, SMS Spam Collection).
- Preprocessing recorded: scaling fitted on training data only; haversine radians for geospatial data; MinMax for SOM/RNN.
- Train/validation/test protocols recorded (stratified, chronological, early-stopping split, 5-fold CV).
- Hyperparameters documented per experiment in the notebooks and this report.
- Metrics defined before final testing; no test-set model selection.
- Figures and tables saved inside the executed notebooks and the printed PDFs.
- Code runs from the committed `.venv` with the listed dependencies; GPU optional (CPU fallback automatic).

# Appendix C: Suggested Assessment Rubric (self-assessment)

| Component | Weight | Our evidence |
|---|---|---|
| Pre-lab preparation / algorithm understanding | 10% | Step-by-step algorithm blocks in every notebook + `VIVA_GUIDE.md` |
| Correct implementation | 25% | 19 experiments implemented and executed (0 errors) |
| Experimental design and preprocessing | 15% | Documented splits, scaling, leakage avoidance, haversine handling |
| Evaluation and visualization | 15% | Appropriate metrics per task; plots and metric tables in each notebook |
| Analysis and interpretation | 20% | Discussion/limitations sections in every experiment + mini project |
| Report quality / reproducibility | 10% | This completed manual + PDFs + seeds/datasets committed |
| Viva / discussion | 5% | `VIVA_GUIDE.md` Q&A + per-experiment viva answers |

# Appendix D: Student Experiment Record

| Field | Student Entry |
|---|---|
| Name | ____________________________ |
| ID | ____________________________ |
| Section | ____________________________ |
| Experiment No. | 1-19 (all algorithms of the assignment list) |
| Date | ____________________________ |
| Dataset | Mall Customers, USGS earthquakes 2023, Breast Cancer, Heart Disease, California Housing, Digits, Airline Passengers, Wine, DAX 1991-98, Old Faithful, Motorcycle, SMS Spam |
| Algorithm / Version | scikit-learn 1.9.1, XGBoost 3.4.1, CatBoost 1.2.10, PyTorch 2.14.0, transformers 5.17.0, MiniSom 2.3.6, hmmlearn 0.3.3 |
| Key Hyperparameters | K=5; DBSCAN ε=0.03 rad/MinPts=10; HDBSCAN mcs=50; threshold τ=0.80; 150-200 trees/boosting rounds; hidden (128,64)/(64-128-64-10); σ=0.08 |
| Random Seed | 42 everywhere |
| Main Result | All algorithms implemented and evaluated on real data; best classification AUC 0.871 (AdaBoost), best regression R² 0.803 (CatBoost), LSTM RMSE 44.1 vs RNN 69.9, SMS spam F1 0.959 |
| Observation | Model choice must match task and data size; gradient boosting leads tabular regression, ensembles tie on small classification data, density clustering finds real seismic zones without labels, and GRNN's global bandwidth limits it on discontinuous data |
| Instructor Signature | ____________________________ |
