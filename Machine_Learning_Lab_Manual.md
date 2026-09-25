

**MACHINE LEARNING LABORATORY MANUAL**

Practical Guide to Classical, Ensemble, Neural, Probabilistic,  
Density-Based, Semi-Supervised and Generative ML

Prepared by  
**Dr. Ohidujjaman Tuhin**   
Associate Professor, Dept. of CSE, UIU

**Department of Computer Science & Engineering**

Machine Learning Lab

| Item | Details |
| :---- | :---- |
| Course | Machine Learning Laboratory |
| Programming Language | Python 3.x |
| Primary Platform | Jupyter Notebook / Google Colab / VS Code |
| Recommended Level | Undergraduate / Graduate |
| Assessment | Pre-lab preparation, implementation, analysis, report and viva |

*Prepared as a practical, reproducible lab guide*

# **Preface**

This manual is designed around the algorithm list supplied for the Machine Learning assignment. It turns that list into a structured laboratory sequence emphasizing implementation, experimentation, model evaluation and interpretation rather than only code execution. Students should run the programs, record outputs, vary important hyperparameters, and explain why the observed results change.

## **Important terminology correction**

The supplied list contains “Random Forest Regression (RFR)” and “Random Forest Regression (RFC)”. In standard machine-learning terminology, RFC normally means Random Forest Classifier. This manual therefore treats RFR as Random Forest Regressor and RFC as Random Forest Classifier.

## **Scope of the laboratory**

| Module | Algorithms / Topics |
| :---- | :---- |
| Clustering | K-means, Modified K-means, Hierarchical Clustering, Fuzzy C-means |
| Density-based learning | DBSCAN, HDBSCAN |
| Semi-supervised learning | Self-training |
| Ensemble learning | Random Forest Regression, Random Forest Classification, XGBoost, AdaBoost, CatBoost |
| Neural networks | MLP, RNN |
| Unsupervised neural learning | SOM |
| Probabilistic sequence model | HMM |
| Margin-based learning | SVM |
| Generative / foundation models | LLM |
| Non-parametric regression | GRNN |

## **General learning outcomes**

* Implement and execute major machine-learning algorithms using Python.  
* Select an appropriate algorithm based on data type, task and assumptions.  
* Preprocess data, handle categorical variables, scale numerical features and split datasets correctly.  
* Evaluate clustering with internal measures and supervised models with appropriate metrics.  
* Interpret model behavior through decision boundaries, feature importance, learning curves and error analysis.  
* Compare algorithms using a fair experimental protocol rather than relying on a single score.  
* Document reproducible experiments, hyperparameters, random seeds, observations and limitations.

## **Recommended software environment**

\# Core  
pip install numpy pandas matplotlib seaborn scikit-learn  
\# Specialized algorithms  
pip install scikit-fuzzy hdbscan hmmlearn minisom  
\# Gradient boosting libraries  
pip install xgboost catboost  
\# LLM / transformer experiment  
pip install transformers torch

Laboratory Guidelines

| Before Lab | During Lab | After Lab |
| :---- | :---- | :---- |
| Read theory and algorithm steps. | Run the baseline program. | Submit code and results. |
| Know the dataset and target. | Record hyperparameters and seed. | Explain observations. |
| Predict expected behavior. | Perform at least one controlled variation. | Answer viva questions. |
| Prepare key formulas. | Check train/test leakage. | State limitations and improvements. |

## **Minimum report format**

1. Experiment title and student information.  
2. Problem statement and objective.  
3. Dataset description and preprocessing.  
4. Algorithm principle and pseudocode / flowchart.  
5. Python implementation.  
6. Hyperparameters and experimental setup.  
7. Results: tables, plots and relevant metrics.  
8. Discussion: what the results mean and why they occurred.  
9. Conclusion and limitations.  
10. Answers to viva questions.

## **Recommended evaluation metrics**

| Task |  | Suggested Metrics |
| :---- | :---- | :---- |
| Classification | Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix |  |
| Regression | MAE, MSE, RMSE, R² |  |
| Clustering | Silhouette score, Calinski–Harabasz, Davies–Bouldin; visual inspection |  |
| Sequence modeling | Negative log-likelihood / log-likelihood, sequence accuracy where appropriate |  |
| Language generation | Task-specific automatic metrics plus qualitative/error analysis |  |

# **Experiment 1: K-means Clustering**

**Category:** Clustering

## **1\. Objective**

Partition unlabeled observations into K clusters by iteratively assigning points to the nearest centroid and updating centroids.

## **2\. Suggested Dataset**

Use sklearn's Iris dataset after removing the target column, or a 2-D synthetic dataset for visualization.

## **3\. Required Libraries**

numpy, pandas, matplotlib, scikit-learn

## **4\. Brief Theory**

K-means minimizes within-cluster sum of squared distances. The main choices are K, initialization and number of restarts. Because Euclidean distance is scale-sensitive, numerical features should usually be standardized.

## **5\. Procedure**

1) Load the dataset and retain numerical features.  
2) Standardize the features.  
3) Choose a candidate K.  
4) Fit K-means with a fixed random\_state.  
5) Visualize clusters when the data can be projected to two dimensions.  
6) Compute silhouette score for several K values and discuss the choice.

## **6\. Reference Implementation**

from sklearn.datasets import load\_iris  
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette\_score  
import matplotlib.pyplot as plt

X \= load\_iris().data  
X \= StandardScaler().fit\_transform(X)

scores \= {}  
for k in range(2, 7):  
    model \= KMeans(n\_clusters=k, n\_init=20, random\_state=42)  
    labels \= model.fit\_predict(X)  
    scores\[k\] \= silhouette\_score(X, labels)

print("Silhouette scores:", scores)

best\_k \= max(scores, key=scores.get)  
model \= KMeans(n\_clusters=best\_k, n\_init=20, random\_state=42)  
labels \= model.fit\_predict(X)  
print("Selected K:", best\_k)  
print("Inertia:", model.inertia\_)

## **7\. Evaluation / Expected Output**

* Report the silhouette score for each tested K.  
* Plot clusters or a 2-D projection.  
* Discuss sensitivity to initialization and feature scaling.

## **8\. Viva / Discussion Questions**

* Why does K-means depend on feature scaling?  
* What is inertia?  
* Why can a high silhouette score still be misleading?  
* What happens when clusters are non-spherical?

# **Experiment 2: Modified K-means**

**Category:** Clustering

## **1\. Objective**

Implement a transparent modification of K-means that uses k-means++ initialization and an explicit outlier-distance check before final cluster assignment.

## **2\. Suggested Dataset**

Use a synthetic 2-D dataset containing compact clusters plus a small number of distant points.

## **3\. Required Libraries**

numpy, matplotlib, scikit-learn

## **4\. Brief Theory**

There is no single universally accepted algorithm named “Modified K-means”; variants differ by application. In this lab, the modification is intentionally defined so that students can reproduce it: initialize centroids with k-means++, fit K-means, estimate a distance threshold from training distances, and flag unusually distant observations as potential outliers. This is an experimental variant, not a standard replacement for robust clustering.

## **5\. Procedure**

1) Generate or load 2-D data with a few outliers.  
2) Run standard K-means.  
3) Compute each sample's distance to its assigned centroid.  
4) Set a threshold using a documented percentile on the training data.  
5) Mark points beyond the threshold as candidate outliers.  
6) Compare standard labels with the modified interpretation.

## **6\. Reference Implementation**

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.datasets import make\_blobs  
from sklearn.cluster import KMeans

X, \_ \= make\_blobs(n\_samples=400, centers=3, cluster\_std=0.8, random\_state=42)  
rng \= np.random.default\_rng(42)  
outliers \= rng.uniform(-8, 8, size=(12, 2))  
X \= np.vstack(\[X, outliers\])

km \= KMeans(n\_clusters=3, n\_init=20, random\_state=42)  
labels \= km.fit\_predict(X)

dist \= np.linalg.norm(X \- km.cluster\_centers\_\[labels\], axis=1)  
threshold \= np.percentile(dist, 97\)  
outlier\_flag \= dist \> threshold

print("Distance threshold:", threshold)  
print("Flagged observations:", outlier\_flag.sum())

## **7\. Evaluation / Expected Output**

* Show the threshold and number of flagged points.  
* Plot centroids and candidate outliers.  
* Explain why the modification is application-dependent.

## **8\. Viva / Discussion Questions**

* Why is it important to define a modification precisely?  
* How does k-means++ differ from random initialization?  
* Why might distance-to-centroid fail for elongated clusters?

# **Experiment 3: Hierarchical Clustering**

**Category:** Clustering

## **1\. Objective**

Perform agglomerative hierarchical clustering and interpret the resulting dendrogram and cluster labels.

## **2\. Suggested Dataset**

Iris features or a small standardized 2-D dataset.

## **3\. Required Libraries**

scipy, scikit-learn, matplotlib

## **4\. Brief Theory**

Agglomerative clustering starts with each point as a cluster and repeatedly merges clusters. Linkage choices such as single, complete, average and Ward produce different structures.

## **5\. Procedure**

1) Standardize the features.    
2) Generate a linkage matrix.  
3) Plot a dendrogram.  
4) Select a cut level or number of clusters.  
5) Fit AgglomerativeClustering.  
6) Compare at least two linkage strategies where valid. 

## **6\. Reference Implementation**

from sklearn.datasets import load\_iris  
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import AgglomerativeClustering  
from scipy.cluster.hierarchy import linkage, dendrogram  
import matplotlib.pyplot as plt

X \= StandardScaler().fit\_transform(load\_iris().data)

Z \= linkage(X, method="ward")  
plt.figure(figsize=(9, 4))  
dendrogram(Z, truncate\_mode="level", p=5)  
plt.title("Hierarchical Clustering Dendrogram")  
plt.xlabel("Sample")  
plt.ylabel("Distance")  
plt.show()

model \= AgglomerativeClustering(n\_clusters=3, linkage="ward")  
labels \= model.fit\_predict(X)  
print("Cluster sizes:", np.bincount(labels))

## **7\. Evaluation / Expected Output**

* Include the dendrogram.  
* Compare cluster sizes and silhouette scores.  
* Discuss linkage selection.

## **8\. Viva / Discussion Questions**

* What does a dendrogram show?  
* What is linkage?  
* When is Ward linkage inappropriate?

# **Experiment 4: Fuzzy C-means**

**Category:** Clustering

## **1\. Objective**

Cluster data using fuzzy memberships so that each observation can belong to multiple clusters with different degrees of membership.

## **2\. Suggested Dataset**

Iris numerical features after scaling, or a 2-D synthetic dataset.

## **3\. Required Libraries**

numpy, scikit-fuzzy, matplotlib

## **4\. Brief Theory**

Unlike hard K-means assignments, Fuzzy C-means produces membership values u\_ij. The fuzziness parameter m controls how soft the assignments are.

## **5\. Procedure**

1. Scale the data.  
2. Choose the number of clusters and fuzziness parameter.  
3. Fit Fuzzy C-means.  
4. Obtain membership matrix and hard labels.  
5. Identify ambiguous samples with similar membership values.  
6. Visualize membership or cluster labels.

## **6\. Reference Implementation**

import numpy as np  
import skfuzzy as fuzz  
from sklearn.datasets import load\_iris  
from sklearn.preprocessing import StandardScaler

X \= StandardScaler().fit\_transform(load\_iris().data).T

centers, u, u0, d, jm, p, fpc \= fuzz.cluster.cmeans(  
    X, c=3, m=2.0, error=0.005, maxiter=1000, init=None  
)

labels \= np.argmax(u, axis=0)  
print("Cluster centers (scaled):")  
print(centers)  
print("Fuzzy partition coefficient:", fpc)  
print("First five membership vectors:")  
print(u\[:, :5\])

## **7\. Evaluation / Expected Output**

* Report cluster centers and membership vectors.  
* Discuss ambiguous observations.  
* Repeat with m values such as 1.5 and 2.5.

## **8\. Viva / Discussion Questions**

* How is Fuzzy C-means different from K-means?  
* What does m control?  
* What does a membership vector represent?

# **Experiment 5: DBSCAN**

Category: Density-based learning

## **1\. Objective**

Detect dense groups and noise using DBSCAN without specifying the number of clusters in advance.

## **2\. Suggested Dataset**

Use a 2-D synthetic moons/circles dataset and optionally a real dataset.

## **3\. Required Libraries**

scikit-learn, numpy, matplotlib

## **4\. Brief Theory**

DBSCAN uses eps (neighborhood radius) and min\_samples. Points are classified as core, border or noise. It can discover non-convex clusters.

## **5\. Procedure**

1. Generate two interleaving moons.  
2. Scale features if required.  
3. Fit DBSCAN.  
4. Vary eps and min\_samples.  
5. Plot cluster labels and noise.  
6. Discuss sensitivity to density.

## **6\. Reference Implementation**

from sklearn.datasets import make\_moons  
from sklearn.cluster import DBSCAN  
import matplotlib.pyplot as plt

X, \_ \= make\_moons(n\_samples=500, noise=0.06, random\_state=42)

model \= DBSCAN(eps=0.20, min\_samples=5)  
labels \= model.fit\_predict(X)

print("Clusters:", sorted(set(labels) \- {-1}))  
print("Noise points:", (labels \== \-1).sum())

plt.scatter(X\[:, 0\], X\[:, 1\], c=labels, s=15)  
plt.title("DBSCAN")  
plt.show()

## **7\. Evaluation / Expected Output**

* Plot cluster/noise labels.  
* Report number of clusters and noise points.  
* Show at least two parameter settings.

## **8\. Viva / Discussion Questions**

* Why does DBSCAN not require K?  
* What does label \-1 mean?  
* Why can DBSCAN struggle with varying densities?

# **Experiment 6: HDBSCAN**

**Category:** Density-based learning

## **1\. Objective**

Apply hierarchical density-based clustering to discover clusters of different densities and identify noise.

## **2\. Suggested Dataset**

Use a synthetic dataset with clusters having different spreads.

## **3\. Required Libraries**

hdbscan, numpy, matplotlib

## **4\. Brief Theory**

HDBSCAN builds a hierarchy of density-based clusters and extracts a stable flat clustering. It is often useful when a single global density threshold is unsuitable.

## **5\. Procedure**

1. Create data with unequal cluster density.  
2. Fit HDBSCAN.  
3. Inspect labels and membership probabilities.  
4. Count noise points.  
5. Compare with DBSCAN.  
6. Discuss minimum cluster size.

## **6\. Reference Implementation**

import hdbscan  
from sklearn.datasets import make\_blobs  
import numpy as np

X, \_ \= make\_blobs(  
    n\_samples=\[250, 180, 120\],  
    centers=\[(-3, \-2), (2, 2), (5, \-3)\],  
    cluster\_std=\[0.35, 1.0, 0.55\],  
    random\_state=42  
)

clusterer \= hdbscan.HDBSCAN(min\_cluster\_size=25, min\_samples=10)  
labels \= clusterer.fit\_predict(X)

print("Clusters:", sorted(set(labels) \- {-1}))  
print("Noise:", np.sum(labels \== \-1))  
print("First membership probabilities:", clusterer.probabilities\_\[:10\])

## **7\. Evaluation / Expected Output**

* Report cluster count, noise count and sample probabilities.  
* Compare DBSCAN and HDBSCAN behavior.  
* Explain the effect of min\_cluster\_size.

## **8\. Viva / Discussion Questions**

* What is hierarchical about HDBSCAN?  
* What does membership probability mean?  
* Why is HDBSCAN useful for variable-density data?

# **Experiment 7: Self-training for Semi-supervised Learning**

**Category:** Semi-supervised learning

## **1\. Objective**

Use a small labeled subset and a larger unlabeled subset, iteratively adding high-confidence pseudo-labels.

## **2\. Suggested Dataset**

Iris or another multiclass dataset with an artificial labeled/unlabeled split.

## **3\. Required Libraries**

scikit-learn, numpy

## **4\. Brief Theory**

Self-training fits a base classifier on labeled data, predicts unlabeled samples, and adds sufficiently confident predictions to the labeled set. Pseudo-label errors can propagate, so confidence thresholds and validation are important.

## **5\. Procedure**

1. Split the dataset into train/test.  
2. Hide labels for most training samples.  
3. Train a supervised baseline.  
4. Wrap the classifier with SelfTrainingClassifier.  
5. Evaluate on the untouched test set.  
6. Compare with the baseline using the same test set.

## **6\. Reference Implementation**

from sklearn.datasets import load\_iris  
from sklearn.model\_selection import train\_test\_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.pipeline import make\_pipeline  
from sklearn.linear\_model import LogisticRegression  
from sklearn.semi\_supervised import SelfTrainingClassifier  
from sklearn.metrics import accuracy\_score

X, y \= load\_iris(return\_X\_y=True)  
X\_train, X\_test, y\_train, y\_test \= train\_test\_split(  
    X, y, test\_size=0.30, stratify=y, random\_state=42  
)

rng \= np.random.default\_rng(42)  
y\_partial \= y\_train.copy()  
hide \= rng.random(len(y\_partial)) \< 0.75  
y\_partial\[hide\] \= \-1

base \= make\_pipeline(StandardScaler(), LogisticRegression(max\_iter=1000))  
baseline \= base.fit(X\_train\[y\_partial \!= \-1\], y\_partial\[y\_partial \!= \-1\])  
pred\_base \= baseline.predict(X\_test)

self\_model \= SelfTrainingClassifier(  
    estimator=make\_pipeline(StandardScaler(), LogisticRegression(max\_iter=1000)),  
    threshold=0.90  
)  
self\_model.fit(X\_train, y\_partial)  
pred\_self \= self\_model.predict(X\_test)

print("Supervised accuracy:", accuracy\_score(y\_test, pred\_base))  
print("Self-training accuracy:", accuracy\_score(y\_test, pred\_self))

## **7\. Evaluation / Expected Output**

* Record the number of initially labeled samples and pseudo-labeled samples.  
* Compare test metrics fairly.  
* Repeat with thresholds such as 0.80, 0.90 and 0.95.

## **8\. Viva / Discussion Questions**

* What is a pseudo-label?  
* What is confirmation bias in self-training?  
* Why must the test labels remain hidden during training?

# **Experiment 8: Random Forest Regression (RFR)**

**Category:** Ensemble learning

## **1\. Objective**

Predict a continuous target using an ensemble of randomized decision trees.

## **2\. Suggested Dataset**

California housing dataset or another regression dataset.

## **3\. Required Libraries**

scikit-learn, pandas, numpy

## **4\. Brief Theory**

Random Forest Regression averages predictions from many decision trees trained on bootstrap samples and randomized feature subsets. It can model nonlinear relationships and interactions.

## **5\. Procedure**

1. Load a regression dataset.  
2. Split into training and test sets.  
3. Train RandomForestRegressor.  
4. Evaluate with MAE, RMSE and R².  
5. Inspect feature importances.  
6. Vary number of trees and maximum depth.

## **6\. Reference Implementation**

from sklearn.datasets import fetch\_california\_housing  
from sklearn.model\_selection import train\_test\_split  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean\_absolute\_error, mean\_squared\_error, r2\_score  
import numpy as np

X, y \= fetch\_california\_housing(return\_X\_y=True)  
Xtr, Xte, ytr, yte \= train\_test\_split(X, y, test\_size=0.2, random\_state=42)

model \= RandomForestRegressor(  
    n\_estimators=300, random\_state=42, n\_jobs=-1  
)  
model.fit(Xtr, ytr)  
pred \= model.predict(Xte)

print("MAE:", mean\_absolute\_error(yte, pred))  
print("RMSE:", mean\_squared\_error(yte, pred, squared=False))  
print("R2:", r2\_score(yte, pred))  
print("Feature importances:", model.feature\_importances\_)

## **7\. Evaluation / Expected Output**

* Report MAE, RMSE and R².  
* Plot predicted vs. actual values.  
* Discuss feature importance and overfitting.

## **8\. Viva / Discussion Questions**

* Why does a forest reduce variance?  
* What is bagging?  
* Why can feature importance be misleading?

# **Experiment 9: Random Forest Classification (RFC)**

**Category:** Ensemble learning

## **1\. Objective**

**Classify observations using an ensemble of randomized decision trees.**

## **2\. Suggested Dataset**

**Iris, Breast Cancer Wisconsin, or another multiclass/binary dataset**.

## **3\. Required Libraries**

**scikit-learn, numpy, matplotlib**

## **4\. Brief Theory**

**Random Forest Classification aggregates tree votes. Randomization through bootstrap samples and feature subsampling reduces correlation among trees.**

## **5\. Procedure**

1. **Load a classification dataset.**  
2. **Use a stratified train/test split.**  
3. **Train RandomForestClassifier.**  
4. **Compute accuracy, precision, recall and F1.**  
5. **Display the confusion matrix.**  
6. **Vary max\_depth and n\_estimators.**

## **6\. Reference Implementation**

from sklearn.datasets import load\_breast\_cancer  
from sklearn.model\_selection import train\_test\_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification\_report, confusion\_matrix

X, y \= load\_breast\_cancer(return\_X\_y=True)  
Xtr, Xte, ytr, yte \= train\_test\_split(  
    X, y, test\_size=0.2, stratify=y, random\_state=42  
)

model \= RandomForestClassifier(  
    n\_estimators=300, random\_state=42, n\_jobs=-1  
)  
model.fit(Xtr, ytr)  
pred \= model.predict(Xte)

print(confusion\_matrix(yte, pred))  
print(classification\_report(yte, pred))

## **7\. Evaluation / Expected Output**

* Include confusion matrix and classification report.  
* Discuss class imbalance if present.  
* Compare at least two forest sizes.

## **8\. Viva / Discussion Questions**

* RFR vs RFC?  
* What is majority voting?  
* How can class imbalance affect accuracy?

# **Experiment 10: XGBoost**

**Category:** Ensemble learning / Gradient boosting

## **1\. Objective**

Train a gradient-boosted tree model and analyze the effect of boosting hyperparameters.

## **2\. Suggested Dataset**

Iris for a simple classification exercise, or a larger tabular classification dataset.

## **3\. Required Libraries**

xgboost, scikit-learn, numpy

## **4\. Brief Theory**

Gradient boosting builds trees sequentially, with each new tree attempting to improve the current ensemble. Important controls include learning\_rate, n\_estimators, max\_depth and subsample.

## **5\. Procedure**

1. Prepare a classification dataset.  
2. Split into train/test.  
3. Fit XGBClassifier.  
4. Evaluate using F1 and accuracy.  
5. Change learning rate and number of estimators.  
6. Discuss the trade-off between training complexity and generalization.

## **6\. Reference Implementation**

from sklearn.datasets import load\_breast\_cancer  
from sklearn.model\_selection import train\_test\_split  
from sklearn.metrics import classification\_report  
from xgboost import XGBClassifier

X, y \= load\_breast\_cancer(return\_X\_y=True)  
Xtr, Xte, ytr, yte \= train\_test\_split(  
    X, y, test\_size=0.2, stratify=y, random\_state=42  
)

model \= XGBClassifier(  
    n\_estimators=200,  
    max\_depth=4,  
    learning\_rate=0.05,  
    subsample=0.9,  
    colsample\_bytree=0.9,  
    eval\_metric="logloss",  
    random\_state=42  
)  
model.fit(Xtr, ytr)  
pred \= model.predict(Xte)  
print(classification\_report(yte, pred))

## **7\. Evaluation / Expected Output**

* Report classification metrics.  
* Plot feature importance if required.  
* Experiment with learning\_rate × n\_estimators while keeping the test set untouched.

## **8\. Viva / Discussion Questions**

* Bagging vs boosting?  
* What does learning rate do?  
* Why can boosting overfit?

# **Experiment 11: AdaBoost**

**Category**: Ensemble learning

## **1\. Objective**

Build an adaptive boosting classifier and study how weak learners combine into a stronger ensemble.

## **2\. Suggested Dataset**

Iris or a binary subset of a standard classification dataset.

## **3\. Required Libraries**

scikit-learn

## **4\. Brief Theory**

AdaBoost sequentially emphasizes observations that previous weak learners classified incorrectly. The final prediction combines weak learners using learned weights.

## **5\. Procedure**

1. Prepare a train/test split.  
2. Fit AdaBoost with shallow decision trees.  
3. Evaluate on the test set.  
4. Vary the number of estimators and learning rate.  
5. Compare with a single decision tree.

## **6\. Reference Implementation**

from sklearn.datasets import load\_iris  
from sklearn.model\_selection import train\_test\_split  
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy\_score

X, y \= load\_iris(return\_X\_y=True)  
Xtr, Xte, ytr, yte \= train\_test\_split(  
    X, y, test\_size=0.25, stratify=y, random\_state=42  
)

model \= AdaBoostClassifier(  
    estimator=DecisionTreeClassifier(max\_depth=1, random\_state=42),  
    n\_estimators=100,  
    learning\_rate=0.5,  
    random\_state=42  
)  
model.fit(Xtr, ytr)  
pred \= model.predict(Xte)  
print("Accuracy:", accuracy\_score(yte, pred))

## **7\. Evaluation / Expected Output**

* Compare stump vs. boosted ensemble.  
* Plot validation performance as estimators increase.  
* Discuss the role of learning rate.

## **8\. Viva / Discussion Questions**

* Why use weak learners?  
* How does AdaBoost focus on difficult examples?  
* What is the effect of noisy labels?

# **Experiment 12: CatBoost**

**Category:** Ensemble learning

## **1\. Objective**

Train a gradient-boosted decision-tree model with native handling of categorical variables.

## **2\. Suggested Dataset**

Use a small tabular dataset containing both numerical and categorical columns.

## **3\. Required Libraries**

catboost, pandas, scikit-learn

## **4\. Brief Theory**

CatBoost is a gradient boosting library designed to handle categorical features effectively. The lab should emphasize correct declaration of categorical columns and avoidance of target leakage.

## **5\. Procedure**

1. Create or load a mixed-type table.  
2. Identify categorical columns.  
3. Split into train/test.  
4. Fit CatBoostClassifier.  
5. Evaluate with F1/accuracy.  
6. Inspect feature importance.

## **6\. Reference Implementation**

import pandas as pd  
from sklearn.datasets import load\_iris  
from sklearn.model\_selection import train\_test\_split  
from sklearn.metrics import accuracy\_score  
from catboost import CatBoostClassifier

data \= load\_iris(as\_frame=True)  
df \= data.frame  
df\["species\_group"\] \= pd.cut(  
    df\["sepal length (cm)"\], bins=3, labels=\["short", "medium", "long"\]  
)  
X \= df.drop(columns="target")  
y \= df\["target"\]  
cat\_cols \= \[X.columns.get\_loc("species\_group")\]

Xtr, Xte, ytr, yte \= train\_test\_split(  
    X, y, test\_size=0.2, stratify=y, random\_state=42  
)

model \= CatBoostClassifier(  
    iterations=200, depth=5, learning\_rate=0.05,  
    verbose=False, random\_seed=42  
)  
model.fit(Xtr, ytr, cat\_features=cat\_cols)  
pred \= model.predict(Xte).ravel()  
print("Accuracy:", accuracy\_score(yte, pred))

## **7\. Evaluation / Expected Output**

* State which columns are categorical.  
* Report performance and feature importance.  
* Explain why categorical handling can matter.

## **8\. Viva / Discussion Questions**

* How does CatBoost treat categorical features?  
* Why should categories not be encoded using target leakage?  
* CatBoost vs ordinary one-hot encoding?

# **Experiment 13: Multilayer Perceptron (MLP)**

**Category**: Neural networks

## **1\. Objective**

Implement a feed-forward neural network for classification and study hidden-layer size and regularization.

## **2\. Suggested Dataset**

Iris or handwritten digits.

## **3\. Required Libraries**

scikit-learn, numpy, matplotlib

## **4\. Brief Theory**

An MLP stacks affine transformations and nonlinear activation functions. Training uses backpropagation and an optimizer such as Adam or stochastic gradient descent.

## **5\. Procedure**

1. Scale features.  
2. Split data into training/test sets.  
3. Create an MLPClassifier.  
4. Train and evaluate.  
5. Vary hidden-layer size and alpha.  
6. Plot loss curve.

## **6\. Reference Implementation**

from sklearn.datasets import load\_iris  
from sklearn.model\_selection import train\_test\_split  
from sklearn.pipeline import make\_pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.neural\_network import MLPClassifier  
from sklearn.metrics import classification\_report

X, y \= load\_iris(return\_X\_y=True)  
Xtr, Xte, ytr, yte \= train\_test\_split(  
    X, y, test\_size=0.25, stratify=y, random\_state=42  
)

model \= make\_pipeline(  
    StandardScaler(),  
    MLPClassifier(  
        hidden\_layer\_sizes=(32, 16),  
        activation="relu",  
        solver="adam",  
        max\_iter=1000,  
        random\_state=42  
    )  
)  
model.fit(Xtr, ytr)  
pred \= model.predict(Xte)  
print(classification\_report(yte, pred))

## **7\. Evaluation / Expected Output**

* Report classification metrics.  
* Plot loss\_curve\_ when using MLP directly.  
* Discuss scaling and hidden-layer size.

## **8\. Viva / Discussion Questions**

* Why is scaling important for MLP?  
* What is backpropagation?  
* What causes vanishing/exploding gradients?

# **Experiment 14: Recurrent Neural Network (RNN)**

**Category**: Sequence modeling

## **1\. Objective**

Build a simple RNN for sequence classification and understand recurrent state and temporal dependencies.

## **2\. Suggested Dataset**

A synthetic sequence dataset or a small text/time-series dataset. For a first lab, synthetic sequences are recommended.

## **3\. Required Libraries**

tensorflow/keras, numpy, matplotlib

## **4\. Brief Theory**

An RNN processes a sequence one step at a time while maintaining a hidden state. Vanilla RNNs can struggle with long-term dependencies; gated variants such as LSTM/GRU address this issue.

## **5\. Procedure**

1. Generate labeled sequences.  
2. Split into training/test sets.  
3. Build an RNN-based classifier.  
4. Train for a small number of epochs.  
5. Evaluate accuracy.  
6. Vary sequence length or hidden units.

## **6\. Reference Implementation**

import numpy as np  
import tensorflow as tf  
from tensorflow.keras import Sequential  
from tensorflow.keras.layers import SimpleRNN, Dense

rng \= np.random.default\_rng(42)  
X \= rng.normal(size=(1200, 20, 1)).astype("float32")  
y \= (X.sum(axis=1).ravel() \> 0).astype("int32")

Xtr, Xte \= X\[:1000\], X\[1000:\]  
ytr, yte \= y\[:1000\], y\[1000:\]

model \= Sequential(\[  
    SimpleRNN(32, input\_shape=(20, 1)),  
    Dense(1, activation="sigmoid")  
\])  
model.compile(optimizer="adam", loss="binary\_crossentropy", metrics=\["accuracy"\])  
model.fit(Xtr, ytr, epochs=10, batch\_size=32, validation\_split=0.2, verbose=0)  
print("Test:", model.evaluate(Xte, yte, verbose=0))

## **7\. Evaluation / Expected Output**

* Record training and validation loss/accuracy.  
* Plot learning curves.  
* Discuss sequence length and recurrent state.

## **8\. Viva / Discussion Questions**

* Why does an RNN have memory?  
* What is a hidden state?  
* Why are LSTM/GRU often preferred for long dependencies?

# **Experiment 15: Self-Organizing Map (SOM)**

**Category:** Unsupervised neural learning

## **1\. Objective**

Use a Self-Organizing Map to map high-dimensional observations onto a low-dimensional grid while preserving neighborhood structure.

## **2\. Suggested Dataset**

Iris or another normalized numerical dataset.

## **3\. Required Libraries**

minisom, numpy, matplotlib

## **4\. Brief Theory**

SOM is an unsupervised competitive-learning method. Each input selects a best matching unit (BMU), and nearby units are updated toward the input. Repeated training creates an ordered map.

## **5\. Procedure**

1. Scale the input data.  
2. Create a SOM grid.  
3. Train the map.  
4. Find BMUs for samples.  
5. Visualize the U-matrix or mapped labels.  
6. Vary grid size and sigma.

## **6\. Reference Implementation**

from minisom import MiniSom  
from sklearn.datasets import load\_iris  
from sklearn.preprocessing import StandardScaler  
import numpy as np

X, y \= load\_iris(return\_X\_y=True)  
X \= StandardScaler().fit\_transform(X)

som \= MiniSom(8, 8, X.shape\[1\], sigma=1.0, learning\_rate=0.5, random\_seed=42)  
som.random\_weights\_init(X)  
som.train\_random(X, 2000\)

bmus \= np.array(\[som.winner(x) for x in X\])  
print("First ten BMUs:")  
print(bmus\[:10\])

## **7\. Evaluation / Expected Output**

* Visualize BMU locations or a U-matrix.  
* Discuss neighborhood preservation.  
* Compare 6×6 and 10×10 maps.

## **8\. Viva / Discussion Questions**

* SOM vs K-means?  
* What is a BMU?  
* Why is SOM called a topology-preserving map?

# **Experiment 16: Hidden Markov Model (HMM)**

**Category:** Probabilistic sequence modeling

## **1\. Objective**

Model a sequence using hidden states, transition probabilities and emission probabilities.

## **2\. Suggested Dataset**

A small discrete observation sequence; optionally a speech/weather-state toy example.

## **3\. Required Libraries**

hmmlearn, numpy

## **4\. Brief Theory**

HMM assumes a hidden Markov state sequence that generates observations. Core parameters are initial-state probabilities, transition probabilities and emission distributions. The Viterbi algorithm finds the most likely hidden-state path.

## **5\. Procedure**

1. Define a small observation sequence.  
2. Fit or specify a discrete HMM.  
3. Decode the most likely hidden state sequence.  
4. Inspect transition and emission probabilities.  
5. Discuss the Markov and conditional-independence assumptions.

## **6\. Reference Implementation**

import numpy as np  
from hmmlearn.hmm import CategoricalHMM

\# Observations encoded as integers: 0, 1, 2  
X \= np.array(\[\[0\], \[1\], \[2\], \[1\], \[0\], \[0\], \[1\], \[2\], \[2\], \[1\]\])

model \= CategoricalHMM(n\_components=2, n\_iter=100, random\_state=42)  
model.fit(X)

log\_prob, states \= model.decode(X, algorithm="viterbi")  
print("Log probability:", log\_prob)  
print("Most likely hidden states:", states)  
print("Transition matrix:")  
print(model.transmat\_)

## **7\. Evaluation / Expected Output**

* Report the decoded state sequence.  
* Display transition/emission probabilities.  
* Explain what the hidden states represent in your experiment.

## **8\. Viva / Discussion Questions**

* What is hidden in an HMM?  
* Viterbi vs forward algorithm?  
* What is the Markov assumption?

# **Experiment 17: Support Vector Machine (SVM)**

**Category**: Supervised / margin-based learning

## **1\. Objective**

Train an SVM classifier and investigate the effects of kernel, C and gamma.

## **2\. Suggested Dataset**

Iris, Breast Cancer or a 2-D synthetic dataset for boundary visualization.

## **3\. Required Libraries**

scikit-learn, numpy, matplotlib

## **4\. Brief Theory**

SVM seeks a decision boundary with a large margin. Kernels enable nonlinear decision boundaries. C controls the penalty for training errors; gamma affects the influence of individual samples for RBF kernels.

## **5\. Procedure**

1. Scale features.  
2. Train a linear SVM.  
3. Train an RBF SVM.  
4. Compare metrics.  
5. Vary C and gamma.  
6. Visualize a 2-D decision boundary when possible.

## **6\. Reference Implementation**

from sklearn.datasets import load\_breast\_cancer  
from sklearn.model\_selection import train\_test\_split  
from sklearn.pipeline import make\_pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.svm import SVC  
from sklearn.metrics import classification\_report

X, y \= load\_breast\_cancer(return\_X\_y=True)  
Xtr, Xte, ytr, yte \= train\_test\_split(  
    X, y, test\_size=0.2, stratify=y, random\_state=42  
)

model \= make\_pipeline(  
    StandardScaler(),  
    SVC(kernel="rbf", C=2.0, gamma="scale")  
)  
model.fit(Xtr, ytr)  
print(classification\_report(yte, model.predict(Xte)))

## **7\. Evaluation / Expected Output**

* Compare linear and RBF kernels.  
* Perform a small C × gamma experiment.  
* Discuss margin, support vectors and scaling.

## **8\. Viva / Discussion Questions**

* What is a support vector?  
* Effect of C?  
* Effect of gamma?  
* Why does SVM require careful scaling?

# **Experiment 18: Large Language Model (LLM) Experiment**

Category: Generative / foundation models

## **1\. Objective**

Demonstrate a reproducible LLM workflow for text classification or generation while understanding tokenization, inference and evaluation.

## **2\. Suggested Dataset**

A small text dataset created by students or a public sentiment/topic dataset. Do not upload private or confidential data.

## **3\. Required Libraries**

transformers, torch, pandas

## **4\. Brief Theory**

An LLM is a large neural model trained on massive text corpora. In a laboratory setting, students should focus on inference and controlled evaluation rather than attempting to train a foundation model from scratch. A useful experiment is prompt-based classification or generation with a small, openly licensed model.

## **5\. Procedure**

1. Select a small public dataset.  
2. Define a fixed prompt/template.  
3. Run inference on a fixed test subset.  
4. Store prompts, outputs and model settings.  
5. Evaluate task performance where an objective label exists.  
6. Inspect errors, hallucinations and sensitivity to prompt wording.

## **6\. Reference Implementation**

\# Example inference workflow; model availability depends on the local environment.  
from transformers import pipeline

classifier \= pipeline(  
    "sentiment-analysis",  
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"  
)

texts \= \[  
    "The laboratory session was clear and useful.",  
    "The instructions were difficult to follow."  
\]  
outputs \= classifier(texts)

for text, out in zip(texts, outputs):  
    print(text)  
    print(out)

## **7\. Evaluation / Expected Output**

* Keep the test set fixed.  
* Report the model name/version and inference settings.  
* Evaluate errors rather than only showing attractive examples.  
* Discuss privacy, bias, hallucination and reproducibility.

## **8\. Viva / Discussion Questions**

* What is tokenization?  
* Pretraining vs fine-tuning vs prompting?  
* Why is LLM output not automatically ground truth?  
* How can prompt wording affect results?

# **Experiment 19: Generalized Regression Neural Network (GRNN)**

Category: Non-parametric regression

## **1\. Objective**

Implement GRNN-style kernel regression and study the effect of smoothing parameter sigma.

## **2\. Suggested Dataset**

Use a synthetic nonlinear regression function or a small continuous target dataset.

## **3\. Required Libraries**

numpy, matplotlib, scikit-learn

## **4\. Brief Theory**

GRNN estimates a continuous target using a Gaussian kernel over training observations. The smoothing parameter controls locality: very small values can overfit, while large values can oversmooth.

## **5\. Procedure**

1. Generate a nonlinear regression dataset.  
2. Standardize features if necessary.  
3. Implement the GRNN prediction equation.  
4. Evaluate MAE/RMSE.  
5. Vary sigma.  
6. Plot predicted vs. true values.

## **6\. Reference Implementation**

import numpy as np  
from sklearn.model\_selection import train\_test\_split  
from sklearn.metrics import mean\_squared\_error

rng \= np.random.default\_rng(42)  
X \= np.linspace(-3, 3, 250).reshape(-1, 1\)  
y \= np.sin(X\[:, 0\]) \+ 0.15 \* rng.normal(size=len(X))

Xtr, Xte, ytr, yte \= train\_test\_split(X, y, test\_size=0.25, random\_state=42)

def grnn\_predict(X\_train, y\_train, X\_query, sigma=0.35):  
    d2 \= (X\_query\[:, None, :\] \- X\_train\[None, :, :\]) \*\* 2  
    d2 \= d2.sum(axis=2)  
    W \= np.exp(-d2 / (2 \* sigma\*\*2))  
    return (W @ y\_train) / (W.sum(axis=1) \+ 1e-12)

pred \= grnn\_predict(Xtr, ytr, Xte, sigma=0.35)  
print("RMSE:", mean\_squared\_error(yte, pred, squared=False))

## **7\. Evaluation / Expected Output**

* Report RMSE for several sigma values.  
* Plot the fitted curve against observations.  
* Explain under-smoothing and over-smoothing.

## **8\. Viva / Discussion Questions**

* Why is GRNN called non-parametric?  
* What does sigma control?  
* How is GRNN related to kernel regression?

# **Integrated Mini Project: Comparative Machine Learning Study**

For the final laboratory assessment, students should select one real-world dataset and compare at least three algorithms from different families. The objective is not to produce the largest score but to conduct a controlled, explainable experiment.

## **Suggested workflow**

1. Define a specific prediction or clustering problem.  
2. Identify the target variable (for supervised learning) or justify the absence of labels.  
3. Inspect missing values, duplicates, class balance and feature types.  
4. Create a train/test split before fitting preprocessing steps that learn from data.  
5. Build a simple baseline.  
6. Train at least three candidate algorithms.  
7. Use cross-validation on the training data for model selection where appropriate.  
8. Evaluate once on the untouched test set.  
9. Analyze errors, not only aggregate metrics.  
10. Document limitations, reproducibility settings and possible future improvements.

## **Recommended comparative table**

| Model | Key Hyperparameters | Validation Metric | Test Metric | Training Time | Observation |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Baseline |  |  |  |  |  |
| Model 1 |  |  |  |  |  |
| Model 2 |  |  |  |  |  |
| Model 3 |  |  |  |  |  |
| Model 4 (optional) |  |  |  |  |  |

## **Common experimental mistakes to avoid**

* Scaling the entire dataset before the train/test split, which can leak test-set information.  
* Selecting hyperparameters using the test set repeatedly.  
* Comparing models on different train/test partitions.  
* Reporting only accuracy on imbalanced classification problems.  
* Treating cluster labels as ground truth without understanding the evaluation method.  
* Changing several hyperparameters at once without recording the experiment.  
* Using private or sensitive data in external LLM services.  
* Claiming causation from feature importance or correlation alone.  
* Copying library output without explaining the underlying algorithm.

# **Appendix A: Quick Reference Formulas**

| Concept | Formula / Definition |
| :---- | :---- |
| Euclidean distance | d(x,y) \= sqrt(Σ\_j (x\_j − y\_j)^2) |
| K-means objective | J \= Σ\_i ||x\_i − μ\_{c\_i}||² |
| Regression MAE | MAE \= (1/n) Σ\_i |y\_i − ŷ\_i| |
| Regression RMSE | RMSE \= sqrt\[(1/n) Σ\_i (y\_i − ŷ\_i)²\] |
| R² | R² \= 1 − SS\_res / SS\_tot |
| Precision | TP / (TP \+ FP) |
| Recall | TP / (TP \+ FN) |
| F1 | 2 × Precision × Recall / (Precision \+ Recall) |
| Sigmoid | σ(z) \= 1 / (1 \+ e^(−z)) |
| SVM margin idea | Maximize margin while penalizing violations via C |
| GRNN kernel | w\_i(x) \= exp(−||x−x\_i||² / (2σ²)) |

# **Appendix B: Reproducibility Checklist**

* Python version recorded.  
* Package versions recorded.  
* Random seed recorded where supported.  
* Dataset source and version recorded.  
* Preprocessing steps recorded.  
* Train/validation/test protocol recorded.  
* Hyperparameters recorded.  
* Evaluation metrics defined before final testing.  
* Plots and tables saved with meaningful names.  
* Code runs from a clean environment or requirements file.

# **Appendix C: Suggested Assessment Rubric**

| Component | Suggested Weight |
| :---- | :---- |
| Pre-lab preparation / algorithm understanding | 10% |
| Correct implementation | 25% |
| Experimental design and preprocessing | 15% |
| Evaluation and visualization | 15% |
| Analysis and interpretation | 20% |
| Report quality / reproducibility | 10% |
| Viva / discussion | 5% |

# **Appendix D: Student Experiment Record**

| Field | Student Entry |
| :---- | :---- |
| Name |  |
| ID |  |
| Section |  |
| Experiment No. |  |
| Date |  |
| Dataset |  |
| Algorithm / Version |  |
| Key Hyperparameters |  |
| Random Seed |  |
| Main Result |  |
| Observation |  |
| Instructor Signature |  |

