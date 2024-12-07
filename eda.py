import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ================================
# Step 1: Define Purpose
# ================================
# Objective: Predict student academic success (Graduate, Enrolled, Dropout) using classification models.

# File paths
train_path = '/Users/user/Desktop/USD/ADS 502/final project/playground-series-s4e6/train.csv'
test_path = '/Users/user/Desktop/USD/ADS 502/final project/playground-series-s4e6/test.csv'

# ================================
# Step 2: Obtain Data
# ================================
# Load datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# Separate features and target variable
X = train.drop(columns=['Target'])
y = train['Target']

# ================================
# Step 3: Data Cleaning and Preparation
# ================================
# Manually specify categorical columns
categorical_features = [
    'Marital status', 'Application order', 'Daytime/evening attendance', 
    'Displaced', 'Educational special needs', 'Debtor', 
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 
    'International'
]

# Automatically identify numeric features
numeric_features = [col for col in X.columns if col not in categorical_features]

print(f"Numeric Features: {numeric_features}")
print(f"Categorical Features: {categorical_features}")

# Normalize numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Correlation heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(
    X[numeric_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={'label': 'Correlation Coefficient'},
    annot_kws={'size': 8}
)
plt.title("Correlation Heatmap (Numeric Features)", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data['Feature'] = numeric_features
vif_data['VIF'] = [variance_inflation_factor(X[numeric_features].values, i) for i in range(len(numeric_features))]
print("\nVariance Inflation Factor (VIF):\n", vif_data.sort_values(by='VIF', ascending=False))

# ================================
# Step 4: Dimension Reduction with PCA
# ================================
# Apply PCA to reduce dimensions while retaining 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X[numeric_features])
print(f"Number of components after PCA: {pca.n_components_}")

# Visualize explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='orange')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================
# Step 5: Model Implementation
# ================================
# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Logistic Regression (Baseline Model)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_val)
y_prob_lr = log_reg.predict_proba(X_val) 
print("\nLogistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_lr))
print("Classification Report:\n", classification_report(y_val, y_pred_lr))

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_val)
y_prob_dt = decision_tree.predict_proba(X_val)
print("\nDecision Tree Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_dt))
print("Classification Report:\n", classification_report(y_val, y_pred_dt))

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_val)
y_prob_rf = random_forest.predict_proba(X_val)
print("\nRandom Forest Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Classification Report:\n", classification_report(y_val, y_pred_rf))

# ================================
# Step 6: Model Performance Visualization
# ================================
# Confusion matrices
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=['Dropout', 'Enrolled', 'Graduate'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dropout', 'Enrolled', 'Graduate'], yticklabels=['Dropout', 'Enrolled', 'Graduate'])
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion_matrix(y_val, y_pred_lr, "Logistic Regression")
plot_confusion_matrix(y_val, y_pred_dt, "Decision Tree")
plot_confusion_matrix(y_val, y_pred_rf, "Random Forest")

# Feature importance for Random Forest
feature_importances = pd.Series(random_forest.feature_importances_, index=[f'PC{i+1}' for i in range(pca.n_components_)])
top_features = feature_importances.nlargest(10)

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', title="Top 10 Feature Importances (Random Forest)", color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Principal Components")
plt.tight_layout()
plt.show()

# ================================
# Step 7: Model Comparison
# ================================
model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_val, y_pred_lr),
        accuracy_score(y_val, y_pred_dt),
        accuracy_score(y_val, y_pred_rf)
    ]
})
print("\nModel Comparison:\n", model_comparison)

# Display model comparison as a bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x="Accuracy", y="Model", data=model_comparison, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.tight_layout()
plt.show()


# ================================
# Step 8: ROC Curve Comparisons 
# ================================
# Encode target labels for multi-class ROC
y_val_bin = label_binarize(y_val, classes=['Dropout', 'Enrolled', 'Graduate'])

# ROC Calc Logistic Regression
fpr_lr = {}
tpr_lr = {}
roc_auc_lr = {}
for i in range(3):  
    fpr_lr[i], tpr_lr[i], _ = roc_curve(y_val_bin[:, i], y_prob_lr[:, i])
    roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

# ROC Calc Decision Tree
fpr_dt = {}
tpr_dt = {}
roc_auc_dt = {}
for i in range(3):  
    fpr_dt[i], tpr_dt[i], _ = roc_curve(y_val_bin[:, i], y_prob_dt[:, i])
    roc_auc_dt[i] = auc(fpr_dt[i], tpr_dt[i])

# ROC Calc Random Forest
fpr_rf = {}
tpr_rf = {}
roc_auc_rf = {}
for i in range(3):  
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_val_bin[:, i], y_prob_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])


# Plot ROC curves for Logistic Regression
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr_lr[i], tpr_lr[i], lw=2, label=f'Class {["Dropout", "Enrolled", "Graduate"][i]} (AUC = {roc_auc_lr[i]:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True, linestyle='--', color='gray')  
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Plot ROC curves for Decision Tree
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr_dt[i], tpr_dt[i], lw=2, label=f'Class {["Dropout", "Enrolled", "Graduate"][i]} (AUC = {roc_auc_dt[i]:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.title('ROC Curve - Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True, linestyle='--', color='gray')  
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Plot ROC curves for Random Forest
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr_rf[i], tpr_rf[i], lw=2, label=f'Class {["Dropout", "Enrolled", "Graduate"][i]} (AUC = {roc_auc_rf[i]:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True, linestyle='--', color='gray')  
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
