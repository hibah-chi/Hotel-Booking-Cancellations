# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv("merged_dataset.csv")

# Drop irrelevant columns
df = df.drop(columns=['reservation_status_date'], axis=1, errors='ignore')

# Handle mixed data types by converting all object columns to string if necessary
for column in df.select_dtypes(include=['object']).columns:
    if df[column].apply(type).nunique() > 1:
        # Convert column to string
        df[column] = df[column].astype(str)

# Apply Label Encoding to categorical columns
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

# Features and target
X = df.drop('is_canceled', axis=1)  # Keep the 'is_canceled' column as the target
y = df['is_canceled']

# Impute missing values
imputer = SimpleImputer(strategy='median')  # Use 'mean', 'median', or 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_imputed, y = smote.fit_resample(X_imputed, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Decision Tree and Random Forest classifiers
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predict with both models
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Convert predictions to numpy arrays and ensure they are integers
dt_pred = np.array(dt_pred, dtype=int)
rf_pred = np.array(rf_pred, dtype=int)

# Combine predictions into a 2D array (rows are models, columns are predictions)
predictions = np.vstack((dt_pred, rf_pred))

# Calculate majority vote along the first axis (row-wise)
ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# Calculate metrics for Decision Tree
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Calculate metrics for Random Forest
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Calculate metrics for Ensemble
ensemble_precision = precision_score(y_test, ensemble_pred)
ensemble_recall = recall_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

# Print the results
print("Decision Tree - Precision:", dt_precision, "Recall:", dt_recall, "F1 Score:", dt_f1, "Accuracy:", dt_accuracy)
print("Random Forest - Precision:", rf_precision, "Recall:", rf_recall, "F1 Score:", rf_f1, "Accuracy:", rf_accuracy)
print("Ensemble - Precision:", ensemble_precision, "Recall:", ensemble_recall, "F1 Score:", ensemble_f1, "Accuracy:", ensemble_accuracy)

print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Ensemble Classification Report:\n", classification_report(y_test, ensemble_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ensemble_pred))



#%%

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion Matrix Plot
def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot Decision Tree Confusion Matrix
plot_confusion_matrix(y_test, dt_pred, "Decision Tree")

# Plot Random Forest Confusion Matrix
plot_confusion_matrix(y_test, rf_pred, "Random Forest")

# Plot Ensemble Confusion Matrix
plot_confusion_matrix(y_test, ensemble_pred, "Ensemble")

# Classification Metrics Bar Plot
def plot_classification_metrics(y_test, predictions, model_name):
    report = classification_report(y_test, predictions, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop(['support'], axis=1)  # Drop support as it's not a metric

    # Only plot for 'precision', 'recall', and 'f1-score'
    metrics_to_plot = metrics_df.loc[['0', '1'], ['precision', 'recall', 'f1-score']]

    metrics_to_plot.plot(kind='bar', figsize=(8, 6))
    plt.title(f"Classification Metrics - {model_name}")
    plt.xlabel("Classes (0: Not Canceled, 1: Canceled)")
    plt.ylabel("Metric Score")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plot Decision Tree Metrics
plot_classification_metrics(y_test, dt_pred, "Decision Tree")

# Plot Random Forest Metrics
plot_classification_metrics(y_test, rf_pred, "Random Forest")

# Plot Ensemble Metrics
plot_classification_metrics(y_test, ensemble_pred, "Ensemble")


#%%
'''
# Visualize feature distributions
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.show()
''' 

#%%
from scipy.stats import ttest_rel

# Compare Decision Tree and Random Forest Precision
t_stat, p_val = ttest_rel([dt_precision, dt_recall], [rf_precision, rf_recall])
print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")




