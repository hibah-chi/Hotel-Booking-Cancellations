# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("hotel_booking.csv")

# Drop irrelevant or highly correlated features (e.g., reservation_status, agent, company)
df = df.drop(['reservation_status', 'reservation_status_date', 'agent', 'company'], axis=1)

# Handling missing values appropriately
for column in df.columns:
    if df[column].dtype == 'object':  # Fill missing categorical values with the mode
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # Fill missing numerical values with the median
        df[column].fillna(df[column].median(), inplace=True)

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# Address class imbalance (if needed)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data (optional for tree-based models, but useful for PCA or distance-based models)
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

# Combine predictions using majority voting
predictions = np.vstack((dt_pred, rf_pred))
ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# Evaluate the models
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Ensemble Classification Report:\n", classification_report(y_test, ensemble_pred))

# Confusion Matrix Plot
def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(y_test, dt_pred, "Decision Tree")
plot_confusion_matrix(y_test, rf_pred, "Random Forest")
plot_confusion_matrix(y_test, ensemble_pred, "Ensemble")


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