# Install necessary libraries
%pip install seaborn scikit-learn lightgbm xgboost tabulate optuna

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from xgboost import XGBClassifier
from tabulate import tabulate
import optuna
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Load datasets
try:
    train = pd.read_csv('Train_data.csv')
    test = pd.read_csv('Test_data.csv')
except FileNotFoundError:
    print("Train_data.csv or Test_data.csv not found. Please check file paths.")
    exit()

# Data inspection
print("Train Data Preview:")
print(train.head())
print("\nTrain Data Info:")
print(train.info())
print("\nTrain Data Description (Numerical):")
print(train.describe())
print("\nTrain Data Description (Categorical):")
print(train.describe(include='object'))
print("\nTrain Data Shape:", train.shape)

# Missing values
print("\nMissing Values in Train Data:")
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count / train.shape[0]) * 100
    print(f"{col}: {null_count} missing ({round(per, 3)}%)")

# Duplicate rows
print(f"\nNumber of duplicate rows in train data: {train.duplicated().sum()}")

# Class distribution
if 'class' in train.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=train['class'])
    plt.title("Class Distribution in Train Data")
    plt.show()

    print("\nClass distribution in Training set:")
    print(train['class'].value_counts())
else:
    print("No 'class' column found in train data.")

# Label encoding
def label_encode(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

label_encode(train)
label_encode(test)

# Drop unnecessary column
if 'num_outbound_cmds' in train.columns:
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
if 'num_outbound_cmds' in test.columns:
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Feature selection
if 'class' not in train.columns:
    print("The 'class' column is missing from the train dataset. Exiting.")
    exit()

X_train = train.drop(['class'], axis=1)
Y_train = train['class']

rfc = RandomForestClassifier(random_state=42)
rfe = RFE(rfc, n_features_to_select=10)
rfe.fit(X_train, Y_train)

selected_features = [v for i, v in zip(rfe.get_support(), X_train.columns) if i]

print("\nSelected Features:", selected_features)

X_train = X_train[selected_features]
X_test = test[selected_features]

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=42)
print("\nTrain-Test Split Shapes:")
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Model training and evaluation function
def evaluate_model(model, x_train, y_train, x_test, y_test):
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(x_test)
    testing_time = time.time() - start_time

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print(f"\nModel: {type(model).__name__}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Testing time: {testing_time:.2f} seconds")
    print(f"Training Score: {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")

    return train_score, test_score, y_pred

# Logistic Regression
lr = LogisticRegression(max_iter=1200000, random_state=42)
lg_train, lg_test, lr_preds = evaluate_model(lr, x_train, y_train, x_test, y_test)

# Hyperparameter optimization with Optuna
def optimize_knn(trial):
    n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    return knn.score(x_test, y_test)

print("\nOptimizing KNN with Optuna:")
study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(optimize_knn, n_trials=30)
best_knn_params = study_knn.best_trial.params

knn = KNeighborsClassifier(n_neighbors=best_knn_params['KNN_n_neighbors'])
knn_train, knn_test, knn_preds = evaluate_model(knn, x_train, y_train, x_test, y_test)

# Decision Tree
def optimize_dt(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10)
    dt = DecisionTreeClassifier(max_features=dt_max_features, max_depth=dt_max_depth, random_state=42)
    dt.fit(x_train, y_train)
    return dt.score(x_test, y_test)

print("\nOptimizing Decision Tree with Optuna:")
study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(optimize_dt, n_trials=30)
best_dt_params = study_dt.best_trial.params

dt = DecisionTreeClassifier(max_features=best_dt_params['dt_max_features'], max_depth=best_dt_params['dt_max_depth'], random_state=42)
dt_train, dt_test, dt_preds = evaluate_model(dt, x_train, y_train, x_test, y_test)

# Model performance comparison
data = [
    ["KNN", knn_train, knn_test],
    ["Logistic Regression", lg_train, lg_test],
    ["Decision Tree", dt_train, dt_test]
]

col_names = ["Model", "Train Score", "Test Score"]
print("\nModel Performance Comparison:")
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

# Confusion matrix and classification report
target_names = ["normal", "anomaly"]
for name, preds in zip(["KNN", "Logistic Regression", "Decision Tree"], [knn_preds, lr_preds, dt_preds]):
    print(f"\n{name} Model:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=target_names))

# F1-score
f1_scores = {name: f1_score(y_test, preds, average='weighted') * 100
             for name, preds in zip(["KNN", "Logistic Regression", "Decision Tree"], [knn_preds, lr_preds, dt_preds])}
print("\nF1 Scores:")
print(f1_scores)

# F1-score plot
f1_df = pd.DataFrame(f1_scores.values(), index=f1_scores.keys(), columns=["F1-score"])
f1_df.plot(kind="bar", ylim=[80, 100], figsize=(10, 6), rot=0, color='skyblue')
plt.title("F1-Scores")
plt.ylabel("Score (%)")
plt.show()
