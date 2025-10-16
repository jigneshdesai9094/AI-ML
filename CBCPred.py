import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. Load dataset
# =========================
df = pd.read_excel("./data/CBC-Dataset.xlsx", sheet_name="cbc-dataset")

# =========================
# 2. Drop duplicates
# =========================
df.drop_duplicates(inplace=True)

# =========================
# 3. Handle missing values
# =========================
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())   # numeric → median
    else:
        df[col] = df[col].fillna(df[col].mode()[0])  # categorical → mode

# =========================
# 4. Fix invalid values / outliers
# =========================
df = df[df['wbc'] >= 1000]
df.loc[df['neutrophils'] > 100, 'neutrophils'] = 100
df.loc[df['neutrophils'] < 0, 'neutrophils'] = 0
df = df[(df['mchc'] >= 20) & (df['mchc'] <= 40)]
df = df[df['Age'] > 0]

# =========================
# 5. Encode categorical variables
# =========================
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Anemia_Status'] = df['Anemia_Status'].map({"Anemic": 0, "NonAnemic": 1})


# =========================
# 5.1 Print counts after encoding
# =========================
print("\n--- Category Counts ---")
print("Total Males   :", (df['Sex'] == 1).sum())
print("Total Females :", (df['Sex'] == 0).sum())
print("Total Anemic (0)     :", (df['Anemia_Status'] == 0).sum())
print("Total Non-Anemic (1) :", (df['Anemia_Status'] == 1).sum())

# =========================
# 6. Reset index
# =========================
df.reset_index(drop=True, inplace=True)

# =========================
# 7. Split features & target
# =========================
X = df.drop(columns=['Anemia_Status'])
y = df['Anemia_Status']


# =========================
# 8. Min-Max Scaling
# =========================
numeric_cols = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and col != 'Sex']

X_scaled = X.copy()
scaler = MinMaxScaler()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print("Original:\n", df.head(5))
print("Scaled:\n", X_scaled.head(5))


# =========================
# 9. K-Fold 
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)

fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y), 1):
    # Split data
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train model
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")

# =========================
# 10. Results
# =========================
print("Accuracy per fold:", fold_accuracies)
print("Mean Accuracy:", np.mean(fold_accuracies))

# =========================
# Function to assign TP, TN, FP, FN
# =========================
def add_confusion_labels(y_true, y_pred):
    results = []
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            results.append("TP")
        elif actual == 0 and pred == 0:
            results.append("TN")
        elif actual == 0 and pred == 1:
            results.append("FP")
        elif actual == 1 and pred == 0:
            results.append("FN")
    return pd.DataFrame({"Actual": y_true.values, "Predicted": y_pred, "Result": results})

result=add_confusion_labels(y_test,y_pred)
print(result.head(60))

# =========================
# 11. Count TP, TN, FP, FN
# =========================
tp = (result['Result'] == 'TP').sum()
tn = (result['Result'] == 'TN').sum()
fp = (result['Result'] == 'FP').sum()
fn = (result['Result'] == 'FN').sum()

print("Confusion Matrix Counts:")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

accuracy=(tp+tn)/(tp+tn+fp+fn)
print("Accuracy :: ",accuracy)

Sensitivity=tp/(tp+fn)

print("Sensitivity :: ",Sensitivity)

Specificity=tn/(tn+fp)

print("Specificity  ::  ",Specificity)

type_1_error=fp/(fp+tn)
print("Type 1 error : ",type_1_error)

type_2_error=fn/(fn+tp)
print("Type 2 Error : ",type_2_error)