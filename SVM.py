import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import pickle


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
# 5. Reset index
# =========================
df.reset_index(drop=True, inplace=True)

# =========================
# 6. Encode categorical variables
# =========================
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Anemia_Status'] = df['Anemia_Status'].map({"Anemic": 0, "NonAnemic": 1})


# =========================
# 6.1 Print counts after encoding
# =========================
print(df)
print("\n--- Category Counts ---")
print("Total Males   :", (df['Sex'] == 1).sum())
print("Total Females :", (df['Sex'] == 0).sum())
print("Total Anemic (0)     :", (df['Anemia_Status'] == 0).sum())
print("Total Non-Anemic (1) :", (df['Anemia_Status'] == 1).sum())



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

with open("cbc_scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

# =========================
# 9. Train and Test Splitting
# =========================
# x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# =========================
# 9. Appply SMOTE to balace data
# =========================
smote=SMOTE(random_state=42,sampling_strategy=0.8)
x_balanced,y_balanced=smote.fit_resample(X_scaled,y)

print("---After Balacing---")
print("Total Anemic (0) : ",(y_balanced==0).sum())
print("Total Non Anemic (1) : ",(y_balanced==1).sum())


# =========================
# 10. K-Fold Cross validation & Implement SVM
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []
fold = 1  

for train_idx, test_idx in kf.split(x_balanced, y_balanced):
    print(f"\n===== Fold {fold} =====")
    
    # Split into train/test based on current fold
    X_train, X_test = x_balanced.iloc[train_idx], x_balanced.iloc[test_idx]
    y_train, y_test = y_balanced.iloc[train_idx], y_balanced.iloc[test_idx]
    
    # Train SVM
    model = svm.LinearSVC(random_state=0, max_iter=10000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    fold += 1   



print("Fold Accuracy : ", fold_accuracies)
print("Average Accuracy : ",np.mean(fold_accuracies))


# =========================
# 11. Function For assign TP,TN,FP,FN
# =========================

def add_confusion_labels(y_actual,y_predicted):
    results=[]
    for actual,pred in zip(y_actual,y_predicted):
        if actual == 1 and pred==1:
            results.append("TP")
        elif actual==0 and pred==0:
            results.append("TN")
        elif actual==1 and pred==0:
            results.append("FN")
        elif actual==0 and pred==1:
            results.append("FP")
    return pd.DataFrame({"Actual":y_actual.values,"Predicted":y_predicted,"Result":results})

conf_mat=add_confusion_labels(y_test,y_pred)
print(conf_mat)

tp=(conf_mat["Result"]=='TP').sum()
tn=(conf_mat['Result']=='TN').sum()
fp=(conf_mat['Result']=='FP').sum()
fn=(conf_mat['Result']=='FN').sum()

print("Confusion Metrix Counts : ")
print(f"TP : {tp}")
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")


Sensitivity=tp/(tp+fn)

print("Sensitivity :: ",Sensitivity)

Specificity=tn/(tn+fp)

print("Specificity  ::  ",Specificity)

type_1_error=fp/(fp+tn)
print("Type 1 error : ",type_1_error)

type_2_error=fn/(fn+tp)
print("Type 2 Error : ",type_2_error)

with open("cbc_svm_model.pkl","wb") as f:
    pickle.dump(model,f)
print("Model saved.")