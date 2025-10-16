# Step 1: Import libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Read CSV
data = pd.read_csv('datafiles/Advertising.csv')  # replace with your file path

# Step 3: Select numeric features
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Step 4: Separate features and target
X = numeric_data.drop('sales', axis=1)  # replace 'target_column' with your target
y = numeric_data['sales']

# Step 5: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Apply PCA (e.g., 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 7: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 8: Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = lr.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
