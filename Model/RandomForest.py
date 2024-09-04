# Obesity Prediction Model Training and Evaluation

## Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Step 1: Load the Dataset
df = pd.read_csv('obesity_data.csv')

# Display the first few rows of the dataset
df.head()

# Step 2: Data Exploration
# Display the basic information about the dataset
df.info()

# Display the summary statistics of the dataset
df.describe()

# Check for any missing values
df.isnull().sum()

# Step 3: Data Visualization
# Plot the distribution of the ObesityCategory
plt.figure(figsize=(8, 6))
sns.countplot(x='ObesityCategory', data=df, palette='viridis')
plt.title('Distribution of Obesity Categories')
plt.show()

# Pairplot of all the features
sns.pairplot(df, hue='ObesityCategory', palette='viridis')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Feature Selection and Preprocessing
# Define features (X) and target variable (y)
X = df[['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']]
y = df['ObesityCategory']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training
# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Model Evaluation
# Predict the test set results
y_pred = model.predict(X_test_scaled)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Feature Importance
feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features, palette='viridis')
plt.title('Feature Importance')
plt.show()

# Step 7: Save the Model and Scaler
# Save the trained model
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Step 8: Load the Model and Scaler (for future use)
# Example of loading the model and scaler back into memory
with open('rf_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Optionally, you can test the loaded model to ensure it works correctly
test_accuracy = loaded_model.score(X_test_scaled, y_test)
print(f'Loaded model accuracy: {test_accuracy:.2f}')
