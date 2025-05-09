# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# # Load dataset
# df = pd.read_csv('myapp/model/dataset.csv')

# # Create binary target from popularity column
# df['popularity_binary'] = df['popularity'].apply(lambda x: 1 if x >= 70 else 0)

# # Select features and target
# features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']
# X = df[features]
# y = df['popularity_binary']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Save model
# joblib.dump(model, 'popularity_model.pkl')
# joblib.dump(scaler, 'scaler.pkl') 
# print("Model trained and saved as 'popularity_model.pkl'")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv('myapp/model/dataset.csv')  # full relative path

# Create binary target
df['popularity_binary'] = df['popularity'].apply(lambda x: 1 if x >= 70 else 0)

# Select features and target
features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']
X = df[features]
y = df['popularity_binary']

# Scale features
scaler = StandardScaler()  # ✅ define the scaler
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'popularity_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved.")
