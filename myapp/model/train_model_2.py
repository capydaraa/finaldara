import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('dataset.csv')

# Create binary target from popularity column
df['popularity_binary'] = df['popularity'].apply(lambda x: 1 if x >= 70 else 0)

# Select features and target
features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']
X = df[features]
y = df['popularity_binary']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'popularity_model.pkl')
print("Model trained and saved as 'popularity_model.pkl'")
