import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the data
df = pd.read_csv('dataset.csv')


# Prepare the data
X = df[['danceability', 'energy', 'loudness', 'valence', 'tempo']]
y = df['popularity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'mse': mse,
        'r2': r2,
        'predictions': y_pred
    }
    
    print(f"\n{name} Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Identify the best model
best_model_name = min(results, key=lambda x: results[x]['mse'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name}")

# Feature importance for Random Forest
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

# Save the best model and scaler using pickle
model_data = {
    'model': best_model,
    'scaler': scaler,
    'features': ['danceability', 'energy', 'loudness', 'valence', 'tempo']
}

with open('song_popularity_predictor.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel and scaler saved to 'song_popularity_predictor.pkl'")

# Example of how to use the saved model
def predict_popularity(danceability, energy, loudness, valence, tempo):
    # Load the model
    with open('song_popularity_predictor.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Prepare input data
    input_data = np.array([[danceability, energy, loudness, valence, tempo]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

# Example usage
print("\nExample prediction:")
example_song = {
    'danceability': 0.7,
    'energy': 0.5,
    'loudness': -8.0,
    'valence': 0.6,
    'tempo': 120.0
}
predicted_popularity = predict_popularity(**example_song)
print(f"Predicted popularity: {predicted_popularity:.2f}")