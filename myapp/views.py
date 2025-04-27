# from django.shortcuts import render
# from django.conf import settings
# import os
# import joblib
# import numpy as np

# # Load the model once at the start of the application
# MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp', 'model', 'random_forest_hit_predictor.pkl')
# model = joblib.load(MODEL_PATH)

# def predict_popularity(request):
#     prediction = None
#     if request.method == 'POST':
#         try:
#             # Collect features from the form
#             features = [
#                 float(request.POST.get('danceability')),
#                 float(request.POST.get('energy')),
#                 float(request.POST.get('tempo')),
#                 float(request.POST.get('valence')),
#                 float(request.POST.get('loudness'))
#             ]

#             # Reshape features into a 2D array for prediction
#             prediction = model.predict([features])[0]

#         except Exception as e:
#             prediction = f"Error: {str(e)}"

#     return render(request, 'index.html', {'prediction': prediction})

from django.shortcuts import render
import pickle
import os
import numpy as np
from django.conf import settings

def predict_popularity(request):
    """
    View function for the music popularity predictor home page.
    Handles both GET and POST requests.
    """
    prediction = None
    
    if request.method == 'POST':
        # Extract feature values from the form
        try:
            danceability = float(request.POST.get('danceability', 0))
            energy = float(request.POST.get('energy', 0))
            tempo = float(request.POST.get('tempo', 0))
            valence = float(request.POST.get('valence', 0))
            loudness = float(request.POST.get('loudness', 0))
            
            # Load the model
            model_path = os.path.join(settings.BASE_DIR, 'song_popularity_predictor.pkl')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Get the model components
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Scale the input features
            input_data = np.array([[danceability, energy, loudness, valence, tempo]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Round to 2 decimal places for display
            prediction = round(prediction, 2)
            
        except Exception as e:
            # Log the error (in a real app, use proper logging)
            print(f"Error making prediction: {e}")
            # Provide a fallback
            prediction = "Error: Could not make prediction"
    
    # Render the template with the prediction result
    return render(request, 'index.html', {
        'prediction': prediction,
    })


def about(request):
    """
    View function for the about page explaining the model.
    """
    return render(request, 'about.html')