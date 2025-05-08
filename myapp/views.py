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

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect
import pickle
import os
import numpy as np
from django.conf import settings
from .forms import CustomUserForm
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
import pandas as pd
import io
import base64
from django.views.decorators.cache import never_cache
from django.contrib.auth import logout

def landing_view(request):
    return render(request, 'index.html')

def register(request):
    if request.method == 'POST':
        form = CustomUserForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password']) 
            user.save()
            return redirect('login')
    else:
        form = CustomUserForm()
    return render(request, 'register.html', {'form': form})


@login_required(login_url='login')
@never_cache
def dashboard(request):
    # Load dataset
    dataset_path = os.path.join(settings.BASE_DIR, 'myapp', 'model', 'dataset.csv')
    df = pd.read_csv(dataset_path)

    # Dataset Summary
    total_tracks = len(df)
    avg_tempo = round(df['tempo'].mean(), 2)
    avg_loudness = round(df['loudness'].mean(), 2)
    avg_valence = round(df['valence'].mean(), 2)

    # Bar Chart: Popular vs Unpopular
    df['popularity_binary'] = df['popularity'].apply(lambda x: 1 if x >= 70 else 0)
    pop_counts = df['popularity_binary'].value_counts()
    pop_chart = plot_bar_chart(['Unpopular', 'Popular'], pop_counts.sort_index().values, "Popularity Distribution")

    # Feature Importance Chart (dummy data for now)
    feature_importance_chart = plot_bar_chart(
        ['danceability', 'energy', 'loudness', 'valence', 'tempo'],
        [0.25, 0.2, 0.15, 0.25, 0.15],  # update later with real values
        "Feature Importance"
    )

    # Histogram: Valence Distribution
    valence_hist = plot_histogram(df['valence'], "Valence Distribution", bins=10)

    accuracy = "92%" 

    return render(request, 'dashboard.html', {
        'total_tracks': total_tracks,
        'avg_tempo': avg_tempo,
        'avg_loudness': avg_loudness,
        'avg_valence': avg_valence,
        'pop_chart': pop_chart,
        'feature_chart': feature_importance_chart,
        'valence_hist': valence_hist,
        'accuracy': accuracy
    })

def plot_bar_chart(labels, values, title):
    fig, ax = plt.subplots()
    ax.bar(labels, values, color='skyblue')
    ax.set_title(title)
    return fig_to_base64(fig)

def plot_histogram(data, title, bins=10):
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, color='salmon', edgecolor='black')
    ax.set_title(title)
    return fig_to_base64(fig)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{image_base64}"

def predict_popularity(request):
    """
    View function for the music popularity predictor form page.
    Handles both GET and POST requests.
    """
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        try:
            danceability = float(request.POST.get('danceability', 0))
            energy = float(request.POST.get('energy', 0))
            tempo = float(request.POST.get('tempo', 0))
            valence = float(request.POST.get('valence', 0))
            loudness = float(request.POST.get('loudness', 0))
            
            model_path = os.path.join(settings.BASE_DIR, 'myapp', 'model', 'popularity_model.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'myapp', 'model', 'scaler.pkl')

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            input_data = np.array([[danceability, energy, loudness, valence, tempo]])
            input_scaled = scaler.transform(input_data)

            prob = model.predict_proba(input_scaled)[0][1]
            prediction = 1 if prob >= 0.5 else 0
            confidence = round(prob * 100, 2)


        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = "Error: Could not make prediction"
            confidence = None

    return render(request, 'predict_form.html', {
        'prediction': prediction,
        'confidence': confidence
    })

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('index') 
    return render(request, 'logout.html')
