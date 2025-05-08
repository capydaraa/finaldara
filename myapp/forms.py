# predictor/forms.py

from django import forms
from django.contrib.auth.models import User

class CustomUserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

class SongFeatureForm(forms.Form):
    track_name = forms.CharField(max_length=255, required=True)
    artist = forms.CharField(max_length=255, required=True)
    duration_ms = forms.IntegerField(min_value=1000, required=True)
    explicit = forms.IntegerField(min_value=0, max_value=1, required=True)  # Changed to IntegerField for simplicity
    danceability = forms.FloatField(min_value=0, max_value=1, required=True)
    energy = forms.FloatField(min_value=0, max_value=1, required=True)
    key = forms.IntegerField(min_value=0, max_value=11, required=True)
    loudness = forms.FloatField(required=True)
    mode = forms.IntegerField(min_value=0, max_value=1, required=True)
    speechiness = forms.FloatField(min_value=0, max_value=1, required=True)
    acousticness = forms.FloatField(min_value=0, max_value=1, required=True)
    instrumentalness = forms.FloatField(min_value=0, max_value=1, required=True)
    liveness = forms.FloatField(min_value=0, max_value=1, required=True)
    valence = forms.FloatField(min_value=0, max_value=1, required=True)
    tempo = forms.FloatField(min_value=0, required=True)
    time_signature = forms.IntegerField(min_value=1, max_value=7, required=True)

    