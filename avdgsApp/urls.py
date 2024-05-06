# urls.py

from django.urls import path
from avdgsApp import views

urlpatterns = [
    path('camera_feed/', views.camera_feed, name='camera_feed'),
]
