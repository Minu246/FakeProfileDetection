from django.conf.urls import url
from user_app.views import UserPredictionView

urlpatterns = [
    url(r'^predict/$', UserPredictionView.as_view(), name='user_prediction'),
]
