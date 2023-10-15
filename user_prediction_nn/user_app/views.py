import os
from django.shortcuts import render
from django.views.generic import View
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.datasets import ClassificationDataSet
import pandas as pd
import numpy as np
import sexmachine.detector as gender
import cPickle
from sklearn.ensemble import RandomForestClassifier


class UserPredictionView(View):
    def get(self, request):
        return render(request, 'predict_form.html')

    def post(self, request):
        # Get the model selection and user data from the form
        model_selection = request.POST.get('model_selection')
        name = request.POST.get('name')
        statuses_count = int(request.POST.get('statuses_count'))
        followers_count = int(request.POST.get('followers_count'))
        friends_count = int(request.POST.get('friends_count'))
        favourites_count = int(request.POST.get('favourites_count'))
        listed_count = int(request.POST.get('listed_count'))
        lang = request.POST.get('lang')

        # Create a DataFrame from the user data
        user_data = pd.DataFrame({'statuses_count': [statuses_count],
                                  'followers_count': [followers_count],
                                  'friends_count': [friends_count],
                                  'favourites_count': [favourites_count],
                                  'listed_count': [listed_count],
                                  'lang': [lang],
                                  'name': [name]})

        # Make predictions based on the selected model
        predicted_labels = []
        if model_selection == 'neural_network':
            predicted_labels = predict_user_type_neural_network(user_data)
        elif model_selection == 'random_forest':
            predicted_labels = predict_user_type_random_forest(user_data)
        elif model_selection == 'support_vector_machine':
            predicted_labels = predict_user_type_support_vector_machine(user_data)

        # Render the result template with the predicted labels
        context = {
            'model_selected': model_selection,
            'name': name,
            'statuses_count': statuses_count,
            'followers_count': followers_count,
            'friends_count': friends_count,
            'favourites_count': favourites_count,
            'listed_count': listed_count,
            'lang': lang,
            'predicted_labels': predicted_labels
        }
        return render(request, 'result.html', context)


def predict_user_type_neural_network(user_data):
    fnn = NetworkReader.readFrom(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fnn.xml'))

    lang_list = list(enumerate(pd.unique(user_data['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    user_data['lang_code'] = user_data['lang'].map(lambda x: lang_dict.get(x, 0)).astype(int)
    user_data['sex_code'] = user_data['name'].apply(predict_sex)
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count',
                              'listed_count', 'sex_code', 'lang_code']
    user_data = user_data[feature_columns_to_use]

    input_data = ClassificationDataSet(len(user_data.columns), 1, nb_classes=2)
    input_data.addSample(user_data.values.flatten(), [0])
    input_data._convertToOneOfMany()

    predictions = fnn.activateOnDataset(input_data)
    predicted_classes = np.argmax(predictions, axis=1)

    labels = ['Genuine' if pred == 0 else 'Fake' for pred in predicted_classes]

    return labels


def predict_user_type_random_forest(user_data):
    # Load the trained Random Forest model from random_forest_model.pkl
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'random_forest_model.pkl'), 'rb') as f:
        model = cPickle.load(f)

    # Preprocess the user data
    user_data = preprocess_data(user_data)

    # Make predictions using the Random Forest model
    predictions = model.predict(user_data)
    labels = ['Genuine' if pred == 0 else 'Fake' for pred in predictions]

    return labels

def predict_user_type_support_vector_machine(user_data):
    # Load the trained Random Forest model from svm_model.pkl
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'svm_model.pkl'), 'rb') as f:
        model = cPickle.load(f)

    # Preprocess the user data
    user_data = preprocess_data(user_data)

    # Make predictions using the Random Forest model
    predictions = model.predict(user_data)
    labels = ['Genuine' if pred == 0 else 'Fake' for pred in predictions]

    return labels


def preprocess_data(user_data):
    # Extract the necessary features from the user data
    lang_list = list(enumerate(pd.unique(user_data['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    user_data['lang_code'] = user_data['lang'].map(lambda x: lang_dict[x]).astype(int)
    user_data['sex_code'] = user_data['name'].apply(predict_sex)
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count',
                              'listed_count', 'sex_code', 'lang_code']
    user_data = user_data[feature_columns_to_use]
    return user_data


def predict_sex(name):
    sex_predictor = gender.Detector(unknown_value=u"unknown", case_sensitive=False)
    first_name = name.split(' ')[0]
    sex = sex_predictor.get_gender(first_name)
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    sex_code = sex_dict.get(sex, 0)
    return sex_code

