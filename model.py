from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# shap explainer
import shap

# printing
import pprint

import os

dirpath = os.getcwd()
local_gcs = dirpath + '/gcs/'

## train the model
def _train():
    # import data files
    train_data = pd.read_csv('gcs/data/adult.data.csv', index_col=False,
                            names=['age', 'workclass', 'fnlwgt', 'education',
                                    'education-num', 'marital-status', 'occupation',
                                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                    'hours-per-week', 'native-country', 'salary'])
    test_data = pd.read_csv('gcs/data/adult.test.csv', index_col=False,
                            names=['age', 'workclass', 'fnlwgt', 'education',
                                'education-num', 'marital-status', 'occupation',
                                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                'hours-per-week', 'native-country', 'salary'])

    # create label field as integer in train data
    train_data['over50K'] = (train_data['salary'] == ' >50K').astype(int)
    # also create label field in test data
    test_data['over50K'] = (test_data['salary'] == ' >50K').astype(int)

    # create training labels dataset
    train_labels = train_data.iloc[:, -1]
    # also create test labels dataset
    test_labels = test_data.iloc[:, -1]

    # remove labels and other fields from training data and test data
    train_data.drop(['salary', 'over50K', 'fnlwgt', 'capital-gain',
                    'capital-loss'], axis=1, inplace=True)
    test_data.drop(['salary', 'over50K', 'fnlwgt', 'capital-gain',
                    'capital-loss'], axis=1, inplace=True)

    # create a dataframe for object data types so we can aggregate categories
    obj_df = train_data.select_dtypes(include=['object']).copy()

    # define number of feature values to include for each feature
    n_items = 5

    # create a function to group items based on most frequently occurring
    def item_map(series):
        n = 0
        while n < n_items - 1:
            if (series[field] == top_items[n]):
                return str.lower(top_items[n])
            n += 1
        else:
            return 'other'


    # group categorical data into broader buckets
    for _ in obj_df.columns:
        field = _
        top_items = obj_df[field].value_counts().keys()[0:n_items]
        df_mapped = train_data[[field]].apply(item_map, axis='columns')
        train_data[field] = df_mapped

    # do the same for test data
    for _ in obj_df.columns:
        field = _
        top_items = obj_df[field].value_counts().keys()[0:n_items]
        df_mapped = test_data[[field]].apply(item_map, axis='columns')
        test_data[field] = df_mapped

    # visualize new classification
    obj_df2 = train_data.select_dtypes(include=['object']).copy()

    # create a new dataframe that converts to one hot encoding
    train_data_input = pd.get_dummies(train_data)
    # also one hot encode the test data
    test_data_input = pd.get_dummies(test_data)

    # configure model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(40,)),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    # compile model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # fit model
    model.fit(train_data_input, train_labels, epochs=5)

    # evaluate model
    test_loss, test_acc = model.evaluate(test_data_input, test_labels)

    print('Test accuracy:', test_acc)

    with open(local_gcs + 'model.json', 'w+') as json_file:
        json_file.write(model.to_json())
    json_file.close()

    test_data_input.to_csv(local_gcs + 'test_data_input.csv', index=False)
    return [model, test_data_input, train_data_input]


def _getPrediction(model, prediction, train_data_input):
        # rename columns
    train_data_input.columns = (['age', 'education-num', 'hours-per-week', 'workclass_unknown',
                                 'workclass_local-gov', 'workclass_private',
                                 'workclass_self-emp-not-inc', 'workclass_other',
                                 'education_bachelors', 'education_hs-grad', 'education_masters',
                                 'education_some-college', 'education_other',
                                 'marital-status_divorced', 'marital-status_married-civ-spouse',
                                 'marital-status_never-married', 'marital-status_separated',
                                 'marital-status_other', 'occupation_adm-clerical',
                                 'occupation_craft-repair', 'occupation_exec-managerial',
                                 'occupation_prof-specialty', 'occupation_other',
                                 'relationship_husband', 'relationship_not-in-family',
                                 'relationship_own-child', 'relationship_unmarried',
                                 'relationship_other', 'race_amer-indian-eskimo',
                                 'race_asian-pac-islander', 'race_black', 'race_white', 'race_other',
                                 'sex_female', 'sex_male', 'native-country_unknown',
                                 'native-country_mexico', 'native-country_philippines',
                                 'native-country_united-states', 'native-country_other'])
    prediction = dict(zip(['age', 'education-num', 'hours-per-week', 'workclass_unknown',
                      'workclass_local-gov', 'workclass_private',
                      'workclass_self-emp-not-inc', 'workclass_other',
                      'education_bachelors', 'education_hs-grad', 'education_masters',
                      'education_some-college', 'education_other',
                      'marital-status_divorced', 'marital-status_married-civ-spouse',
                      'marital-status_never-married', 'marital-status_separated',
                      'marital-status_other', 'occupation_adm-clerical',
                      'occupation_craft-repair', 'occupation_exec-managerial',
                      'occupation_prof-specialty', 'occupation_other',
                      'relationship_husband', 'relationship_not-in-family',
                      'relationship_own-child', 'relationship_unmarried',
                      'relationship_other', 'race_amer-indian-eskimo',
                      'race_asian-pac-islander', 'race_black', 'race_white', 'race_other',
                      'sex_female', 'sex_male', 'native-country_unknown',
                      'native-country_mexico', 'native-country_philippines',
                      'native-country_united-states', 'native-country_other'],
                      prediction))
    
    prediction = pd.DataFrame(prediction, index=[0])
    # print(type(prediction))

    # converting the dataframe into an np array
    shap_train_data = np.array(train_data_input[:100])

    # prediction on this line instead of test_data_input
    shap_test_data = np.array(prediction)

    # we use the first 100 training examples as our background dataset to integrate over
    explainer = shap.DeepExplainer(model, shap_train_data)

    # explain the first 10 predictions
    # explaining each prediction requires 2 * background dataset size runs
    shap_values = explainer.shap_values(shap_test_data)

    # create list of feature names
    feature_name_list = train_data_input.columns

    # get shap explainer values for one prediction
    _predictionResults = explainer.shap_values(shap_test_data)

    # print(_predictionResults[0])
    probability_class_0= (explainer.expected_value[0] + sum(_predictionResults[0][0]))
    p = (probability_class_0 > .5)
    obj = {
               'prediction': str(p),
               'baseline': str(explainer.expected_value[0]),
               'shift': str(sum(_predictionResults[0][0])),
               'probability': str(probability_class_0),
               'features': []
           }

    for j in range(0, len(_predictionResults[0][0])):
        obj['features'].append({
            'name': str(feature_name_list[j]),
            'value': str(prediction.iloc[0][j]),
            'weight': str(_predictionResults[0][0][j])
        })
    # print(obj)
    
    # return object built
    return obj
