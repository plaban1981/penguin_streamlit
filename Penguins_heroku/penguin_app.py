# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:47:56 2020

@author: Plaban Nayak
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

url = 'https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv'

st.write("""
# Penguin Gender Prediction App
This app predicts the **Palmer Penguin** sex!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Encoding of ordinal features
encode = ['species','island']

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    for col in encode:
        dummy = pd.get_dummies(input_df[col])
        input_df = pd.concat([input_df,dummy],axis=1)
    
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        species = st.sidebar.selectbox('Species',('Adelie', 'Gentoo', 'Chinstrap'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'species': species}
        data['Biscoe'] =0
        data['Dream'] = 0
        data['Torgersen'] =0
        
        #
        data['Adelie'] =0
        data['Gentoo'] = 0
        data['Chinstrap'] =0
        #
        #    
        if data['species'] == 'Adelie':
            data['Adelie'] =1
        elif data['species'] == 'Gentoo':
            data['Gentoo'] = 1
        else:
            data['Chinstrap'] =1
        #
        if data['island'] == 'Biscoe':
            data['Biscoe'] = 1
        elif data['island'] == 'Dream':
            data['Dream'] == 1
        else:
            data['Torgersen'] =1
        
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
#penguins_raw = pd.read_csv(url)
#penguins = penguins_raw.drop(columns=['sex'])
#df = pd.concat([input_df,penguins],axis=0)
df = input_df.copy()

df = df.drop(encode,axis=1)



df = df[:1] # Selects only the first row (the user input data)


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)



prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_sex = np.array(['female','male'])
st.write(penguins_sex[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba) 