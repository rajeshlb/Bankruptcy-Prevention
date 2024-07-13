

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: ')

st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk = st.sidebar.selectbox('Industrial Risk', [0, 0.5, 1])
    management_risk = st.sidebar.selectbox('Management Risk', [0, 0.5, 1])
    financial_flexibility = st.sidebar.selectbox('Financial Flexibility', [0, 0.5, 1])
    credibility = st.sidebar.selectbox('Credibility', [0, 0.5, 1])
    competitiveness = st.sidebar.selectbox('Competitiveness', [0,0.5,  1])
    operating_risk = st.sidebar.selectbox('Operating Risk', [0, 0.5, 1])

    # Prepare the input data
    # input_data = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])
    i={
        'industrial_risk':industrial_risk,
            ' management_risk':management_risk,
            ' financial_flexibility':financial_flexibility,
            ' credibility':credibility,
            ' competitiveness':competitiveness,
            ' operating_risk':operating_risk

            }
    features = pd.DataFrame(i,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('Model1.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes, it is bankruptcy' if prediction_proba[0][1] > 0.5 else 'No, it is not bankruptcy')

st.subheader('Prediction Probability')
st.write(prediction)
