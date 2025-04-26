import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('credit_risk_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("Credit Risk Prediction App")
st.subheader("Predict whether a loan applicant is a Good or Bad Credit Risk")

# Sidebar for user input
st.sidebar.header("Applicant Information")

def user_input_features():
    # REMOVE Age input because model does not expect it
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    job = st.sidebar.selectbox('Job Type', (0, 1, 2, 3))
    housing = st.sidebar.selectbox('Housing', ('own', 'free', 'rent'))
    saving_accounts = st.sidebar.selectbox('Saving Accounts', ('little', 'moderate', 'quite rich', 'rich'))
    checking_account = st.sidebar.selectbox('Checking Account', ('little', 'moderate', 'rich', 'no_checking'))
    credit_amount = st.sidebar.number_input('Credit Amount', min_value=0, value=1000)
    duration = st.sidebar.slider('Duration (months)', 4, 72, 12)
    purpose = st.sidebar.selectbox('Purpose', ('radio/TV', 'education', 'furniture/equipment', 'new car', 'used car', 
                                                'business', 'domestic appliances', 'repairs', 'vacation/others'))

    # Create a dataframe
    data = {
        'Sex': sex,
        'Job': job,
        'Housing': housing,
        'Saving accounts': saving_accounts,
        'Checking account': checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Main panel - Display input
st.write("### Applicant details entered:")
st.write(input_df)

# Preprocess the input data
def preprocess_input(input_data):
    # Map categorical values
    input_data['Sex'] = input_data['Sex'].map({'male': 1, 'female': 0})
    input_data['Housing'] = input_data['Housing'].map({'own': 0, 'free': 1, 'rent': 2})
    input_data['Saving accounts'] = input_data['Saving accounts'].map({'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3})
    input_data['Checking account'] = input_data['Checking account'].map({'no_checking': 0, 'little': 1, 'moderate': 2, 'rich': 3})
    input_data['Purpose'] = input_data['Purpose'].map({
        'radio/TV': 0,
        'education': 1,
        'furniture/equipment': 2,
        'new car': 3,
        'used car': 4,
        'business': 5,
        'domestic appliances': 6,
        'repairs': 7,
        'vacation/others': 8
    })
    
    # Arrange the columns
    columns = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
    input_data = input_data[columns]
    
    return input_data

# Predict button
if st.button('Predict Credit Risk'):
    processed_input = preprocess_input(input_df.copy())
    
    if processed_input.shape[1] == 8:
        prediction = model.predict(processed_input)

        if prediction[0] == 1:
            st.success('✅ Applicant is a **Good Credit Risk**')
        else:
            st.error('⚠️ Applicant is a **Bad Credit Risk**')
    else:
        st.error("Error: Mismatch in feature count. Please ensure the input data is processed correctly.")