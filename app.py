import sklearn
import streamlit as st
import pandas as pd
import pickle

# Function to load serialized Python objects with pickle
def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load the TF-IDF vectorizer
tfidf_vectorizer = load_pickle('/kaggle/input/pickle-vectorizer/scikitlearn/picklevectorizer/1/tfidf_vectorizer.pkl')

# Load the trained model
loaded_model = load_pickle('/kaggle/input/picklesvm/scikitlearn/picklesvm/1/linear_svc_model.pkl')

# Mapping of category numbers to category names
category_mapping = {
    0: 'Credit reporting, repair, or other',
    1: 'Debt collection',
    2: 'Consumer Loan',
    3: 'Credit card or prepaid card',
    4: 'Mortgage',
    5: 'Vehicle loan or lease',
    6: 'Student loan',
    7: 'Payday loan, title loan, or personal loan',
    8: 'Checking or savings account',
    9: 'Bank account or service',
    10: 'Money transfer, virtual currency, or money service',
    11: 'Money transfers',
    12: 'Other financial service'
}

# Description for each complaint category
category_descriptions = {
    0: 'Complaints related to credit reporting, repair, or other financial matters.',
    1: 'Complaints related to debt collection practices.',
    2: 'Complaints related to consumer loans such as personal loans or installment loans.',
    3: 'Complaints related to credit cards or prepaid cards.',
    4: 'Complaints related to mortgages.',
    5: 'Complaints related to vehicle loans or leases.',
    6: 'Complaints related to student loans.',
    7: 'Complaints related to payday loans, title loans, or personal loans.',
    8: 'Complaints related to checking or savings accounts.',
    9: 'Complaints related to bank accounts or banking services.',
    10: 'Complaints related to money transfer services, virtual currency, or money services.',
    11: 'Complaints related to money transfers.',
    12: 'Complaints related to other financial services not covered by the above categories.'
}

# Create a DataFrame to display categories and descriptions
categories_df = pd.DataFrame({
    'Category': [category_mapping[i] for i in range(13)],
    'Description': [category_descriptions[i] for i in range(13)]
})

# Define function to preprocess text data and make predictions
def predict_category(text):
    # Preprocess the text
    preprocessed_text = tfidf_vectorizer.transform([text])
    # Make prediction using the loaded model
    predicted_category = loaded_model.predict(preprocessed_text)
    # Map the predicted category number to category name
    predicted_category_name = category_mapping[predicted_category[0]]
    return predicted_category_name

# Streamlit UI
st.title('Classifying Customer Complaints by Type')

# Text area for the user to enter a customer complaint
user_input = st.text_area("Enter a customer complaint", "Type your complaint here...")

# Button to classify complaint
if st.button('Classify Complaint'):
    if user_input:
        # Predict the type of complaint
        result = predict_category(user_input)
        # Display the result
        st.write(f'Predicted Complaint Type: {result}')
    else:
        st.write("Please enter a valid complaint to classify.")

# Display categories and descriptions in a neat table
st.write(categories_df)
