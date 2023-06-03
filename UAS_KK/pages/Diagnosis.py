import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Set page config
st.set_page_config(page_title='Diagnosa Diabetes Mellitus', layout='wide')

def run():
    @st.cache_data
    def load_data():
        data = pd.read_csv('pages/datastreamlit.csv')
        return data

    # Title
    st.title("Diagnosa Diabetes Mellitus")

    # Load data
    data = load_data()

    # Split dataset into features and target variable
    X = data.drop('Penyakit', axis=1)
    y = data['Penyakit']

    # Create Decision Tree classifier
    clf = DecisionTreeClassifier()

    # Train the classifier
    clf.fit(X, y)

    def user_input_features():
        # Create input fields based on the columns in the dataset
        features = {}
        for column in X.columns:
            feature_value = st.number_input(column, step=0.01)
            features[column] = feature_value

        input_data = pd.DataFrame(features, index=[0])
        return input_data

    input_df = user_input_features()

    # CSS styling to center the input fields
    st.markdown(
        """
        <style>
        .element-container {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display user input
    st.subheader("User Input:")
    st.dataframe(input_df)

    # Make prediction on user input
    prediction = clf.predict(input_df)

    # Mapping label
    label_mapping = {'Diabetes Mellitus': 'Anda didiagnosis dengan Diabetes Mellitus',
                     'Hepatitis B': 'Anda didiagnosis dengan Hepatitis B',
                     'Stroke': 'Anda didiagnosis dengan Stroke',
                     'Anemia': 'Anda didiagnosis dengan Anemia',
                     'Cardiomegali': 'Anda didiagnosis dengan Cardiomegali',
                     'Hipoglikemi': 'Anda didiagnosis dengan Hipoglikemi',
                     'Sepsis': 'Anda didiagnosis dengan Sepsis',
                     'Scabies': 'Anda didiagnosis dengan Scabies',
                     'Hyperglikemi': 'Anda didiagnosis dengan Hyperglikemi'}

    # Display prediction result
    st.subheader("Diagnosis Result:")
    if prediction[0] in label_mapping:
        st.success(label_mapping[prediction[0]])
    else:
        st.warning("Tidak dapat melakukan diagnosis")

run()
