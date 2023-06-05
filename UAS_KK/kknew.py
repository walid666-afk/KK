import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title='Diagnosa Diabetes Mellitus', layout='wide')

def run():
    @st.cache_(allow_output_mutation=True)
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

    # Load the trained model
    @st.cache_resource(allow_output_mutation=True)
    def load_model():
        with open('DT.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    clf = load_model()

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
    label_mapping = {'Diabetes Mellitus': 'Anda didiagnosis Diabetes Mellitus',
                     'Hepatitis B': 'Anda didiagnosis Hepatitis B',
                     'Stroke': 'Anda didiagnosis Stroke',
                     'Anemia': 'Anda didiagnosis Anemia',
                     'Cardiomegali': 'Anda didiagnosis Cardiomegali',
                     'Hipoglikemi': 'Anda didiagnosis Hipoglikemi',
                     'Sepsis': 'Anda didiagnosis Sepsis',
                     'Scabies': 'Anda didiagnosis Scabies',
                     'Hyperglikemi': 'Anda didiagnosis Hyperglikemi'}

    # Display prediction result
    st.subheader("Diagnosis Result:")
    if prediction[0] in label_mapping:
        st.success(label_mapping[prediction[0]])
    else:
        st.warning("Tidak dapat melakukan diagnosis")

run()
