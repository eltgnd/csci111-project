# Packages
import streamlit as st
import pickle
import numpy
import pandas as pd

# Variables
ss = st.session_state

# Callable

# Initialize
page_title = 'CSCI 111 Project'
page_icon = '🤖'
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered", initial_sidebar_state="auto", menu_items=None)


# Main
st.title('Adult Census Income Binary Classifier')

# Load the machine learning model
@st.cache_resource
def load_variables():
    with open("models.pkl", "rb") as f:
        models = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("preprocessing_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("xtrain_df.pkl", "rb") as f:
        xtrain_df = pd.compat.pickle_compat.load(f)

    return models, xtrain_df, preprocessor, feature_names

models, xtrain_df, preprocessor, feature_names = load_variables()
logreg = models['logreg']
rfc = models['rfc']

def display_to_col(s):
    return s.lower().replace(' ', '_')

def preprocess(a):
    preprocessor_features = preprocessor.get_feature_names_out()
    total_columns = []
    for i in preprocessor_features:
        total_columns.append(i.split('__')[-1])
    x_processed = preprocessor.transform(a)

    df = pd.DataFrame(0, index=[0], columns=total_columns)
    # Populate the DataFrame with the nonzero values
    for col_idx, value in zip(x_processed.indices, x_processed.data):
        df.iloc[0, col_idx] = value

    selected_features = xtrain_df.columns.tolist()
    x_feature_selected = df[selected_features]
    return x_feature_selected.to_numpy().reshape(1,-1)

# Define the features and their input types
categorical_features = {
    "Workclass": ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay'],
    "Education": ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school', '5th-6th', '10th', 'Preschool', '12th', '1st-4th'],
    "Marital Status": ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
    "Occupation": ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'],
    "Relationship": ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
    "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    "Sex": ["Male", "Female"],
    "Native Country": ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']
}

numeric_features = ["Age", "Education Num", "Capital Gain", "Capital Loss", "Hours Per Week"]


st.write("Enter the details below to predict whether the income exceeds $50,000.")

# Create an input form
with st.form("prediction_form"):
    user_input = {}

    # user_input['model_choice'] = st.radio('Choose model to use', ['Logistic Regression', 'Random Forest Classifier'])

    # Numeric features
    for feature in numeric_features:
        col_name = display_to_col(feature)
        user_input[col_name] = st.number_input(feature, value=0)

    # Categorical features
    for feature, options in categorical_features.items():
        col_name = display_to_col(feature)
        user_input[col_name] = st.selectbox(feature, sorted(options))

    # Submit button
    submitted = st.form_submit_button("Predict")

# Process and predict when the form is submitted
if submitted:
    # model = logreg if user_input['model_choice'] == 'Logistic Regression' else rfc
    order_of_features = ['age',
        'fnlwgt',
        'education_num',
        'capital_gain',
        'capital_loss',
        'hours_per_week',
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country'
    ]

    user_input['fnlwgt'] = 0 # Will be removed after feature selection anyway
    input_dict = {}
    for i in order_of_features:
        input_dict[i] = user_input[i]
    input_df = pd.DataFrame(input_dict, index=[0])

    preprocessed_array = preprocess(input_df)

    # Make prediction
    # prediction = model.predict(preprocessed_array)
    pred_logreg = logreg.predict(preprocessed_array)
    proba_logreg = logreg.predict_proba(preprocessed_array)
    pred_rfc = rfc.predict(preprocessed_array)
    proba_rfc = rfc.predict_proba(preprocessed_array)

    output = [['Logistic Regression', pred_logreg, proba_logreg], ['Random Forest Classifier', pred_rfc, proba_rfc]]
    for i in output:
        if i[1] == '<=50K':
            st.info(f'{i[0]} predicts: Income <= $50,000 with a confidence of {round(i[2][0][0] * 100)}%', icon='👎')
        else:
            st.success(f'{i[0]} predicts: Income > $50,000 with a confidence of {round(i[2][0][0] * 100)}%', icon='👍')

