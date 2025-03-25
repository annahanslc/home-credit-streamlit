import streamlit as st
import pandas as pd
import joblib

from sklearn import set_config
set_config(transform_output='pandas')

# Step 1: Title your streamlit
st.title('Home Loan Credit Default Risk')

# Step 2: User input
name = st.text_input('Enter your name')
if name:
  st.write(f'Hello, {name}')

# Step 3: Get the data
@st.cache_data
def load_data():
  return pd.read_csv('https://raw.githubusercontent.com/annahanslc/home-credit-streamlit/refs/heads/main/cleaned_sample.csv')

loan_data = load_data()

# Step 4: Create a expander containing the dataframe
with st.expander('Home Loan Credit DataFrame', expanded=False):
  st.dataframe(loan_data)

  if st.button('Reload Data', key='reload_data'):
    load_data.clear()

# Step 5: Load the model

@st.cache_resource
def load_model():
  return joblib.load('lr_pipe.joblib')

@st.cache_resource
def load_columns():
  return joblib.load('model_columns.pkl')

model=load_model()
columns=load_columns()

def build_model_input(user_input: dict, columns: list) -> pd.DataFrame:
    data = {str(col): user_input.get(col, float('nan')) for col in columns}
    return pd.DataFrame([data])


# Step 6: Create a sidebar

with st.sidebar:
  # prediction = pd.DataFrame([{}], columns=loan_data.columns)
  user_input = {}


  for feat in loan_data.columns:
    if loan_data[feat].name == 'AMT_CREDIT':
       user_input[feat] = st.slider('Select the loan amounts',
                                           min_value = loan_data[feat].min(),
                                           max_value = loan_data[feat].max(),
                                           step=100.00,
                                           value=loan_data[feat].min())
    elif loan_data[feat].name == 'AMT_INCOME_TOTAL':
       user_input[feat] = st.slider('Select the income of the client',
                                           min_value = loan_data[feat].min(),
                                           max_value = loan_data[feat].max(),
                                           step=100.00,
                                           value=loan_data[feat].min())
    elif loan_data[feat].name == 'FLAG_OWN_CAR':
       user_input[feat] = st.radio('Select if you own a car',
                                           ['Yes','No'])
    elif loan_data[feat].name == 'NAME_EDUCATION_TYPE':
       user_input[feat] = st.selectbox('Select your level of education',
                                             loan_data[feat].unique())
    elif loan_data[feat].name == 'FLAG_EMAIL':
       user_input[feat] = 1 if st.toggle('Provide your email?', value=False) else 0
    elif loan_data[feat].name == 'REGION_RATING_CLIENT':
       rating = st.radio("Rate the region you live in:",
       options=['⭐️','⭐️⭐️','⭐️⭐️⭐️'], index=1)
       star_map = {"⭐️": 1, "⭐️⭐️": 2, "⭐️⭐️⭐️": 3}
       user_input[feat] = star_map[rating]
    elif loan_data[feat].name == 'CNT_CHILDREN':
       user_input[feat] = st.number_input('How many children do you have?',
                                           min_value = loan_data[feat].min(),
                                           max_value = loan_data[feat].max(),
                                           step=1,
                                           value=loan_data[feat].min())


  input_df = build_model_input(user_input, columns)
  input_df.columns = input_df.columns.astype(str)


  st.write('Prediction input DataFrame:')
  st.dataframe(input_df)

  if st.button('Predict if you will have payment difficulties'):

    loan_status = model.predict(input_df)

    if loan_status == 1:
      st.markdown('## **You will have payment difficulties** 😢')
    elif loan_status == 0:
      st.markdown('## **You will not have payment difficulties!** 💸')
