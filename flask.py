# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Load data
# heart_data = 'data.csv'
# df = pd.read_csv(heart_data)
# X = df.drop(columns='target', axis=1)
# Y = df['target']

# # Train-test split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=23)

# # Train model
# mod = LogisticRegression()
# mod.fit(X_train, Y_train)

# # Define function for prediction
# # Define function for prediction
# def predict(input_value):
#     # Convert 'Sex' feature to numeric value
#     sex_mapping = {'Male': 1, 'Female': 0}
#     input_value = list(input_value)  # Convert tuple to list for modification
#     input_value[1] = sex_mapping[input_value[1]]  # Encode 'Sex' feature
#     input_value = tuple(map(float, input_value))  # Convert all input values to float
    
#     arr = np.array(input_value)
#     prediction = mod.predict(arr.reshape(1, -1))
#     if prediction == 0:
#         return 'You are healthy'
#     else:
#         return 'Unhealthy! Consult a Doctor'



# # Streamlit app
# def main():
#     st.title('Heart Disease Prediction')
#     st.sidebar.title('User Input')

#     # Sidebar inputs
#     age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
#     sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
#     cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
#     trestbps = st.sidebar.number_input('Resting Blood Pressure', min_value=0, max_value=250, value=120)
#     chol = st.sidebar.number_input('Cholesterol Level', min_value=0, max_value=600, value=200)
#     fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
#     restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
#     thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
#     exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
#     oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=2.0)
#     slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
#     ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])
#     thal = st.sidebar.selectbox('Thalassemia Type', [0, 1, 2, 3])

#     # Predict button
#     if st.sidebar.button('Predict'):
#         input_value = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
#         result = predict(input_value)
#         st.write(result)

# if __name__ == '__main__':
#     main()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
heart_data = 'data.csv'
df = pd.read_csv(heart_data)
X = df.drop(columns='target', axis=1)
Y = df['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=23)

# Train model
mod = LogisticRegression()
mod.fit(X_train, Y_train)

# Define function for prediction
def predict(input_value):
    # Convert 'Sex' feature to numeric value
    sex_mapping = {'Male': 1, 'Female': 0}
    input_value = list(input_value)  # Convert tuple to list for modification
    input_value[1] = sex_mapping[input_value[1]]  # Encode 'Sex' feature
    input_value = tuple(map(float, input_value))  # Convert all input values to float
    
    arr = np.array(input_value)
    prediction = mod.predict(arr.reshape(1, -1))
    if prediction == 0:
        return 'You are healthy'
    else:
        return 'Unhealthy! Consult a Doctor'

# Streamlit app
def main():
    st.title('Heart Disease Prediction')
    st.sidebar.title('User Input')

    # Sidebar inputs
    with st.sidebar:
        st.subheader("Enter Patient Details")
        age = st.number_input('Age', min_value=0, max_value=120, value=50)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=250, value=120)
        chol = st.number_input('Cholesterol Level', min_value=0, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
        restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
        exang = st.selectbox('Exercise Induced Angina', [0, 1])
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=2.0)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia Type', [0, 1, 2, 3])

    # Predict button
    if st.sidebar.button('Predict'):
        input_value = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        result = predict(input_value)
        st.subheader('Prediction Result')
        st.write(result)

if __name__ == '__main__':
    main()

