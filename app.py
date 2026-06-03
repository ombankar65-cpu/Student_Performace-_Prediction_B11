import streamlit as st
import pickle
import numpy as np

# Set up page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title and description
st.title("🎓 Student Exam Performance Predictor")
st.write("Enter the student's daily metrics below to estimate their performance score.")

# 1. Load the trained model safely
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure it is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 2. Build the User Input Form if model loads successfully
if model is not None:
    st.markdown("---")
    st.subheader("📊 Input Student Metrics")
    
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.number_input(
            "Hours Studied (Daily)", 
            min_value=0.0, 
            max_value=24.0, 
            value=5.0, 
            step=0.5,
            help="Average hours spent studying per day."
        )
        
        sleep_hours = st.number_input(
            "Sleep Hours (Daily)", 
            min_value=0.0, 
            max_value=24.0, 
            value=7.0, 
            step=0.5,
            help="Average hours of sleep per night."
        )

    with col2:
        previous_scores = st.number_input(
            "Previous Exam Scores", 
            min_value=0.0, 
            max_value=100.0, 
            value=70.0, 
            step=1.0,
            help="The score achieved in the most recent exam."
        )
        
        papers_practiced = st.number_input(
            "Sample Question Papers Practiced", 
            min_value=0, 
            max_value=50, 
            value=2, 
            step=1,
            help="Number of full-length mock papers completed."
        )

    st.markdown("---")
    
    # 3. Prediction Button
    if st.button("🔮 Predict Performance Score", type="primary", use_container_width=True):
        # Format the features into the shape the model expects: [[feat1, feat2, feat3, feat4]]
        input_features = np.array([[hours_studied, previous_scores, sleep_hours, papers_practiced]])
        
        try:
            prediction = model.predict(input_features)[0]
            
            # Constrain the output if your target metric usually falls between 0 and 100
            # (Linear regression can sometimes overshoot bounds if edge inputs are given)
            final_score = max(0.0, min(100.0, float(prediction)))
            
            # Display Result
            st.success("### Prediction Complete!")
            st.metric(label="Estimated Performance Score", value=f"{final_score:.2f}%")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
