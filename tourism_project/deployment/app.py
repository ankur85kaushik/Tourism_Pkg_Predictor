import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_predictor_model_v1.joblib")
model = joblib.load(model_path)

# --- App Header ---
st.title("🧳 Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase a newly pitched holiday package
based on their demographic and interaction data.
""")

# --- Sidebar / Form Inputs ---
st.header("👤 Customer Demographics & Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age_group = st.selectbox("Age Group", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65'])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

with col2:
    monthly_income = st.number_input("Monthly Income (INR)", min_value=0, max_value=200000, value=25000, step=500)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    passport = st.selectbox("Has a Passport?", ["No", "Yes"])

st.header("📞 Sales Pitch Details")
col3, col4 = st.columns(2)

with col3:
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15, step=1)

with col4:
    num_person_visiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    num_children_visiting = st.slider("Number of Children Visiting", 0, 3, 0)
    num_followups = st.slider("Number of Follow-ups Done", 1, 6, 3)
    num_trips = st.slider("Number of Historical Trips", 1, 10, 2)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
    preferred_stars = st.slider("Preferred Hotel Star Rating (3-5)", 3, 5, 3)

# --- Process & Transform Inputs (Match your Training Script exactly) ---
if st.button("🔮 Predict Package Purchase"):

    # 1. Standardize binary fields to match your LabelEncoder mappings (0 or 1)
    # Hint: Ensure these match whatever order your LabelEncoder used during training
    gender_encoded = 0 if gender == "Female" else 1
    contact_encoded = 1 if typeof_contact == "Self Enquiry" else 0

    # 2. Build the base row with numerical columns
    raw_input = {
        'TypeofContact': contact_encoded,
        'CityTier': city_tier,
        'DurationOfPitch': float(duration_of_pitch),
        'Gender': gender_encoded,
        'NumberOfPersonVisiting': num_person_visiting,
        'NumberOfFollowups': float(num_followups),
        'PreferredPropertyStar': float(preferred_stars),
        'NumberOfTrips': float(num_trips),
        'Passport': 1 if passport == "Yes" else 0,
        'PitchSatisfactionScore': pitch_satisfaction,
        'OwnCar': 1 if own_car == "Yes" else 0,
        'NumberOfChildrenVisiting': float(num_children_visiting),
        'MonthlyIncome': float(monthly_income)
    }

    # 3. Handle the One-Hot Encoded columns manually to prevent shape mismatch errors!
    # These must be named exactly like the outputs from your `pd.get_dummies()` command.
    all_occupations = ["Free Lancer", "Large Business", "Salaried", "Small Business"]
    for occ in all_occupations:
        raw_input[f"Occupation_{occ}"] = 1 if occupation == occ else 0

    all_products = ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]
    for prod in all_products:
        raw_input[f"ProductPitched_{prod}"] = 1 if product_pitched == prod else 0

    all_marital = ["Divorced", "Married", "Single"]
    for mar in all_marital:
        raw_input[f"MaritalStatus_{mar}"] = 1 if marital_status == mar else 0

    all_designations = ["AVP", "Executive", "Manager", "Senior Manager", "VP"]
    for des in all_designations:
        raw_input[f"Designation_{des}"] = 1 if designation == des else 0

    all_ages = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65']
    for ag in all_ages:
        raw_input[f"AgeGroup_{ag}"] = 1 if age_group == ag else 0

    # Convert dictionary into a Single Row DataFrame
    input_df = pd.DataFrame([raw_input])

    # --- Prediction Execution ---
    try:

        # Ensure column alignment matches the trained model layout
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success("🎉 **High Probability:** The customer is likely to **BUY** the tourism package!")
        else:
            st.error("❌ **Low Probability:** The customer is likely to **REJECT** the tourism package.")

    except FileNotFoundError:
        st.warning("⚠️ Model file `tourism_model.pkl` not found. Please train and save your model first before trying to predict.")
