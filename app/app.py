import streamlit as st
import numpy as np


# CORE MATHEMATICAL FUNCTIONS 
def trimf(x, params):
    """Triangular membership function."""
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if a != b else 1.0
    elif b < x < c:
        return (c - x) / (c - b) if b != c else 1.0
    return 0.0

def trapmf(x, params):
    """Trapezoidal membership function."""
    a, b, c, d = params
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if a != b else 1.0
    elif b <= x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c) if c != d else 1.0
    return 0.0


# STREAMLIT UI & USER INPUTS
st.set_page_config(page_title="Stroke Risk Prediction", layout="wide")
st.title("🧠 Stroke Risk Assessment Using Fuzzy Logic")
st.markdown("By Tuku Lovers")
st.markdown("---")

st.sidebar.header("Patient Clinical Data")

age = st.sidebar.slider("Age", min_value=0, max_value=100, value=65)
glucose = st.sidebar.slider("Average Glucose Level", min_value=0.0, max_value=350.0, value=155.0)
bmi = st.sidebar.slider("BMI", min_value=0.0, max_value=100.0, value=32.0)

smoking_dict = {'Unknown': 0, 'Never smoked': 1, 'Formerly smoked': 2, 'Smokes': 3}
smoking_input = st.sidebar.selectbox("Smoking Status", list(smoking_dict.keys()))
smoking_val = smoking_dict[smoking_input]

hypertension_dict = {'Absent': 0, 'Present': 1}
ht_input = st.sidebar.selectbox("Hypertension", list(hypertension_dict.keys()))
ht_val = hypertension_dict[ht_input]


# FUZZIFICATION
age_young_params, age_middle_params, age_senior_params = [0, 0, 30, 45], [30, 50, 70], [55, 70, 100, 100]
gluc_normal_params, gluc_prediab_params, gluc_diabetic_params = [0, 0, 90, 140], [100, 150, 200], [150, 200, 350, 350]
bmi_low_norm_params, bmi_overwt_params, bmi_obese_params = [0, 0, 18.5, 25], [20, 27.5, 35], [30, 35, 100, 100]
smoke_low_params, smoke_mod_params, smoke_high_params = [-1, 0, 1, 1.5], [1, 2, 3], [2.5, 3, 4, 4]
ht_absent_params, ht_present_params = [-1, 0, 0.5], [0.5, 1, 2]

mu_age = {
    "Young": trapmf(age, age_young_params),
    "Middle": trimf(age, age_middle_params),
    "Senior": trapmf(age, age_senior_params)
}
mu_gluc = {
    "Normal": trapmf(glucose, gluc_normal_params),
    "Prediabetic": trimf(glucose, gluc_prediab_params),
    "Diabetic": trapmf(glucose, gluc_diabetic_params)
}
mu_bmi = {
    "Low/Norm": trapmf(bmi, bmi_low_norm_params),
    "Overweight": trimf(bmi, bmi_overwt_params),
    "Obese": trapmf(bmi, bmi_obese_params)
}
mu_smoke = {
    "Low/Never": trapmf(smoking_val, smoke_low_params),
    "Moderate/Former": trimf(smoking_val, smoke_mod_params),
    "High/Smokes": trapmf(smoking_val, smoke_high_params)
}
mu_ht = {
    "Absent": trimf(ht_val, ht_absent_params),
    "Present": trimf(ht_val, ht_present_params)
}

st.subheader("Patient Profile")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Age", f"{max(mu_age, key=mu_age.get)} ({max(mu_age.values()):.2f})")
col2.metric("Glucose", f"{max(mu_gluc, key=mu_gluc.get)} ({max(mu_gluc.values()):.2f})")
col3.metric("BMI", f"{max(mu_bmi, key=mu_bmi.get)} ({max(mu_bmi.values()):.2f})")
col4.metric("Smoking", f"{max(mu_smoke, key=mu_smoke.get)} ({max(mu_smoke.values()):.2f})")
col5.metric("Hypertension", f"{max(mu_ht, key=mu_ht.get)} ({max(mu_ht.values()):.2f})")


# RULE BASE INFERENCE (MIN/MAX OPERATORS)
def evaluate_rules():
    # HIGH RISK RULES
    rule1 = min(mu_age["Senior"], mu_ht["Present"])
    rule2 = mu_gluc["Diabetic"]
    rule3 = min(mu_age["Senior"], mu_bmi["Obese"])
    rule4 = min(mu_smoke["High/Smokes"], mu_ht["Present"])
    high_risk_firing = max(rule1, rule2, rule3, rule4)
    
    # MEDIUM RISK RULES
    rule5 = min(mu_age["Middle"], mu_bmi["Overweight"], mu_gluc["Prediabetic"])
    rule6 = min(mu_smoke["Moderate/Former"], mu_ht["Absent"])
    medium_risk_firing = max(rule5, rule6)
    
    # LOW RISK RULES
    rule7 = min(mu_age["Young"], mu_bmi["Low/Norm"], mu_smoke["Low/Never"])
    rule8 = min(mu_gluc["Normal"], mu_ht["Absent"])
    low_risk_firing = max(rule7, rule8)
    
    return low_risk_firing, medium_risk_firing, high_risk_firing

f_low, f_med, f_high = evaluate_rules()


# DEFUZZIFICATION
st.subheader("Patient Risk Score")

# SUGENO METHOD
z_low_sugeno = 0.1
z_med_sugeno = 0.5
z_high_sugeno = 0.9

numerator = (f_low * z_low_sugeno) + (f_med * z_med_sugeno) + (f_high * z_high_sugeno)
denominator = f_low + f_med + f_high

if denominator == 0:
    sugeno_risk = 0.0
else:
    sugeno_risk = numerator / denominator

# MAMDANI METHOD (CENTROID)
x_out = np.linspace(0, 1, 100)
mamdani_numerator = 0.0
mamdani_denominator = 0.0

for x in x_out:
    mu_low_out = min(f_low, trapmf(x, [0, 0, 0.2, 0.4]))
    mu_med_out = min(f_med, trimf(x, [0.2, 0.5, 0.8]))
    mu_high_out = min(f_high, trapmf(x, [0.6, 0.8, 1.0, 1.0]))

    agg_mu = max(mu_low_out, mu_med_out, mu_high_out)

    mamdani_numerator += x * agg_mu
    mamdani_denominator += agg_mu

if mamdani_denominator == 0:
    mamdani_risk = 0.0
else:
    mamdani_risk = mamdani_numerator / mamdani_denominator

# Displaying results side by side for comparison
c1, c2 = st.columns(2)

with c1:
    st.info(f"### Mamdani Risk Score\n# {mamdani_risk * 100:.1f}%")
    st.caption("*Calculates physical center of gravity of overlapping geometric membership functions. Slower but more intuitive.*")

with c2:
    st.success(f"### Sugeno Risk Score\n# {sugeno_risk * 100:.1f}%")
    st.caption("*Utilizes a zero-order weighted average of fixed constants ($z\_low=0.1$, $z\_med=0.5$, $z\_high=0.9$). Fast and computationally efficient.*")