
# Introduction to AI Final Project

## Comparative Analysis of Mamdani and Sugeno Fuzzy Inference Systems for Stroke Risk Assessment

This repository contains the implementation and documentation of a fuzzy logic-based stroke risk assessment system developed for the Introduction to Artificial Intelligence final project.

The project compares two fuzzy inference approaches, Mamdani and Sugeno, using real-world healthcare data. Both systems are implemented from scratch without using fuzzy logic libraries, allowing the inference process, membership functions, rule evaluation, and defuzzification methods to be clearly demonstrated.

## Project Objective

The main objective of this project is to evaluate how Mamdani and Sugeno fuzzy inference systems perform in assessing stroke risk based on selected clinical and lifestyle-related variables.

The comparison focuses on:

- system design
- implementation process
- output characteristics
- prediction accuracy
- interpretability of results

## Dataset

The dataset used in this project is the Healthcare Stroke Dataset from Kaggle, containing 5,110 patient records.

Selected input variables:

- Age
- Average glucose level
- Body Mass Index
- Smoking status
- Hypertension status

Output variable:

- Stroke risk classification

## Fuzzy System Design

The fuzzy system uses linguistic variables to represent gradual health conditions instead of strict numerical boundaries.

Input linguistic variables include:

- Age: Young, Middle-aged, Senior
- Glucose Level: Normal, Prediabetic, Diabetic
- BMI: Low/Normal, Overweight, Obese
- Smoking Status: Low Risk, Moderate Risk, High Risk
- Hypertension: Absent, Present

The output variable is stroke risk, categorized as:

- Low Risk
- Moderate Risk
- High Risk

## Implementation

The system was implemented manually in Python, including:

- triangular membership functions
- trapezoidal membership functions
- fuzzification
- rule evaluation
- Mamdani inference
- Sugeno inference
- Mamdani centroid defuzzification
- Sugeno weighted average defuzzification
- accuracy evaluation
- risk score visualization

## Rule Base

The fuzzy inference system uses 15 IF-THEN rules based on general medical reasoning related to stroke risk factors.

The rules consider the influence of:

- advanced age
- hypertension
- high glucose level
- obesity
- smoking behavior

The same rule base is applied to both Mamdani and Sugeno systems to ensure a fair comparison.

## Project Structure

```text
.
├── app/
│   └── Application source code
├── data/
│   └── Dataset and processed data
├── images/
│   └── Output figures and visualizations
├── notebooks/
│   └── Jupyter Notebook implementation
├── report/
│   └── Final project report
└── README.md
