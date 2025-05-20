"""
synthetic_data_generator.py

This script generates realistic synthetic patient data for hospital length of stay modeling.
The data includes demographic information, admission details, medical conditions, and other
factors that influence length of stay in a hospital setting.

The relationships between variables are based on realistic medical correlations and
hospital operational patterns.
"""

import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path

def generate_synthetic_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic hospital patient data with realistic relationships.
    
    Parameters:
    -----------
    n_samples : int
        Number of patient records to generate
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic patient data
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate basic patient demographics
    age = np.random.normal(65, 15, n_samples)  # Age centered around 65
    gender = np.random.binomial(1, 0.48, n_samples)  # 48% male
    bmi = np.random.normal(27, 5, n_samples)  # BMI centered around 27
    
    # Generate vital signs
    systolic_bp = np.random.normal(130, 20, n_samples)  # Systolic BP
    diastolic_bp = systolic_bp * 0.7 + np.random.normal(5, 5, n_samples)  # Related to systolic
    heart_rate = np.random.normal(75, 15, n_samples)  # Heart rate
    
    # Generate comorbidity information
    # Base probabilities for conditions
    base_diabetes_prob = 0.15
    base_hypertension_prob = 0.30
    base_heart_disease_prob = 0.12
    base_copd_prob = 0.08
    base_renal_disease_prob = 0.05
    
    # Adjust probabilities based on age
    age_factor = (age - 50) / 30  # Normalized age factor
    
    # Generate conditions with age-adjusted probabilities
    diabetes = np.random.binomial(1, np.clip(base_diabetes_prob + 0.1 * age_factor, 0, 1), n_samples)
    hypertension = np.random.binomial(1, np.clip(base_hypertension_prob + 0.15 * age_factor, 0, 1), n_samples)
    heart_disease = np.random.binomial(1, np.clip(base_heart_disease_prob + 0.2 * age_factor, 0, 1), n_samples)
    copd = np.random.binomial(1, np.clip(base_copd_prob + 0.1 * age_factor, 0, 1), n_samples)
    renal_disease = np.random.binomial(1, np.clip(base_renal_disease_prob + 0.08 * age_factor, 0, 1), n_samples)
    
    # Calculate total number of conditions for each patient
    num_conditions = diabetes + hypertension + heart_disease + copd + renal_disease
    
    # Generate admission details
    emergency_admission = np.random.binomial(1, 0.3, n_samples)  # 30% emergency admissions
    
    # Generate insurance information with age-based adjustments
    insurance_probs = {
        'young': [0.6, 0.05, 0.3, 0.05],  # [private, medicare, medicaid, uninsured]
        'middle': [0.5, 0.3, 0.15, 0.05],
        'elderly': [0.1, 0.8, 0.08, 0.02]
    }
    
    insurance = []
    for a in age:
        if a < 50:
            probs = insurance_probs['young']
        elif a < 65:
            probs = insurance_probs['middle']
        else:
            probs = insurance_probs['elderly']
        
        insurance_type = np.random.choice(['private', 'medicare', 'medicaid', 'uninsured'], p=probs)
        insurance.append(insurance_type)
    
    # Generate admission dates over a one-year period
    base_date = dt.datetime(2024, 1, 1)
    admission_dates = [base_date + dt.timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
    # Generate department data with realistic distributions
    departments = ['cardiology', 'orthopedics', 'general_surgery', 'internal_medicine', 'neurology']
    
    # Adjust probabilities based on emergency status and conditions
    department_probs = []
    
    for i in range(n_samples):
        if emergency_admission[i] == 1:
            # Emergency cases are more likely to go to internal medicine or cardiology
            if heart_disease[i] == 1:
                probs = [0.4, 0.05, 0.05, 0.45, 0.05]  # Higher chance of cardiology
            else:
                probs = [0.2, 0.1, 0.1, 0.5, 0.1]  # Higher chance of internal medicine
        else:
            # Non-emergency distribution
            if heart_disease[i] == 1:
                probs = [0.5, 0.05, 0.1, 0.3, 0.05]  # Higher chance of cardiology
            elif renal_disease[i] == 1:
                probs = [0.1, 0.05, 0.05, 0.75, 0.05]  # Higher chance of internal medicine
            else:
                probs = [0.2, 0.25, 0.15, 0.3, 0.1]  # Standard distribution
        
        department_probs.append(probs)
    
    dept = [np.random.choice(departments, p=probs) for probs in department_probs]
    
    # Create the base length of stay with realistic relationships
    # Base formula incorporates all key medical and operational factors
    base_los = (
        3 +                                         # Base stay
        0.05 * age +                                # Age effect
        0.1 * bmi +                                 # BMI effect
        0.02 * systolic_bp +                        # BP effect
        1.5 * num_conditions +                      # Comorbidities effect
        2 * emergency_admission +                   # Emergency effect
        diabetes * 1.2 +                            # Diabetes effect
        hypertension * 0.8 +                        # Hypertension effect
        heart_disease * 2.5 +                       # Heart disease effect
        copd * 3.0 +                                # COPD effect
        renal_disease * 2.8                         # Renal disease effect
    )
    
    # Add insurance type effect
    insurance_effect = {
        'private': 0, 
        'medicare': 1, 
        'medicaid': 1.5, 
        'uninsured': 2.0
    }
    
    for i in range(n_samples):
        base_los[i] += insurance_effect[insurance[i]]
    
    # Add department-specific effects
    dept_effect = {
        'cardiology': 1.5,
        'orthopedics': 2.0,
        'general_surgery': 0.5,
        'internal_medicine': 0.0,
        'neurology': 2.5
    }
    
    for i in range(n_samples):
        base_los[i] += dept_effect[dept[i]]
    
    # Add weekend effect (longer stays if admitted on weekend)
    weekday = [d.weekday() for d in admission_dates]
    is_weekend = [1 if w >= 5 else 0 for w in weekday]
    base_los += np.array(is_weekend) * 0.8
    
    # Add seasonal effect (winter months have longer stays)
    month = [d.month for d in admission_dates]
    is_winter = [1 if m in [12, 1, 2] else 0 for m in month]
    base_los += np.array(is_winter) * 0.7
    
    # Add realistic non-linear effects for high BMI
    high_bmi_effect = [(b - 30) * 0.2 if b > 30 else 0 for b in bmi]
    base_los += np.array(high_bmi_effect)
    
    # Add effect for COPD patients during winter (interaction)
    copd_winter_interaction = [1.5 if copd[i] == 1 and is_winter[i] == 1 else 0 for i in range(n_samples)]
    base_los += np.array(copd_winter_interaction)
    
    # Add noise
    length_of_stay = base_los + np.random.normal(0, 1.5, n_samples)
    
    # Ensure length of stay is positive and realistic
    length_of_stay = np.maximum(1, length_of_stay)
    
    # Create DataFrame with all features
    data = pd.DataFrame({
        # Demographics
        'age': age,
        'gender': gender,
        'bmi': bmi,
        
        # Vitals
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        
        # Medical conditions
        'num_conditions': num_conditions,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'copd': copd,
        'renal_disease': renal_disease,
        
        # Administrative data
        'emergency_admission': emergency_admission,
        'insurance': insurance,
        'length_of_stay': length_of_stay,
        'admission_date': admission_dates,
        'department': dept
    })
    
    # Add time-based features
    data['admission_month'] = [d.month for d in data['admission_date']]
    data['admission_day_of_week'] = [d.weekday() for d in data['admission_date']]
    data['is_weekend'] = [1 if w >= 5 else 0 for w in data['admission_day_of_week']]
    data['is_winter'] = [1 if m in [12, 1, 2] else 0 for m in data['admission_month']]
    
    # Add derived features
    data['copd_winter'] = data['copd'] * data['is_winter']
    data['age_group'] = pd.cut(data['age'], bins=[0, 50, 70, 100], labels=['young', 'middle', 'elderly'])
    data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
    data['bp_category'] = pd.cut(data['systolic_bp'], bins=[0, 120, 140, 180, 300], labels=['normal', 'elevated', 'high', 'crisis'])
    
    return data

def save_data(data, filename='hospital_los_data.csv', output_dir='data'):
    """
    Save the generated data to a CSV file
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to save
    filename : str
        Name of the output file
    output_dir : str
        Directory to save the data in
    """
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the data
    output_path = Path(output_dir) / filename
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Generate 1000 synthetic patient records
    data = generate_synthetic_data(n_samples=1000)
    
    # Print basic information about the generated data
    print(f"Generated {len(data)} patient records")
    print("\nData summary:")
    print(data.describe())
    
    # Save the generated data
    save_data(data)