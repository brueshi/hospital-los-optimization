# Healthcare Length of Stay Optimization

![](https://img.shields.io/badge/Python-3.8+-blue.svg)
![](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![](https://img.shields.io/badge/pandas-1.4+-green.svg)
![](https://img.shields.io/badge/R²-0.86-brightgreen.svg)

A machine learning solution that predicts how long patients will stay in the hospital and provides recommendations to optimize these stays. This project combines predictive modeling with practical implementation strategies that healthcare organizations can use to improve operations and reduce costs.

## Project Overview

Hospital length of stay (LOS) directly impacts both patient care and hospital finances. Each additional day adds approximately $2,000 in costs, while excessive stays can increase risks of hospital-acquired conditions. This project addresses this challenge by:

1. Predicting patient length of stay with high accuracy
2. Identifying which factors most strongly influence longer stays
3. Creating a system to identify high-risk patients early
4. Providing targeted recommendations for different hospital departments
5. Outlining a practical implementation plan with expected financial benefits

## Key Findings

![Feature Importance](https://github.com/user-attachments/assets/35d305f8-0d70-45d9-a2b1-b80c715b5d1f)

![length_of_stay_by_risk](https://github.com/user-attachments/assets/456250ba-51c9-4a72-9545-8566f8089a40)


Our analysis reveals several factors that significantly impact how long patients stay in the hospital:

- **Emergency admissions** add approximately 3 days to a patient's hospital stay compared to planned admissions
- **Patients with respiratory conditions (COPD)** typically require 2.9 days of additional care
- **Kidney-related issues (renal disease)** extend hospitalization by about 2.8 days
- **Department matters** - patients in specialty units like neurology have different patterns than those in general medicine
- **Weekend admissions** result in longer stays across all departments, presenting a clear opportunity for process improvement

The model effectively categorizes patients into risk groups, showing a clear difference of about 5 days between the highest and lowest risk patients. This allows hospitals to:

- Identify high-risk patients at admission
- Deploy targeted interventions for those most likely to have extended stays
- Allocate resources more effectively across departments

## Model Performance in Plain English

Our best model can predict hospital length of stay with high accuracy:

- **86% accuracy in predicting variations in length of stay** (R² of 0.86)
- **Average prediction error of just 2.6 days** (RMSE of 2.57)
- **Consistent performance across different patient groups** (Cross-validation score of 0.86 ± 0.01)

In practical terms, this means:
- Hospital administrators can rely on these predictions for capacity planning
- Care teams can identify high-risk patients with confidence
- Resource allocation decisions can be made with greater precision

### Understanding the Metrics

- **R² (R-squared)**: Think of this as a percentage of accuracy. Our R² of 0.86 means the model explains 86% of the variation in patient length of stay. The closer to 100%, the better the predictions.

- **RMSE (Root Mean Square Error)**: This represents the average prediction error in days. Our RMSE of 2.57 means that, on average, our predictions are within about 2.6 days of the actual length of stay.

- **Cross-validation**: This shows how well the model performs on different subsets of data. Our score of 0.86 ± 0.01 means the model consistently performs well regardless of which patients it's tested on, indicating it will work reliably on new patients.

## How Hospitals Can Use This Model

This predictive approach can be integrated into hospital operations to create real improvements:

1. **Early Identification System**: Flag high-risk patients during admission with a simple scoring tool based on the top 5 factors

2. **Department-Specific Approaches**:
   - **Emergency Department**: Focus on expediting transfers for patients with COPD and renal disease
   - **Internal Medicine**: Implement standardized care pathways for patients with multiple conditions
   - **Specialty Units**: Customize discharge planning timelines based on department-specific patterns

3. **Weekend Process Improvement**:
   - Enhance staffing during critical weekend periods
   - Develop Friday planning protocols to prevent unnecessary weekend delays
   - Create Monday acceleration processes to address weekend backlogs

4. **Financial Benefits**:
   - Reduce average stays by 1-1.5 days across all patients
   - Free up approximately 7,300 bed-days annually in a mid-sized hospital
   - Generate savings of $2.5-3.8 million while maintaining quality of care

## Implementation Roadmap

The project includes a practical, phased implementation plan:

### Phase 1: Patient Risk Scoring System (Months 1-3)
- Integrate prediction tool into admission processes
- Train staff on risk factor identification
- Begin targeted interventions for highest-risk patients

### Phase 2: Department-Specific Optimizations (Months 4-6)
- Customize approaches based on department needs
- Implement condition-specific care pathways
- Enhance discharge planning processes

### Phase 3: Weekend Effect Mitigation (Months 7-9)
- Optimize weekend staffing patterns
- Implement pre-weekend planning protocols
- Enhance diagnostic service availability

### Phase 4: Continuous Improvement (Months 10-12)
- Retrain model with actual hospital data
- Refine intervention strategies
- Expand to additional facilities or departments

## Technical Approach

Multiple regression models were evaluated, with Ridge Regression providing the best balance of accuracy and interpretability:

| Model | R² | RMSE | CV Score |
|-------|-----|------|----------|
| Ridge Regression | 0.86 | 2.57 | 0.86 ± 0.01 |
| Random Forest | 0.85 | 2.64 | 0.84 ± 0.02 |
| Gradient Boosting | 0.84 | 2.71 | 0.83 ± 0.01 |
| Linear Regression | 0.86 | 2.57 | 0.86 ± 0.01 |
| Lasso | 0.86 | 2.62 | 0.86 ± 0.01 |
| ElasticNet | 0.86 | 2.62 | 0.86 ± 0.01 |

The modeling process included:
- Advanced feature engineering to capture medical relationships
- Cross-validation to ensure reliable performance
- Detailed evaluation of model predictions across departments and risk categories

## Repository Structure

```
healthcare-los-optimization/
├── data/
│   └── hospital_los_data.csv
├── models/
│   ├── ridge_model.pkl
│   ├── ridge_metrics.json
│   └── ...
├── visualizations/
│   ├── feature_importance.png
│   ├── length_of_stay_by_risk.png
│   └── ...
├── synthetic_data_generator.py
├── hospital_los_model.py
├── model_evaluation.ipynb
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: scikit-learn, pandas, numpy, matplotlib, seaborn

### Installation

```bash
# Clone the repository
git clone https://github.com/brueshi/healthcare-los-optimization.git
cd healthcare-los-optimization

# Install required packages
pip install -r requirements.txt
```

### Usage

1. Generate synthetic data (or use your own hospital data):
   ```python
   from synthetic_data_generator import generate_synthetic_data, save_data
   
   data = generate_synthetic_data(n_samples=1000)
   save_data(data, output_dir='data')
   ```

2. Train the model:
   ```python
   from hospital_los_model import LengthOfStayModel
   
   model = LengthOfStayModel(model_type='ridge')
   X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = model.prepare_data(data)
   model.fit(X_train, y_train)
   
   # Evaluate performance
   metrics = model.evaluate(X_test, y_test, X_test_original=X_test_orig)
   print(f"R² score: {metrics['r2_score']:.2f}")
   ```

3. Run the Jupyter notebook for detailed evaluation:
   ```bash
   jupyter notebook model_evaluation.ipynb
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed for healthcare analytics applications
- While using synthetic data, the relationships are based on realistic healthcare patterns
- The implementation recommendations are derived from healthcare operations best practices
