"""
hospital_los_model.py

This module contains the implementation of predictive models for hospital length of stay.
It includes data preprocessing, feature engineering, model training, evaluation, and
prediction functions.

The models are designed to help hospitals optimize resource allocation by accurately
predicting patient length of stay and identifying key factors that contribute to
extended hospitalizations.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin

class LengthOfStayModel:
    """A class for building and evaluating hospital length of stay prediction models."""
    
    def __init__(self, model_type='ridge', random_state=42):
        """
        Initialize the length of stay model.
        
        Parameters:
        -----------
        model_type : str
            Type of regression model to use ('linear', 'ridge', 'lasso', 'elasticnet', 
            'randomforest', 'gradientboosting')
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.feature_importances_ = None
        
    def _create_model(self):
        """Create the regression model based on the specified model type."""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=self.random_state)
        elif self.model_type == 'lasso':
            return Lasso(alpha=0.01, random_state=self.random_state)
        elif self.model_type == 'elasticnet':
            return ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=self.random_state)
        elif self.model_type == 'randomforest':
            return RandomForestRegressor(n_estimators=100, max_depth=15, random_state=self.random_state)
        elif self.model_type == 'gradientboosting':
            return GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                            random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def preprocess_data(self, data):
        """
        Preprocess the data for modeling, including handling dates, encoding categoricals,
        and creating derived features.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw hospital patient data
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed data ready for feature engineering and modeling
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Convert admission_date to datetime if it's not already
        if 'admission_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['admission_date']):
            df['admission_date'] = pd.to_datetime(df['admission_date'])
        
        # Add time-based features if they don't exist
        if 'admission_date' in df.columns:
            if 'admission_month' not in df.columns:
                df['admission_month'] = df['admission_date'].dt.month
            if 'admission_day_of_week' not in df.columns:
                df['admission_day_of_week'] = df['admission_date'].dt.dayofweek
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = (df['admission_date'].dt.dayofweek >= 5).astype(int)
            if 'is_winter' not in df.columns:
                df['is_winter'] = df['admission_date'].dt.month.isin([12, 1, 2]).astype(int)
        
        # Drop the admission_date column since we've extracted the needed features
        if 'admission_date' in df.columns:
            df = df.drop(columns=['admission_date'])
        
        # Create interaction terms for known important combinations
        if 'copd' in df.columns and 'is_winter' in df.columns and 'copd_winter' not in df.columns:
            df['copd_winter'] = df['copd'] * df['is_winter']
        
        # Create age categories if they don't exist
        if 'age' in df.columns and 'age_group' not in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 50, 70, 100], labels=['young', 'middle', 'elderly'])
        
        # Create BMI categories if they don't exist
        if 'bmi' in df.columns and 'bmi_category' not in df.columns:
            df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                       labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Create blood pressure categories if they don't exist
        if 'systolic_bp' in df.columns and 'bp_category' not in df.columns:
            df['bp_category'] = pd.cut(df['systolic_bp'], bins=[0, 120, 140, 180, 300], 
                                      labels=['normal', 'elevated', 'high', 'crisis'])
        
        return df
    
    def engineer_features(self, X, y=None):
        """
        Apply feature engineering to the preprocessed data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Preprocessed features
        y : pandas.Series, optional
            Target variable (only needed for fit)
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            Engineered features ready for modeling
        """
        # First time setup of feature engineering pipeline
        if self.preprocessor is None:
            # Identify numeric and categorical features
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Store for later reference
            self.numeric_features = numeric_features
            self.categorical_features = categorical_features
            
            # Create transformers for different feature types
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False))
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combine transformers in a column transformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )
            
            # Fit the preprocessing pipeline
            if y is not None:
                self.preprocessor.fit(X, y)
        
        # Transform the data
        X_transformed = self.preprocessor.transform(X)
        
        # Get feature names
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            self.feature_names = self.preprocessor.get_feature_names_out()
        
        return X_transformed
    
    def prepare_data(self, data, target='length_of_stay', test_size=0.2):
        """
        Prepare data for modeling by preprocessing, splitting, and engineering features.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw hospital patient data
        target : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test) - Prepared data splits
        """
        # Handle missing target column
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Preprocess the data
        preprocessed_data = self.preprocess_data(data)
        
        # Split features and target
        X = preprocessed_data.drop(columns=[target])
        y = preprocessed_data[target]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Apply feature engineering
        X_train_processed = self.engineer_features(X_train, y_train)
        X_test_processed = self.engineer_features(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test, X_train, X_test
    
    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target values
            
        Returns:
        --------
        self
            The fitted model instance
        """
        # Create the model if it doesn't exist
        if self.model is None:
            self.model = self._create_model()
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Extract feature importances if available
        self._extract_feature_importances()
        
        return self
    
    def _extract_feature_importances(self):
        """Extract feature importances from the model if available."""
        if self.model is None:
            return
        
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importances = self.model.feature_importances_
            self.feature_importances_ = pd.DataFrame({
                'Feature': self.feature_names if self.feature_names is not None else range(len(importances)),
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coefs = self.model.coef_
            self.feature_importances_ = pd.DataFrame({
                'Feature': self.feature_names if self.feature_names is not None else range(len(coefs)),
                'Coefficient': coefs,
                'Importance': np.abs(coefs)
            }).sort_values('Importance', ascending=False)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        array-like
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, X_test_original=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test features (processed)
        y_test : array-like
            Test target values
        X_test_original : pandas.DataFrame, optional
            Original test features (before processing) for additional analysis
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create evaluation results
        results = {
            'mean_squared_error': mse,
            'root_mean_squared_error': rmse,
            'mean_absolute_error': mae,
            'r2_score': r2,
            'predictions': y_pred
        }
        
        # Add departmental analysis if original data is provided
        if X_test_original is not None and 'department' in X_test_original.columns:
            dept_analysis = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'department': X_test_original['department']
            })
            
            dept_metrics = dept_analysis.groupby('department').apply(
                lambda x: pd.Series({
                    'mae': mean_absolute_error(x['actual'], x['predicted']),
                    'r2': r2_score(x['actual'], x['predicted']) if len(x) > 1 else np.nan,
                    'mean_los': x['actual'].mean(),
                    'count': len(x)
                })
            )
            
            results['department_analysis'] = dept_metrics
        
        # Add risk stratification if original data has needed features
        if X_test_original is not None:
            risk_factors = ['emergency_admission', 'num_conditions', 'age']
            if all(factor in X_test_original.columns for factor in risk_factors):
                # Create a simple risk score
                risk_score = (
                    (X_test_original['emergency_admission'] * 3) + 
                    (X_test_original['num_conditions'] * 1.5) + 
                    ((X_test_original['age'] > 70).astype(int) * 1)
                )
                
                # Define risk categories
                risk_categories = pd.qcut(risk_score, 4, labels=['Low', 'Medium', 'High', 'Very High'])
                
                risk_analysis = pd.DataFrame({
                    'actual': y_test,
                    'predicted': y_pred,
                    'risk_category': risk_categories
                })
                
                risk_metrics = risk_analysis.groupby('risk_category').apply(
                    lambda x: pd.Series({
                        'mae': mean_absolute_error(x['actual'], x['predicted']),
                        'mean_los': x['actual'].mean(),
                        'count': len(x)
                    })
                )
                
                results['risk_analysis'] = risk_metrics
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Dictionary of cross-validation results
        """
        # Create model for cross-validation
        model = self._create_model()
        
        # Set up cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
        cv_neg_mse = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        cv_neg_mae = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        
        # Convert negative MSE and MAE to positive values
        cv_mse = -cv_neg_mse
        cv_rmse = np.sqrt(cv_mse)
        cv_mae = -cv_neg_mae
        
        # Create results dictionary
        results = {
            'r2_scores': cv_r2,
            'mse_scores': cv_mse,
            'rmse_scores': cv_rmse,
            'mae_scores': cv_mae,
            'mean_r2': cv_r2.mean(),
            'std_r2': cv_r2.std(),
            'mean_rmse': cv_rmse.mean(),
            'mean_mae': cv_mae.mean()
        }
        
        return results
    
    def tune_hyperparameters(self, X_train, y_train, X_test=None, y_test=None, cv=5):
        """
        Tune model hyperparameters using grid search.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target values
        X_test : array-like, optional
            Test features for final evaluation
        y_test : array-like, optional
            Test target values for final evaluation
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Dictionary of tuning results
        """
        # Define parameter grids for different model types
        param_grids = {
            'ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'elasticnet': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'randomforest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradientboosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'linear': {}  # No hyperparameters to tune for LinearRegression
        }