# app/services/preprocessing.py
"""
Data preprocessing service for transforming input data to model-ready format.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from app.models.schemas import PredictionRequest
from app.core.config import settings, FEATURE_MAPPINGS


logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for preprocessing input data."""
    
    def __init__(self):
        self.feature_mappings = FEATURE_MAPPINGS
        self.required_features = settings.GA_SELECTED_FEATURES
        
    def preprocess_input(self, request: PredictionRequest) -> Dict[str, Any]:
        """
        Preprocess input request to model-ready features.
        
        Args:
            request: PredictionRequest object
            
        Returns:
            Dictionary of preprocessed features
        """
        features = {}
        
        # Extract demographics
        features.update(self._process_demographics(request.demographics))
        
        # Extract reproductive history
        features.update(self._process_reproductive_history(request.reproductive_history))
        
        # Extract body metrics
        features.update(self._process_body_metrics(request.body_metrics))
        
        # Extract medical history
        features.update(self._process_medical_history(request.medical_history))
        
        # Extract lifestyle factors (if provided)
        if request.lifestyle:
            features.update(self._process_lifestyle(request.lifestyle))
        
        # Add study-related features (defaults for production use)
        features.update(self._add_study_features())
        
        # Handle missing features
        features = self._handle_missing_features(features)
        
        # Validate features
        self._validate_features(features)
        
        return features
    
    def _process_demographics(self, demographics) -> Dict[str, Any]:
        """Process demographics data."""
        features = {
            'age': demographics.age,
            'bq_age': demographics.age,  # Age at baseline questionnaire
            'entryage_dhq': demographics.age,  # Entry age at DHQ
        }
        
        # Education mapping (if provided)
        if demographics.education:
            education_mapping = {
                'less_than_high_school': 1,
                'high_school': 3,
                'some_college': 5,
                'college_graduate': 6,
                'postgraduate': 7
            }
            features['educat'] = education_mapping.get(demographics.education, 5)
        else:
            features['educat'] = 5  # Default: some college
        
        # Marital status mapping (if provided)
        if demographics.marital_status:
            marital_mapping = {
                'married': 1,
                'widowed': 2,
                'divorced': 3,
                'separated': 4,
                'never_married': 5
            }
            features['marital'] = marital_mapping.get(demographics.marital_status, 1)
        else:
            features['marital'] = 1  # Default: married
        
        return features
    
    def _process_reproductive_history(self, reproductive) -> Dict[str, Any]:
        """Process reproductive history data."""
        # Family history of breast cancer is a major risk factor
        has_bc_family_history = reproductive.first_degree_bc > 0 if reproductive.first_degree_bc else False
        
        features = {
            'fmenstr': self._map_age_at_menarche(reproductive.age_at_menarche),
            'livec': min(reproductive.number_of_live_births, 5),  # Cap at 5+
            'fh_cancer': 1 if has_bc_family_history else 0,
            'breast_fh_cnt': min(reproductive.first_degree_bc, 3) if reproductive.first_degree_bc else 0,
            'sisters': reproductive.first_degree_bc if reproductive.first_degree_bc else 0,  # Approximate sisters with BC
        }
        
        # Age at first birth
        if reproductive.age_at_first_birth is not None:
            features['fchilda'] = self._map_age_at_first_birth(reproductive.age_at_first_birth)
        else:
            # Nulliparous - use current age as per preprocessing logic
            features['fchilda'] = 99  # Special code for nulliparous
        
        # Number of pregnancies (estimated if not available)
        features['pregc'] = reproductive.number_of_live_births + reproductive.miscarriages if reproductive.miscarriages else reproductive.number_of_live_births
        
        # Miscarriages
        if reproductive.miscarriages is not None:
            features['miscar'] = min(reproductive.miscarriages, 2)  # 0, 1, or 2+
        
        # Menopause status (derived from age if not provided)
        # This is simplified - in production, you'd ask directly
        if hasattr(reproductive, 'menopausal_status'):
            features['menstrs'] = 1 if reproductive.menopausal_status == 'natural' else 2
        else:
            # Estimate based on age
            features['menstrs'] = 1 if features.get('age', 50) >= 50 else 0
        
        return features
    
    def _process_body_metrics(self, body_metrics) -> Dict[str, Any]:
        """Process body metrics data."""
        features = {
            'bmi_curr': body_metrics.current_bmi,
            'height_f': body_metrics.height_cm / 2.54,  # Convert to inches
        }
        
        # BMI at age 20
        if body_metrics.bmi_at_age_20:
            features['bmi_20'] = body_metrics.bmi_at_age_20
        else:
            # Estimate BMI at 20 as slightly lower than current
            features['bmi_20'] = body_metrics.current_bmi * 0.85
        
        return features
    
    def _process_medical_history(self, medical) -> Dict[str, Any]:
        """Process medical history data."""
        features = {
            'ph_any_trial': 1 if medical.personal_cancer_history else 0,
            'ph_any_bq': 1 if medical.personal_cancer_history else 0,
            'ph_any_dhq': 1 if medical.personal_cancer_history else 0,
            'ph_any_sqx': 1 if medical.personal_cancer_history else 0,
            'ph_any_muq': 1 if medical.personal_cancer_history else 0,
            'bbd': 1 if medical.benign_breast_disease else 0,
        }
        
        # Hormone therapy - set BOTH horm_f and curhorm for model compatibility
        is_on_hormones = medical.hormone_therapy_current
        features['horm_f'] = 1 if is_on_hormones else 0  # THIS WAS MISSING!
        features['curhorm'] = 1 if is_on_hormones else 0
        
        if medical.hormone_therapy_years:
            features['thorm'] = self._categorize_hormone_years(medical.hormone_therapy_years)
        else:
            features['thorm'] = 0 if not is_on_hormones else 1  # At least 1 if currently using
        
        # NSAID use
        features['asp'] = 1 if medical.aspirin_use else 0
        features['ibup'] = 1 if medical.ibuprofen_use else 0
        
        # Map NSAID frequency (simplified)
        features['asppd'] = 1 if medical.aspirin_use else 0
        features['ibuppd'] = 1 if medical.ibuprofen_use else 0
        
        return features
    
    def _process_lifestyle(self, lifestyle) -> Dict[str, Any]:
        """Process lifestyle factors."""
        features = {}
        
        # Smoking
        if lifestyle.smoking_status == 'never':
            features['smoked_f'] = 0
            features['rsmoker_f'] = 0  # Not recent smoker
            features['cigpd_f'] = 0
            features['filtered_f'] = 0
            features['cig_years'] = 0
        elif lifestyle.smoking_status == 'former':
            features['smoked_f'] = 1
            features['rsmoker_f'] = 0  # Not currently smoking
            features['cigpd_f'] = 10  # Average cigarettes per day when smoked
            features['filtered_f'] = 1  # Assume filtered
            features['cig_years'] = lifestyle.pack_years if lifestyle.pack_years else 10
        else:  # current
            features['smoked_f'] = 1
            features['rsmoker_f'] = 1  # Recent/current smoker
            features['cigpd_f'] = 15
            features['filtered_f'] = 1
            features['cig_years'] = lifestyle.pack_years if lifestyle.pack_years else 15
        
        # Birth control
        features['bcontr_f'] = 1 if lifestyle.birth_control_ever else 0
        
        return features
    
    def _add_study_features(self) -> Dict[str, Any]:
        """Add study-related features with reasonable defaults."""
        return {
            'center': 5,  # Default study center
            'arm': 1,  # Intervention arm (doesn't affect risk)
            'rndyear': 2000,  # Default randomization year
            'reconsent_outcome': 1,  # Active consent
            'reconsent_outcome_days': 365,  # Days to consent
            'in_tgwas_population': 1,  # Include in analysis
            'bq_compdays': 30,  # Days to complete questionnaire
            'entrydays_bq': 30,
            'entrydays_dhq': 60,
            'entrydays_sqx': 90,
            'entrydays_dqx': 120,
            'entrydays_muq': 150,
            'ph_breast_bq': 0,  # No breast cancer history
            'ph_breast_dhq': 0,
            'ph_breast_sqx': 0,
            'ph_breast_muq': 0,
        }
    
    def _handle_missing_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing features with appropriate defaults."""
        # Define defaults for all 42 features needed by the GA model (in exact order)
        defaults = {
            # Demographics
            'educat': 5,  # Education level (some college)
            'marital': 1,  # Married
            
            # Smoking - tobacco
            'pipe': 0,  # Never smoked pipe
            'cigar': 0,  # Never smoked cigars
            
            # Family
            'sisters': 1,  # Number of sisters
            
            # Reproductive
            'fmenstr': 3,  # Age at menarche category (12-13)
            'menstrs': 1,  # Menstrual status
            'miscar': 0,  # Number of miscarriages
            'tubal': 0,  # No tubal pregnancies
            'uterine_fib': 0,  # No uterine fibroids
            'lmenstr': 50,  # Age at last period
            'prega': 25,  # Age at first pregnancy
            
            # Hormone therapy
            'thorm': 0,  # Hormone therapy years category
            
            # Medical history flags
            'hyperten_f': 0,  # No hypertension
            'bronchit_f': 0,  # No bronchitis
            'diabetes_f': 0,  # No diabetes
            'arthrit_f': 0,  # No arthritis
            'gallblad_f': 0,  # No gallbladder issues
            
            # Age at baseline questionnaire
            'bq_age': 55,  # Age at questionnaire
            
            # Surgical history
            'hyster_f': 0,  # No hysterectomy
            'ovariesr_f': 0,  # Ovaries not removed
            
            # Birth control and hormones
            'bcontr_f': 0,  # No birth control
            'horm_f': 0,  # No hormone use
            
            # Smoking - cigarettes
            'smoked_f': 0,  # Never smoked
            'rsmoker_f': 0,  # Not recent smoker
            'cigpd_f': 0,  # Cigarettes per day
            'filtered_f': 0,  # Filtered cigarettes
            'cig_years': 0,  # Years of smoking
            
            # BMI
            'bmi_20': 22.0,  # BMI at age 20
            'bmi_curr': 25.0,  # Current BMI
            
            # Physical
            'height_f': 65.0,  # Height in inches (~165cm)
            
            # Comorbidities
            'colon_comorbidity': 0,  # No colon comorbidity
            
            # Family history
            'fh_cancer': 0,  # No family cancer history
            
            # Study features
            'entryage_dhq': 55,  # Entry age at DHQ
            'ph_any_bq': 0,  # Personal history any BQ
            'ph_any_dhq': 0,  # Personal history any DHQ
            'ph_any_sqx': 0,  # Personal history any SQX
            'ph_any_trial': 0,  # Personal history trial
            'entrydays_bq': 30,  # Entry days BQ
            'entrydays_dhq': 60,  # Entry days DHQ
            'arm': 1,  # Study arm
            
            # Age
            'age': 55,  # Patient age
        }
        
        # Add defaults for missing features
        for feature, default in defaults.items():
            if feature not in features:
                features[feature] = default
        
        return features
    
    def _validate_features(self, features: Dict[str, Any]):
        """Validate that all required features are present."""
        missing_features = []
        for feature in self.required_features:
            if feature not in features:
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add zeros for missing features (not ideal but allows prediction)
            for feature in missing_features:
                features[feature] = 0
    
    # Helper methods
    def _map_age_at_menarche(self, age: int) -> int:
        """Map age at menarche to categorical value."""
        if age < 10:
            return 1
        elif age <= 11:
            return 2
        elif age <= 13:
            return 3
        elif age <= 15:
            return 4
        else:
            return 5
    
    def _map_age_at_first_birth(self, age: int) -> int:
        """Map age at first birth to categorical value."""
        if age < 16:
            return 1
        elif age <= 19:
            return 2
        elif age <= 24:
            return 3
        elif age <= 29:
            return 4
        elif age <= 34:
            return 5
        elif age <= 39:
            return 6
        else:
            return 7
    
    def _categorize_bmi(self, bmi: float) -> int:
        """Categorize BMI."""
        if bmi < 18.5:
            return 1
        elif bmi < 25:
            return 2
        elif bmi < 30:
            return 3
        else:
            return 4
    
    def _categorize_hormone_years(self, years: float) -> int:
        """Categorize hormone therapy years."""
        if years <= 1:
            return 5
        elif years <= 3:
            return 4
        elif years <= 5:
            return 3
        elif years <= 9:
            return 2
        else:
            return 1
    
    def _categorize_bc_years(self, years: float) -> int:
        """Categorize birth control years."""
        if years <= 1:
            return 5
        elif years <= 3:
            return 4
        elif years <= 5:
            return 3
        elif years <= 9:
            return 2
        else:
            return 1
    
    def _estimate_weight_from_bmi(self, bmi: float, height_cm: float) -> float:
        """Estimate weight in pounds from BMI and height."""
        height_m = height_cm / 100
        weight_kg = bmi * (height_m ** 2)
        return weight_kg * 2.205