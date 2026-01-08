# app/utils/validators.py
"""
Input validation utilities.
"""
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from app.models.schemas import PredictionRequest
from app.core.config import settings


def validate_prediction_request(request: PredictionRequest) -> List[str]:
    """
    Validate prediction request inputs.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate demographics
    if request.demographics.age < 20 or request.demographics.age > 100:
        errors.append("Age must be between 20 and 100 years")
    
    valid_races = ["white", "black", "hispanic", "asian", "other"]
    if request.demographics.race.lower() not in valid_races:
        errors.append(f"Race must be one of: {', '.join(valid_races)}")
    
    # Validate reproductive history
    if request.reproductive_history.age_at_menarche < 8 or request.reproductive_history.age_at_menarche > 20:
        errors.append("Age at menarche must be between 8 and 20 years")
    
    if request.reproductive_history.age_at_first_birth is not None:
        if request.reproductive_history.age_at_first_birth < 15 or request.reproductive_history.age_at_first_birth > 50:
            errors.append("Age at first birth must be between 15 and 50 years")
        
        # Logical check
        if request.reproductive_history.age_at_first_birth <= request.reproductive_history.age_at_menarche:
            errors.append("Age at first birth must be greater than age at menarche")
    
    if request.reproductive_history.number_of_live_births < 0 or request.reproductive_history.number_of_live_births > 20:
        errors.append("Number of live births must be between 0 and 20")
    
    # Nulliparous check
    if request.reproductive_history.number_of_live_births == 0 and request.reproductive_history.age_at_first_birth is not None:
        errors.append("Age at first birth should be null for women with no live births")
    
    if request.reproductive_history.first_degree_bc < 0 or request.reproductive_history.first_degree_bc > 10:
        errors.append("Number of first-degree relatives with breast cancer must be between 0 and 10")
    
    # Validate body metrics
    if request.body_metrics.current_bmi < 15 or request.body_metrics.current_bmi > 60:
        errors.append("BMI must be between 15 and 60")
    
    if request.body_metrics.height_cm < 120 or request.body_metrics.height_cm > 220:
        errors.append("Height must be between 120 and 220 cm")
    
    if request.body_metrics.weight_kg < 30 or request.body_metrics.weight_kg > 300:
        errors.append("Weight must be between 30 and 300 kg")
    
    # Validate BMI calculation
    calculated_bmi = request.body_metrics.weight_kg / ((request.body_metrics.height_cm / 100) ** 2)
    if abs(calculated_bmi - request.body_metrics.current_bmi) > 0.5:
        errors.append("BMI doesn't match height and weight")
    
    # Validate medical history
    if request.medical_history.breast_biopsies < 0 or request.medical_history.breast_biopsies > 10:
        errors.append("Number of breast biopsies must be between 0 and 10")
    
    # Validate lifestyle factors (if provided)
    if request.lifestyle:
        valid_smoking_status = ["never", "current", "former"]
        if request.lifestyle.smoking_status not in valid_smoking_status:
            errors.append(f"Smoking status must be one of: {', '.join(valid_smoking_status)}")
        
        if request.lifestyle.pack_years is not None:
            if request.lifestyle.pack_years < 0 or request.lifestyle.pack_years > 100:
                errors.append("Pack years must be between 0 and 100")
            
            # Logical check
            if request.lifestyle.smoking_status == "never" and request.lifestyle.pack_years > 0:
                errors.append("Pack years should be 0 for never smokers")
    
    return errors


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> List[str]:
    """
    Validate password strength.
    
    Returns:
        List of validation errors
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("Password must contain at least one special character")
    
    return errors


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize text input."""
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit length
    text = text[:max_length]
    
    # Strip whitespace
    text = text.strip()
    
    return text


def validate_date_range(start_date: Optional[datetime], 
                       end_date: Optional[datetime],
                       max_days: int = 365) -> List[str]:
    """Validate date range."""
    errors = []
    
    if start_date and end_date:
        if start_date > end_date:
            errors.append("Start date must be before end date")
        
        if (end_date - start_date).days > max_days:
            errors.append(f"Date range cannot exceed {max_days} days")
    
    if end_date and end_date > datetime.utcnow():
        errors.append("End date cannot be in the future")
    
    return errors


def validate_feature_values(features: Dict[str, Any]) -> List[str]:
    """
    Validate feature values for drift detection.
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Define expected ranges for key features
    feature_ranges = {
        "age": (20, 100),
        "bmi_curr": (15, 60),
        "bmi_20": (15, 60),
        "bmi_50": (15, 60),
        "fmenstr": (8, 20),
        "fchilda": (15, 50),
        "livec": (0, 20),
        "breast_fh_cnt": (0, 10),
        "pack_years": (0, 100),
        "height_f": (48, 84),  # inches
        "weight_f": (60, 600),  # pounds
    }
    
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in features:
            value = features[feature]
            if isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    errors.append(f"{feature} value {value} is outside expected range [{min_val}, {max_val}]")
    
    return errors


def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    # Model name should be alphanumeric with underscores
    pattern = r'^[a-zA-Z0-9_]+$'
    return re.match(pattern, model_name) is not None


def validate_model_version(version: str) -> bool:
    """Validate model version format."""
    # Version should be semantic versioning or simple numeric
    pattern = r'^(\d+\.)?(\d+\.)?(\*|\d+)$|^\d+$'
    return re.match(pattern, version) is not None