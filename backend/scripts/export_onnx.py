#!/usr/bin/env python3
"""
Export sklearn GA-optimized Random Forest model to ONNX format for browser inference.

Usage:
    python export_onnx.py --model-path /path/to/ga_model.pkl --output-dir ../frontend/public/models

Requirements:
    pip install skl2onnx onnx joblib numpy
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
except ImportError:
    print("Please install required packages: pip install skl2onnx onnx")
    sys.exit(1)


# Feature configuration from config.py
GA_SELECTED_FEATURES = [
    "educat", "marital", "pipe", "cigar", "sisters", "fmenstr", "menstrs",
    "miscar", "tubal", "uterine_fib", "lmenstr", "prega", "thorm", "hyperten_f",
    "bronchit_f", "diabetes_f", "arthrit_f", "gallblad_f", "bq_age", "hyster_f",
    "ovariesr_f", "bcontr_f", "horm_f", "smoked_f", "rsmoker_f", "cigpd_f",
    "filtered_f", "cig_years", "bmi_20", "bmi_curr", "height_f", "colon_comorbidity",
    "fh_cancer", "entryage_dhq", "ph_any_bq", "ph_any_dhq", "ph_any_sqx",
    "ph_any_trial", "entrydays_bq", "entrydays_dhq", "arm", "age"
]

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.035  # 3.5%
MODERATE_RISK_THRESHOLD = 0.018  # 1.8%


def load_model(model_path: Path):
    """Load the sklearn model from pickle file."""
    print(f"Loading model from: {model_path}")
    
    try:
        data = joblib.load(model_path)
    except Exception as e:
        print(f"joblib failed, trying pickle: {e}")
        import pickle
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(data, dict):
        if 'best_model' in data:
            model = data['best_model']
        elif 'models' in data:
            model = data['models'][0] if isinstance(data['models'], list) else data['models']
        else:
            raise ValueError("Could not find model in pickle dict")
    else:
        model = data
    
    print(f"Loaded model type: {type(model).__name__}")
    return model


def load_scaler(scaler_path: Path):
    """Load the MinMaxScaler if available."""
    if scaler_path.exists():
        print(f"Loading scaler from: {scaler_path}")
        return joblib.load(scaler_path)
    print("No scaler file found, using default parameters")
    return None


def export_model_to_onnx(model, output_path: Path):
    """Convert sklearn model to ONNX format."""
    print("Converting model to ONNX...")
    
    # Define input type (42 features as float32)
    n_features = len(GA_SELECTED_FEATURES)
    initial_type = [('input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={id(model): {'zipmap': False}}  # Return array instead of dict
    )
    
    # Save the model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(output_path))
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model saved to: {output_path} ({size_mb:.2f} MB)")
    
    return onnx_model


def export_scaler_params(scaler, output_path: Path):
    """Export scaler parameters to JSON for JavaScript preprocessing."""
    print("Exporting scaler parameters...")
    
    if scaler is not None:
        params = {
            'feature_names': GA_SELECTED_FEATURES,
            'min_': scaler.min_.tolist() if hasattr(scaler, 'min_') else [0.0] * len(GA_SELECTED_FEATURES),
            'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [1.0] * len(GA_SELECTED_FEATURES),
            'data_min_': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else [0.0] * len(GA_SELECTED_FEATURES),
            'data_max_': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else [1.0] * len(GA_SELECTED_FEATURES),
        }
    else:
        # Default scaling parameters (no scaling - identity transform)
        params = {
            'feature_names': GA_SELECTED_FEATURES,
            'min_': [0.0] * len(GA_SELECTED_FEATURES),
            'scale_': [1.0] * len(GA_SELECTED_FEATURES),
            'data_min_': [0.0] * len(GA_SELECTED_FEATURES),
            'data_max_': [1.0] * len(GA_SELECTED_FEATURES),
        }
    
    # Add risk thresholds for client-side categorization
    params['thresholds'] = {
        'high': HIGH_RISK_THRESHOLD,
        'moderate': MODERATE_RISK_THRESHOLD
    }
    
    # Add feature mapping hints for form -> model features
    params['feature_mapping'] = {
        'age': 'age',
        'education_level': 'educat',
        'marital_status': 'marital',
        'age_at_menarche': 'fmenstr',
        'number_of_live_births': 'prega',
        'hormone_therapy_years': 'thorm',
        'current_bmi': 'bmi_curr',
        'bmi_at_20': 'bmi_20',
        'family_history_cancer': 'fh_cancer',
        'number_of_relatives_with_bc': 'sisters',
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Scaler parameters saved to: {output_path}")


def create_demo_model(output_dir: Path):
    """Create a demo ONNX model for development/testing when real model unavailable."""
    print("Creating demo model for development...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    # Create a simple demo model
    n_features = len(GA_SELECTED_FEATURES)
    np.random.seed(42)
    
    # Generate synthetic training data
    X = np.random.rand(1000, n_features)
    # Create labels based on simplified risk logic
    y = ((X[:, -1] > 0.6) & (X[:, 32] > 0.5)).astype(int)  # age and fh_cancer
    
    # Train a small random forest
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Create and fit scaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    # Export
    onnx_path = output_dir / 'candetect.onnx'
    scaler_path = output_dir / 'scaler.json'
    
    export_model_to_onnx(model, onnx_path)
    export_scaler_params(scaler, scaler_path)
    
    print("\nDemo model created successfully!")
    print("Note: This is a simplified model for development. Replace with real model for production.")


def main():
    parser = argparse.ArgumentParser(description='Export sklearn model to ONNX for browser inference')
    parser.add_argument('--model-path', type=str, help='Path to the sklearn model pickle file')
    parser.add_argument('--scaler-path', type=str, help='Path to the scaler pickle file (optional)')
    parser.add_argument('--output-dir', type=str, default='../../frontend/public/models',
                        help='Output directory for ONNX model and scaler JSON')
    parser.add_argument('--demo', action='store_true', help='Create a demo model for development')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    
    if args.demo:
        create_demo_model(output_dir)
        return
    
    if not args.model_path:
        print("Error: --model-path is required unless using --demo")
        parser.print_help()
        sys.exit(1)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Load model
    model = load_model(model_path)
    
    # Load scaler if provided
    scaler = None
    if args.scaler_path:
        scaler_path = Path(args.scaler_path)
        if scaler_path.exists():
            scaler = load_scaler(scaler_path)
    
    # Export to ONNX
    onnx_path = output_dir / 'candetect.onnx'
    export_model_to_onnx(model, onnx_path)
    
    # Export scaler parameters
    scaler_json_path = output_dir / 'scaler.json'
    export_scaler_params(scaler, scaler_json_path)
    
    print("\nExport complete!")
    print(f"ONNX model: {onnx_path}")
    print(f"Scaler params: {scaler_json_path}")


if __name__ == '__main__':
    main()
