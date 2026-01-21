#!/usr/bin/env python3
"""
Script to extract the sklearn model from the GA experiment pickle file
and save it in a format that can be loaded without the custom classes.
"""
import sys
import os

# Add the thesis experiments directory to path so we can import custom classes
sys.path.insert(0, "/Users/saadiahumayun/Documents/Thesis experiments")

import joblib
import pickle
from pathlib import Path

def extract_and_save_model():
    """Extract the sklearn model and save it cleanly."""
    
    # Source pickle file
    source_path = Path("/Users/saadiahumayun/Documents/Thesis experiments/ga_f1_multi_population_experiment.pkl")
    
    # Destination path
    dest_path = Path("/Users/saadiahumayun/Documents/breast cancer risk predictor/backend/models/ga_model.pkl")
    
    print(f"Loading experiment data from: {source_path}")
    
    # Load the experiment data (this works because we added the path above)
    experiment_data = joblib.load(source_path)
    
    print(f"Experiment data type: {type(experiment_data)}")
    
    if isinstance(experiment_data, dict):
        print(f"Keys in experiment data: {list(experiment_data.keys())}")
        
        # Try to find the model
        model = None
        
        if 'best_model' in experiment_data:
            model = experiment_data['best_model']
            print("Found 'best_model'")
        elif 'model' in experiment_data:
            model = experiment_data['model']
            print("Found 'model'")
        elif 'models' in experiment_data and len(experiment_data['models']) > 0:
            model = experiment_data['models'][0]
            print("Found model in 'models' list")
        elif 'population_results' in experiment_data:
            for pop_result in experiment_data.get('population_results', []):
                if 'model' in pop_result:
                    model = pop_result['model']
                    print("Found model in 'population_results'")
                    break
                if 'best_model' in pop_result:
                    model = pop_result['best_model']
                    print("Found best_model in 'population_results'")
                    break
        
        if model is None:
            # Let's explore deeper
            print("\nExploring structure...")
            for key, value in experiment_data.items():
                print(f"  {key}: {type(value)}")
                if isinstance(value, dict):
                    for subkey in list(value.keys())[:5]:
                        print(f"    {subkey}: {type(value[subkey])}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"    First item type: {type(value[0])}")
                    if hasattr(value[0], '__dict__'):
                        print(f"    First item attrs: {list(value[0].__dict__.keys())[:10]}")
    else:
        # Maybe it's the model directly
        model = experiment_data
        print(f"Direct model type: {type(model)}")
    
    if model is not None:
        print(f"\nModel type: {type(model)}")
        print(f"Has predict: {hasattr(model, 'predict')}")
        print(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
        
        if hasattr(model, 'predict'):
            # Save just the sklearn model
            print(f"\nSaving model to: {dest_path}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, dest_path)
            print("Model saved successfully!")
            
            # Verify it can be loaded
            print("\nVerifying saved model...")
            loaded = joblib.load(dest_path)
            print(f"Loaded type: {type(loaded)}")
            print(f"Has predict: {hasattr(loaded, 'predict')}")
            print("✅ Model exported successfully!")
        else:
            print("❌ No valid sklearn model found with predict method")
    else:
        print("❌ Could not find model in experiment data")

if __name__ == "__main__":
    extract_and_save_model()

