# ML Models Directory

Place your trained model files here for Railway deployment.

## Required Files

Copy your trained model pickle file to this directory:

```bash
cp "/Users/saadiahumayun/Documents/Thesis experiments/ga_f1_multi_population_experiment.pkl" ./ga_model.pkl
```

## Supported Model Files

- `ga_model.pkl` - GA-optimized Random Forest model (primary)
- `ga_f1_multi_population_experiment.pkl` - Alternative name

## Important

⚠️ Make sure to add this to your `.gitignore` if the model file is too large:

```
# Large model files (use Git LFS instead)
models/*.pkl
!models/README.md
```

For large models (>100MB), consider:
1. Using Git LFS
2. Storing in cloud storage (S3, GCS) and downloading on startup
3. Using Railway volumes for persistent storage

