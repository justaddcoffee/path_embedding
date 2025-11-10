# Models Directory

This directory stores trained classifier models in pickle format.

## Usage

After training, models are saved here as `.pkl` files:

```bash
uv run path-embedding train \
  --data data/drugmechdb.yaml \
  --output models/classifier.pkl
```

Models can then be loaded and used for prediction with the saved pickle files.

## Model Format

Trained models are saved as pickle files containing:
- RandomForestClassifier (sklearn)
- Feature extraction pipeline
- Training metadata

## Loading Models

```python
import pickle

with open('models/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Use model for predictions
predictions = model.predict(embeddings)
```
