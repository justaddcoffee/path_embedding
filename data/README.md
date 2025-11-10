# Data Directory

This directory contains DrugMechDB YAML files and other source data used for training the path embedding classifier.

## Usage

Place your DrugMechDB YAML files here and reference them when running the train command:

```bash
uv run path-embedding train \
  --data data/drugmechdb.yaml \
  --output models/classifier.pkl \
  --api-key-path /path/to/openai.key
```

## Sample Data

The `tests/data/sample_drugmechdb.yaml` file contains 2 example indications for development and testing.

## Full Data

Download the full DrugMechDB dataset from: https://github.com/harinder-singh/DM_project
