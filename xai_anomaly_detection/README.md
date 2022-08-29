# XAI anomaly detection
## Description
This package contains several classes to generate explainations by explainable AI techniques for a fully connected DNN and the NSL-KDD data set it is trained on. Each class can be used indpendent of the others.

## Installation (currently)
´´´bash
pip install -e .
´´´

## Testing
`python -m pytest`

just `pytest` causes `ImportError` because `pytest` does not add the current directory to the PYTHONPATH
