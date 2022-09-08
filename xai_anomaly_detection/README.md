# XAI anomaly detection
## Description
This package contains several classes to generate explanations by explainable AI techniques for a fully connected DNN and the NSL-KDD data set it is trained on. Each class can be used independent of the others.

## Installation
Get git repository
´´´bash
git clone https://github.com/TobiasSchalau/XAI_anomaly_detection.git
´´´
(optionally) Setup virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Change to package directory
´´´bash
cd xai_anomaly_detection
´´´
install package and dependencies
```bash
pip install -e .
```

## Testing
### Dependencies
* `pytest`
* `pytest-black`

### Run
```bash
python -m pytest
```

just `pytest` causes `ImportError` because `pytest` does not add the current directory to the `PYTHONPATH`
