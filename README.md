# XAI_anomaly_detection

## Seminar Cybersecurity and AI
#### Subject: Explainable AI in context of anomaly detection

## Description
The paper of Mane and Rao (Explaining Network Intrusion Detection System Using Explainable AI Framework, 2021) presented a framework with different explainable AI techniques for analyzing a fully connected deep neural network. Implemented AI techniques are SHAP, Protodash, BRCG, LIME and CEM.

The goal of this project is to reconstruct the explainable AI methods proposed in the paper to get explanation for NSL-KDD data set and a fully connected DNN trained on this data set.
Implemented techniques are SHAP, Protodash, BRCG and LIME. CEM is the only technique currently not integrated.

## Structure
Basically, the project has three parts:
1. Data Preprocessing
2. Model
3. Explanations

Each step is implemented in a separated module. A Jupyter notebook is implemented to invokes the different steps and present the results.

### Installation
See [here](xai_anomaly_detection/README.md)

## Results
### Preprocessing

### Model performance
Model performance after 5 epochs
```bash
acc: 81.83%
precision_m: 43.08%
recall_m: 100.00%
f1_m: 59.67%
```

### Explanations

#### SHAP
![Alt text](/docs/force_plot.png "Example force plot")
![Alt text](/docs/force_plot.png "Example summary plot")

#### LIME
![Alt text](/docs/LIME.png "Example LIME result")

#### BRCG
```bash
Accuracy: 0.7955462893137559
Balanced accuracy: 0.8168909733184287
Precision: 0.6854193923535398
Recall: 0.9710637421480794
{'isCNF': False,
 'rules': ['dst_bytes > 0.00 AND hot <= 0.00',
  'count <= 0.03 AND dst_host_serror_rate <= 0.08 AND level > 0.86',
  'level > 0.86 AND service_smtp not  AND src_bytes > 0.00']}
```
#### ProtoDash
![Alt text](/docs/ProtoDash.PNG "Example snip from ProtoDash output")
