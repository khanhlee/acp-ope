# ACP-OPE
## Prediction of Anticancer Peptides Based on an Ensemble Model of Deep Learning and Machine Learning Using Ordinal Positional Encoding

This code is to reproduce the anticancer peptide prediction model that has been used in the paper "Prediction of Anticancer Peptides Based on an Ensemble Model of Deep Learning and Machine Learning Using Ordinal Positional Encoding"

### Dependencies
- Python 3.7
- Tensorflow 2.8.2
- Keras 2.8.0

### Using our final model to evaluate your sequence:
Using "final_model_evaluation.py" to run, example FASTA testing file is in "data" folder (*ACP20mainTest.fasta*). The output results: 1 is anticancer peptide & 0 is non-anticancer peptide.

### Re-training our models:
Using files in "training" folder which included our training setting for CNN, Bi-LSTM, RNN, or ensemble models. Training data is located in "data" folder (*ACP20mainTrain.fasta*).

## Citation
Please cite our paper as:
>Yuan Q, Chen K, Yu Y, Le NQK, & Chua MCH (2020). Prediction of Anticancer Peptides Based on an Ensemble Model of Deep Learning and Machine Learning Using Ordinal Positional Encoding. [Under Review]


