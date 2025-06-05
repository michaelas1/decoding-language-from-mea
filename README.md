# Decoding language from microelectrode array data

This folder contains the code for my master's thesis.

## Installation

TBA

## How to reproduce the results
1. put the experiment data in the appropriate folder
2. extract the spikerates and transcribe the audio by executing the `experiment2_data_extraction.ipynb` notebook
3. run the hyperparameter searches with the `hyperparameter_searches.ipynb` notebook
4. visualize the hyperparameter search results with the `hyperparameter_search_results.ipynb` notebook
5. retrain the models with the optimal hyperparameter configurations with the `best_model_retraining.ipynb` notebook
6. visualize the retrained model results by running the `best_model_results.ipynb` notebook
7. conduct permutation tests and visualize the sequence processing results with the `sequence_processing_results.ipynb` notebook

## Using already trained models
The optimal weights for each model and session combination can be found in the folder "TBA". \
The models can be loaded and used for inference in the following way:

```python
from thesis_project.evaluation.model_inference import ModelInference
model_inference = ModelInference(model_type, task_type, session_id, label_name)
model_inference.chunked_inference(features, targets)
```
