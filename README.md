## ALTEGRAD - Advanced Learning for Text and Graph Data ()

### Project Description

This project is developed for the course "Advanced Learning for Text and Graph Data" of Prof. Michalis Vazirgiannis at the Master MVA ENS Paris-Saclay. The code is designed for a Kaggle challenge where the objective is to retrieve a molecule from a given text query and a list of molecules represented as graphs. Notably, no reference or textual information about the molecules is provided, making the task challenging.

### Files

1. **f_during_train.py:**
   - Python script for training the model with a procedure to increase the batch size during learning. We implemented several contrastive losses. 

2. **main.py:**
   - Python script for training the model with a fixed batch size.

3. **test_models.ipynb:**
   - Jupyter notebook for testing the models and analyzing the dataset.

4. **Model.py:**
   - Contains the implementation of the model with different architecture (a Graph encoder and a Text encoder).

### Usage

1. Run `f_during_train.py` for training the model (all the argument are available in the parser function):

   ```bash
  python f_during_train.py -loss_type lifted_structured_loss -init_bs 80 -final_bs 210 -n_ep_update 2 -conv_layer ChebConv -lr 3e-5



For any inquiries or issues, please contact Lucas Gascon, Hippolyte Pilchen or Pierre Fihey at <forename>.<name>@polytechnique.edu
