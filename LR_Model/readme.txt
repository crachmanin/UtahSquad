######################################################################
#																	 #
#							Utah SQuAD QA							 #
#																	 #
# Christian Rachmaninoff								Madhur Pandey#
######################################################################

1. Pre-requisites:

   Spacy  : NLP package. Download using the below command:
   			python -m pip install --user spacy

   Sklearn: Machine Learning package, install using below command:
   			python -m pip install --user sklearn

   The training and development dataset should downloaded from the
   below link before running the QA system.

   DataSet: https://rajpurkar.github.io/SQuAD-explorer/

2. Run Instruction:

   To run the basic feature-based Question Answering system, first,
   we need to extract features from the dataset and create a feature
   file. Then, we need to train and predict the output of the model 
   on the test set. This is achieved by running the below commands in 
   the sequence:

   > python sent_features_lower.py <<train set>>

   For generating development set features, <<train set>> = dev
   else, <<train set>> is not required.

   > python predict_dev.py

3. Execution:

   The program is successfully executed on the below CADE machine:
   Lab: 1
   Machine: 5

4. Miscellaneous:

   A lot of code used for Neural Network implementation is also 
   available in the main folder. However, the files for the Basic
   QA systems can be found in the folder 'LR_Model'