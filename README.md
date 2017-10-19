# Melbourne University AES-MathWorks-NIH Seizure Prediction Challenge

This repository contains the second place submission for Melbourne University AES-MathWorks-NIH Seizure Prediction Challenge on Kaggle.

https://www.kaggle.com/c/melbourne-university-seizure-prediction

##Dependencies

###Required

 * Python 2.7
 * scikit_learn-0.18.1
 * numpy-1.11.1
 * pandas-0.19.1
 * scipy-0.18.0

##Train the model and make predictions

Obtain all the competition data.  This includes the training data, the old testing data as well as the new testing data.  Also, obtain the train_and_test_data_labels_safe.csv; remove all the unsafe training data plus move all the safe files from the old testing data to their corresponding training directory and append the correct corresponding label (e.g. 1_665.mat from test_1 became 1_665.mat_1.mat in train_1).

We took the method of using a hold-out set for testing our model predictions.  As such, 10% of the training data was removed for testing and not used for training; that is we only used 90% of the available data for model generation.  To replicate our results one must move/remove these files from training.  The files in question are located and tabulate in pat_1_hold_out_test_files.txt, pat_2_hold_out_test_files.txt and pat_3_hold_out_test_files.txt.
 
The directory name of the data, the patient in question, etc. are controlled by the SETTINGS.json file.

To run, features must be generated, the model must be trained and then prediction generated for each patient.  Generated features per patient are included that were used to generate the second place submission.
To run from scratch in an ipython session:
```
%run get_all_features.py
%run train.py
%run predict.py
```

One classifier is trained for each patient and dumped to a pickle file.  Trained models that generate the second place submission is included.

```
modeldump_1_ef.pkl
modeldump_2_ef.pkl
modeldump_3_ef.pkl
```

To generate new predictions for existing patients, all that is needed is for features to be generated using the get_all_features.py routine.   Afterwards predict.py can be run on the new generated features.
