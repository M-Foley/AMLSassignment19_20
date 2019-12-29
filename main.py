import sys
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from data_preprocessing import get_data
from A1 import A1
from A2 import A2
from B1 import B1
from B2 import B2



# ======================================================================================================================
# Data preprocessing
smile_X_train, smile_y_train, smile_X_test, smile_y_test, gender_X_train, gender_y_train, gender_X_test, gender_y_test, eye_X_train, eye_y_train, eye_X_test, eye_y_test, face_X_train, face_y_train, face_X_test, face_y_test = get_data()
# ======================================================================================================================
# Task A1
model_A1 = A1.create_model()            # Build model object.
acc_A1_train, model_A1_trained = A1.train_model(model_A1, gender_X_train, gender_y_train) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = A1.test_model(model_A1_trained, gender_X_test, gender_y_test)   # Test model based on the test set.

#Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
model_A2 = A2.create_model()
acc_A2_train, model_A2_trained = A2.train_model(model_A2, smile_X_train, smile_y_train)
acc_A2_test = A2.test_model(model_A2_trained, smile_X_test, smile_y_test)

#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
model_B1 = B1.create_model()
acc_B1_train, model_B1_trained = B1.train_model(model_B1, face_X_train, face_y_train)
acc_B1_test = B1.test_model(model_B1_trained, face_X_test, face_y_test)
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
model_B2 = B2.create_model()
acc_B2_train, model_B2_trained = B2.train_model(model_B2, eye_X_train, eye_y_train)
acc_B2_test = B2.test_model(model_B2_trained, eye_X_test, eye_y_test)
#Clean up memory/GPU etc...


# ======================================================================================================================
# Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'