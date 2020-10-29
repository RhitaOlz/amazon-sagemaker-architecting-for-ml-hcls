
import argparse
import pandas as pd
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    regr = joblib.load(os.path.join(model_dir, "model.joblib"))
    return regr

def predict_fn(input_data, model):
    '''return the class and the probability of the class'''
    prediction = model.predict(input_data)
    pred_prob = model.predict_proba(input_data) #a numpy array
    return np.array(pred_prob)

def parse_args():
    
    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_leaf_nodes', type=int, default=-1)

   # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default = os.environ['SM_CHANNEL_TEST'])
    
    # hyperparameters for tuning
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default = 0.001)
    
    args = parser.parse_args()
    
    return args

def train(args):
    
   # Take the set of files and read them all into a single pandas dataframe
    train_data = pd.read_csv('s3://sagemaker-us-east-1-138066269299/model-hosting/train/train_set.csv', engine='python')

    # labels are in the first column
    train_y = train_data['truth']
    train_X = train_data[train_data.columns[1:len(train_data)]]

    # Now use scikit-learn's MLP Classifier to train the model.

    regr = MLPClassifier(random_state=1, max_iter=500, batch_size = args.batch_size, learning_rate_init = args.lr, solver='lbfgs').fit(train_X, train_y)
    regr.get_params()

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(regr, os.path.join(args.model_dir, "model.joblib")) 
    
    return regr
    
def accuracy(y_pred, y_true):
    
    cm = confusion_matrix(y_pred, y_true)
    
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    
    rt = diagonal_sum / sum_of_all_elements
    
    print ('Accuracy: {}'.format(rt))
    
    return rt
    
    
def test(regr, args):
    test_data=pd.read_csv(os.path.join(args.test, 'test_set.csv'), engine='python')

    # labels are in the first column
    y_true = test_data['truth']
    test_x = test_data[test_data.columns[1:len(test_data)]]
    
    y_pred = regr.predict(test_x)
    
    accuracy(y_pred, y_true)
    
if __name__ == '__main__':

#     args = parse_args()
    
#     regr = train(args)
    
#     test(regr, args)
    
    
    start = timer()

    args = parse_args()
        
    # Print SageMaker args
    print('\n====== args ======')
    for k,v in vars(args).items():
        print(f'{k},  type: {type(v)},  value: {v}')
    
    train(args)

    # Package inference code with model export
    subprocess.call('mkdir /opt/ml/model/code'.split())
    subprocess.call('cp /opt/ml/code/inference.py /opt/ml/model/code/'.split())
    
    elapsed_time = round(timer()-start,3)
    print(f'Elapsed time: {elapsed_time} seconds')  
    print('===== Training Completed =====')
