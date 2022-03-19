## main libraries
import numpy as np
import pandas as pd
from scipy.stats import boxcox
## for preprocessing and preparing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Reading the Dataset --> New Dataset insted of DELTA, givem (IRF (target) & IR0 (Feature)) 
## This Dataset will outperform the old one
df = pd.read_excel('dataset.xlsx', sheet_name='Sheet1')
## Shuffling the Dataset to split randomly --> helping the model for good training 
df = utils.shuffle(df, random_state=123)

## Split to target and Features --> Note Impotant --> split the original data and then do the transformation on splitted data
X = df.drop(columns=['IRIF(in/mile)'], axis=1)  ## drop the target (Features)
y = df['IRIF(in/mile)']  ## target

## split to training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

## BoxCox for both (X_train) and (X_test) for both two columns --> Doint this step to get (lambda)
X_train['AGE'], lmd_age_train = boxcox(X_train['AGE'])            ## AGE  -- train
X_train['RUT(in)'], lmd_rut_train = boxcox(X_train['RUT(in)'])    ## RUT(in) -- train

## Scaling the Features to (mean=0, std=1) --> standard Normal Distr.
scaler = StandardScaler()
scaler.fit(X_train.values)  ## fit on train and transform on train 

### Function for processing one point
def process_one(X_new):
    ''' This Function tries to process new point before the model take it
    Args:
    ****** 
        (X_new: list): [IRI0(in/mile)', 'AGE', 'FC%', 'LC(ft/mile)', 'TC(ft/mile)', 'RUT(in)']
    Returns:
    *******
        The processed point ready for the model to predict
    '''
    X_new = np.array(X_new, dtype='float32')
    X_new[0] = np.log(X_new[0])   ## take the log of IRI0
    X_new = pd.DataFrame(X_new.reshape(1, -1))  ## to DF
    X_new = scaler.transform(X_new)  ## Scaling
    return X_new


### Function for processi
def process_batch(X_new):
    ''' This Function tries to process batch of points before the model take it
    Args:
    ****** 
        (X_new: batch of instances): each record [IRI0(in/mile)', 'AGE', 'FC%', 'LC(ft/mile)', 'TC(ft/mile)', 'RUT(in)']
    Returns:
    *******
        The processed batch of points ready for the model to predict
    '''
    X_new = np.array(X_new, dtype='float32')
    iri0 = np.log(X_new[:, 0])  ## taking the log
    age = X_new[:, 1]
    fc = X_new[:, 2]
    lc = X_new[:, 3]
    tc = X_new[:, 4]
    rut = X_new[:, 5]
    age = boxcox(age, lmbda=lmd_age_train)
    rut = boxcox(rut, lmbda=lmd_rut_train)
    X_new = np.column_stack((iri0, age, fc, lc, tc, rut))  ## concatenating 
    X_new = scaler.transform(X_new)  ## Scaling
    return X_new