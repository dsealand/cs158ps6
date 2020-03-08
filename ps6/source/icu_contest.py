"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2020 Feb 27
Description : Survival of ICU Patients

This code is adapted from course material by Jenna Wiens (UMichigan).
"""

# python libraries
import os
from joblib import Parallel, delayed
from joblib import dump, load

# progress monitoring libraries
try :
    from tqdm import tqdm
    #raise
except :
    def tqdm(items, **kwargs) :
        if 'desc' in kwargs:
            print(kwargs['desc'])
        for it in items :
            yield it

# data science libraries
import numpy as np
import pandas as pd

# scikit-learn libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

# yaml
import yaml
config = yaml.safe_load(open('config.yaml'))

# project-specific helper libraries
import icu_config
from icu_featurize_soln import get_raw_data
from icu_tune_soln import make_pipeline_and_grid
import preprocessors
import classifiers



######################################################################
# globals
######################################################################

NJOBS = 4           # number of jobs to run in parallel

NRECORDS = 12000    # number of patient records
FEATURES_TRAIN_FILENAME, LABELS_TRAIN_FILENAME, \
    FEATURES_TEST_FILENAME, LABELS_TEST_FILENAME = \
        icu_config.get_filenames(nrecords=NRECORDS, test_data=True, challenge=True)



######################################################################
# functions
######################################################################

def write_predictions(ids, y_score, uniqname) :
    """Write out predictions to csv file.
    
    Please make sure that you do not change the order of the test examples in the held-out set 
    since we will this file to evaluate your classifier.
    """
    
    df_labels = pd.DataFrame(y_score)
    df_labels.columns = ['30-day_mortality']
    df_labels.insert(0, 'RecordID', ids)
    df_labels.to_csv(f'{uniqname}_icu.csv', index=False)



######################################################################
# classes
######################################################################

class Vectorizer(BaseEstimator, TransformerMixin):
    """Convert a patient record to matrix (numpy array) of feature values."""
    
    def __init__(self) :
        pass
    
    
    def fit(self, X, y=None) :
        """Does nothing: this transformer is stateless.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        
        ### ========== TODO : START ========== ###
        
        ### ========== TODO : END ========== ###
        
        return self
    
    
    def _process_record(self, df) :
        """Transform raw records to a feature matrix.
        
        Parameters
        ----------
        df : pandas DataFrame
            Columns (Time, Variable, Value).
        
        Returns
        ----------
        features : dictionary
            Dictionary of format {feature_name : feature_value}.
        """
        
        features = {}
        
        ### ========== TODO : START ========== ###
        
        ### ========== TODO : END ========== ###
        
        return features
    
    
    def transform(self, X) :
        """Transform raw records to a feature matrix.
        
        Parameters
        ----------
        X : pandas DataFrame
            Raw data to be featuerized.  See get_raw_data.
        
        Returns
        -------
        X : numpy array of shape (n_samples, n_features)
            Feature matrix.
        """
        
        df = X
        ids = df['RecordID'].unique()
        
        features = Parallel(n_jobs=NJOBS)(delayed(self._process_record)(df[df['RecordID'] == i]) for i in tqdm(ids, desc='Generating feature vectors'))
        df_features = pd.DataFrame(features).sort_index(axis=1)     # sort by feature name
        self.feature_names_ = df_features.columns.tolist()
        
        return df_features.values
    
    
    def get_feature_names(self) :
        """Array mapping from feature integer indices to feature name.
        
        Returns
        -------
        feature_names : list
            Feature names.
        """
        
        return self.feature_names_



######################################################################
# main
######################################################################

def featurize() :
    #========================================
    # read raw data
    
    print('Reading data...')
    
    df_features, df_labels = get_raw_data(icu_config.RAW_DATA_DIR, n=NRECORDS)

    #========================================
    # extract features and labels
    
    print('Extracting features and labels...')
    
    # extract features for all records
    avg_vect = Vectorizer()
    X = avg_vect.fit_transform(df_features)
    feature_names = avg_vect.get_feature_names()
    
    # get labels
    y = df_labels['30-day_mortality'].values
    
    # get record ids
    ids = df_labels['RecordID'].values

    #========================================
    # setup for contest
    
    print('Splitting data into training and test set...')
    
    X_train = X[:10000,:]
    y_train = y[:10000]
    ids_train = ids[:10000]
    
    X_test = X[10000:]
    ids_test = ids[10000:]
    
    #========================================
    # write to file (so we only have to do this preprocessing once)
    # files include headers and record ids
    
    print('Writing to file...')
        
    df_features = pd.DataFrame(X_train)
    df_features.columns = avg_vect.get_feature_names()
    df_features.insert(0, 'RecordID', ids_train)
    df_features.to_csv(FEATURES_TRAIN_FILENAME, index=False)
    print(f'\{FEATURES_TRAIN_FILENAME}')
    
    df_features = pd.DataFrame(X_test)
    df_features.columns = avg_vect.get_feature_names()
    df_features.insert(0, 'RecordID', ids_test)
    df_features.to_csv(FEATURES_TEST_FILENAME, index=False)
    print(f'\{FEATURES_TEST_FILENAME}')
    
    df_labels = pd.DataFrame(y_train)
    df_labels.columns = ['30-day_mortality']
    df_labels.insert(0, 'RecordID', ids_train)
    df_labels.to_csv(LABELS_TRAIN_FILENAME, index=False)
    print(f'\{LABELS_TRAIN_FILENAME}')
    
    print()


def tune() :
    #========================================
    # read training set
    
    print('Reading training set...')
    
    df_features_train = pd.read_csv(FEATURES_TRAIN_FILENAME)
    X_train = df_features_train.drop('RecordID', axis=1).values
    
    df_labels_train = pd.read_csv(LABELS_TRAIN_FILENAME)
    y_train = df_labels_train['30-day_mortality'].values
    
    X = np.copy(X_train)
    y = np.copy(y_train)
    
    print()
    
    #========================================
    # setup experiments
    
    print('Making and tuning pipeline...')
    
    ### ========== TODO : START ========== ###

    ### ========== TODO : END ========== ###
    
    # dump to file
    filename = os.path.join(icu_config.PICKLE_DIR, 'challenge_soln.joblib')
    #dump(search.best_estimator_, filename)


def predict() :
    #========================================
    # read held-out test set
    
    print('Reading test set...')
    
    df_features_test = pd.read_csv(FEATURES_TEST_FILENAME)
    ids = df_features_test['RecordID']
    X_test = df_features_test.drop('RecordID', axis=1).values
    
    #========================================
    # predict on held-out test set
    
    print('Predicting test set...')
    
    filename = os.path.join(icu_config.PICKLE_DIR, 'challenge_soln.joblib')
    pipe = load(filename)
    
    # compute scores
    y_score = pipe.decision_function(X_test) 
    
    #========================================
    # write predictions
    
    print('Writing test set predictions...')
    
    ### ========== TODO : START ========== ###
    # update with your username(s)
    
    write_predictions(ids, y_score, "../data/challenge/yjw")
    ### ========== TODO : END ========== ###


if __name__ == '__main__' :
    ### ========== TODO : START ========== ###
    # do one step at a time (comment / uncomment)

    featurize()
    tune()
    predict()
    ### ========== TODO : END ========== ###
