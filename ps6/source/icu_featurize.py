"""
Author      : Arjun Natarajan, Daniel Sealand
Class       : HMC CS 158
Date        : 11 March 2020
Description : Survival of ICU Patients

This code is adapted from course material by Jenna Wiens (UMichigan).
"""

# python libraries
import os
from joblib import Parallel, delayed

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
import pandas as pd
import numpy as np

# scikit-learn libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

# yaml
import yaml
config = yaml.safe_load(open('config.yaml'))

# project-specific helper libraries
import icu_config
import tests



######################################################################
# globals
######################################################################

NJOBS = 4           # number of jobs to run in parallel
RANDOM_SEED = 3     # seed for train_test_split

# after part a : change to 2500 records
NRECORDS = 2500        # number of patient records

FEATURES_TRAIN_FILENAME, LABELS_TRAIN_FILENAME, \
    FEATURES_TEST_FILENAME, LABELS_TEST_FILENAME = \
        icu_config.get_filenames(nrecords=NRECORDS, test_data=True)



######################################################################
# functions
######################################################################

def get_raw_data(path, n=None) :
    """Read raw data from <path>/labels.csv and <data>/files/*.csv,
    keeping only the first n examples.
    
    Parameters
    --------------------
    path : string
        Data directory.
    
    n : int
        Number of examples to retain.
    
    Returns
    --------------------
    df_features : pandas DataFrame of shape (?,3)
        Features.
        The number of rows depends on the number of values per patient.
        columns:
            RecordID (int)
            Time (object)
            Variable (object)
            Value (float)
    
    df_labels : pandas DataFrame of shape (n_samples,3)
        Labels.
        columns:
            RecordID (int)
            In-hospital_death (int, -1 for survived, +1 for died)
            30-day_mortality (int, -1 for survived, +1 for died)
    """
    
    # read labels and keep first n
    df_labels = pd.read_csv(os.path.join(path, 'labels.csv'))
    if n is not None :
        df_labels = df_labels[:n]
    ids = df_labels['RecordID']
    
    # read features
    data = []
    for i in tqdm(ids, desc='Loading files from disk'):
        df = pd.read_csv(os.path.join(path, f'files/{i}.csv'))
        df.insert(0, 'RecordID', i)     # add RecordID
        data.append(df)
    df_features = pd.concat(data, axis=0, ignore_index=True)
    
    return df_features, df_labels



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
            Dictionary of format {feature_name : feature_value},
            e.g. {'Age': 32, 'Gender': 0, 'mean_HR': 84, ...}.
        """
        
        features = {}
        
        
        static_vars = config['static']
        timeseries_vars = config['timeseries']

        # replace unknown values (-1) with np.nan
        # hint: use df.replace(dict)
        # professor's solution: 1 line

        df = df.replace(-1, np.nan)

        # process time-invariant variables
        # keep time-invariant series
        # nothing to implement here
        static = df[df['Variable'].isin(static_vars)]
        for var in static_vars :
            values = static[static['Variable'] == var].Value
            features[var] = values.iloc[0]

        # process time series variables
        # convert time series values to single mean
        # use above code as a template
        # professor's solution: 4 lines
        timeseries = df[df['Variable'].isin(timeseries_vars)]
        for var in timeseries_vars :
            values = timeseries[timeseries['Variable'] == var]
            if values.empty:
                mean = np.nan
            else:
                mean = values['Value'].mean()
            features['mean_{}'.format(var)] = mean

        
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

#========================================
# part a : featurize data record of one patient

# This is a sandbox for you to play around with the pandas library.
# Uncomment this block to test, and run in interactive mode.
#     python -i icu_featurize.py
#
# Test your featurization with a dataframe with one record.
# Once correct, copy code into Vectorizer._process_record(...)
# and recomment this block.

'''
df_features, df_labels = get_raw_data(icu_config.RAW_DATA_DIR, n=1)

rid = df_labels['RecordID'][0] # 132539
df = df_features[df_features['RecordID'] == rid]

features = {}

static_vars = config['static']
timeseries_vars = config['timeseries']

# replace unknown values (-1) with np.nan
# hint: use df.replace(dict)
# professor's solution: 1 line

df = df.replace(-1, np.nan)

# process time-invariant variables
# keep time-invariant series
# nothing to implement here
static = df[df['Variable'].isin(static_vars)]
for var in static_vars :
    values = static[static['Variable'] == var].Value
    features[var] = values.iloc[0]

# process time series variables
# convert time series values to single mean
# use above code as a template
# professor's solution: 4 lines
timeseries = df[df['Variable'].isin(timeseries_vars)]
for var in timeseries_vars :
    values = timeseries[timeseries['Variable'] == var]
    if values.empty:
        mean = np.nan
    else:
        mean = values['Value'].mean()
    features['mean_{}'.format(var)] = mean

# test answer
tests.test_process_record(features)
'''


def main() :
    
    #========================================
    # read raw data
    
    print('Reading data...')
    
    df_features, df_labels = get_raw_data(icu_config.RAW_DATA_DIR, n=NRECORDS)
    
    print()
    
    #========================================
    # extract features and labels
    
    print('Extracting features and labels...')
    
    # test feature extraction for one record
    tests.test_Vectorizer(df_features, df_labels)
    
    # extract features for all records
    avg_vect = Vectorizer()
    X = avg_vect.fit_transform(df_features)
    feature_names = avg_vect.get_feature_names()
    
    # get labels
    y = df_labels['In-hospital_death'].values
    
    # get record ids
    ids = df_labels['RecordID'].values
    
    print()
    
    #========================================
    # setup for ML
    
    print('Setting up training and test set...')
    
    # create hold-out test set
    X_train, X_test, y_train, y_test, ids_train, ids_test = \
        train_test_split(X, y, ids, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    
    # print feature matrix information
    n, d = X_train.shape
    print('number of samples (training):', n)
    print('number of features (training):', d)
    
    n, d = X_test.shape
    print('number of samples (test):', n)
    print('number of features (test):', d)
    
    # compute stats on training data set
    print('number of samples missing at least one value (training):', np.count_nonzero(np.isnan(X_train).any(axis=1)))
    print('number of features missing at least one value (training):', np.count_nonzero(np.isnan(X_train).any(axis=0)))
    
    # compute average feature vector
    feature_avg = np.nanmean(X_train, axis=0)
    print('average feature vector (training):')
    df = pd.DataFrame({'feature': pd.Series(feature_names),
                       'average': pd.Series(feature_avg)})
    print(df)
    
    print()
    
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
    df_labels.columns = ['In-hospital_death']
    df_labels.insert(0, 'RecordID', ids_train)
    df_labels.to_csv(LABELS_TRAIN_FILENAME, index=False)
    print(f'\{LABELS_TRAIN_FILENAME}')
    
    df_labels = pd.DataFrame(y_test)
    df_labels.columns = ['In-hospital_death']
    df_labels.insert(0, 'RecordID', ids_test)
    df_labels.to_csv(LABELS_TEST_FILENAME, index=False)
    print(f'\{LABELS_TEST_FILENAME}')


if __name__ == '__main__' :
    main()
