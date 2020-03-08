"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2020 Feb 18
Description : Survival of ICU Patients

This code is adapted from course material by Jenna Wiens (UMichigan).
Docstrings based on scikit-learn format.
"""

# python libraries
import os



######################################################################
# globals
######################################################################

#========================================
# data files

DATA_DIR = '../data'

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

PROCESSED_DATA_DIR= os.path.join(DATA_DIR, 'processed')
if not os.path.exists(PROCESSED_DATA_DIR) :
    os.mkdir(PROCESSED_DATA_DIR)

CHALLENGE_DATA_DIR = os.path.join(DATA_DIR, 'challenge')
if not os.path.exists(CHALLENGE_DATA_DIR) :
    os.mkdir(CHALLENGE_DATA_DIR)


def get_filenames(nrecords=2500, test_data=False, challenge=False) :
    """
    Get filenames.
    
    Parameters
    ----------
    n_records : bool
        Number of patient records.
    
    test_data : bool
        Set to return test set filenames.
    
    Returns
    ----------
    features_train_filename : string
    
    labels_train_filename : string
    
    features_test_filename : string
    
    labels_test_filename : string
    """
    
    if not challenge :
        path = PROCESSED_DATA_DIR
    else :
        path = CHALLENGE_DATA_DIR
    
    features_train_filename = os.path.join(path, f'features{nrecords}_train.csv')
    labels_train_filename = os.path.join(path, f'labels{nrecords}_train.csv')
    
    if not test_data :
        return features_train_filename, labels_train_filename
    
    features_test_filename = os.path.join(path, f'features{nrecords}_test.csv')
    labels_test_filename = os.path.join(path, f'labels{nrecords}_test.csv')
    
    return features_train_filename, labels_train_filename, \
        features_test_filename, labels_test_filename

#========================================
# pickle files (for trained models)

PICKLE_DIR = '../pickle'
if not os.path.exists(PICKLE_DIR) :
    os.mkdir(PICKLE_DIR)
