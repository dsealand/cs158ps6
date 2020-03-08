"""
Author      : Yi-Chieh Wu
Class       : HMC CS 121
Date        : 2020 Feb 20
Description : ML Classifiers
"""

# python libraries
from abc import ABC

# numpy libraries
import numpy as np

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC, SVC


######################################################################
# classes
######################################################################

class Classifier(ABC):
    """Base class for classifier with hyper-parameter optimization.
    See sklearn.model_selection._search.
    
    Attributes
    -------
    estimator_ : estimator object
        This is assumed to implement the scikit-learn estimator interface.
    
    param_grid_ : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    
    Parameters
    -------
    n : int
        Number of samples.
    
    d : int
        Number of features.
    """
    
    def __init__(self, n, d):
        self.estimator_ = None
        self.param_grid_ = None


class Dummy(Classifier):
    """A Dummy classifier."""
    
    def __init__(self, n, d):
        self.estimator_ = DummyClassifier(strategy='stratified')
        self.param_grid_ = {}


class LinearSVM(Classifier):
    """A SVM classifier."""
    
    def __init__(self, n, d):
        self.estimator_ = SVC(kernel='linear', class_weight='balanced')
        self.param_grid_ = {'C': np.logspace(-3, 3, 7)}


class RbfSVM(Classifier):
    """A SVM classifier."""
    
    def __init__(self, n, d):
        self.estimator_ = SVC(kernel='rbf', class_weight='balanced',
                              tol=1e-3, max_iter=1e6)
        self.param_grid_ = {'gamma': np.logspace(-3, 3, 7), 'C': np.logspace(-3, 3, 7)}


######################################################################
# globals
######################################################################

CLASSIFIERS = [c.__name__ for c in Classifier.__subclasses__()]
