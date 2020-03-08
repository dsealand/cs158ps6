"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2020 Feb 13
Description : ML Preprocessors
"""

# python libraries
from abc import ABC

# scikit-learn libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


######################################################################
# classes
######################################################################

class Preprocessor(ABC):
    """Base class for preprocessor with hyper-parameter optimization.
    Attributes
    --------------------
    transformer_  : transformer object
        This is assumed to implement the scikit-learn transformer interface.
    
    param_grid_ : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    """
    def __init__(self):
        self.transformer_ = None
        self.param_grid_ = None


class Imputer(Preprocessor):
    """Impute missing (NaN) entries."""
    
    def __init__(self):
        # impute missing (NaN) entries
        self.transformer_ = SimpleImputer(strategy='mean')
        self.param_grid_ = {}


class Scaler(Preprocessor):
    """Scale each feature to given range."""
    
    def __init__(self):
        self.transformer_ = MinMaxScaler(feature_range=(-1,1))
        self.param_grid_ = {}


######################################################################
# globals
######################################################################

PREPROCESSORS = [c.__name__ for c in Preprocessor.__subclasses__()]
