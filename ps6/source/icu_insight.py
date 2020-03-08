"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2020 Feb 19
Description : Survival of ICU Patients

This code is adapted from course material by Jenna Wiens (UMichigan).
Docstrings based on scikit-learn format.
"""

# python libraries
import os
from joblib import load

# data science libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import resample
from sklearn.pipeline import Pipeline

# project-specific helper libraries
import icu_config
from icu_practice import score, METRICS
import classifiers



######################################################################
# globals
######################################################################

NRECORDS = 2500     # number of patient records
FEATURES_TRAIN_FILENAME, LABELS_TRAIN_FILENAME, \
    FEATURES_TEST_FILENAME, LABELS_TEST_FILENAME = \
        icu_config.get_filenames(nrecords=NRECORDS, test_data=True)



######################################################################
# functions
######################################################################

def get_test_scores(clf, X, y, n_bootstraps=1, metrics=['accuracy']) :
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
    clf : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        The estimator must already be fitted to data.
    
    X : numpy array of shape (n_samples,n_features)
        Feature vectors of test set.
    
    y : numpy array of shape (n_samples,)
        Ground truth labels of test set.
    
    n_bootstraps : int
        Number of bootstrapping iterations.
    
    metrics : list
        Performance metrics.
    
    Returns
    --------------------
    scores : dict
        Dictionary of (metric, list of score) pairs.
        For instance, if n_bootstraps = 3 and metrics = ['accuracy', 'auroc'],
        then scores will be represented as a dict of
        {
            'accuracy'      : 0.81
            'accuracy_boot' : [0.74, 0.70, 0.90]
            'auroc'         : 0.81
            'auroc_boot'    : [0.60, 0.75, 0.85]
        }
    """
    
    # make predictions
    try :
        y_pred = clf.decision_function(X) 
    except :  # for dummy classifiers
        y_pred = clf.predict(X)
    
    # initialize dictionary
    scores = {}
    
    ### ========== TODO : START ========== ###
    # part a : find score on on full data set
    #          find bootstrap scores on resampled data set
    # professor's solution: 7 lines
    #
    # hint: use sklearn.utils.resample to sample
    #       set random_state to the bootstrap iteration
    #           to generate same sampling across metrics
    

    
    ### ========== TODO : END ========== ###
    
    return scores


def plot_results(clf_strs, score_names, scores) :
    """
    Plot results as grouped bar plot,
    with metric along x-axis and model as groups.
    
    You do NOT have to understand the implementation of this function.
    
    Parameters
    ----------
    clf_strs : list
        List of strings, one per classifier.
    
    score_names : list
        List of scorer names.
    
    scores : dict
        Dictionary of (clf_str, score) pairs.
        For instance, if clf_strs == ['Dummy'] and score_names = ['score'],
        then scores will be represented as a dict of
        { 'Dummy' :
            {
                'score'       : 0.70
                'lower_score' : 0.00
                'upper_score' : 0.05
            }
        }
    """
    
    # text annotation
    def autolabel(rects) :
        """Attach a text label above each bar displaying its height"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2., height),
                        xytext=(0, 3), textcoords='offset points', # 3 points vertical offset
                        ha='center', va='bottom')
    
    scorers = sorted(score_names)
    n_scorers = len(scorers)
    ind = np.arange(n_scorers)      # x locations for groups
    width = 1 / (len(clf_strs) + 1) # width of the bars
    
    # bar plot with error bars
    fig = plt.figure(figsize=[12.8, 9.6])
    ax = plt.gca()
    for j, clf_str in enumerate(clf_strs) :
        results = scores[clf_str]
        heights = np.empty((n_scorers),)
        errs = np.empty((2, n_scorers))
        
        for k, scorer in enumerate(scorers) :
            height = results[f'{scorer}']
            lower = results[f'lower_{scorer}']
            upper = results[f'upper_{scorer}']
            
            heights[k] = height
            errs[:,k] = (height - lower, upper - height)
        
        rects = ax.bar(ind + width * j, heights, width, yerr=errs, label=clf_str)
        autolabel(rects)
    
    # x-axis
    ax.set_xticks(ind + width * (len(clf_strs) - 1) / 2.)
    ax.set_xticklabels(scorers)
    
    # title
    ax.set_title('Test Performance')
        
    # y-axis
    ax.set_ylabel('score')
    ax.set_ylim(0, 1)
    
    # legend
    ax.legend(title='model',
              bbox_to_anchor=(1.04,.5), loc='center left')
    
    fig.tight_layout()
    plt.show()



######################################################################
# main
######################################################################

def main():
    np.random.seed(42)
    
    #========================================
    # read data
    
    print('Reading data...')
    
    df_features_test = pd.read_csv(FEATURES_TEST_FILENAME)
    X_test = df_features_test.drop('RecordID', axis=1).values
    
    df_labels_test = pd.read_csv(LABELS_TEST_FILENAME)
    y_test = df_labels_test['In-hospital_death'].values
    
    print()
    
    #========================================
    # evaluate on test data
    
    print('Evaluating on test data...')
    
    clf_strs = classifiers.CLASSIFIERS
    n_bootstraps = 100
    scores = {}
    
    for clf_str in clf_strs :
        # load pipelines from file
        # use the pipeline like any regular classifier
        # pipelines have already been refit on full training set using best found parameters
        # no need to retrain here
        filename = os.path.join(icu_config.PICKLE_DIR, f'{clf_str}_soln.joblib')
        pipe = load(filename)
        
        # compute scores
        test_scores = get_test_scores(pipe, X_test, y_test, n_bootstraps, METRICS)
        
        ### ========== TODO : START ========== ###
        # part b : summarize to dictionary
        # professor's solution: 6 lines
        #
        # We will use this dictionary to visualize performance.
        # Example: scores_clf_str is a dictionary that looks like
        #          {
        #              'accuracy'       : 0.70
        #              'lower_accuracy' : 0.00
        #              'upper_accuracy' : 0.05
        #          }
        #          
        #          You will have three dict items per score metric.
        # 
        # The first element (e.g. 'accuracy') is the score on the full test set.
        # The lower and upper bound are based on the 95% confidence interval.
        # That is, lower value (e.g. 'lower_accuracy') corresponds to the 2.5-percentile,
        # and upper value (e.g. 'upper_accuracy') corresponds to the 97.5-percentile.
        #
        # hint: use np.percentile to compute percentiles
        
        scores_clf = {}
        
        
        ### ========== TODO : END ========== ###
        
        # save scores for current classifier
        scores[clf_str] = scores_clf
    
    # plot test performance
    plot_results(clf_strs, METRICS, scores)
    
    print()
    
    #========================================
    # feature importances
    
    print('Evaluating feature importance...')
    
    clf_str = 'LinearSVM'
    filename = os.path.join(icu_config.PICKLE_DIR, f'{clf_str}_soln.joblib')
    pipe = load(filename)
    
    feature_names = df_features_test.drop('RecordID', axis=1).columns.tolist()
    coef = pipe['clf'].coef_[0]
    
    ### ========== TODO : START ========== ###
    # part e : identify important features
    #          print to screen
    # professor's solution: 8 lines
    
    
    
    ### ========== TODO : START ========== ###
    
    print()


if __name__ == '__main__':
    main()
