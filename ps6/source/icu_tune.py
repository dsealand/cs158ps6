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
from joblib import dump

# data science libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# scikit-learn libraries
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

# project-specific libraries
import icu_config
from icu_practice import score, METRICS
import preprocessors
import classifiers

# setup
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)



######################################################################
# globals
######################################################################

NRECORDS = 2500     # number of patient records
FEATURES_TRAIN_FILENAME, LABELS_TRAIN_FILENAME = \
    icu_config.get_filenames(nrecords=NRECORDS)



######################################################################
# functions
######################################################################

def make_pipeline_and_grid(steps) :
    """Make composite pipeline and parameter grid from list of estimators.
    
    You do NOT have to understand the implementation of this function.
    It stitches together the input steps and generates the nested parameter grid.
    
    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are chained,
        in the order in which they are chained, with the last object an estimator.
        
        Each transform should have either a transformer_ or an estimator_ attribute.
        These attributes store a sklearn object that can transform or predict data.
        
        Each transform should have a param_grid.
        This attribute stores the hyperparameter grid for the transformer or estimator.
    """
    
    pipe_steps = []
    pipe_param_grid = {}
    
    # chain transformers
    for (name, transform) in steps[:-1] :
        transformer = transform.transformer_
        pipe_steps.append((name, transformer))
        for key, val in transform.param_grid_.items() :
            pipe_param_grid[name + "__" + key] = val
    
    # chain estimator
    name, transform = steps[-1]
    estimator = transform.estimator_
    pipe_steps.append((name, estimator))
    for key, val in transform.param_grid_.items() :
        pipe_param_grid[name + "__" + key] = val
    
    # stitch together preprocessors and classifier
    pipe = Pipeline(pipe_steps)
    return pipe, pipe_param_grid


def plot_results(clf_strs, score_names, scores) :
    """
    Plot results as grouped bar plot.
    Make two plots, one for training and one for testing,
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
                'mean_train_score' : 0.75
                'mean_test_score'  : 0.70
                'std_train_score'  : 0.00
                'std_test_score'   : 0.05
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
    ind = np.arange(len(scorers))   # x locations for groups
    width = 1 / (len(clf_strs) + 1) # width of the bars
    
    # bar plot with error bars
    fig, axes = plt.subplots(1, 2,
                             sharex=True, sharey=True,
                             figsize=[12.8, 9.6])
    samples = ('train', 'test')
    for i, sample in enumerate(samples) :
        ax = axes[i]
        
        for j, clf_str in enumerate(clf_strs) :
            results = scores[clf_str]
            means = [results[f'mean_{sample}_{scorer}'] for scorer in scorers]
            stds = [results[f'std_{sample}_{scorer}'] for scorer in scorers]
            rects = ax.bar(ind + width * j, means, width, yerr=stds, label=clf_str)
            autolabel(rects)
        
        # x-axis
        ax.set_xticks(ind + width * (len(clf_strs) - 1) / 2.)
        ax.set_xticklabels(scorers)
        
        # title
        if sample == 'test' :
            sample = 'cross-validation'
        ax.set_title(f'{sample.capitalize()} Performance')
        
    # y-axis
    axes[0].set_ylabel('score')
    axes[0].set_ylim(0, 1)
    
    # legend
    axes[1].legend(title='model',
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
    
    df_features_train = pd.read_csv(FEATURES_TRAIN_FILENAME)
    X_train = df_features_train.drop('RecordID', axis=1).values
    
    df_labels_train = pd.read_csv(LABELS_TRAIN_FILENAME)
    y_train = df_labels_train['In-hospital_death'].values
    
    # make copies so we do not risk changing training set
    X = np.copy(X_train)
    y = np.copy(y_train)
    
    print()
    
    #========================================
    # setup scores and cross-validation folds for experiments
    
    print('Setting up experiment...')
    
    # create scoring dictionary
    scoring = {}
    for metric in METRICS :
        scoring[metric] = make_scorer(score, needs_threshold=True, metric=metric)
    
    # there is a lot of data now, so run a single trial (without shuffling data)
    cv = StratifiedKFold(n_splits=5)
    
    print()
    
    #========================================
    # make and tune pipelines
    
    print('Making and tuning pipeline...')
    
    n, d = X.shape
    
    ### ========== TODO : START ========== ###
    # part a : make and tune pipelines
    # professor's solution: 15-25 lines
    #
    # There are several steps here:
    #
    # (1) Make a pipeline and hyperparameter grid.
    #
    #     The whole point of this block is to get preprocessors
    #     and classifiers from preprocessors.py and classifiers.py.
    #     This way, you can edit those files to add or remove elements.
    #     This high-level script remains unchanged.
    #
    #     All pipelines start with the same preprocessing steps.
    #         preprocessors.Imputer()
    #         preprocessors.Scaler()
    #
    #     Each pipeline ends with a different classifier.
    #     Consider all classifiers in 'clf_strs'.
    #     Use the following fancy Python to get 'clf' from 'clf_str':
    #         clf = getattr(classifiers, clf_str)(n, d)
    #
    #     If you have a list of (name, transform) pairs in a variable
    #     called'steps', call
    #         pipe, param_grid = make_pipeline_and_grid(steps)
    #     to make a Pipeline object with nested parameter grid.
    #
    # (2) Tune the pipeline with GridSearchCV.
    #
    #     Use multiple metrics and StratifiedKFold (see above variables).
    #     Return training scores.
    #     Set refit='auroc'.
    #         The search will refit the pipeline on the whole dataset
    #         using the parameters that optimized test auroc.
    #     Set n_jobs=-1 to use all processors.
    #
    #     With refit, your GridSearchCV object stores a lot more useful things.
    #     You will use cv_results_, best_esimator_, best_params_, and best_index_
    #     in a later step. See GridSearchCV documentation for a full list.
    #
    #  (3) Print the current classifier and optimal hyperparameter setting.
    #
    #  (4) Store results to 'scores'.
    #      We will pass 'scores' to a plotting function to visualize performance.
    #      Key is the current 'clf_str'. Value is a dict object of optimal scores.
    #
    #      Example: If clf_str == 'Dummy' and score_names = ['accuracy'],
    #               then scores[clf_str] is a dict that looks like
    #               {
    #                   'mean_train_accuracy' : 0.75
    #                   'mean_test_accuracy'  : 0.70
    #                   'std_train_accuracy'  : 0.00
    #                   'std_test_accuracy'   : 0.05
    #               }
    #
    #               If score_names has multiple elements,
    #               you will have four dict items per score metric.
    # 
    #      Hint: For a fitted GridSearchCV object 'search', access performance scores
    #            through 'search.cv_results_' together with 'search.best_index_'.
    #
    # More Hints
    # - We provide skeleton code below. Feel free to use it or not.
    # - We recommend that you consider a single classifier first.
    #   Once you get the plot working for a single classifier,
    #   consider all classifiers. (Otherwise, you might have an error,
    #   and you will not know until the plot fails.)
    
    # classifiers and scores
    clf_strs = classifiers.CLASSIFIERS
    scores = {}
    
    for clf_str in clf_strs :
        # step 1
        # make pipeline and parameter grid
        # professor's solution: 3-5 lines
        pass
        
        # step 2
        # tune hyperparameters using CV
        # professor's solution: 3-5 lines
        pass
        
        # step 3
        # print optimal hyperparameter setting
        # professor's solution: 1 line
        pass
        
        # step 4
        # store results
        dct = {}
        for scorer in sorted(scoring) :
            # professor's solution: 3 lines
            pass
            
            scores[clf_str] = dct
        
        # dump to file
        # uncomment the line to dump best estimator to file
        # assumes you named your GridSearchCV object 'search'
        filename = os.path.join(icu_config.PICKLE_DIR, f'{clf_str}.joblib')
        # dump(search.best_estimator_, filename)
    ### ========== TODO : END ========== ###
    
    print()
    
    #========================================
    # plot results
    
    plot_results(clf_strs, list(scoring.keys()), scores)


if __name__ == '__main__':
    main()
