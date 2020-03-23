"""
Author      : Arjun Natarajan, Daniel Sealand
Class       : HMC CS 158
Date        : 23 March 2020
Description : Survival of ICU Patients

This code is adapted from course material by Jenna Wiens (UMichigan).
Docstrings based on scikit-learn format.
"""

# python libraries
import sys

# data science libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# scikit-learn libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

# project-specific helper libraries
import icu_config
import tests



######################################################################
# globals
######################################################################

RANDOM_SEED = 42    # seed for RepeatedStratifiedKFold
EPS = 10 * sys.float_info.epsilon

NRECORDS = 100      # number of patient records
FEATURES_TRAIN_FILENAME, LABELS_TRAIN_FILENAME = \
    icu_config.get_filenames(nrecords=NRECORDS)

METRICS = ["accuracy", "auroc", "f1_score", "sensitivity", "specificity", "precision"] # sensitivity = recall



######################################################################
# functions
######################################################################

def score(y_true, y_score, metric='accuracy') :
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
    y_true : numpy array of shape (n_samples,)
        Ground truth (correct) labels.
    
    y_score : numpy array of shape (n_samples,)
        Target scores (continuous-valued) predictions.
    
    metric : {'accuracy', 'auroc', 'f1_score', 'sensitivity', 'specificity', 'precision', 'recall'}
        Performance metric.
    
    Returns
    --------------------
    score : float
        Performance score.
    """
    
    # map continuous-valued predictions to binary labels
    y_pred = np.sign(y_score)
    y_pred[y_pred == 0] = 1 # map points on hyperplane to +1
    
    # part a : compute classifier performance for specified metric
    # professor's solution: 16 lines
    
    # compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1,1]).ravel()

    # compute scores
    if metric == "accuracy": 
        x = (tp + tn)
        y = (tp + fn + fp + tn)
    elif metric == "auroc":
        return roc_auc_score(y_true, y_score)
    elif metric == "f1_score":
        if tp+fp == 0:
            precision = 0
        else:
            precision = tp/(tp+fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp/(tp+fn)
        x = 2 * (precision * recall)
        y = precision + recall
    elif metric == "sensitivity" or metric == 'recall':
        x = tp
        y = (tp + fn)
    elif metric == "specificity":
        x = tn
        y = (tn + fp)
    elif metric == "precision":
        x = tp
        y = (tp + fp)
    if y == 0:
        return 0
    else:
        return x/y
    


def plot_cv_results(results, scorers, param_name) :
    """Plot performance for tuning LinearSVC.
    
    You do NOT have to understand the implementation of this function.
    It basically pulls together data from GridSearch.cv_results_ and feeds it into matplotlib.
    """
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    
    # hyperparameter values along x-axis
    X_axis = np.array(results[param_name].data, dtype=float)
    
    # metrics in different colors
    cm = plt.get_cmap('tab10')
    scorers_colors = [(scorer, cm(i)) for i, scorer in enumerate(sorted(scorers))]
    
    # samples in different line styles
    samples = ('train', 'test')
    
    # plot
    fig, axes = plt.subplots(1, 2,  # 1 row, 2 columns
                             sharex=True, sharey=True,
                             figsize=[12.8, 9.6])
    for i, sample in enumerate(samples) :
        ax = axes[i]
        
        for scorer, color in scorers_colors :
            sample_score_mean = results[f'mean_{sample}_{scorer}']
            sample_score_std = results[f'std_{sample}_{scorer}']
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, color=color,
                    alpha=1 if sample == 'test' else 0.7) # suppress label
                    #label="%s (%s)" % (scorer, sample))
        
        # title
        if sample == 'test' :
            sample = 'cross-validation'
        ax.set_title(f'{sample.capitalize()} Performance')
    
    # axes
    axes[0].set_ylabel('score')
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('C'); axes[1].set_xlabel('C')
    axes[0].set_xscale('log')
    
    # legend
    lines = []
    labels = []
    for scorer, color in scorers_colors :
        lines.append(Line2D([0], [0], linestyle='-', color=color))
        labels.append(scorer)
    axes[1].legend(lines, labels, title='metric',
                   bbox_to_anchor=(1.04,.5), loc='center left')
    
    # title and show
    fig.tight_layout()
    plt.show()



######################################################################
# main
######################################################################

def main() :
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
    # try out preprocessors and classifiers
    # as always, make sure to set the correct hyperparameters for each estimator
    
    print('Preprocessing and classifying...')
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_train)
    assert np.any(~np.isnan(X_imputed))         # sanity check
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_imputed_scaled = scaler.fit_transform(X_imputed)
    assert X_imputed_scaled.ravel().min() > -1-EPS and \
        X_imputed_scaled.ravel().max() < 1+EPS  # sanity check
    
    clf = SVC(kernel='linear', class_weight='balanced', max_iter=1e6)
    clf.fit(X_imputed_scaled, y)
    
    #========================================
    # score
    
    print('Computing multiple metrics...')
    
    tests.test_score()
    
    print()
    
    y_score = clf.decision_function(X_imputed_scaled)
    for metric in METRICS :
        print(f'{metric}: ', score(y, y_score, metric))
    
    print()
    
    #========================================
    # search over Linear SVM hyperparameters using various scoring metrics
    
    print('Tuning Linear SVM...')
    
    # create scoring dictionary to maps scorer name (metric) to scoring function
    # make_scorer(score_func, needs_threshold=True, **kwargs)
    #   score_func has signature score_func(y, y_pred, **kwargs) and returns a scalar score
    #   needs_threshold is True says score_func requires continuous decisions
    #   **kwargs allows us to pass additional arguments to score_func (e.g. metric)
    scoring = {}
    for metric in METRICS :
        scoring[metric] = make_scorer(score, needs_threshold=True, metric=metric)
    
    # run exhaustive grid search
    # GridSearchCV(..., scoring, refit, ...)
    #   you should be familiar with most of the parameters to GridSearchCV
    #   scoring with a list or dict allows us to specify multiple metrics
    #   refit allows us to refit estimator on whole dataset
    #     refit=False says do NOT refit estimator
    #     if want to refit when using multiple metrics, use string corresponding to key in scoring dict
    param_grid = {'C': np.logspace(-3, 3, 7)}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_SEED)
    search = GridSearchCV(clf, param_grid,
                          scoring=scoring, cv=cv, refit=False,
                          return_train_score=True)
    search.fit(X_imputed_scaled, y)
    results = search.cv_results_
    
    # plot results
    # param_name specifies the hyperparameter of interest (the key to look up in results)
    plot_cv_results(results, search.scorer_, param_name='param_C')
    
    print()
        
    #========================================
    # put everything together in a pipeline
    
    print('Making and tuning pipeline...')
    
    # make pipeline
    # Pipeline(...) takes in list of (name, transform) pairs
    #   where name is name of the step and transform is corresponding transformer or estimator
    steps = [
        ('imputer', imputer),
        ('scaler', scaler),
        ('clf', clf)
    ]
    pipe = Pipeline(steps)
    
    # make parameter grid
    # nested parameters use the syntax <estimator>__<parameter>
    # estimator is the name of the step
    param_grid = {'clf__C' : np.logspace(-3, 3, 7)}
    
    # tune pipeline
    search = GridSearchCV(pipe, param_grid,
                          scoring=scoring, cv=cv, refit=False,
                          return_train_score=True)
    search.fit(X, y)
    results = search.cv_results_
    
    # plot results
    plot_cv_results(results, search.scorer_, param_name='param_clf__C')
    
    # part c : find optimal hyperparameter setting for each metric
    #          report corresponding mean train score and test score
    #          everything you need is in results variable
    # professor's solution: 12 lines
    
    for scorer in sorted(scoring) :
        print("{}............................".format(scorer))
        train = results["mean_train_{}".format(scorer)]
        test = results["mean_test_{}".format(scorer)]
        maxTest = np.max(test)
        maxTestIndex = np.argmax(test)
        maxTrain = train[maxTestIndex]
        optC = param_grid['clf__C'][maxTestIndex]
        print("Train Score {}".format(maxTrain))
        print("Test Score {}".format(maxTest))
        print("C {}".format(optC))
    
    
    print()


if __name__ == '__main__' :
    main()
