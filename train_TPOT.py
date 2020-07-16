#! /usr/bin/env python
from __future__ import print_function
import os
import gc
import argparse
import numpy as np
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, GroupKFold



# Config dicts from https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier.py

Classifiers = {

    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },
}

Transformers = {
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False]
    }
}

Selectors = {
    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}

parser = argparse.ArgumentParser("Run TPOT to find a good machine learning pipeline on training data")
parser.add_argument("--training_data", required=True, help="Features with training labels. Columns: ID1,ID2,feat1,feat2...featN,label")
parser.add_argument("--outfile", required=True, help="File name to write the output pipeline to")
parser.add_argument("--classifier_subset", default=None, nargs="+", choices=Classifiers.keys(), help="Use a subset of sklearn's classifiers in search")
parser.add_argument("--transformer_subset", default=None, nargs="+", choices=Transformers.keys(), help="Use a subset of sklearn's preprocessors in search")
parser.add_argument("--selector_subset", default=None, nargs="+", choices=Selectors.keys(), help="Use a subset of sklearn's preprocessors in search")
parser.add_argument("--template", default = 'Selector-Transformer-Classifier', help = "Organization of training pipeline")

parser.add_argument("--score", default="average_precision", help="Which scoring function to use, default=average_precision")
parser.add_argument("--generations", type=int, default=100, help="How many generations to run, default=100")
parser.add_argument("--population_size", type=int, default=100, help="Size of the recombining population, default=100")
parser.add_argument("--n_jobs", type=int, default=1, help="How many jobs to run in parallel. Warning, n_jobs>1 is crashy")
parser.add_argument("--id_cols", type=int, nargs="+", default=[0,1], help="Which column(s) contain(s) row identifiers, default=[0,1]")
parser.add_argument("--labels", type=int, nargs="+", default=[0,1], help="Which labels to retain, default=[0,1]")
parser.add_argument("--delimiter", default=",", help="Delimiter of training data, default=','")
parser.add_argument("--temp_dir", default="tpot_tmp", help="Temporary directory to stash intermediate results")
parser.add_argument("--warm_start", action='store_true', help="Flag: Whether to re-start TPOT from the results in the temp_dir")
parser.add_argument("--cv", default=5, type = int, help="cv fold")
parser.add_argument("--groupcol", default = None, help="Optional column containing group identifiers for row, to be used for GroupKFold crossvalidation")
parser.add_argument("--labelcol", default = 'label', help="Name of column containing label")



args = parser.parse_args()


tpot_config = {}
if args.selector_subset != None:
    selectors = {i:Selectors[i] for i in args.selector_subset}
    tpot_config.update(selectors)
else:
    tpot_config.update(Selectors) # use all

# Watch out for OHE bug, https://github.com/EpistasisLab/tpot/pull/552:
if args.transformer_subset != None:
    transformers = {i:Transformers[i] for i in args.transformer_subset}
    tpot_config.update(transformers)
else:
    tpot_config.update(Transformers) # use all
 

if args.classifier_subset != None:
    classifiers = {i:Classifiers[i] for i in args.classifier_subset}
    tpot_config.update(classifiers)
else:
    tpot_config.update(Classifiers) # use all
   



if not os.path.exists(args.temp_dir):
    os.makedirs(args.temp_dir) 
    
print("Loading data")
df = pd.read_csv(args.training_data, sep=args.delimiter, index_col=args.id_cols)
#label_name = df.columns[-1]
label_name = args.labelcol
print("Using '{}' as label column".format(label_name))

print("Dropping unlabeled rows")
df = df[df[label_name].isin(args.labels)]

labels = df.pop(label_name)
       
print("Running TPOT")       
print("Requires > 0.10.0")

if args.groupcol:
    groups = df.pop(args.groupcol)
    data = df.values

    print(data)
    print(labels)
    print(groups) 
    print(data.shape)

    tpot = TPOTClassifier(template = args.template, 
             verbosity=2, scoring=args.score, config_dict=tpot_config,
                        generations=args.generations, population_size=args.population_size,
                        memory=args.temp_dir, n_jobs=args.n_jobs, warm_start=args.warm_start, subsample=1.0, cv = GroupKFold(n_splits=args.cv))
    tpot.fit(data, labels, groups = groups)

else:
    data = df.values
    tpot = TPOTClassifier(template = args.template, 
             verbosity=2, scoring=args.score, config_dict=tpot_config,
                        generations=args.generations, population_size=args.population_size,
                        memory=args.temp_dir, n_jobs=args.n_jobs, warm_start=args.warm_start, subsample=0.5, cv = args.cv)
    tpot.fit(data, labels)




tpot.export(args.outfile)
