#! /usr/bin/env python
from __future__ import print_function
import os
import gc
import argparse
import numpy as np
import pandas as pd
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from config_dicts import Classifiers_dict, Transformers_dict, Selectors_dict, Regressors_dict



Regressors = Regressors_dict()
Classifiers = Classifiers_dict()
Selectors = Selectors_dict()
Transformers = Transformers_dict()

print(Classifiers)

parser = argparse.ArgumentParser("Run TPOT to find a good machine learning pipeline on training data")
parser.add_argument("--training_data", required=True, help="Features with training labels. Columns: ID1,ID2,feat1,feat2...featN,label")
parser.add_argument("--outfile", required=True, help="File name to write the output pipeline to")
parser.add_argument("--classifier_subset", default=None, nargs="+", choices=Classifiers.keys(), help="Use a subset of sklearn's classifiers in search")
parser.add_argument("--transformer_subset", default=None, nargs="+", choices=Transformers.keys(), help="Use a subset of sklearn's preprocessors in search")
parser.add_argument("--selector_subset", default=None, nargs="+", choices=Selectors.keys(), help="Use a subset of sklearn's preprocessors in search")
parser.add_argument("--regressor_subset", default=None, nargs="+", help="Use a subset of sklearn's preprocessors in search")
parser.add_argument("--style", choices = ["classify", "regress"], default = "classify", help = "Whether to classify or regress")

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
parser.add_argument("--max_features_to_select", default = None, type = int, help = "Optional, Limit maximum number of features selected for training to this number")


args = parser.parse_args()


tpot_config = {}



if args.max_features_to_select != None:
    Selectors = Selectors_dict(args.max_features_to_select + 1) 

if args.selector_subset != None:
    selectors = {i:Selectors[i] for i in args.selector_subset}
    tpot_config.update(selectors)
else:
    tpot_config.update(Selectors) # use all

print(tpot_config)

# Watch out for OHE bug, https://github.com/EpistasisLab/tpot/pull/552:
if args.transformer_subset != None:
    transformers = {i:Transformers[i] for i in args.transformer_subset}
    tpot_config.update(transformers)
else:
    tpot_config.update(Transformers) # use all
 

if args.style == "classify":
    if args.classifier_subset != None:
        classifiers = {i:Classifiers[i] for i in args.classifier_subset}
        tpot_config.update(classifiers)
    else:
        tpot_config.update(Classifiers) # use all
   

if args.style == "regress":
    if args.regressor_subset != None:
        regressors = {i:Regressors[i] for i in args.regressor_subset}
        tpot_config.update(regressors)
    else:
        tpot_config.update(Regressors) # use all
 

if not os.path.exists(args.temp_dir) and args.temp_dir != "auto":
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



    if args.style == "classify":
        tpot = TPOTClassifier(template = args.template, 
                 verbosity=3, scoring=args.score, config_dict=tpot_config,
                            generations=args.generations, population_size=args.population_size,
                            memory=args.temp_dir, n_jobs=args.n_jobs, warm_start=args.warm_start, subsample=1.0, cv = GroupKFold(n_splits=args.cv))
    if args.style == "regress":
        tpot = TPOTRegressor(template = args.template, 
                 verbosity=3, scoring=args.score, config_dict=tpot_config,
                            generations=args.generations, population_size=args.population_size,
                            memory=args.temp_dir, n_jobs=args.n_jobs, warm_start=args.warm_start, subsample=1.0, cv = GroupKFold(n_splits=args.cv))


    tpot.fit(data, labels, groups = groups)

else:

    data = df.values
    if args.style == "classify":
        tpot = TPOTClassifier(template = args.template, 
                 verbosity=2, scoring=args.score, config_dict=tpot_config,
                            generations=args.generations, population_size=args.population_size,
                            memory=args.temp_dir, n_jobs=args.n_jobs, warm_start=args.warm_start, subsample=1.0, cv = args.cv)
    if args.style == "regress":

        tpot = TPOTRegressor(template = args.template, 
                 verbosity=2, scoring=args.score, config_dict=tpot_config,
                            generations=args.generations, population_size=args.population_size,
                            memory=args.temp_dir, n_jobs=args.n_jobs, warm_start=args.warm_start, subsample=1.0, cv = args.cv)
    tpot.fit(data, labels)




tpot.export(args.outfile)
