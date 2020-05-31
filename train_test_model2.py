from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
from tpot_utils import trim_pipeline
from tpot_utils import eval_train
from tpot_utils import eval_test
from breakup_exported_pipeline import trim_pipeline

import sklearn.metrics
from sklearn.externals import joblib


def eval_train(exported_pipeline, training_infile, serialized_trained_model="fitted_model.p", index_cols=[0,1]):
    
    assert os.path.exists(training_infile), "{} not found".format(training_infile)
    
    print("Reading training data")
    train = pd.read_csv(training_infile, index_col=index_cols)
    train_label = train.pop("label").values
    train_data = train.values
    
    print("Training model")
    exported_pipeline.fit(train_data, train_label)
    joblib.dump(exported_pipeline, serialized_trained_model_outfile)
    
    train_fit_probs = exported_pipeline.predict_proba(train_data)[:,1]
    train_aps = sklearn.metrics.average_precision_score(train_label,train_fit_probs)
    print("Training set average precision score: {}".format(train_aps))
    
    del train
    del train_data
    return(exported_pipeline)

def eval_test(exported_pipeline, test_infile, pr_curve_outfile="test_PRC.csv", results_df_outfile="test_resultsDF.csv", index_cols=[0,1] ):    


    assert os.path.exists(test_infile), "{} not found".format(test_infile)
    print("Reading test data", index_cols=[0,1])
    test = pd.read_csv(test_infile, index_col=index_cols)
    test_label = test.pop("label")
    test_data = test.values
    
    test_probs = exported_pipeline.predict_proba(test_data)[:,1]
    
    test_aps = sklearn.metrics.average_precision_score(test_label, test_probs)
    print("Test set average precision score: {}".format(test_aps))
    
    test_p, test_r, thresholds = sklearn.metrics.precision_recall_curve(test_label, test_probs)
    
    test_PRC = pd.DataFrame({"precision": test_p, "recall": test_r, "threshold": thresholds}).sort_values("recall")
    test_PRC.to_csv(pr_curve_outfile,index=False)
    
    test_DF = pd.DataFrame({"label":test_label,"P_1":test_probs}, index=test.index).sort_values("P_1", ascending=False)
    test_DF["FDR"] = 1 - (test_DF.label.cumsum() / (np.arange(test_DF.shape[0]) + 1))
    test_DF.to_csv(results_df_outfile)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge complexes given a similarity threshold")
    parser.add_argument("--training_infile", action="store", dest="training_infile", required=True,
                                            help="training and test data are csv files with first two columns as IDs and label column called 'label")
    parser.add_argument("--test_infile", action="store", dest="test_infile", required=False,
                                            help="training and test data are csv files with first two columns as IDs and label column called 'label")
    parser.add_argument("--exported_pipeline", action="store", dest="exported_pipeline", required=True,
                                            help="Exported pipeline (.py) from train_tpot.py")
    parser.add_argument("--output_basename", action="store", dest="output_basename", required=False, default = "TPOT",
                                            help="training and test data are csv files with first two columns as IDs and label column called 'label")
    parser.add_argument("--id_cols", type=int, nargs="+", dest="index_cols", required=False, default=[0,1], help="Which column(s) contain(s) row identifiers, default=[0,1]")


    args = parser.parse_args()
    

    # The python output by TPOT contains steps we won't use
    trimmed_pipeline_outfile = "{}_trim.py".format(args.exported_pipeline.replace(".py", ""))

    trim_pipeline(args.exported_pipeline, trimmed_pipeline_outfile)

    # This is done to load in the set of imported modules in the exported pipeline
    # object `exported_pipeline` is loaded from this execfile
    execfile(trimmed_pipeline_outfile)

    serialized_trained_model_outfile = args.output_basename + "_fitted_model.p"
    pr_curve_outfile =  args.output_basename + "_test_PRC.csv"
    results_df_outfile =  args.output_basename + "_test_resultsDF.csv"

    exported_pipeline = eval_train(exported_pipeline, 
                        args.training_infile, 
                        serialized_trained_model_outfile, 
                        args.index_cols)
   
    if args.test_infile:
        eval_test(exported_pipeline, args.test_infile, pr_curve_outfile, results_df_outfile, args.id_cols)


