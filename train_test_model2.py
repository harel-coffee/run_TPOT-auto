from __future__ import print_function
#import os
import argparse
#import numpy as np
#import pandas as pd
from tpot_utils import trim_pipeline
from tpot_utils import eval_train
from tpot_utils import eval_test

#import sklearn.metrics
#from sklearn.externals import joblib
    
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
 
    #python2/3 change
    #execfile(trimmed_pipeline_outfile)
    exec(open(trimmed_pipeline_outfile).read())

    serialized_trained_model_outfile = args.output_basename + "_fitted_model.p"
    pr_curve_outfile =  args.output_basename + "_test_PRC.csv"
    results_df_outfile =  args.output_basename + "_test_resultsDF.csv"

    exported_pipeline = eval_train(exported_pipeline, 
                        args.training_infile, 
                        serialized_trained_model_outfile, 
                        args.index_cols)
   
    if args.test_infile:
        eval_test(exported_pipeline, args.test_infile, pr_curve_outfile, results_df_outfile, args.id_cols)


