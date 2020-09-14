# `Simple wrapper for TPOT`

[TPOT](https://epistasislab.github.io/tpot/ "TPOT homepage") is an "auto ML" package that finds 
a good machine learning pipeline for your data. It is a genetic algorithm that searches
over the sklearn classifier algorithms. It does not train the model or do prediction, it's just
for finding an appropriate model and set of hyperparameters, as well as preprocessors.

Here are three scripts that run TPOT, train the model, and do prediction. This is a brief outline
of what you'll do to run them, but there's more documentation in the scripts themselves.

1.) train_TPOT.py: Run TPOT on training data to find a good pipeline. This can take several days, reduced by limiting the number of classifiers, selectors, transformers, etc to explore. 
It can be sped up by using more jobs, but we've noticed crashes from too many jobs before. Note that TPOT outputs a python 
script. I suggest telling TPOT to look at only a subset of the classifier models in sklearn with 
the flag --classifier_subset, as in the example below. I've found 50 generations with a population 
size of 50 works pretty well, but results may vary.

```bash
python train_TPOT.py --training_data path/to/training_data.csv \
--outfile tpot_output.py \
--classifier_subset sklearn.ensemble.ExtraTreesClassifier sklearn.ensemble.RandomForestClassifier \
--generations 50 --population_size 50 --n_jobs 10 --template Selector-Classifier
```

2.) After running TPOT, it's now time to train the model and test it on your hold-out set. TPOT has
written a python script (called 'tpot_output.py' in the example above).

 Option A) You should now cut and paste the appropriate parts into the script called train_test_model.py, which will train the model using
pipeline from TPOT and test it against your holdout set. It writes a serialized model object, a 
csv file with precision-recall results on the test set, and a csv file with the prediction results
on the test set (ID1,ID2,known_label,classifier_score,FDR)

  Option B) Use wrapper script python train_test_model2.py --training_infile /path/to/training_data.csv --test_infile /path/to/test_data.csv --exported_pipeline tpot_test_all.py --id_cols 0


3.) If you like the results of testing on your leaveout set, you can predict on your full dataset 
using the trained, serialized model with tpot_predict.py


-Benjamin Liebeskind and Claire McWhite

Example flow:

SELECTORS_FORMATTED=\$(cat example_files/selector_subset.txt | tr '\n' ' ')
CLASSIFIERS_FORMATTED=\$(cat example_files/classifier_subset.txt | tr '\n' ' ')
  

python train_TPOT.py --training_data featmat_labeled1 --outfile pipeline.py --template 'Selector-Classifier' --selector_subset $SELECTORS_FORMATTED --classifier_subset $CLASSIFIERS_FORMATTED --style "classify" --id_cols 0 --n_jobs 10 --10 --population_size 100 --labels -1 1 --temp_dir auto --groupcol traincomplexgroups --max_features_to_select 2


python train_test_model2.py --training_infile example_files/featmat_labeled1 --exported_pipeline pipeline --id_cols 0 --output_basename tpot --groupcol traincomplexgroups


python tpot_predict.py --datafile example_files/featmat --serialized_model model --outfile scored_interactions --id_cols 0


