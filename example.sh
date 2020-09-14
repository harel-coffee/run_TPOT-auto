#Note not a great dataset, just to demo how pipeline works

SELECTORS_FORMATTED=$(cat example_files/selector_subset.txt | tr '\n' ' ')
CLASSIFIERS_FORMATTED=$(cat example_files/classifier_subset.txt | tr '\n' ' ')
  

python train_TPOT.py --training_data featmat_labeled1 --outfile pipeline.py --template 'Selector-Classifier' --selector_subset $SELECTORS_FORMATTED --classifier_subset $CLASSIFIERS_FORMATTED --style "classify" --id_cols 0 --n_jobs 10  --generations 10 --population_size 20 --labels -1 1 --temp_dir auto --groupcol traincomplexgroups --max_features_to_select 2


python train_test_model2.py --training_infile example_files/featmat_labeled1 --exported_pipeline pipeline.py --id_cols 0 --output_basename tpot --groupcol traincomplexgroups


python tpot_predict.py --datafile example_files/featmat --serialized_model tpot_fitted_model.p --outfile scored_interactions --id_cols 0


