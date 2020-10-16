# System for SRST2020 

This branch contains our system for the **Surface Realisation Shared Task 2020**. The 2019 version of our system can be found on the **srst2019** branch.

## Training

To train a model for a given language on the SRST2019 train and dev datasets, run
```
lang=en; python surface/grammar.py --model_file models/$lang.bin --train_files data/T1-train/${lang}_* data/T1-dev/${lang}_* --test_files data/T1-test/${lang}_* data/T1-dev/${lang}_* --port 4780
```

Note that you have to specify the test files also so that the vocabulary of the
grammar can include them, as there is currently no treatment for OOVs.

## Testing

