# System for SRST2020 

This branch contains our system for the **Surface Realisation Shared Task 2020**. The 2019 version of our system can be found on the **srst2019** branch.

## Training a model

To train a model for a given language on the SRST2019 train and dev datasets and then host it on a given port:
```
lang=en; python surface/grammar.py --model_file models/$lang.bin --train_files data/T1-train/${lang}_* data/T1-dev/${lang}_* --test_files data/T1-test/${lang}_* data/T1-dev/${lang}_* --port 4780
```

Note that you have to specify the test files also so that the vocabulary of the
grammar can include them, as there is currently no treatment for OOVs.

## Loading a model
To load a pretrained model and host it, run the same command without specifying
input files:
```
lang=en; python surface/grammar.py --model_file models/$lang.bin --port 4780
```

## Surface realization

Running surface realization for a given language:
```
lang=en; for fn in data/T1-test/${lang}_*; do f=`basename $fn | cut -d'.' -f1`; echo $f; mkdir -p gen/$f; python surface/surface_realization.py --gen_dir gen/$f --test_file $fn --output_file output/$f.conllu --timeout 120 --port 4780; done &> log/20201016_${lang}_test.log &
```

