> Template from https://github.com/paperswithcode/releasing-research-code.

# Reproducing Chen’s Work on Assertion Detection 

This repository is a reproduction of [Assertion Detection in Clinical Natural Language Processing: A Knowledge-Poor Machine Learning Approach](https://ieeexplore.ieee.org/document/8710921/).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Preprocessing

The training and testing dataset are from the 2010 [i2b2/VA challenge](https://www.i2b2.org/NLP/DataSets/Main.php). You need to apply for the data access and download the data into `ib2b_va_data/`, and rename the directories into: `BIDPH_assertion`, `BIDPH_txt`, `UPMC_assertion`, `UPMC_txt`. 

Run this command to pre-process the data: 
```preprocess
python preprocess.py
```

## Training and evaluation

First, download the pre-trained word embedding:

```load_embedding
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin -O biowordvec.bin
```

To train and evaluate the model in the paper, run this command:

```train_eval
python train_eval.py
```

## NegEx evaluation

To evaluate NegEx with the testing dataset, you should downlaod the NegEx implementation from  https://github.com/chapmanbe/negex/tree/master/negex.python and put it in `open_negex/`.

Then, run this command:

```evaluate_negex
python eval_negex.py
```
