# LAUR
- This is the code for Lifelong Language Learning with Adaptive Uncertainty Regularization.

### DATA
You need to download these data and set them into data/ .
| Datasets | link |
| ---- | ------- |
| Text Classification | [AGNews, Yelp, Amazon, DBPedia, Yahoo](http://goo.gl/JyCnZq) |
| Different Types Language Tasks | [SQuAD, WikiSQL, SST, QA-SRL](https://drive.google.com/file/d/1A1Q4P1LiwVEHEwVfAuRhN5fWF3LMDw3G/view?usp=sharing) |
| Sentiment Classification | [16 tasks sentiment](https://drive.google.com/file/d/1lgT2ieGn5sAXwtF_nFH4ee0ZmNC_hFnY/view?usp=sharing) |

we converted QA-SRL into Squad-like format, you can get original QA-SRL data from [here](https://dada.cs.washington.edu/qasrl/).
 

### Run our code
- Text classification: 
```bash
CUDA_VISIBLE_DEVICES=0 python train_text.py --approach LAUR --logname 'order1_seed42' --seed 42 --tasks_order 1 --min_import 1.05 --alpha 1 --beta 1 --gamma 1
```
  -`--approach`: select method.
  -`--logname`: the log name which save in result_data/csvdata.
  -`--seed`: Set the seed of random generator.
  -`--tasks_order 1`: select order to train, there are four orders in paper.
  -`--min_import`: set min for rho_import in Eq.10 of paper.
  -`--alpha` `--beta` `--gamma`: the hyperparameter for UR. Usually, the best effect can be achieved by setting all of them to 1
  
- Different Types Language Tasks:
```bash
sh run_train.sh
```

- Sentiment Classification:
```bash
CUDA_VISIBLE_DEVICES=0 python train_sentiment.py --approach LAUR --logname '1_LAUR' --seed 1 --tasks_order 1 --alpha 1 --beta 1 --gamma 0.5
```

### Requirements

- Python >=3.6
- Pytorch 1.6.0+cudatoolkit10.1 / CUDA 10.1
- transformers


Reference

BERT Base network is from https://github.com/huggingface/transformers
