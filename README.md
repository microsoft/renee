# Renee: End-to-end training of extreme classification models

Official PyTorch implementation for the paper: "Renee: End-to-end training of extreme classification models" accepted at MLSys 2023.

## Abstract

The goal of Extreme Multi-label Classification (XC) is to learn representations that enable mapping input texts to the most relevant subset of labels selected from an extremely large label set, potentially in hundreds of millions.

We identify challenges in the end-to-end training of XC models and devise novel optimizations that improve training speed over an order of magnitude, making end-to-end XC model training practical. Renee delivers state-of-the-art accuracy in a wide variety of XC benchmark datasets.

## Requirements

Run the below command, this will create a new conda environment with all the dependencies required to run Renee.

```bash
bash install1.sh
conda activate renee
bash install2.sh
```

## Data Preparation
You can download the datasets from the XML repo (http://manikvarma.org/downloads/XC/XMLRepository.html).

A dataset folder should have the following directory structure. Below we show it for LF-AmazonTitles-131K dataset:

```bash
📁 LF-AmazonTitles-131K/
    📄 trn_X_Y.txt # contains mappings from train IDs to label IDs
    📄 trn_filter_labels.txt # this contains train reciprocal pairs to be ignored in evaluation
    📄 tst_X_Y.txt # contains mappings from test IDs to label IDs
    📄 tst_filter_labels.txt # this contains test reciprocal pairs to be ignored in evaluation
    📄 trn_X.txt # each line contains the raw input train text, this needs to be tokenized
    📄 tst_X.txt # each line contains the raw input test text, this needs to be tokenized
    📄 Y.txt # each line contains the raw label text, this needs to be tokenized
```

To tokenize the raw train, test and label texts, we can use the following command (change the path of the dataset folder accordingly):
```bash
python -W ignore -u utils/CreateTokenizedFiles.py \
--data-dir xc/Datasets/LF-AmazonTitles-131K \
--max-length 32 \
--tokenizer-type bert-base-uncased \
--tokenize-label-texts
```

To create a dataset having label-text augmentation, we can use the following command:
```bash
python utils/CreateAugData.py \
--data-dir xc/Datasets/LF-AmazonTitles-131K \
--tokenization-folder bert-base-uncased-32 \
--max-len 32
```

Above command will create a folder named `xc/Datasets/LF-AmazonTitles-131K-Aug`, now we can refer to this dataset directory in our training script to train with label-text augmentation.

## Training

Train Renee on LF-AmazonTitles-131K dataset using label-text augmentation, you can use the following command (make sure you modify `data-dir`, `use-ngame-encoder` accordingly; keep in mind that you need to generate label-text augmentation dataset folder first, refer to Data Preparation section of README)
```bash
python main.py \
--epochs 100 \
--batch-size 32 \
--lr1 0.05 \
--lr2 1e-5 \
--warmup 5000 \
--data-dir xc/Datasets/LF-AmazonTitles-131K-Aug \
--maxlen 32 \
--tf sentence-transformers/msmarco-distilbert-base-v4 \
--dropout 0.85 \
--pre-tok \
--wd1 1e-4 \
--noloss \
--fp16xfc \
--use-ngame-encoder xc/ngame_pretrained_models/LF-AmazonTitles-131K/state_dict.pt \
--expname lfat-131k-aug-1.0
```
To change hyperparameters, you can refer to the various arguments provided in `main.py` file or you can do `python main.py --help` to list out the all the arguments.

Training commands for other datasets are provided in `scripts/train_commands.md`.

## Citation

If you find our work/code useful in your research, please cite the following:

```bibtex
@article{renee_2023,
  title={Renee: End-to-end training of extreme classification models},
  author={Jain, Vidit and Prakash, Jatin and Saini, Deepak and Jiao, Jian and Ramjee, Ramachandran and Varma, Manik},
  journal={Proceedings of Machine Learning and Systems},
  year={2023}
}
```

## References