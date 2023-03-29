# Training scripts

Below we provide sample train scripts for some of the datasets.

> All batch-sizes are intended for 16 GPU runs. If you are running the code on fewer GPUs, adjust the (per-gpu) batch size accordingly. For example, if you are running on 8 GPUs, double the (per-gpu) batch-size. 

> In case the intended `batch-size` is unable to fit on the configuration being used, please make use of the `--accum` flag to enable gradient accumulation. 


## non-Label Features Datasets

AmazonTitles-670K:
```bash
python main.py \
--epochs 60 \
--batch-size 16 \
--lr1 0.005 \
--lr2 5e-5 \
--warmup 10000 \
--data-dir xc/Datasets/AmazonTitles-670K \
--maxlen 32 \
--tf sentence-transformers/all-roberta-large-v1 \
--dropout 0.8 \
--wd1 0.001 \
--noloss \
--fp16xfc
```

Amazon-670K:
```bash
python main.py \
--epochs 70 \
--batch-size 16 \
--lr1 0.001 \
--lr2 4e-5 \
--warmup 10000 \
--data-dir xc/Datasets/Amazon-670K \
--maxlen 512 \
--tf sentence-transformers/all-roberta-large-v1 \
--dropout 0.8 \
--wd1 0.001 \
--noloss \
--fp16xfc
```

AmazonTitles-3M

Amazon-3M

Wiki-500K
```bash
python main.py \
--epochs 100 \
--batch-size 128 \
--lr1 0.002 \
--lr2 2e-4 \
--warmup 5000 \
--data-dir xc/Datasets/Wiki-500K \
--maxlen 256 \
--tf sentence-transformers/paraphrase-distilroberta-base-v2 \
--dropout 0.75 \
--wd1 0.001 \
--noloss \
--fp16xfc
```


## Label Features Datasets

LF-AmazonTitles-131K
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
--wd1 1e-4 \
--noloss \
--fp16xfc \
--use-ngame-encoder ngame_pretrained_models/LF-AmazonTitles-131K/state_dict.pt
```

LF-Wikipedia-500K
```bash
python main.py \
--epochs 100 \
--batch-size 128 \
--lr1 0.002 \
--lr2 1e-4 \
--warmup 5000 \
--data-dir xc/Datasets/LF-Wikipedia-500K-Aug \
--maxlen 256 \
--tf sentence-transformers/paraphrase-distilroberta-base-v2 \
--dropout 0.7 \
--wd1 0.001 \
--noloss \
--fp16xfc
```

LF-AmazonTitles-3M


LF-WikiSeeAlso-320K
```bash
python main.py \
--epochs 100 \
--batch-size 128 \
--lr1 0.1 \
--lr2 2e-4 \
--warmup 5000 \
--data-dir xc/Datasets/LF-WikiSeeAlso-320K-Aug \
--maxlen 128 \
--tf sentence-transformers/paraphrase-distilroberta-base-v2 \
--dropout 0.75 \
--wd1 1e-4 \
--noloss \
--fp16xfc
```

LF-Amazon-131K






