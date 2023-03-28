"""
For a dataset, create tokenized files in the folder `{tokenizer-type}-{maxlen}` folder inside the `output-path` (default is `data-dir`) folder.

Sample usage: 

python -W ignore -u CreateTokenizedFiles.py \
--data-dir /home/t-japrakash/xc/Datasets/LF-AmazonTitles-131K \
--max-length 32 \
--tokenizer-type bert-base-uncased \
--tokenize-label-texts

"""
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import os
import numpy as np
import time
import functools
import argparse
from tqdm import tqdm

def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer

def _tokenize(batch_input):
    tokenizer, max_len, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])

def convert(corpus, tokenizer, max_len, num_threads, bsz=100000): 
    batches = [(tokenizer, max_len, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids, attention_mask

@timeit
def tokenize_dump_memmap(corpus, tokenization_dir, tokenizer, max_len, prefix, num_threads, batch_size=10000000):
    ii = np.memmap(f"{tokenization_dir}/{prefix}_input_ids.dat", dtype='int64', mode='w+', shape=(len(corpus), max_len))
    am = np.memmap(f"{tokenization_dir}/{prefix}_attention_mask.dat", dtype='int64', mode='w+', shape=(len(corpus), max_len))
    for i in tqdm(range(0, len(corpus), batch_size)):
        _input_ids, _attention_mask = convert(corpus[i: i + batch_size], tokenizer, max_len, num_threads)
        ii[i: i + _input_ids.shape[0], :] = _input_ids
        am[i: i + _input_ids.shape[0], :] = _attention_mask

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", type=str, required=True, help="Data directory path - with {trn,tst}_X.txt and Y.txt where each line contains train/test and label text respectively")
parser.add_argument("--max-length", type=int, help="Max sequence length for tokenizer", default=32)
parser.add_argument("--tokenizer-type", type=str, help="Tokenizer to use", default="bert-base-uncased")
parser.add_argument("--num-threads", type=int, help="Number of threads to use", default=32)
parser.add_argument("--output-folder", type=str, help="Folder to dump the tokenization to, if not provided, uses data-dir", default="")
parser.add_argument("--tokenize-label-texts", action="store_true", default=False, help="Whether to tokenize label texts or not")

args = parser.parse_args()

DATA_DIR = args.data_dir

trnX = [x.strip() for x in open(f'{DATA_DIR}/trn_X.txt', "r", encoding="utf-8").readlines()]
tstX = [x.strip() for x in open(f'{DATA_DIR}/tst_X.txt', "r", encoding="utf-8").readlines()]

if args.tokenize_label_texts:
    Y = [x.strip() for x in open(f'{DATA_DIR}/Y.txt', "r", encoding="utf-8").readlines()]
    print(len(Y))
    print(Y[:10])

print(len(trnX), len(tstX))
print(trnX[:10], tstX[:10])

max_len = args.max_length
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type, do_lower_case=True)

if len(args.output_folder) == 0:
    args.output_folder = args.data_dir
tokenization_dir = f"{args.output_folder}/{args.tokenizer_type}-{max_len}"
os.makedirs(tokenization_dir, exist_ok=True)

print(f"Dumping files in {tokenization_dir}...")
print("Dumping for trnX...")
tokenize_dump_memmap(trnX, tokenization_dir, tokenizer, max_len, "trn_doc", args.num_threads)
print("Dumping for tstX...")
tokenize_dump_memmap(tstX, tokenization_dir, tokenizer, max_len, "tst_doc", args.num_threads)

if args.tokenize_label_texts:
    print("Dumping for Y...")
    tokenize_dump_memmap(Y, tokenization_dir, tokenizer, max_len, "lbl", args.num_threads)