"""
This will create a dataset augmented with label texts, treating them as inputs as explained in the paper.

Make sure the `tokenization-folder` also contains tokenized label texts, if not use `CreateTokenizedFiles.py` with `--tokenize-label-texts` flag to do so.

Sample usage:

python CreateAugData.py \
--data-dir /home/t-japrakash/xc/Datasets/LF-AmazonTitles-131K \
--tokenization-folder bert-base-uncased-32 \
--max-len 32

"""

import os
import numpy as np
import argparse
import shutil
import scipy.sparse as sp

import xclib.data.data_utils as data_utils
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", type=str, required=True, help="Data directory path - with {trn_X_Y,tst_X_Y}.txt and the tokenization folder")
parser.add_argument("--tokenization-folder", type=str, help="Folder that contains the path to tokenized texts", required=True)
parser.add_argument("--max-len", type=int, help="sequence length of tokenization of the tokenization-folder", required=True)
args = parser.parse_args()

parent_dir = os.path.dirname(args.data_dir)
dataset_name = args.data_dir.split("/")[-1]

# create Aug folder
aug_folder = os.path.join(parent_dir, dataset_name + "-Aug")
os.makedirs(aug_folder, exist_ok=True)

# transfer the train and test files
command_to_run = f'cp {str(os.path.join(args.data_dir, "trn_X_Y.txt"))} {str(aug_folder)}'
print("✅ Now running:", command_to_run)
subprocess.Popen(command_to_run, shell=True)
command_to_run = f'cp {str(os.path.join(args.data_dir, "trn_filter_labels.txt"))} {str(aug_folder)}'
print("✅ Now running:", command_to_run)
subprocess.Popen(command_to_run, shell=True)

command_to_run = f'cp {str(os.path.join(args.data_dir, "tst_X_Y.txt"))} {str(aug_folder)}'
print("✅ Now running:", command_to_run)
subprocess.Popen(command_to_run, shell=True)
command_to_run = f'cp {str(os.path.join(args.data_dir, "tst_filter_labels.txt"))} {str(aug_folder)}'
print("✅ Now running:", command_to_run)
subprocess.Popen(command_to_run, shell=True)

command_to_run = f'cp -r {str(os.path.join(args.data_dir, args.tokenization_folder))} {str(aug_folder)}'
print("✅ Now running:", command_to_run)
subprocess.Popen(command_to_run, shell=True)


# add identity matrix in trn_X_Y

trn_X_Y = data_utils.read_sparse_file(f'{args.data_dir}/trn_X_Y.txt')
N, L = trn_X_Y.shape[0], trn_X_Y.shape[1]

L_cross_L_matrix = sp.identity(L, dtype="float32")
print("Label text targets shape:", L_cross_L_matrix.shape)

trn_X_Y_aug = sp.vstack((trn_X_Y, L_cross_L_matrix))
trn_X_Y_aug = trn_X_Y_aug.tocsr()
print("new trn_X_Y shape:", trn_X_Y_aug.shape)

data_utils.write_sparse_file(trn_X_Y_aug, os.path.join(aug_folder, "trn_X_Y.txt"))
print("✅ Saving trn_X_Y with augmentation done...")

# add label text tokenization
print("✅ Creating tokenization for the augmented trn data...")
mmap_trn_doc_path = os.path.join(aug_folder, args.tokenization_folder, "trn_doc_input_ids.dat")
mmap_lbl_path = os.path.join(aug_folder, args.tokenization_folder, "lbl_input_ids.dat")
mmap_trn_am = os.path.join(aug_folder, args.tokenization_folder, "trn_doc_attention_mask.dat")
mmap_lbl_am = os.path.join(aug_folder, args.tokenization_folder, "lbl_attention_mask.dat")

print("✅ Loading tokenization at:", mmap_trn_doc_path)
trn_doc_input_ids = np.array(np.memmap(mmap_trn_doc_path, dtype=np.int64, mode="r", shape=(N, args.max_len)))
trn_doc_attention_masks = np.array(np.memmap(mmap_trn_am, dtype=np.int64, mode="r", shape=(N, args.max_len)))

print("✅ Loading tokenization at:", mmap_lbl_path)
lbl_input_ids = np.array(np.memmap(mmap_lbl_path, dtype=np.int64, mode="r", shape=(L, args.max_len)))
lbl_attention_masks = np.array(np.memmap(mmap_lbl_am, dtype=np.int64, mode="r", shape=(L, args.max_len)))

trn_aug_doc_input_ids = np.vstack((trn_doc_input_ids, lbl_input_ids))
trn_aug_attention_masks = np.vstack((trn_doc_attention_masks, lbl_attention_masks))

assert trn_aug_doc_input_ids.shape[0] == N+L
assert trn_aug_doc_input_ids.shape[1] == args.max_len

assert trn_aug_attention_masks.shape[0] == N+L
assert trn_aug_attention_masks.shape[1] == args.max_len

print(trn_aug_doc_input_ids.shape)
print(trn_aug_attention_masks.shape)

print("✅ Saving new tokenization at:", mmap_trn_doc_path)
trn_aug_mmap_ii = np.memmap(mmap_trn_doc_path, dtype=np.int64, mode="w+", shape=(N+L, args.max_len))
trn_aug_mmap_ii[:, :] = trn_aug_doc_input_ids[:, :]

print("✅ Saving new tokenization at:", mmap_trn_doc_path)
trn_aug_mmap_am = np.memmap(mmap_trn_am, dtype=np.int64, mode="w+", shape=(N+L, args.max_len))
trn_aug_mmap_am[:, :] = trn_aug_attention_masks[:, :]

print("✅ Augmented data created at:", aug_folder)