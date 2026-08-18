# !/bin/bash
# This script computes UMPIRE and evaluates the results.
# Change the paths as necessary.
SPLIT="okvqa" # dataset name split
CKPT="llava-v1.5-13b"
generation_file="output/${SPLIT}/generation_embedding/${CKPT}.pkl"
output_dir="output/${SPLIT}/results"

# Compute UMPIRE and evaluate
CUDA_VISIBLE_DEVICES=0 python pipeline/compute_umpire_and_evaluate.py \
        --generation_file=$generation_file \
        --output_dir=$output_dir
