# !/bin/bash
# This script computes UMPIRE and evaluates the results.
# Change the paths as necessary.
SPLIT="llava_OpenEnded_mscoco_val2014"
CKPT="llava-v1.5-13b"
generation_file="output_dir/${SPLIT}/generation_embedding/${CKPT}.pkl"
output_dir="output_dir/${SPLIT}/results"

# Compute UMPIRE and evaluate
python pipeline/compute_umpire_and_evaluate.py \
        --generation_file=$generation_file \
        --output_dir=$output_dir