# !/bin/bash
# This script computes UMPIRE and evaluates the results.
# Change the paths as necessary.
generation_file='output/okvqa/generation_embedding/llava-v1.5-13b.pkl'
output_dir='output/okvqa/results'

# Compute UMPIRE and evaluate
python pipeline/compute_umpire_and_evaluate.py \
        --generation_file=$generation_file \
        --output_dir=$output_dir