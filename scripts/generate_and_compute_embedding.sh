# Initialize
# Initialize model variables
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7,4,5,6,7}" # GPU for parallel processing
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
CKPT="llava-v1.5-13b"
MODEL_PATH='liuhaotian/llava-v1.5-13b'
SPLIT="llava_OpenEnded_mscoco_val2014" # dataset name split
IMG_DIR='/home/daohieu/suvlm/data/vqav2/val2014'
QUES_FILE='data/okvqa/okvqa_processed_25.jsonl' # preprocessed question file
OUTDIR="output_dir" # for saving 

# Process the first chunk (use for small datasets or debugging, or the first run to download the model).
# Uncomment this and run for downloading the model and processor if it is the first run
# Comment this if you want to run the full pipeline, but need to download the model first.
IDX=0
CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python pipeline/generate_and_compute_emb_hf.py \
        --model_path $MODEL_PATH \
        --question_file $QUES_FILE \
        --image_folder $IMG_DIR \
        --outdir  $OUTDIR/$SPLIT/generation_embedding/$CKPT/${CHUNKS}_${IDX} \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 1 \
        --top_p='0.9' \
        --num_generations_per_prompt=50


# Generate and compute embeddings parallelly in background
# Uncomment the following lines to run in parallel, but need to download the model first.
# for IDX in $(seq 0 $((CHUNKS-1))); do
# CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python pipeline/generate_and_compute_emb_hf.py \
#         --model_path $MODEL_PATH \
#         --question_file $QUES_FILE \
#         --image_folder $IMG_DIR \
#         --outdir  $OUTDIR/$SPLIT/generation_embedding/$CKPT/${CHUNKS}_${IDX} \
#         --num_chunks $CHUNKS \
#         --chunk_idx $IDX \
#         --temperature 1 \
#         --top_p='0.9' \
#         --num_generations_per_prompt=50 &
# done

wait
# Merge the generated embeddings
python pipeline/merge_generation.py \
        --generation_dir $OUTDIR/$SPLIT/generation_embedding/$CKPT