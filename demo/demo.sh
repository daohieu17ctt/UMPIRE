# Correct cases
python demo/predict_and_compute_umpire.py \
    --model_name 'liuhaotian/llava-v1.5-13b' \
    --max_new_tokens 256 \
    --image_path demo/images/COCO_val2014_000000525732.jpg \
    --prompt "What color strip does the surfboard have?" \
    --num_generations 10 \
    --temperature 1 \
    --alpha 0.96 \
    --entailment_model 'deberta' \
    --reason 'none'

# Wrong cases
python demo/predict_and_compute_umpire.py \
    --model_name 'liuhaotian/llava-v1.5-13b' \
    --max_new_tokens 256 \
    --image_path demo/images/COCO_val2014_000000525732.jpg \
    --prompt "What is the brand name of the surfboard?" \
    --num_generations 10 \
    --temperature 1 \
    --alpha 0.96 \
    --entailment_model 'deberta' \
    --reason 'none'
