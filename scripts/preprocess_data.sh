# Initialize dataset variables
ds_name='okvqa'
ds_question_file="data/${ds_name}/okvqa_OpenEnded_mscoco_val2014_questions.json"
ds_answer_file="data/${ds_name}/okvqa_mscoco_val2014_annotations.json"
ds_outfile="data/${ds_name}/okvqa_processed.jsonl"

# Preprocess VQA data
echo "Processing dataset: $dataset_name"
python pipeline/preprocess_data.py \
        --question_file=$ds_question_file \
        --answer_file=$ds_answer_file \
        --outfile=$ds_outfile