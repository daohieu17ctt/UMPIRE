# Initialize dataset variables
ds_name='okvqa'
ds_question_file='/home/daohieu/maplecg_nfs/research/VLM/su_vlm/data/okvqa/okvqa_OpenEnded_mscoco_val2014_questions.json'
ds_answer_file='/home/daohieu/maplecg_nfs/research/VLM/su_vlm/data/okvqa/okvqa_mscoco_val2014_annotations.json'
ds_outfile='/home/daohieu/maplecg_nfs/research/VLM/su_vlm/data/okvqa/okvqa_processed.jsonl'

# Preprocess VQA data
echo "Processing dataset: $dataset_name"
python pipeline/preprocess_data.py \
        --question-file= $ds_question_file\
        --answer-file= $ds_answer_file\
        --outfile= $ds_outfile