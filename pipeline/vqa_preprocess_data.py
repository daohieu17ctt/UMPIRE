# Preprocess original vqa question format into llava vqa question format:
    # Original file:
        # infor, task_type, data_type, license, data_subtype, questions(image_id, question, question_id)
    # into
    # Llava format:
        # question_id, image(image_file_name), text, category


import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--question-file", type=str, default="data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json")
parser.add_argument("--answer-file", type=str, default="data/vqav2/v2_mscoco_val2014_annotations.json")
parser.add_argument("--outfile", type=str, default="data/vqav2/llava_OpenEnded_mscoco_val2014_questions.jsonl")
args = parser.parse_args()

objs = json.load(open(args.question_file, 'r'))
answer_objs = json.load(open(args.answer_file, 'r'))
data_type = objs['data_type']
if data_type != "mscoco":
    print("data type is not mscoco")
    exit()
data_subtype = objs['data_subtype']
question_list = objs['questions']

# map answer with question
answers_dict = {}
for answer_obj in answer_objs['annotations']:
    ques_id = answer_obj['question_id']
    answers = [x['answer'] for x in answer_obj['answers']]
    answers_dict[ques_id] = answers

# create new dataset
new_question_list = []
for q_obj in question_list:
    q_id = q_obj['question_id']
    img_id = q_obj['image_id']
    q_text = q_obj['question']
    if 'advqa' in args.question_file:
        # Because of val2017 split, advqa dataset has image_id in the format of 12 digits only, e.g: 000000466986.jpg
        img_file_name = f"{img_id:012}.jpg"
    else:
        # vqav2 and okvqa have this val2014 format, e.g: COCO_val2014_000000297147.jpg
        img_file_name = f"COCO_{data_subtype}_{img_id:012}.jpg"
    new_question_list.append({
        "question_id": q_id,
        "image": img_file_name,
        "text": q_text,
        "category": "default",
        "answers": answers_dict[q_id]   
    })


# import pdb; pdb.set_trace()
import jsonlines
with jsonlines.open(args.outfile, 'w') as writer:
    writer.write_all(new_question_list)
