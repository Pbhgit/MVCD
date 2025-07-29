import sys
import os 
import torch
import argparse
import numpy as np
from vqa_metric import compute_vqa_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blip_result", type=str, default='../result/instrucutblip_result.json')
    parser.add_argument("--question_path", type=str, default='../datasets/vqa-v2/v2_OpenEnded_mscoco_val2014_questions.json')  
    parser.add_argument("--annotations_path", type=str, default= '../datasets/vqa-v2/v2_mscoco_val2014_annotations.json')
    parser.add_argument("--ok_question_path", type=str, default="../data/OK-VQA/OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--ok_annotations_path", type=str, default="../data/OK-VQA/mscoco_val2014_annotations.json")
    # parser.add_argument()
    args = parser.parse_args()


    # args.accuracy = compute_vqa_accuracy(args.llama_result, args.ok_question_path, args.ok_annotations_path)
    args.accuracy = compute_vqa_accuracy(args.blip_result, args.question_path, args.annotations_path)
    accuracy = args.accuracy

    # print(f"OKVQA Accuracy: {accuracy}")
    print(f"VQAV2 Accuracy: {accuracy}")


