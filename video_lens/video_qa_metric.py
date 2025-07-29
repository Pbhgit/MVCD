import os
import json
import ast
import numpy as np
import argparse
import torch
from tqdm import tqdm
from openai import AzureOpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-4")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--api_base", default="https://openai-vlaa-westus.openai.azure.com", help="OpenAI API base.")
    parser.add_argument("--result_path", default="../result/qwenvd_result_0.json")
    args = parser.parse_args()
    return args

def annotate(result_path, args):
    client = AzureOpenAI(  
        azure_endpoint = args.api_base,  # west us   
        api_key = args.api_key,  # west us
        api_version = "2023-12-01-preview",
    )
    with open(result_path, "r") as f:
        results = json.load(f)
    result_qa_pair = []
    for result in results:
        question = result['question']
        answer = result['actual_answer']
        pred = result['answer']
        try:
            response = client.chat.completions.create(
            model="gpt-4-1106-preview-nofilter",
            messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms, paraphrases, or related concepts as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer, where semantic alignment can yield high scores."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match, including cases where the predicted answer semantically aligns with the correct answer through synonyms or related meanings."
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ]
            )
            response_message = response.choices[0].message.content
            # print(response_message)
            response_dict = ast.literal_eval(response_message)
            print(response_dict)
            result_qa_pair.append([response_dict, result])
            # print(result_qa_pair)
        except Exception as e:
            print(f"Error processing result with question '{question}': {e}")
    with open('output.json', "w") as f:
        json.dump(result_qa_pair, f)

def main():
    args = parse_args()
    annotate(args.result_path, args)
    with open('output.json', 'r') as f:
        result_qa_pair = json.load(f)
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for result_qa in result_qa_pair:
        try:
            count += 1
            score_match = result_qa[0]['score']
            score = int(score_match)
            score_sum += score

            pred = result_qa[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result_qa)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)

    results_data = {
        "result_path": args.result_path,
        "yes_count": yes_count,
        "no_count": no_count,
        "accuracy": accuracy,
        "average_score": average_score
    }
    with open('../result/results.json', 'a') as file:
        json.dump(results_data, file)
        file.write('\n')  
   
        
if __name__ == "__main__":
    main()



        