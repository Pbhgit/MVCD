import os
import sys
import json
import re
import time
import random
import numpy as np
import argparse
from PIL import Image 
import torch
from tqdm import tqdm
from openai import AzureOpenAI
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import init_process_group

from eval_datasets import VQADataset
from contextlib import contextmanager
@contextmanager
def extend_sys_path(path):
    original_path = sys.path.copy()
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = original_path
with extend_sys_path(".../MVCD"):
    from cedecoding import CEDecoding
    from lens import Lens1, LensProcessor   
    from lens.utils import (      
        create_sampler,
        create_dataloader,        
    )

os.environ["HF_DATASETS_OFFLINE"] = "1"

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def prepare_sequential_smples(eval_dataset, num_samples, batch_size):
    sequential_indices = np.arange(num_samples)
    dataset = torch.utils.data.Subset(eval_dataset, sequential_indices)
    sampler = create_sampler(dataset, distributed=True)
    loader = create_dataloader(
        dataset,
        batch_size = batch_size,    
        sampler = sampler,        
    )
    return loader

def get_batch_same_type(train_dataset, batch, query_set_size, num_samples):
    same_type_questions = [[] for _ in range(len(batch['image']))]
    for i, question_type in enumerate(batch['question_type']):
        filtered_samples = [sample for sample in train_dataset if sample['question_type'] == question_type] 
        actual_query_set_size = min(query_set_size, len(filtered_samples))
        if actual_query_set_size < query_set_size:
            print(f"Warning: query_set_size {query_set_size} is reduced to {actual_query_set_size} due to the limited number of available samples.")
        query_set_indices = np.random.choice(len(filtered_samples), actual_query_set_size, replace=False)
        query_set = [filtered_samples[i] for i in query_set_indices]
        same_type_questions[i] = random.sample(query_set, num_samples)
    return same_type_questions


def sample_batch_demos_from_query_set(query_set,  batch_size, num_samples,):
    return [random.sample(query_set, num_samples) for _ in range(len(batch_size['image']))]

def compute_effective_num_shots(num_shots):
    return num_shots if num_shots > 0 else 2


def llm_generate_answers(args, rank):
    lens = Lens1(device=f'cuda:{rank}')
    processor = LensProcessor()
    end_prompt = "Directly answer the above question with one word or phrase."
    system_prompt = "This is a question and answer task. Please analyze the provided Tags, and Captions to deduce the short answer to the given question. Ensure your response is succinct, limited to one word or phrase, and directly relates to the information provided. "
    
    if args.llama2_7b:
        with torch.no_grad():
            try:
                with open(args.incontext_examples_path, 'r') as f:
                    train_dataset = json.load(f)
            except IOError as e:
                print(f"Error opening context.json: {e}")
                return []                           
            predictions = [] 
            for batch in tqdm(args.small_dataset, desc='Processing batches llama2_7b'):
                ids = batch['question_id']
                images, questions, question_types = batch['image'], batch['question'], batch['question_type'] 
                batch_demo_samples = sample_batch_demos_from_query_set(train_dataset, batch, args.num_shots)                             
                batch_images, icl_examples = [], []
                for i, (image, question, question_type) in enumerate(zip(images, questions, question_types)):                                 
                    if args.num_shots > 0:
                        context_images = [Image.open(x['image_path']) for x in batch_demo_samples[i]]
                    else:
                        context_images = []
                    batch_images.append(context_images +[image]) 

                    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<<SYS>>\n\n']              
                    for x in batch_demo_samples[i]:   
                        texts.append(f"{system_prompt}{x['del_attr_prompts']}{end_prompt} [/INST] {x['answer'][0]}</s><s> [INST]")                                                        
                    prompt_text = ''.join(texts)
                    icl_examples.append(prompt_text)
                        
                samples = processor(images, questions)
                output = lens(samples) 
                icl = [icl_examples[i] +  system_prompt + output['del_attr_prompts'][i] + end_prompt + "[/INST]"  for i in range(len(images))]                    
                qas = [system_prompt + output['del_attr_prompts'][i] + end_prompt + "[/INST]"  for i in range(len(images))]
                for icl_example, qa, sample_id in zip(icl, qas, ids): 
                    generate_kwargs = dict(max_length=args.max_length, relative_top=args.relative_top, stop_gen=args.stop_gen, relative_top_value=args.relative_top_value, do_sample=args.do_sample)
                    result = args.llm.generate(qa, icl_example,  **generate_kwargs) 
                    pred = result.replace("\n", "").replace("\\", "").strip() 
                    match = re.search(r':(.+)', pred)
                    if match:
                        result = match.group(1).strip()
                    if result:  
                        result = result.strip()
                        res = result[0].lower() + result[1:] 
                        res  = res.strip('"')
                        if res.endswith('.'):
                            res = res[:-1]  
                    predictions.append({"answer":res, "question_id":sample_id})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return  predictions
        
    elif args.llama2_13b:
        with torch.no_grad():
            try:
                with open(args.incontext_examples_path, 'r') as f:
                    train_dataset = json.load(f)
            except IOError as e:
                print(f"Error opening context.json: {e}")
                return []       
            predictions = [] 
            for batch in tqdm(args.small_dataset, desc='Processing batches llama2_13b...'):
                ids = batch['question_id']
                images, questions, question_types = batch['image'], batch['question'], batch['question_type'] 
                batch_demo_samples = get_batch_same_type(train_dataset, batch, args.query_set_size, args.num_shots)                              
                batch_images, icl_examples = [], []
                for i, (image, question, question_type) in enumerate(zip(images, questions, question_types)):                                 
                    if args.num_shots > 0:
                        context_images = [Image.open(x['image_path']) for x in batch_demo_samples[i]]
                    else:
                        context_images = []
                    batch_images.append(context_images +[image]) 

                    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<<SYS>>\n\n']              
                    for x in batch_demo_samples[i]:   
                        texts.append(f"{system_prompt}{x['prompts']}{end_prompt} [/INST] {x['answer'][0]}</s><s> [INST]")                                                        
                    prompt_text = ''.join(texts)
                    icl_examples.append(prompt_text)
                        
                samples = processor(images, questions)
                output = lens(samples) 
                icl = [icl_examples[i] + system_prompt + output['prompts'][i] + end_prompt + "[/INST]"  for i in range(len(images))]
                qas = [system_prompt + output['prompts'][i] + end_prompt + "[/INST]"  for i in range(len(images))]
                                    
                for icl_example, qa, sample_id in zip(icl, qas, ids): 
                    generate_kwargs = dict(max_length=args.max_length, relative_top=args.relative_top, stop_gen=args.stop_gen, relative_top_value=args.relative_top_value, do_sample=args.do_sample)
                    result = args.llm.generate(qa, icl_example, **generate_kwargs) 
                    pred = result.replace("\n", "").replace("\\", "").lstrip() 
                    match = re.search(r':(.+)', pred)
                    if match:
                        result = match.group(1).strip()
                    if result:  
                        res = result[0].lower() + result[1:] 
                        res  = res.strip('"')
                        if res.endswith('.'):
                            res = res[:-1]     
                    predictions.append({"answer":res, "question_id":sample_id})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return  predictions 
        
    elif args.llama3_8b:
        with torch.no_grad():
            try:
                with open(args.incontext_examples_path, 'r') as f:
                    train_dataset = json.load(f)
            except IOError as e:
                print(f"Error opening context.json: {e}")
                return []       
            predictions = []
            for batch in tqdm(args.small_dataset, desc='Processing batches llama3_8b...'):
                ids = batch['question_id']
                images, questions, question_types = batch['image'], batch['question'], batch['question_type'] 
                batch_demo_samples = get_batch_same_type(train_dataset, batch, args.query_set_size, args.num_shots)                              
                batch_images, icl_examples = [], []
                for i, (image, question, question_type) in enumerate(zip(images, questions, question_types)):                                 
                    if args.num_shots > 0:
                        context_images = [Image.open(x['image_path']) for x in batch_demo_samples[i]]
                    else:
                        context_images = []
                    batch_images.append(context_images +[image]) 

                    messages = [{"role": "sysetem", "content": system_prompt}]
                    for x in batch_demo_samples[i]:
                        messages.append({"role": "user", "content": f"{system_prompt}{x['prompts']}{end_prompt}"})
                        messages.append({"role": "assistant", "content": f"{x['answer'][0]}"})
                    icl_examples.append(messages)

                samples = processor(images, questions)
                output = lens(samples)

                icl = [icl_examples[i] +  [{"role": "user", "content":f"{output['prompts'][i]}{end_prompt}"}]  for i in range(len(images))]
                qas = [[{"role": "user", "content":f"{output['prompts'][i]}{end_prompt}"}]  for i in range(len(images))]
                for message, qa, sample_id in zip(icl, qas, ids):
                    icl_example, _ = args.llm.transform_message(message)
                    qa_, _ = args.llm.transform_message(qa)
                    generate_kwargs = dict(max_length=args.max_length, relative_top=args.relative_top, stop_gen=args.stop_gen, relative_top_value=args.relative_top_value, do_sample=args.do_sample)
                    result= args.llm.generate(prompt=qa_, context_example=icl_example, **generate_kwargs)
                    pred = result.replace("\n", "").replace("\\", "").strip() 
                    match = re.search(r':(.+)', pred)
                    if match:
                        result = match.group(1).strip()     
                    if result:  
                        res = result[0].lower() + result[1:] 
                        res  = res.strip('"')
                        if res.endswith('.'):
                            res = res[:-1]          
                    predictions.append({"answer":res, "question_id":sample_id})

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return  predictions   
    elif args.mistral:
        with torch.no_grad():
            try:
                with open(args.incontext_examples_path, 'r') as f:
                    train_dataset = json.load(f)
            except IOError as e:
                print(f"Error opening context.json: {e}")
                return []       
            predictions = []
            for batch in tqdm(args.small_dataset, desc='Processing batches mistral...'):
                ids = batch['question_id']
                images, questions, question_types = batch['image'], batch['question'], batch['question_type'] 
                batch_demo_samples = get_batch_same_type(train_dataset, batch, args.query_set_size, args.num_shots)                              
                batch_images, icl_examples = [], []
                for i, (image, question, question_type) in enumerate(zip(images, questions, question_types)):                                 
                    if args.num_shots > 0:
                        context_images = [Image.open(x['image_path']) for x in batch_demo_samples[i]]
                    else:
                        context_images = []
                    batch_images.append(context_images +[image]) 

                    messages = []
                    for x in batch_demo_samples[i]:
                        messages.append({"role": "user", "content": f"{system_prompt}{x['prompts']}{end_prompt}"})
                        messages.append({"role": "assistant", "content": f"{x['answer'][0]}"})
                    icl_examples.append(messages)

                samples = processor(images, questions)
                output = lens(samples)

                icl = [icl_examples[i] +  [{"role": "user", "content":f"{output['prompts'][i]}{end_prompt}"}]  for i in range(len(images))]
                qas = [[{"role": "user", "content":f"{system_prompt}{output['prompts'][i]}{end_prompt}"}]  for i in range(len(images))]
                for message, qa, sample_id in zip(icl, qas, ids):
                    icl_example, _ = args.llm.mistral_transform(message)
                    qa_, _ = args.llm.mistral_transform(qa)
                    generate_kwargs = dict(max_length=args.max_length, relative_top=args.relative_top, stop_gen=args.stop_gen, relative_top_value=args.relative_top_value, do_sample=args.do_sample)
                    result= args.llm.generate(prompt=qa_, context_example=icl_example, **generate_kwargs)
                    pred = result.replace("\n", "").replace("\\", "").strip() 
                    match = re.search(r':(.+)', pred)
                    if match:
                        result = match.group(1).strip()     
                    if result:  
                        res = result[0].lower() + result[1:] 
                        res  = res.strip('"')
                        if res.endswith('.'):
                            res = res[:-1]          
                    predictions.append({"answer":res, "question_id":sample_id})

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return  predictions   
        
    elif args.qwen2_7b:
        with torch.no_grad():
            try:
                with open(args.incontext_examples_path, 'r') as f:
                    train_dataset = json.load(f)
            except IOError as e:
                print(f"Error opening context.json: {e}")
                return []       
            predictions = []
            for batch in tqdm(args.small_dataset, desc='Processing batches qwen2_7b...'):
                ids = batch['question_id']
                images, questions, question_types = batch['image'], batch['question'], batch['question_type'] 
                batch_demo_samples = get_batch_same_type(train_dataset, batch, args.query_set_size, args.num_shots)                              
                batch_demo_samples = sample_batch_demos_from_query_set(train_dataset, batch, args.num_shots)                             

                batch_images, icl_examples = [], []
                for i, (image, question, question_type) in enumerate(zip(images, questions, question_types)):                                 
                    if args.num_shots > 0:
                        context_images = [Image.open(x['image_path']) for x in batch_demo_samples[i]]
                    else:
                        context_images = []
                    batch_images.append(context_images +[image]) 

                    messages = [{"role": "sysetem", "content": system_prompt}]
                    for x in batch_demo_samples[i]:
                        messages.append({"role": "user", "content": f"{system_prompt}{x['prompts']}{end_prompt}"})
                        messages.append({"role": "assistant", "content": f"{x['answer'][0]}"})
                    icl_examples.append(messages)

                samples = processor(images, questions)
                output = lens(samples)

                icl = [icl_examples[i] +  [{"role": "user", "content":f"{output['prompts'][i]}{end_prompt}"}]  for i in range(len(images))]
                qas = [[{"role": "user", "content":f"{system_prompt}{output['prompts'][i]}{end_prompt}"}]  for i in range(len(images))]
                for message, qa, sample_id in zip(icl, qas, ids):
                    icl_example = args.llm.qwen_transformer(message)
                    qa_ = args.llm.qwen_transformer(qa)
                    generate_kwargs = dict(max_length=args.max_length, relative_top=args.relative_top, stop_gen=args.stop_gen, relative_top_value=args.relative_top_value, do_sample=args.do_sample)
                    result= args.llm.generate(prompt=qa_, context_example=icl_example, **generate_kwargs)
                    pred = result.replace("\n", "").replace("\\", "").strip() 
                    match = re.search(r':(.+)', pred)
                    if match:
                        result = match.group(1).strip()     
                    if result:  
                        res = result[0].lower() + result[1:] 
                        res  = res.strip('"')
                        if res.endswith('.'):
                            res = res[:-1]          
                    predictions.append({"answer":res, "question_id":sample_id})

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return  predictions   

    else:
        print("Don't find the corresponding model.")

def main(rank, world_size):
    parser = argparse.ArgumentParser("Generate answers for VQA questions.")
    parser.add_argument("--incontext_examples_path", type=str, default='vqa_context.json')
    parser.add_argument("--query_set_size", default=20, type=int)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--num_shots", default=1, type=int)
    parser.add_argument("--qwen2_7b", action='store_true', help='Whether to use mistral for evaluation.')
    parser.add_argument("--qwen_model_path", type=str, default='../models/Qwen2-7B-Instruct')   
    parser.add_argument("--image_dir_path", type=str, default='../datasets/mscoco/val2014/', help='Path to the image directory')
    parser.add_argument("--question_path", type=str, default='../datasets/vqa-v2/v2_OpenEnded_mscoco_val2014_questions.json', help='Path to the question file.')
    parser.add_argument("--annotations_path", type=str, default='../datasets/vqa-v2/v2_mscoco_val2014_annotations.json', help='Path to the annnotation file.')
    parser.add_argument("--is_train", type=bool, default=True, help='Indicates if the dataset is for training of not')
    parser.add_argument("--dataset_name", type=str, default='vqav2', help='Name of the dataset.')
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--mistral", action='store_true', help='Whether to use mistral for evaluation.')
    parser.add_argument("--mistral_model_path", type=str, default="../models/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--llama2_7b", action='store_true', help='Whether to use llama2-7b-chat for evaluation.')
    parser.add_argument("--llama2_13b", action='store_true', help='Whether to use llama2-13b-chat for evaluation.')
    parser.add_argument("--llama_model_name_or_path",  type=str, default='../models/llama2-chat-7b/')
    parser.add_argument("--llama_model_path", type=str, default='../models/Llama-2-13b-chat-hf')
    parser.add_argument("--llama3_8b_instruct_path", type=str, default="../models/Meta-Llama-3-8B-Instruct/")
    parser.add_argument("--llama3_8b", action="store_true", help="Whether to use llama3-8b for evaluation.")
    parser.add_argument("--no_cuda", default=False, action='store_true')
    parser.add_argument("--num_gpus", default=1, type=int) 
    parser.add_argument("--max_gpu_memory", default=27, type=int)
    parser.add_argument("--relative_top_value", default=-1000.0, type=float)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--stop_gen", action="store_true")
    parser.add_argument("--instance_id", type=int, default=1, help="Unique identifier for this script instance.")
   
    args = parser.parse_args()
   
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device("cuda:{}".format(rank))

    ######################################### dataset  ###################################   
    args.dataset = VQADataset(
        image_dir_path = args.image_dir_path,
        question_path = args.question_path,
        annotations_path = args.annotations_path,
        is_train = args.is_train,
        dataset_name = args.dataset_name
    )
    args.small_dataset = prepare_sequential_smples(args.dataset, args.num_samples, args.batch_size)

    ################################  download model  ##############################################
    if args.llama2_7b:
        args.llm = CEDecoding(args.llama_model_name_or_path, args.device, args.num_gpus, args.max_gpu_memory)  

    elif args.llama2_13b:
        args.llm = CEDecoding(args.llama_model_path, args.device, args.num_gpus, args.max_gpu_memory)
      
    elif args.llama3_8b:
        args.llm = CEDecoding(args.llama3_8b_instruct_path, args.device, args.num_gpus, args.max_gpu_memory)  
    elif args.mistral:
        args.llm = CEDecoding(args.mistral_model_path, args.device, args.num_gpus, args.max_gpu_memory)  
    elif args.qwen2_7b:
        args.llm = CEDecoding(args.qwen_model_path, args.device, args.num_gpus, args.max_gpu_memory)  

    else:
        print("Don't need to load the model")

    ###################################  generat answers ############################################  
    args.all_predictions = llm_generate_answers(args, rank)
    predictions = args.all_predictions    
   
    ################################################ save results ######################################################################
    with open(f"../result_{args.instance_id}.json", "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

########################################################################################
    
def spawn_func():
    size = 1
    mp.spawn(main, args=(size, ), nprocs=size, join=True)

if __name__ == "__main__":
    spawn_func()
        

