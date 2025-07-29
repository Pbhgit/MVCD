import os
import sys
import json
import numpy as np
import argparse
import open_clip
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from eval_datasets import VQADataset
from tqdm import tqdm 


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
    from lens import Lens1, LensProcessor
    from lens.utils import (
        # ddp_setup, 
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

def costum_collated_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def generate_text(rank, args):
    dataset = VQADataset(
        image_dir_path = args.image_dir_path,
        question_path = args.question_path,
        annotations_path = args.annotations_path,
        is_train = True,
        dataset_name = args.dataset_name
    )
    sequential_indices = np.arange(args.numples)
    train_dataset = torch.utils.data.Subset(dataset, sequential_indices)
    sampler = create_sampler(train_dataset, distributed=True)
    train_data_loader =create_dataloader(
        train_dataset,     
        sampler=sampler,
        batch_size=args.batch_size,         
          )
    lens = Lens1(device=f'cuda:{rank}')
    processor = LensProcessor()
    with torch.no_grad():
        result = []
        for batch in tqdm(train_data_loader, desc="Preprocessing batches"):
            images, image_paths, questions, question_types, ids, answers = batch['image'], batch['image_path'], batch['question'], batch['question_type'], batch['question_id'], batch['answers']
            samples = processor(images, questions)
            output = lens(samples)

            for i in range(len(images)):
                sample_result = {
                    # 'image': images[i],
                    'image_path': image_paths[i], 
                    'question_type': question_types[i],
                    'question': questions[i],
                    'answer': answers[i],
                    'question_id': ids[i],
                    'tags': output['tags'][i],
                    'attributes': output['attributes'][i],
                    'caption': output['caption'][i],
                    'intensive_captions': output['intensive_captions'][i],
                    # 'prompt': output['prompts'][i] + answers[i][0],
                    'prompts': output['prompts'][i],
                    'del_tag_prompts': output['del_tag_prompts'][i],
                    'del_attr_prompts': output['del_attr_prompts'][i],
                    'del_caption_prompts': output['del_caption_prompts'][i]

                }
                result.append(sample_result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result




def main(rank, world_size):
    parser = argparse.ArgumentParser("Generate answers for VQA prompt.")
    parser.add_argument("--image_dir_path", type=str, default="../datasets/mscoco/train2014/")
    parser.add_argument("--question_path", type=str, default="../datasets/vqa-v2/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--annotations_path", type=str, default="../datasets/vqa-v2/v2_mscoco_train2014_annotations.json")
    parser.add_argument("--dataset_name", type=str, default="vqav2")
    parser.add_argument("--batch_size", type=int, default=170)
    parser.add_argument("--numples", type=int, default=443700)
    args = parser.parse_args()

    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    context = generate_text(rank, args)

    # with open (f"update_context.json", "w") as f:
    #     json.dump(context, f, ensure_ascii=False, indent=4)

    with open (f"v2_all_context_{rank}.json", "w") as f:
        json.dump(context, f, ensure_ascii=False, indent=4)
    torch.distributed.barrier()

    if rank == 0:
        all_result = []
        for i in range(world_size):
            with open (f"v2_all_context_{i}.json", "r") as f:
                result = json.load(f)
                all_result.extend(result)
        with open("v2_all_context.json", "w") as f:
            json.dump(all_result, f, ensure_ascii=False, indent=4)
       

def spawn_func():
    size = 2   
    mp.spawn(main, args=(size, ), nprocs=size, join=True)

if __name__ == "__main__":
    spawn_func()

