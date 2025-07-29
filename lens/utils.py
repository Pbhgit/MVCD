import datetime
import os

import torch
from torch.distributed import init_process_group
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

default_device = "cuda:2" if torch.cuda.is_available() else "cpu"


MAP_CLIP_NAME = {
    "openai/clip-vit-large-patch14": "ViT-L-14",
    "openai/clip-vit-base-patch16": "ViT-B-16",
    "openai/clip-vit-base-patch32": "ViT-B-32",
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "laion-ViT-H-14-2B",
    "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": "laion-ViT-bigG-14-2B",
}

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def costum_collated_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def create_sampler(dataset, distributed=False):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler

#batch_size=8
def create_dataloader(dataset, sampler, batch_size, num_workers=0):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=costum_collated_fn,
        # num_workers=num_workers,
        # pin_memory=True,
        sampler=sampler,
        shuffle=False,
        # drop_last=False,
    )
    return loader


def is_main_process():
    if int(os.environ["RANK"]) == 0:
        return True
    else:
        return False


def get_llm_model(version, load_8bit, device_map=None):
    if load_8bit:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                version,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={"": device_map},
            )
        except:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                version,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={"": device_map},
            )
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(version).to(device_map)
        except:
            model = AutoModelForCausalLM.from_pretrained(version).to(device_map)
    model = model.eval()
    return model


def create_prompt_sample(
    samples,
    idx,
    tags_col="tags",
    attributes_col="attributes",
    caption_col="caption",
    intensive_captions_col="intensive_captions",
    question_col="questions",
    question_prompt=None,
    num_intensive_captions=50,
    mode="all",
):
    prompt = ""
    del_tag_prompt  = ""
    del_attr_prompt = ""
    del_caption_prompt = ""
    if question_prompt is not None:
        question = question_prompt
    else:
        question = samples[question_col][idx]

    if mode == "vqa":
        prompt += "Image:\n"
        prompt += "Captions:"
        prompt += ".".join(
            samples[intensive_captions_col][idx][:num_intensive_captions]
        )
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "vision":
        prompt += "Tag: "
        prompt += ",".join(samples[tags_col][idx])
        prompt += "\nAttributes: "
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "hm":
        prompt += "Image:\n"
        prompt += "Caption:"
        prompt += samples[caption_col][idx]
        prompt += "\nAttributes:"
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nTags:"
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nQuestion: Is the image hateful or not-hateful?"
        prompt += "\nShort Answer:"

    elif mode == "all":
        # all information
        prompt += "Tags:\n-"
        prompt += "\n-".join(samples[tags_col][idx])
        prompt += "\nAttributes:\n-"
        prompt += "\n-".join(samples[attributes_col][idx])
        prompt += "\nCaptions:\n-"
        prompt += "\n-".join(
            samples[intensive_captions_col][idx][:num_intensive_captions]
        )
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

        # delete tags
        del_tag_prompt += "Attributes:\n-"
        del_tag_prompt += "\n-".join(samples[attributes_col][idx])
        del_tag_prompt += "\nCaptions:\n-"
        del_tag_prompt += "\n-".join(
            samples[intensive_captions_col][idx][:num_intensive_captions]
        )
        del_tag_prompt += "\nQuestion:"
        del_tag_prompt += question
        del_tag_prompt += "\nShort Answer:"

        # delete attributes
        del_attr_prompt += "Tags:\n-"
        del_attr_prompt += "\n-".join(samples[tags_col][idx])
        del_attr_prompt += "\nCaptions:\n-"
        del_attr_prompt += "\n-".join(
            samples[intensive_captions_col][idx][:num_intensive_captions]
        )
        del_attr_prompt += "\nQuestion:"
        del_attr_prompt += question
        del_attr_prompt += "\nShort Answer:"

        # delete captions
        del_caption_prompt += "Tags:\n-"
        del_caption_prompt += "\n-".join(samples[tags_col][idx])
        del_caption_prompt += "\nAttributes:\n-"
        del_caption_prompt += "\n-".join(samples[attributes_col][idx])
        del_caption_prompt += "\nQuestion:"
        del_caption_prompt += question
        del_caption_prompt += "\nShort Answer:"

    else:
        raise Exception("Mode not available")
    return prompt, del_tag_prompt, del_attr_prompt, del_caption_prompt
