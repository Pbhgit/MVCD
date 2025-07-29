import os
import glob
import json
import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, List, Optional
from datasets import Dataset, load_dataset

default_device = torch.device("cuda:0" )
class VIDEOLens():
    def __init__(
        self, 
        attribute_weights: str = "../LanguageBind/attribute_features.pt", 
        tag_weights: str = "../LanguageBind/tag_features.pt",
        vocab_attributes: str = "../data/vocab/vocab_attributes",
        vocab_tags: str = "../data/vocab/vocab_tags",
        video_path: str = "../data/MSRVTT/ValVideo",
        video_weights: str = "..LanguageBind/video_features.pt",
        split_tags: str = "train",
        split_attributes: str = "train",
        load_8bit: bool = False,
        device: torch.device = default_device,      
    ):       
        self.device = device
        self.attribute_weights = torch.load(attribute_weights, map_location=self.device)
        self.tag_weights = torch.load(tag_weights, map_location=self.device)
        self.vocab_tags = load_dataset(vocab_tags, split=split_tags)['prompt_descriptions']
        self.video_path = self._read_video_path(video_path)
        self.vocab_attributes = self._flatten(load_dataset(vocab_attributes, split=split_attributes)['prompt_descriptions'])
        self.video_weights = torch.load(video_weights, map_location=self.device)

    def _flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def _read_video_path(self, path):
        path_pattern = os.path.join(path, '**', '*.mp4')
        mp4_paths = glob.glob(path_pattern, recursive=True)
        return mp4_paths 


    def __call__(
        self, 
        samples: dict,
        num_tags: int = 5,
        num_attributes: int = 5,
        contrastive_th: float = 0.2,
        # top_k: int = 50, 
        return_tags: bool = True,
        return_attributes: bool = True,
        return_complete_prompt: bool = True,       
    ):
        if return_tags:
            samples = self.get_top_tags(
                samples, num_tags=num_tags, contrastive_th=contrastive_th
            )
        if return_attributes:
            samples = self.get_top_attributes(
                samples, num_attributes=num_attributes, contrastive_th=contrastive_th
            )
        if return_complete_prompt:
            samples = self.create_prompt_from_samples(samples)

        return samples

    def get_top_tags(
        self,
        samples = dict,
        num_tags: int = 5,
        contrastive_th: float = 0.02,
    ):
        tags = []
        video_features = self.video_weights.clone()
        video_features /= video_features.norm(dim=-1, keepdim=True)  
        tag_features = self.tag_weights.clone()
        tag_features /= tag_features.norm(dim=-1, keepdim=True)
        tag_text_scores = torch.matmul(video_features, tag_features.T)    
                
        top_tag_scores, top_tag_indices = tag_text_scores.float().cpu().topk(k=num_tags, dim=-1)
        for scores, indexes in tqdm(zip(top_tag_scores, top_tag_indices),  desc="Processing Tags"):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_tags = [self.vocab_tags[index] for index in filter_indexes]
            else:
                top_k_tags = []
            tags.append(top_k_tags)    
        samples["tags"] = tags 
        return samples

    def get_top_attributes(
        self,
        samples: dict,
        num_attributes: int = 5,
        contrastive_th: float = 0.02,
    ):
        attributes = []
        video_features = self.video_weights.clone()
        video_features /= video_features.norm(dim=-1, keepdim=True)
        attribute_features = self.attribute_weights.clone()
        attribute_features /= attribute_features.norm(dim=-1, keepdim=True)
        attribute_text_scores = torch.matmul(video_features, attribute_features.T)
        top_attribute_scores, top_attribute_indices = attribute_text_scores.float().cpu().topk(k=num_attributes, dim=-1)
        for scores, indexes in tqdm(zip(top_attribute_scores, top_attribute_indices), desc="Processing Attributes"):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_attributes = [self.vocab_attributes[index] for index in filter_indexes]
            else:
                top_k_attributes = []
            attributes.append(top_k_attributes)    
        samples["attributes"] = attributes 
        return samples   

    

    # def get_top_captions(
    #         self,
    #         samples: dict,
    #         num_attributes: int = 5,
    # ):
    #     captions = []






    def create_prompt_from_samples(
        self,
        samples: dict,
        ):
        video_prompts = []
        for idx, video_path in enumerate(self.video_path):
            video_id = video_path.strip("/").split("/")[-1]
            video_tags = samples['tags'][idx] 
            video_attributes = samples['attributes'][idx] 
            prompt = {
            "video_id": video_id,
            "Tags": video_tags,
            "Attributes": video_attributes
                }
            video_prompts.append(prompt)        
            samples['video_prompts'] = video_prompts
        return samples  



if __name__ == "__main__":
    video_lens = VIDEOLens(
        attribute_weights = "../LanguageBind/attribute_features.pt", 
        tag_weights = "../LanguageBind/tag_features.pt",
        vocab_attributes = "../data/vocab/vocab_attributes",
        vocab_tags = "../data/vocab/vocab_tags",
        video_path = "../data/MSRVTT/ValVideo",
        video_weights = "../LanguageBind/valvideo_features.pt",
        split_tags = "train",
        split_attributes = "train",  
        device = torch.device("cuda:0"),
    )

    samples = {}

    results = video_lens(
        samples = samples,
        num_tags=5,          
        num_attributes=5,   
        contrastive_th=0.1, 
        return_tags=True,    
        return_attributes=True,  
        return_complete_prompt=True  
    )

    with open ("val_prompts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("标签和属性已保存到：{}".format("val_prompts.json"))