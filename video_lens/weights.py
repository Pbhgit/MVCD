import os
import torch
import json
import glob
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from languagebind import (
    LanguageBindVideo, 
    LanguageBindVideoTokenizer, 
    LanguageBindVideoProcessor,  
    LanguageBindVideoProcessor1,  
    LanguageBindVideoProcessor2, 
    transform_dict, 
    to_device
    )



class Features:
    def __init__(
        self,
        # dataset,
        device,
        batch_size,
        vocab_tags = "../data/vocab/vocab_tags",
        vocab_attributes = "../data/vocab/vocab_attributes",
        split_tags = "train",
        split_attributes = "full",
        model_path = "../models/LanguageBind_Video_FT/",
        video_path = "../data/MSRVTT/ValVideo",
        cached_features=None,
        ):
        # self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.vocab_tags = vocab_tags
        self.vocab_attributes = vocab_attributes
        self.split_tags = split_tags
        self.split_attributes = split_attributes
        self.model = LanguageBindVideo.from_pretrained(model_path).to(device)
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(model_path)
        self.video_process = LanguageBindVideoProcessor(self.model.config, self.tokenizer)
        self.video_process1 = LanguageBindVideoProcessor1(self.model.config, self.tokenizer)
        self.video_process2 = LanguageBindVideoProcessor2(self.model.config)
        self.video_path = video_path
        if cached_features is None:
            # self.tag_features = self._tag_features()
            # self.attribute_features = self._attribute_features()
            self.video_features = self._video_features()
        else:
            # self.tag_features = cached_features
            # self.attribute_features = cached_features
            self.video_features = cached_features



    
    def _costum_collated_fn(self, batch):
        return batch
     

    def _tag_features(self):
        self.model.eval()
        vocab_tags = load_dataset(self.vocab_tags, split=self.split_tags)['prompt_descriptions'] 
        loader = torch.utils.data.DataLoader(
            vocab_tags,
            batch_size = self.batch_size,          
            collate_fn = self._costum_collated_fn,       
        )
       
        tag_features = []
        for bath in tqdm(loader, desc="tags features"):
            features = []
            for tag in bath:
                dummy_pixel_values = torch.zeros(1, 3, 8, 224, 224).to(device)
                data = self.video_process1(tag, return_tensors='pt').to(device)
                data['pixel_values'] = dummy_pixel_values
                with torch.no_grad():
                    out = self.model(**data)
                feature = out['text_embeds'].squeeze(0)
                features.append(feature)

            batch_features = torch.stack(features)
            tag_features.append(batch_features)

        tag_features = torch.cat(tag_features, dim=0)
        return tag_features

    def _attribute_features(self):
        self.model.eval()
        vocab_attributes = load_dataset(self.vocab_attributes, split=self.split_attributes)['prompt_descriptions'] 
        vocab_attributes = [attribute for attributes in vocab_attributes for attribute in attributes]
        loader = torch.utils.data.DataLoader(
            vocab_attributes,
            batch_size = self.batch_size,          
            collate_fn = self._costum_collated_fn,       
        )
       
        attribute_features = []
        for bath in tqdm(loader, desc="attributes features"):
            features = []
            for attribute in bath:
                dummy_pixel_values = torch.zeros(1, 3, 8, 224, 224).to(device)
                data  = self.video_process1(attribute, return_tensors='pt').to(device)
                data['pixel_values'] = dummy_pixel_values
                with torch.no_grad():
                    out = self.model(**data)
                feature = out['text_embeds'].squeeze(0)
                features.append(feature)

            batch_features = torch.stack(features)
            attribute_features.append(batch_features)

        attribute_features = torch.cat(attribute_features, dim=0)
        return attribute_features
        
    def _video_features(self):
        path_pattern = os.path.join(self.video_path, '**', '*.mp4')
        mp4_paths = glob.glob(path_pattern, recursive=True)
        loader = torch.utils.data.DataLoader(
            mp4_paths,
            batch_size = self.batch_size,          
            collate_fn = self._costum_collated_fn,       
        )
        video_features  = []
        for bath in tqdm(loader, desc="valvideo features"):
            features = []
            for video_path in bath:
                data = self.video_process2(video_path, return_tensors='pt')
                data['pixel_values'] = data['pixel_values'].to(device)
                dummy_input_ids = torch.zeros((data['pixel_values'].shape[0], 77), dtype=torch.long).to(device)
                dummy_attention_mask = torch.zeros((data['pixel_values'].shape[0], 77), dtype=torch.long).to(device)
                data['input_ids'] = dummy_input_ids
                data['attention_mask'] = dummy_attention_mask
                with torch.no_grad():
                    out = self.model(**data)
                feature = out['image_embeds'].squeeze(0)
                features.append(feature)
                # video_num = video_path.strip("/").split("/")[-1]
            batch_features = torch.stack(features)
            video_features.append(batch_features)

        video_features = torch.cat(video_features, dim=0)
        return video_features

        

device = torch.device("cuda:0")
features_extractor = Features(
    device=device,
    batch_size=7, 
    vocab_tags="../data/vocab/vocab_tags", 
    vocab_attributes="../data/vocab/vocab_attributes",  
    split_tags="train",
    split_attributes="train",
    model_path="../models/LanguageBind_Video_FT/",  
    video_path = "../data/MSRVTT/ValVideo",
    cached_features=None  
)


val_video_features = features_extractor.video_features

torch.save(val_video_features, 'valvideo_features.pt')


