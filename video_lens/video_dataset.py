import os
import torch
import json
import glob
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
class MSRVTT_dataset(Dataset):
    def __init__(self, qa_path, video_path, train_val_info_path, caption_path, tags_atrributes_path, is_train, **kwargs):
        self.video_path = video_path
        self.is_train = is_train
        if qa_path is not None:
            with open(qa_path, 'r') as f:
                self.qa_datas = json.load(f)
        else:
            self.qa_datas = None
       

        self.videos = {}
        if train_val_info_path is not None:
            with open (train_val_info_path, 'r') as f:
                train_val = json.load(f)
            for video in train_val['videos']:
                if video['split'] == 'validate':
                    self.videos[video['video_id']] = {
                        'category': video['category'],
                        'video_id': video['video_id'],
                        'Captions': [],
                        'Tags': [],
                        'Attributes': [],
                    }

        if caption_path is not None:
            with open(caption_path, 'r') as f:
                val_captions = json.load(f)
            for caption_data in val_captions:
                video_id = caption_data['video_id'].split(".")[0].strip("")
                captions = caption_data['captions']
                if video_id in self.videos:
                    self.videos[video_id]['Captions'].extend(captions)


        if tags_atrributes_path is not None:
            with open (tags_atrributes_path, 'r') as f:
                tags_atrributes = json.load(f)
            for tag_atrribute in tags_atrributes['video_prompts']:
                video_id = tag_atrribute['video_id'].split(".")[0].strip()
                tags = tag_atrribute['Tags']
                attributes = tag_atrribute['Attributes']
                if video_id in self.videos:
                    self.videos[video_id]['Tags'].extend(tags)
                    self.videos[video_id]['Attributes'].extend(attributes)


    def __len__(self):
        return len(self.qa_datas)

    def __getitem__(self, index):
        qa_data = self.qa_datas[index]
        video_id = f"video{qa_data['video_id']}"
        video_name = f"{self.video_path}{video_id}.mp4"
        question = qa_data['question']
      
        answer = qa_data['answer']
        question_id = qa_data['id']
        category_id = qa_data['category_id']
        captions = self.videos[video_id]['Captions'] if video_id in self.videos else []  
        tags = self.videos[video_id]['Tags'] if video_id in self.videos else []
        attributes = self.videos[video_id]['Attributes'] if video_id in self.videos else []

        prompt = ""
        prompt = "Tags:\n- " + "\n- ".join(tags)
        prompt += "\nAttributes:\n- " + "\n- ".join(attributes)
        prompt += "\nCaptions:\n- " + "\n- ".join(captions)
        prompt += "\nQuestion: " + question
        prompt += "\nShort Answer: " 

        results = {
            "video_name": video_name,
            "video_id": video_id,
            "Question": question,
            "Short answer": answer,
            "question_id": question_id,
            "category_id": category_id,
            "Captions": captions, 
            "Tags": tags,
            "Attributes": attributes, 
            "prompt": prompt, 
            "prompts" : prompt+answer      
        }
        return results 

def save_dataset_to_json(dataset, file_path):
    data_to_save = []
    for i in range(len(dataset)):
        data_to_save.append(dataset[i])
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)


if __name__ == "__main__": 

    dataset = MSRVTT_dataset(
        qa_path = "../data/MSRVTT/QA/val_qa.json",
        video_path = "../data/MSRVTT/ValVideo/",
        train_val_info_path = "../data/MSRVTT/QA/train_val_videodatainfo.json",
        caption_path = "../experiments/Video-LLaVA/all_captions.json",
        tags_atrributes_path = "../experiments/LanguageBind/val_prompts.json",
        is_train =False
    )

    save_dataset_to_json(dataset, "val_output_file.json")
