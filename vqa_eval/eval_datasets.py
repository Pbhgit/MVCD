import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, image_dir_path, question_path, annotations_path, is_train, dataset_name):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers =  json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
#         self.transform = transform
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014","test2015"}

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
            ) 
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "image_path": img_path,
            "question": question["question"],
            "question_id": question["question_id"],
            "question_type": answer["question_type"]
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        return results
