import os
import json
import glob
import torch
import ipdb

from tqdm import tqdm

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def read_video_path(path):
    path_pattern = os.path.join(path, '**', '*.mp4')
    map4_paths = glob.glob(path_pattern, recursive=True)
    return map4_paths

def costum_collated_fn(batch):
    return batch

def generate_caption(batch_size):
    disable_torch_init()  
    video_path = "../data/MSRVTT/ValVideo"
    mp4_paths = read_video_path(video_path)   
    # inp = 'Please provide a short caption of the video.'
    model_path = '../models/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda:0'    
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    # conv_mode = "llava_v1"
    # conv = conv_templates[conv_mode].copy()
    # roles = conv.roles
    loader =  torch.utils.data.DataLoader(
        mp4_paths,
        batch_size = batch_size,
        collate_fn = costum_collated_fn
    )
    results = []
    for batch in tqdm(loader, desc="generate captons..."):
        for video_pth in batch:                       
            video_id = video_pth.strip("/").split("/")[-1] 
            try:
                video_tensor = video_processor(video_pth, return_tensors='pt')['pixel_values']
                # __import__("ipdb").set_trace()
                captions = []
                if type(video_tensor) is list:
                    tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                else:
                    tensor = video_tensor.to(model.device, dtype=torch.float16)
                for _ in range(5): 
                    inp = 'Please provide a short caption of the video.' 
                    conv_mode = "llava_v1"
                    conv = conv_templates[conv_mode].copy()
                    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], None)                                           
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=tensor,
                            do_sample=True,
                            temperature=0.9,
                            top_k = 50,
                            top_p = 0.95,
                            max_new_tokens=1024,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria])
                    # __import__("ipdb").set_trace()
                    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace('</s>', '').strip()
                    captions.append(output)
                results.append({
                    'video_id': video_id, 
                    'captions': captions
                })
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
    return results



if __name__ == '__main__':
    results = generate_caption(batch_size=7)
    with open('all_captions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)