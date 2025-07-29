import torch
import random
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np

class CEDecoding():
    def __init__(self, model_name, device, num_gpus, max_gpu_memory):
        self.model_name = model_name 
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name_path):
        kwargs = {
            "torch_dtype":  "auto",# torch.float16,
            "offload_folder": f"{model_name_path}/offload"  
        }

        # if "cuda" in self.device:
        if self.device.type == "cuda":
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            elif int(self.num_gpus) > 1:
                kwargs.update({  
                    "device_map": "auto",
                    "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(int(self.num_gpus))}
                })
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_path,  **kwargs).to(self.device)

        model.eval()
        print("Loading LLM...")

        return model, tokenizer
    
    def transform_message(self, messages):
        self.messages = messages
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(self.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        return input_ids, terminators
    
    def mistral_transform(self, messages):
        self.messages = messages
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(self.device)
        return input_ids
    
    def qwen_transformer(self, messages):
        self.messages = messages
        text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_ids = model_inputs.input_ids
        return input_ids

    
    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh
    
    # way1: don't be based on contrastive decoding
    def generate(self, prompt, context_example=None, max_length=50, relative_top=0.1, stop_gen=False, relative_top_value=-1000.0, do_sample=False):
        if isinstance(prompt, str):
            student_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            student_input_ids = prompt.to(self.device)
            
        if isinstance(context_example, str):
            expert_input_ids = self.tokenizer(context_example, return_tensors="pt").input_ids.to(self.device)
        else:
            expert_input_ids = context_example.to(self.device)
   
        output_tokens = []
        with torch.no_grad():            
            while len(output_tokens) < max_length:
                expert_outputs = self.model(expert_input_ids) 
                student_outputs = self.model(student_input_ids)

                expert_logits = expert_outputs.logits[:, -1, :]
                student_logits = student_outputs.logits[:, -1, :]

                expert_next_token = torch.argmax(expert_logits, dim=-1)                 
                student_next_token = torch.argmax(student_logits, dim=-1)
 
                if not stop_gen:
                    if  (expert_next_token == self.tokenizer.eos_token_id) or (student_next_token == self.tokenizer.eos_token_id):
                        break
                else:
                    terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]    
                    if (expert_next_token in terminators ) or (student_next_token in terminators ):
                        break              

                student_logits = student_logits.log_softmax(dim=-1)
                expert_logits = expert_logits.log_softmax(dim=-1)
                diff_logits = expert_logits - student_logits

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(expert_logits, relative_top) 
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)                   
                
                if do_sample:
                    probabilities = torch.softmax(diff_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, 1)
                else:
                    next_token = torch.argmax(diff_logits, dim=-1).unsqueeze(0)


                output_tokens.append(next_token.item())

                student_input_ids = torch.cat([student_input_ids, next_token], dim=1)            
                expert_input_ids = torch.cat([expert_input_ids, next_token], dim=1)
        generated_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return generated_text
            
    # way2: Have been based on the contrastive decoding
    @torch.no_grad()
    def fast_demo_contrastive_search(self, prompt, context_example, beam_width, alpha, decoding_len, relative_top=0.1,  relative_top_value=-1000.0, end_of_sequence_token_id = None, early_stop = False, do_sample=False):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        assert alpha >= 0. and alpha <= 1.0
        if isinstance(prompt, str):
            student_tokens = self.tokenizer.tokenize(prompt)
            student_input_ids = self.tokenizer.convert_tokens_to_ids(student_tokens)
            student_input_ids = torch.LongTensor(student_input_ids).view(1,-1).to(self.device)
        else:
            student_input_ids = prompt.to(self.device)

        if isinstance(context_example, str):
            expert_tokens = self.tokenizer.tokenize(context_example)
            expert_input_ids = self.tokenizer.convert_tokens_to_ids(expert_tokens)
            expert_input_ids = torch.LongTensor(expert_input_ids).view(1,-1).to(self.device)
        else:
            expert_input_ids = expert_input_ids.to(self.device)
        
        output_tokens = []

        # fast mode
        batch_size, seqlen = student_input_ids.size()
        prefix_len = seqlen
        
        student_past_key_values = expert_past_key_values = None
        student_last_hidden_states = expert_last_hidden_states = None
        student_logits = expert_logits = None
        for step in range(decoding_len):
            student_past_key_values, student_last_hidden_states, student_logits = self.ContrastiveDecodingOneStepFast(
                self.model,
                student_input_ids,
                beam_width,
                alpha,
                student_past_key_values,
                student_last_hidden_states,
                self.tokenizer,
                student_logits,
                first_step=step == 0,
            )
            expert_past_key_values, expert_last_hidden_states, expert_logits = self.ContrastiveDecodingOneStepFast(
                self.model,
                expert_input_ids,
                beam_width,
                alpha,
                expert_past_key_values,
                expert_last_hidden_states,
                self.tokenizer,
                expert_logits,
                first_step=step == 0,
            )
           
            # Handle logits
            student_logits_ = student_logits.log_softmax(dim=-1)
            expert_logits_ = expert_logits.log_softmax(dim=-1)
            expert_next_token = torch.argmax(expert_logits, dim=-1) 
            student_next_token = torch.argmax(student_logits, dim=-1)

            if (expert_next_token == self.tokenizer.eos_token_id) or \
            (student_next_token == self.tokenizer.eos_token_id):
                break
            diff_logits = expert_logits_ - student_logits_

            if relative_top > 0.0:
                relative_top_mask = self.get_relative_top_filter(expert_logits, relative_top) 
                diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits) 

            if do_sample:
                probabilities = torch.softmax(diff_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
            else:
                next_token = torch.argmax(diff_logits, dim=-1).unsqueeze(0)
                # print(next_token)

            
            next_token_id = next_token.squeeze().item()
            output_tokens.append(next_token_id)

            # Update input ids
            student_input_ids = torch.cat([student_input_ids, next_token], dim=1)
            expert_input_ids = torch.cat([expert_input_ids, next_token], dim=1)

            # Update generated list
            tokens = student_input_ids.squeeze(dim=-1).tolist()
            generated = [item for item in student_input_ids.tolist()]

            # Early stopping check
            if early_stop and (end_of_sequence_token_id in tokens):
                break
       
        generated_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return generated_text

    def ranking_fast(self, context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):    
        _, context_len, embed_dim = context_hidden.size() 
        norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
        norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True) 
        cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1) 
        scores, _ = torch.max(cosine_matrix, dim=-1) 
        next_top_k_probs = next_top_k_probs.view(-1)
        scores = (1.0 - alpha) * next_top_k_probs - alpha * scores
        scores = torch.stack(torch.split(scores, beam_width)) 
        selected_idx = scores.max(dim=-1)[1]
        return selected_idx

    def ContrastiveDecodingOneStepFast(
        self, 
        model, 
        ids, 
        beam_width, 
        alpha, 
        past_key_values,
        last_hidden_states,
        vocab,
        logit_for_next_step,
        first_step=False,
        ):
        if first_step:
            output = self.model(
                input_ids=ids, 
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = output.past_key_values 
            last_hidden_states = output.hidden_states[-1] 
            logit_for_next_step = output.logits[:, -1, :] 

        bsz, seqlen, embed_dim = last_hidden_states.size()
        p = random.uniform(0, 1)

        next_probs = F.softmax(logit_for_next_step, dim=-1) 
        _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)
        top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids) 
        # compute new hidden
        past_key_values = self.enlarge_past_key_values(past_key_values, beam_width)
        output = model(
            input_ids=top_k_ids.view(-1, 1), 
            attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]  
        next_hidden = output.hidden_states[-1] 
        context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim) 

        selected_idx = self.ranking_fast(
            context_hidden, 
            next_hidden, 
            top_k_probs,    
            alpha,
            beam_width,
        )     
        # prepare for the next step
        next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1) 
        next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width)) 
        next_hidden = next_hidden[range(bsz), selected_idx, :] 
        last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1) 
        past_key_values = self.select_past_key_values(past_key_values, beam_width, selected_idx)
        logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :] 
        # next_id: [bsz, 1]
        return past_key_values, last_hidden_states, logits


    def enlarge_past_key_values(self, past_key_values, beam_width):
        new_key_values = []
        for layer in past_key_values:
            items = []
            for item in layer:
                # item is the key and value matrix
                bsz, num_head, seq_len, esz = item.size()
                item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    
                items.append(item)
            new_key_values.append(items)
        return new_key_values

    def select_past_key_values(self, past_key_values, beam_width, selected_idx):
        '''select_idx: [B]'''
        new_key_values = []
        for layer in past_key_values:
            items = []
            for item in layer:
                bsz_and_beam, num_head, seq_len, esz = item.size()
                bsz = int(bsz_and_beam//beam_width)
                item = torch.stack(torch.split(item, beam_width, dim=0))    
                item = item[range(bsz), selected_idx, :, :, :]   
                items.append(item)
            new_key_values.append(items)
        return new_key_values

        



        
