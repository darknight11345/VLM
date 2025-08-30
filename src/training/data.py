import copy
import os
from dataclasses import dataclass, field
from typing import Dict

import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from .constants import *
import sys

from .params import DataArguments

def encode_video(video_path, max_num_frames=10):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_qa_pairs_from_json(img_file_name: str, qa_json_path: str) -> list[dict]:
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entry = next((item for item in data if item["filename"] == img_file_name), None)
    if entry is None:
        raise ValueError(f"No entry found for {img_file_name} in QA JSON")
    return [{"question": qa["question"], "answer": str(qa["answer"])} for qa in entry["question_answer"]]
    
class LazySupervisedDataset(Dataset):
    
    def __init__(self, data_path: list, processor, data_args: DataArguments):
        super().__init__()
        self.processor = processor
        self.list_data_dict = self._prepare_from_qa_json(data_args.qa_json_path)
        self.data_args = data_args

    def _prepare_from_qa_json(self, qa_json_path):
        with open(qa_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # Load image
        image_file = sources["filename"]
        print(" the file name inside the get item is : ", image_file)
        image_path = os.path.join(self.data_args.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        images = [image]
        

        # Get QA
        qa_pairs = get_qa_pairs_from_json(image_file, self.data_args.qa_json_path)
        #print("qa pairs:", qa_pairs)

        # Create conversation pairs
        conversations = []
        for qa in qa_pairs:
            conversations.append({"from": "user", "value": f"{LLAVA_IMAGE_TOKEN} {qa['question']}"})
            conversations.append({"from": "assistant", "value": qa["answer"]})

        sources = llava_to_openai(conversations)
        print("****************")

        all_input_ids = [torch.tensor([1])]
        all_labels = [torch.tensor([IGNORE_INDEX])]

        for idx in range(0, len(sources), 2):
            user_input = sources[idx]
            gpt_response = sources[idx + 1]

            user_input_text = START_INST + user_input["content"] + END_INST
            gpt_output_text = gpt_response["content"]

            if idx == 0:
                inputs = self.processor(text=user_input_text, images=images, return_tensors='pt')
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values = inputs.get("pixel_values")
            else:
                prompt_input_ids = self.processor(text=user_input_text, return_tensors="pt")["input_ids"]

            response_input_ids = self.processor(text=gpt_output_text, return_tensors="pt")["input_ids"]

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat([
                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                response_input_ids.squeeze(0)
            ], dim=0)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        all_input_ids.append(torch.tensor([2]))
        all_labels.append(torch.tensor([2]))

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids != IGNORE_INDEX).to(torch.long)
        #pixel_values = all_pixel_values[0] if all_pixel_values is not None else None  tharani commented and added the below replacement line
        #pixel_values = all_pixel_values if all_pixel_values is not None else None  # keep shape [1, C, H, W]
        
        pixel_values = None
        if all_pixel_values is not None:
            pv = all_pixel_values
            # unwrap repeated single-element lists
            while isinstance(pv, list) and len(pv) == 1:
                pv = pv[0]
            if not isinstance(pv, torch.Tensor):
                raise TypeError(f"Unexpected pixel_values type after normalization: {type(pv)}")
            # ensure shape is [N_images, C, H, W]
            if pv.dim() == 3:          # [C,H,W] -> add image dim
                pv = pv.unsqueeze(0)   # -> [1,C,H,W]
            elif pv.dim() == 4:
                pass
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pv.shape}")
            pixel_values = pv


        
        print(image_file, len(image_file))
        
        '''print( {
            "input_ids inside LazySupervisedDataset ": input_ids,
            "labels inside LazySupervisedDataset": labels,
            "attention_mask inside LazySupervisedDataset": attention_mask,
            "pixel_values inside LazySupervisedDataset": pixel_values,
        })'''

        data_dict = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, debug: bool = False):  # tharani added debug attribute for debugging
        self.pad_token_id = pad_token_id
        self.debug = debug  # initialize the debug attribute  # tharani added for debugging

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = [] # tharani uncommented for debugging
        
        

        for example in examples:
            print("DEBUG: Example keys:", example.keys())
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            
            pv = example.get("pixel_values")
            
            if pv is None:
                continue

            if isinstance(pv, list):

                if all(isinstance(x, torch.Tensor) for x in pv):
                    pv = torch.stack(pv, dim=0)  # multiple images -> [N_images, C, H, W]
                elif len(pv) == 1:
                    pv = pv[0]
                else:
                    raise TypeError(f"Unexpected list structure for pixel_values at example {example_idx}: {pv}")


            

            # remove singleton batch dim if present ([1, C, H, W] -> [C, H, W])
            #if pv.dim() == 4 and pv.size(0) == 1:
                #pv = pv.squeeze(0)
            
            if pv.dim() == 3:  # [C, H, W] -> add image dim
                pv = pv.unsqueeze(0)  # -> [1, C, H, W]
            elif pv.dim() == 4:
                pass  # already [N_images, C, H, W]
            else:
                raise ValueError(f"pixel_values for example {example_idx} has unexpected shape: {pv.shape}")
            


            batch_pixel_values.append(pv) 
            
            
            #batch_pixel_values.append(example.get("pixel_values")) # tharani uncommented for debugging
            #pixel_values = example.get("pixel_values") # tharani commented for debugging
            
            #### tharani added for debugging start
            
            #if pixel_values is not None:
                #batch_pixel_values.append(pixel_values)
                
            #### tharani added for debugging end 
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        #print("batch_pixel_values: ", batch_pixel_values) # tharani added for debugging
        sys.stdout.flush()
        #pixel_values = torch.cat([pv for pv in batch_pixel_values if pv is not None and pv.numel() > 0], dim=0) if any(pv is not None and pv.numel() > 0 for pv in batch_pixel_values) else None
    
        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            #pixel_values = pixel_values
            
        )
        
        

               
        
        if batch_pixel_values:
            shapes = [pv.shape for pv in batch_pixel_values]
            if self.debug:
                print("DEBUG: individual pixel_values shapes:", shapes)
            if len(set(shapes)) != 1:
                raise ValueError(f"Inconsistent pixel_values shapes in batch: {shapes}")
            pixel_values = torch.stack(batch_pixel_values, dim=0)  # [B, C, H, W]
            batch_dict["pixel_values"] = pixel_values
            if self.debug:
                print(
                    f"batched pixel_values shape: {pixel_values.shape}, "
                    f"min: {pixel_values.min().item()}, max: {pixel_values.max().item()}"
                )
                
         # tharani commented for debugging         
          
        #if pixel_values is not None:
            #batch_dict.update(pixel_values=pixel_values)
            #print(f"the pixel values are {pixel_values}")
            
        # tharani commented for debugging
        # tharani added for debugging --- start
        '''
        print({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        })
        '''
        
        # tharani added for debugging  --- end

        return batch_dict


def replace_image_tokens(input_string, start_count=1):
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN, f"[IMG]", 1)
        count += 1

    return input_string, count

def video_to_image_tokens(input_string, num_frames):

    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)

    return input_string

def llava_to_openai(conversations, is_video=False, num_frames=None):

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1  # Initialize image count here
    for conversation in conversations:
        
        if is_video:
            conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)
        
        transformed_content, image_count = replace_image_tokens(conversation["value"], image_count)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)
    print("transformed_data:", transformed_data)
    return transformed_data

def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    print("Dataset length:", len(sft_dataset))
    print(processor.tokenizer.pad_token_id)
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id, debug=True)
    print(data_collator)
    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)