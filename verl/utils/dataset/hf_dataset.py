# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf, ListConfig
import os
from typing import List, Union, Optional
import copy
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset as HfDataset

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 1280*28*28, min_pixels: int = 256*28*28):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        if image['bytes'] is not None:
            image = Image.open(BytesIO(image['bytes']))
        else:
            image = Image.open(image['path'])

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class HFDataset(Dataset):
    def __init__(self,
                 hf_file: str,
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 system_prompt: str = None,
                 post_prompt: str = None,
                 max_pixels = 1003520,
                 min_pixels = 200704):

        self.hf_file = hf_file
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.post_prompt = post_prompt
        self.serialize_dataset = False

        self._download_and_load_datasets()

    def _download_and_load_datasets(self):
        # self.dataset = load_from_disk(hf_file)
        self.dataset = load_dataset(self.hf_file)['train']

        print(f'original dataset len: {len(self.dataset)}')

        import multiprocessing
        num_cores = multiprocessing.cpu_count()  # 获取可用 CPU 核心数
        num_proc = max(88, num_cores)  # 确保不超过核心数

        if not isinstance(self.dataset[0][self.prompt_key], list):
            self.dataset = self.dataset.map(self.make_conversation, batched=False, num_proc=num_proc, keep_in_memory = True)


        self.dataset = self.dataset.filter(
            lambda doc: len(
                self.tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True)
            ) <= self.max_prompt_length, num_proc=num_proc, keep_in_memory = True
        )

        print(f'filtered dataset len: {len(self.dataset)}')

    def make_conversation(self, row):
        def ensure_period(sentence):
            if sentence.endswith(('.', '?', '!', ".'")):
                return sentence
            else:
                return sentence + '.'

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": ensure_period(row[self.prompt_key]) + ' ' + self.post_prompt,
            },
        ]
        row[self.prompt_key] = prompt
        return row

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        row_dict = self.dataset[item]

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if self.image_key in row_dict:
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            img_data = row_dict.pop(self.image_key)
            if not isinstance(img_data, list):
                img_data = [img_data]
            row_dict['multi_modal_data'] = {
                'image': [
                    process_image(image, max_pixels=self.max_pixels, min_pixels=self.min_pixels)
                    for image in img_data
                ]
            }
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            # row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation
        )

        if self.image_key in row_dict:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        row_dict['input_height'] = image_grid_thw[0][1]*14
        row_dict['input_width'] = image_grid_thw[0][2]*14
        return row_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'dataset' in state and not self.serialize_dataset:
            del state['dataset']
        return state