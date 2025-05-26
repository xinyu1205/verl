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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, get_final_eos_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
import copy
from copy import deepcopy
from verl.utils import hf_processor
import os
from tqdm import tqdm

import re
import json
from typing import Tuple, Optional, List
from verl.utils.dataset.hf_dataset import process_image
from PIL import Image

def truncate_at_json_bbox(
    s: str
) -> Tuple[str, bool, Optional[List[int]]]:
    """
    Search for the first code block in string s that matches the following format:
      ```json
      [
          {"bbox_2d": [int, int, int, int], "label": "string"}
          ...
      ]
      ```
    If found:
      - Truncate the string up to the end of this code block and return the preceding part;
      - Return True to indicate truncation has occurred;
      - Return the bbox_2d list of this entry.
    If not found, return the original string, False, and None.
    """
    # Match all ```json ... ``` code blocks (non-greedy)
    pattern = re.compile(r'```json\n([\s\S]*?)\n```', re.DOTALL)
    for m in pattern.finditer(s):
        content = m.group(1).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        # Must be a list, and each item must be a dict
        if not (isinstance(data, list) and data and all(isinstance(item, dict) for item in data)):
            continue

        # Iterate through each item, check bbox_2d and label
        for item in data:
            bbox = item.get("bbox_2d")
            lbl  = item.get("label")
            if (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(x, int) for x in bbox)
                and isinstance(lbl, str)
            ):
                # Found a valid entry, truncate and return
                end_pos = m.end()  # Includes the ending ```
                return s[:end_pos], True, bbox

        # If no item in this block is valid, continue to the next block

    # No matching code block found
    return s, False, None


def crop_image(image_path, resize_input_height, resize_input_width, bbox):
    """
    Crop the original image at image_path according to bbox in the coordinate space
    of resize_input_width/resize_input_height, and return a PIL.Image object.

    Args:
        image_path (str): Path to the original image.
        resize_input_height (int): Height of the "resized image" where bbox is located.
        resize_input_width (int):  Width of the "resized image" where bbox is located.
        bbox (tuple or list of 4 floats): (x1, y1, x2, y2),
            coordinates in the range [0, resize_input_width] × [0, resize_input_height].

    Returns:
        PIL.Image: The cropped image.
    """
    # Open the image
    img = Image.open(image_path)
    ori_width, ori_height = img.size  # Note: size is an attribute

    x1, y1, x2, y2 = bbox

    x1_ori = int(x1 / resize_input_width  * ori_width)
    x2_ori = int(x2 / resize_input_width  * ori_width)
    y1_ori = int(y1 / resize_input_height * ori_height)
    y2_ori = int(y2 / resize_input_height * ori_height)

    # Calculate the width and height of the cropping area
    w = x2_ori - x1_ori
    h = y2_ori - y1_ori

    # If the aspect ratio exceeds 150, return the original image
    if w >= h * 150 or h >= w * 150:
        return img  # Do not crop, return the original image

    # Otherwise, perform cropping
    cropped = img.crop((x1_ori, y1_ori, x2_ori, y2_ori))
    return cropped


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim==1 for t in tensor_list])
    max_len=max([t.size(0) for t in tensor_list])
    padded_tensor_list=[]
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id]*(max_len-t.size(0)),device=t.device,dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list,dim=dim)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            limit_mm_per_prompt={'image': 2}
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.max_pixels = config.max_pixels
        self.min_pixels = config.min_pixels

        # add tokenizer
        self.tokenizer = tokenizer

        # add processor
        import os
        self.processor = hf_processor(model_path)

        # get valid grounding coordinates
        self.sub_image_token_prompt = config.sub_image_token_prompt
        
        # get invalid grounding coordinates
        self.ori_image_token_prompt = config.ori_image_token_prompt


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        input_height = prompts.batch['input_height']
        input_width = prompts.batch['input_width']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        input_prompt_generation_mask = torch.zeros_like(idx, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        ##### Initialization #####
        vllm_inputs = [] # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [PIL.Image]
        multi_turn_response_mask = [] # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = [] # B*R, list of int, record first round prompt of all trajs

        n = 1 if prompts.meta_info.get('validate', False) else self.config.n  # TODO: for validate, do_sample=False
        do_sample = prompts.meta_info.get('do_sample', True)
        if self.config.n > 1 and do_sample:
            input_height = _repeat_interleave(input_height, self.config.n)
            input_width = _repeat_interleave(input_width, self.config.n)

        if 'multi_modal_data' in non_tensor_batch:
            _multi_modal_data_list = non_tensor_batch['multi_modal_data']
            for raw_prompt_ids, multi_modal_data, image_path, question, choices in zip(non_tensor_batch.pop('raw_prompt_ids'), _multi_modal_data_list, non_tensor_batch['image_path'],non_tensor_batch['question'],non_tensor_batch.pop('choices')):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    # NOTE: use deepcopy to seperate variables
                    vllm_inputs.append(
                        {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': deepcopy(multi_modal_data), 'image_path': deepcopy(image_path), 'question': deepcopy(question), 'choices': deepcopy(choices),} # raw_prompt_ids: list
                    )
                    multi_turn_response_mask.append(
                        [torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)] # USER, Mark as 0
                    ) # [torch.Tensor(prefix_length,)]
                    prefix_prompt_lengths.append(
                        prefix_length
                    )
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        to_generate = list(range(batch_size*n))  # B*R, all trajs' index

        # users can customize different sampling_params at different run
        with self.update_sampling_params(n=1):  # TODO: for validate, do_sample=False
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
                _token_ids = output.outputs[sample_id].token_ids
                filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                if 151645 not in filtered_token_ids: 
                    # replace the last token with <|im_end|> if no <|im_end|> in response,
                    # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                    filtered_token_ids[-1] = 151645

                response.append(filtered_token_ids)

        # attach model responses to vllm_inputs
        assert len(to_generate)==len(response)
        response_filter = []
        
        for i_gen, response_ in zip(to_generate, response): 
            # update conversation
            response_ = list(response_)                
            
            # [SEARCH TRIGGER] We check model's last turn response, if not any <xxx_search>, then remove this traj from to_generate
            decoded_resp_ = self.tokenizer.decode(response_)
            filter_response, hit, bbox_list = truncate_at_json_bbox(decoded_resp_)
            filter_token = self.tokenizer.encode(filter_response)
            if filter_token[-1] != 151645:
                filter_token.append(151645)
            response_filter.append(filter_token)
            vllm_inputs[i_gen]['prompt_token_ids'] += filter_token
            input_height_image = input_height[i_gen]
            input_width_image = input_width[i_gen]
            multi_turn_response_mask[i_gen].append(torch.ones(len(filter_token), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1

            if hit:
                # Extract and convert coordinates
                x1, y1, x2, y2 = map(int, bbox_list[:4])
                # Validity check
                valid = (
                    x1 < x2 and y1 < y2 and  # Top-left corner must be smaller than bottom-right
                    x1 >= 0 and y1 >= 0 and  # Coordinates must be non-negative
                    x2 <= input_width_image and y2 <= input_height_image and  # Must not exceed image boundaries
                    (x2 - x1) > 28 and (y2 - y1) > 28 and  # Size must be no smaller than 28
                    max((x2 - x1) / (y2 - y1), (y2 - y1) / (x2 - x1)) < 100  # Avoid extreme aspect ratios
                )
                if not valid:
                    hit = False

            if not hit:
                # Use the full image
                bbox_list = [0, 0, input_width_image, input_height_image]
                sub_image_token = self.tokenizer.encode(
                    self.ori_image_token_prompt.format(
                        question=vllm_inputs[i_gen]['question'],
                        choices=vllm_inputs[i_gen]['choices']
                    )
                )
                # Append the original image
                vllm_inputs[i_gen]['multi_modal_data']['image'].append(
                    vllm_inputs[i_gen]['multi_modal_data']['image'][0]
                )
            else:
                # Use the cropped image
                # bbox_list[:4] = [x1, y1, x2, y2]  # This line can be omitted
                sub_image_token = self.tokenizer.encode(
                    self.sub_image_token_prompt.format(
                        question=vllm_inputs[i_gen]['question'],
                        choices=vllm_inputs[i_gen]['choices']
                    )
                )
                sub_image = crop_image(
                    vllm_inputs[i_gen]['image_path'],
                    input_height_image,
                    input_width_image,
                    bbox_list
                )
                vllm_inputs[i_gen]['multi_modal_data']['image'].append(
                    process_image(
                        sub_image,
                        max_pixels=self.max_pixels,
                        min_pixels=self.min_pixels
                    )
                )

            # Concatenate token and mask
            vllm_inputs[i_gen]['prompt_token_ids'] += sub_image_token
            multi_turn_response_mask[i_gen].append(
                torch.zeros(len(sub_image_token), dtype=attention_mask.dtype, device=attention_mask.device)
            )

        idx_to_gen = []
        for i in to_generate:
            idx_to_gen.append(vllm_inputs[i])
        with self.update_sampling_params(n=1):  # TODO: for validate, do_sample=False
            outputs = self.inference_engine.generate(
                prompts=idx_to_gen,  # list of dict
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
                _token_ids = output.outputs[sample_id].token_ids
                filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                if 151645 not in filtered_token_ids: 
                    # replace the last token with <|im_end|> if no <|im_end|> in response,
                    # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                    filtered_token_ids[-1] = 151645
                response.append(filtered_token_ids)

        for i_gen, response_ in zip(to_generate, response): 
            # update conversation
            response_ = list(response_)                
            vllm_inputs[i_gen]['prompt_token_ids'] += response_
            multi_turn_response_mask[i_gen].append(torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1

        # regenerate response
        response = [] # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = [] # B'*R, torch.Tensors with unequal lengths but align with 'response'
        for i_ in range(batch_size*n): # 0~15, 4*4
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            resp_mask_device = all_response_masks.device
            first_round_prompt_length = prefix_prompt_lengths[i_]
            response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

            # NOTE: [For Multi-Image] Update response_after_prompt(list of token_ids) and all_response_masks if search tool returned images
            if len(vllm_inputs[i_]['multi_modal_data']['image']) > 1:
                searched_image_inputs = self.processor.image_processor(vllm_inputs[i_]['multi_modal_data']['image'][1], return_tensors='pt') # dict_keys(['pixel_values', 'image_grid_thw'])
                searched_image_grid_thw = searched_image_inputs['image_grid_thw']
                if searched_image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index, image_pad_token, magic_num = 0, 151655, 654321
                    all_response_masks = all_response_masks.tolist() # for convenient modification
                    while image_pad_token in response_after_prompt:
                        # find pos of <|image_pad|>
                        pos = response_after_prompt.index(image_pad_token)
                        replicate_count = searched_image_grid_thw[index].prod() // merge_length
                        # update response_after_prompt
                        response_after_prompt[pos:pos+1] = [magic_num] * replicate_count
                        # update all_response_masks
                        all_response_masks[pos:pos+1] = [0] * replicate_count
                        index += 1
                    response_after_prompt = [image_pad_token if x == magic_num else x for x in response_after_prompt]
                    # print(response_after_prompt)
                    all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=resp_mask_device)
            response_generation_mask.append(all_response_masks) # at least we have single-turn conversation
            all_response = torch.tensor(response_after_prompt, device=idx.device,dtype=idx.dtype)
            response.append(all_response)
            assert response[i_].shape[0] == response_generation_mask[i_].shape[0], f"shape mismatched | response[i_]: {response[i_].shape[0]} | response_generation_mask[i_]: {response_generation_mask[i_].shape[0]}"
        assert len(response)==len(response_generation_mask), "length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim)==response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.response_length)
        if response.shape[1] > self.config.response_length_total:
            response = response[:,:self.config.response_length_total]
            response_generation_mask = response_generation_mask[:,:self.config.response_length_total]
        elif response.shape[1] < self.config.response_length_total:
            response = pad_sequence_to_length(response, self.config.response_length_total, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.response_length_total, 0)

        # All for 1st USER prompt
        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            # NOTE: We repeat 'multi_modal_data'
            if 'multi_modal_data' in non_tensor_batch.keys():
                repeated = []
                _index_br = 0
                for item in non_tensor_batch['multi_modal_data']:
                    for _ in range(self.config.n):
                        new_item = copy.deepcopy(item)
                        if len(vllm_inputs[_index_br]['multi_modal_data']['image']) > 1:
                            new_item['image'] += [vllm_inputs[_index_br]['multi_modal_data']['image'][1]]
                        repeated.append(new_item)
                        _index_br += 1
                non_tensor_batch['multi_modal_data'] = np.array(repeated)

            input_prompt_generation_mask = _repeat_interleave(input_prompt_generation_mask, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0
            non_tensor_batch['image_path'] = _repeat_interleave(non_tensor_batch['image_path'], self.config.n)
            non_tensor_batch['question'] = _repeat_interleave(non_tensor_batch['question'], self.config.n)            

        seq = torch.cat([idx, response], dim=-1) # (B*R, max_prompt_length+max_response_length_total)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        # NOTE: .contiguous() for broadcast
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
                "input_height": input_height,
                "input_width": input_width, 
            },
            batch_size=batch_size
        )

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
