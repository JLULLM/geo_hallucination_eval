import re
from pathlib import Path as p
import pandas as pd
import random
random.seed(42)
from abc import ABC, abstractmethod

import sys
self_path = '/workspace/linjh'
sys.path.append(self_path)
from utils import nlp_tools as nlp
import os



# ==============================================================================
# Aim: Convert opendata to contain three columns: 'question', 'answer', 'image_path', which will be used in tar file generation.
#      Not that this just process q,a,image_path, process_cot_tar will do tar file generation.
# input: dataset_name.jsonl and image_path information
# output: norm_dataset_name.jsonl
# ==============================================================================



# 封装成一个抽象类，然后继承实现这三个方法
def abstract_methods(*method_names):
    def decorator(cls):
        for method_name in method_names:
            setattr(cls, method_name, abstractmethod(getattr(cls, method_name)))
        return cls
    return decorator

@abstract_methods('get_image_path', 'get_question', 'get_answer')
class OpenDataProcessor(ABC):
    def get_image_path(self, input_sample):
        raise NotImplementedError

    def get_question(self, input_sample):
        raise NotImplementedError

    def get_answer(self, input_sample):
        raise NotImplementedError

    def filter_no_image(self, input_df):
        print('Before filtering no image, data shape:', input_df.shape)
        input_df.drop(input_df[input_df['image_path'] == 'no_image'].index, inplace=True)
        print('After filtering no image, data shape:', input_df.shape)
        return input_df
    
    def filter_wired_answer(self, input_df):
        print('Before filtering wired answer, data shape:', input_df.shape)
        input_df.drop(input_df[input_df['answer'] == ''].index, inplace=True)
        print('After filtering wired answer, data shape:', input_df.shape)
        return input_df
    
    def process_dataframe(self, input_df):
        input_df['image_path'] = input_df.apply(self.get_image_path, axis=1)
        self.filter_no_image(input_df)
        input_df['question'] = input_df.apply(self.get_question, axis=1)
        input_df['answer'] = input_df.apply(self.get_answer, axis=1)
        self.filter_wired_answer(input_df)
        return input_df
    
    def __call__(self, input_df, desc=None):
        if desc: nlp.richf(desc)
        return self.process_dataframe(input_df)
        
    


#### ============================================== ####
#### 1. scalequest(query, response, id(i add this)) ####
#### ============================================== ####
dataset = 'None'
if dataset == 'scalequest_keyword_dontcare':
    input_file = '/workspace/linjh/CoT_Factory/MLLM-from-scratch/assets/hf/ScaleQuest-Math/train_with_id.jsonl'  # head => tmp.jsonl
    input_df = pd.read_json(input_file, lines=True)

    instru_list = p('/workspace/linjh/CoT_Factory/assets/instruction_for_scalequest.txt').read_text().split('\n')
    nlp.richf(f'Instruction list length: {len(instru_list)}')

    class ScaleQuestProcessor(OpenDataProcessor):
        def __init__(self):
            self.image_basedir = p('/workspace/linjh/all_assets/ScaleQuest/img')
            self.img_token = '<|ZP_MM_PLH=default|>'

        def get_image_path(self, input_sample):
            cur_id = input_sample['id']
            raw_image_path = self.image_basedir / f'{cur_id}.png'
            if raw_image_path.exists():
                return str(raw_image_path)
            its_latex_image_path = self.image_basedir / f'{cur_id}_latex.png'
            if its_latex_image_path.exists():
                return str(its_latex_image_path)
            return 'no_image'

        def get_question(self, input_sample):
            # most important part: add img_token and random instru
            
            # 5% question = img_token
            # 95 question = img_token + random instru
            if random.random() < 0.05:
                return self.img_token
            else:
                rand_instr = random.choice(instru_list)
                return f'{self.img_token}{rand_instr}'
        
        def get_answer(self, input_sample):
            return input_sample['response']

    scale_quest_processor = ScaleQuestProcessor()
    scale_quest_processor(input_df, desc='Processing ScaleQuest data')
    output_norm_file = input_file.replace('train_with_id.jsonl', 'train_vlm_raw.jsonl')
    input_df.to_json(output_norm_file, lines=True, orient='records')
    nlp.richf(f'Processed data saved to {output_norm_file}')




#### ============================================== ####
####    2. llava reasoner (contain several split)   ####
#### ============================================== ####
dataset = 'None'
if dataset == 'llava_reasoner_sft_keyword_dontcare':
    input_dir = p('/workspace/linjh/MLLM-from-scratch/assets/hf/llava_reasoner_sft/sft')
    items = [x for x in input_dir.glob('*/*.jsonl') if 'direct' not in str(x) and 'text' not in str(x)]

    df_list = []
    for item in items:
        df = pd.read_json(item, lines=True)
        df['source'] = item.name  # add source column
        df_list.append(df)
    input_df = pd.concat(df_list, ignore_index=True)


    class LlavaReasonerSFTProcessor(OpenDataProcessor):
        def __init__(self):
            self.image_basedir = p('/workspace/linjh/MLLM-from-scratch/assets/hf/llava_reasoner_sft/image_data')
            self.user_tag = 'human'
            self.assitant_tag = 'gpt'
            self.img_token = '<|ZP_MM_PLH=default|>'

        def get_image_path(self, input_sample):
            image_path = self.image_basedir / input_sample['image']
            return str(image_path) if image_path.exists() else 'no_image'

        def get_question(self, input_sample):
            """⭐️ 其没有自己的img_token，所以直接返回问题"""
            assert input_sample['conversations'][0]['from'] == self.user_tag
            return f"{self.img_token}{input_sample['conversations'][0]['value']}"
            
        
        def get_answer(self, input_sample):
            assert input_sample['conversations'][1]['from'] == self.assitant_tag
            return input_sample['conversations'][1]['value']

    llava_reasoner_sft_processor = LlavaReasonerSFTProcessor()
    llava_reasoner_sft_processor(input_df, desc='Processing Llava Reasoner SFT data')
    output_norm_file = input_dir.parent / 'llava_reasoner_sft_norm.jsonl'
    input_df.to_json(output_norm_file, lines=True, orient='records')
    nlp.richf(f'Processed data saved to {output_norm_file}')


#### ====================================================== ####
####    3. Rcot GeoMM (circle and poly, re.sub(<image>) )   ####
#### ====================================================== ####


dataset = 'None'
if dataset == 'geomm_aready_done':
    class GeoMMProcessor(OpenDataProcessor):
        def __init__(self):
            self.image_basedir = '/workspace/linjh/CoT_Factory/MLLM-from-scratch/assets/hf/R-CoT/Geometry_generation'
            self.user_tag = 'user'
            self.assitant_tag = 'assistant'
            self.img_token = '<|ZP_MM_PLH=default|>'

        def get_image_path(self, input_sample):
            # sft
            image_path = re.sub(r'Your_path/R-CoT-main/GeoMM', self.image_basedir, input_sample['image'][0])
            # caption
            ori_paddle_caption_dir = '/root/paddlejob/workspace/env_run/dlluo/dle/Geometry_generation'
            image_path = re.sub(rf'{ori_paddle_caption_dir}', self.image_basedir, image_path)
            if re.search(rf'{ori_paddle_caption_dir}', image_path):
                print('image_path:', image_path)
            return image_path if p(image_path).exists() else 'no_image'

        def get_question(self, input_sample):
            """⭐️ 其有自己的img_token，需要sub改掉"""
            assert input_sample['conversations'][0]['from'] == self.user_tag
            raw_question = input_sample['conversations'][0]['value']
            assert re.search(r'<ImageHere>', raw_question)
            question = re.sub(r'<ImageHere>\s*', self.img_token, raw_question).strip()
            return question
            
        def get_answer(self, input_sample):
            assert input_sample['conversations'][1]['from'] == self.assitant_tag
            return input_sample['conversations'][1]['value']

    caption_dir = p('/workspace/linjh/CoT_Factory/MLLM-from-scratch/assets/hf/R-CoT/caption')
    input_file_list = ['/workspace/linjh/CoT_Factory/MLLM-from-scratch/assets/hf/R-CoT/GeoMM.json', 
                    *list(caption_dir.glob('result_*.json'))]
    df_list = []
    for item in input_file_list:
        df = pd.read_json(item)
        df['source'] = p(item).name  # add source column
        df_list.append(df)
    input_df = pd.concat(df_list, ignore_index=True)


    geomm_processor = GeoMMProcessor()
    geomm_processor(input_df, desc='Processing GeoMM data')
    output_norm_file = caption_dir.parent / 'geomm_norm.jsonl'
    input_df.to_json(output_norm_file, lines=True, orient='records')
    nlp.richf(f'Processed data saved to {output_norm_file}')



#### ====================================================== ####
####    4. tiku 1104 (a textual data need render and cot)   ####
#### ====================================================== ####




    
class RenderTikuProcesser(OpenDataProcessor):
    def __init__(self):
        self.image_basedir = p('/workspace/linjh/all_assets/tiku_1104/img')
        self.aug_image_basedir = p('/workspace/linjh/all_assets/tiku_1104/aug_img')
        self.img_token = '<|ZP_MM_PLH=default|>'
        self.init_idx_chosen_aug(aug_percent=0.2)

    def get_image_path(self, input_sample):
        cur_id = input_sample['question_uuid']
        raw_image_path = self.image_basedir / f'{cur_id}.png'
        aug_image_path = self.aug_image_basedir / f'{cur_id}.png'
        if cur_id not in self.idx_chosen_aug:
            return str(raw_image_path)
        else:
            return str(aug_image_path)
    
    def init_idx_chosen_aug(self, aug_percent):
        done_aug_idx = [x.stem for x in self.aug_image_basedir.glob('*.png')]
        n_chosen = int(len(done_aug_idx) * aug_percent)
        assert n_chosen <= len(done_aug_idx), f"you don't have enough aug image: have {len(done_aug_idx)}, need {n_chosen}"
        done_aug_idx = random.sample(done_aug_idx, n_chosen)
        self.idx_chosen_aug = set(done_aug_idx)
        print(f'ten of idx_chosen_aug: {list(self.idx_chosen_aug)[:10]}')
        

    def get_question(self, input_sample):
        # most important part: add img_token and random instru
        question = input_sample['q_text']
        formatted_question = re.sub(r'\s*。\s*str', r'\nstr', question)
        formatted_question = re.sub(r'str\((.)\):str\((.*?)\);', r'\1: \2\n', formatted_question)
        formatted_question = formatted_question.strip()

        return f'{self.img_token}{formatted_question}'
    
    def get_answer(self, input_sample):
        raw_answer = input_sample['cot_result']
        if not raw_answer or raw_answer == '':
            return ''
        answer = nlp.Parser.match_any_in_paragraph(raw_answer) #  default is for markdown
        return answer



input_file = '/workspace/linjh/CoT_Factory/assets/input/tiku_1104/tiku_1104_out.jsonl'
input_df = pd.read_json(input_file, lines=True)
render_tiku_processor = RenderTikuProcesser()
render_tiku_processor(input_df, desc='Processing Tiku 1104 data')
output_name = 'tiku_1104_norm.jsonl'
output_norm_file = p(input_file).with_name(output_name)  # 被覆盖了，幸好只是多了东西
input_df.to_json(output_norm_file, lines=True, orient='records')
nlp.richf(f'Processed data saved to {output_norm_file}')

    
docstring = r"""
# 1. get norm data which could be put into tar file
python src/norm_open_data.py
# 2. put qa and image into tar file
python src/process_cot_tar.py
"""



