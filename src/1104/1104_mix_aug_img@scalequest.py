import re
from pathlib import Path as p
import numpy as np
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
# 现在的pipeline如右侧文件所示：scripts/1104_process_opensource.sh。  先norm，再tar。
# 所以最方便的mix时机就是形成norm文件后，直接修改norm文件，然后将新的norm文件直接tar即可
# ==============================================================================


input_file = '/workspace/linjh/CoT_Factory/MLLM-from-scratch/assets/hf/ScaleQuest-Math/train_vlm_raw.jsonl'  
output_mix_file = input_file.replace('train_vlm_raw', 'train_vlm_mix_20%')

input_df = pd.read_json(input_file, lines=True)
# 从有备选图片的数据中抽取总数的20%，修改图片为修改后的图片
n_row_will_aug = int(input_df.shape[0] * 0.2)
print(f'Will aug {input_df.shape[0]} * 20% = {n_row_will_aug} rows')

print(f'Check if we have augmented image')
have_aug_img = lambda image_path: p(image_path.replace('ScaleQuest/img', 'ScaleQuest/aug_img')).exists()
input_df['have_aug_img'] = input_df['image_path'].apply(have_aug_img)
print(f'We have {input_df["have_aug_img"].sum()} rows with augmented image')

row_idx_have_aug_img = input_df[input_df['have_aug_img']].index
n_row_will_aug = min(n_row_will_aug, len(row_idx_have_aug_img))
row_idx_will_aug = random.sample(row_idx_have_aug_img.tolist(), n_row_will_aug)
print(f'Start to augment {n_row_will_aug} rows')

# 将row_idx_will_aug对应的行的image_path修改为aug_img
# def change_image_path(row):
#     if row.name in row_idx_will_aug:
#         return row['image_path'].replace('ScaleQuest/img', 'ScaleQuest/aug_img')
#     return row['image_path']

# input_df['image_path'] = input_df.apply(change_image_path, axis=1)

row_idx_will_aug = set(row_idx_will_aug)
input_df = input_df.to_dict('records')

# apply不知道为什么很慢，我这里改成并发处理
def process_one(x):
    if x['image_path'] in row_idx_will_aug:
        x['image_path'] = x['image_path'].replace('ScaleQuest/img', 'ScaleQuest/aug_img')
    return None
nlp.Parrallel.process_parallel_in_all_case(item_list = input_df, max_workers=50, process_one=process_one, desc='Augmenting image')


# 保存新的jsonl文件
input_df.to_json(output_mix_file, orient='records', lines=True, force_ascii=False)
print(f'Save to {output_mix_file}')
# save row_idx_have_aug_img

output_row_idx_auged_file = input_file.replace('train_vlm_raw.jsonl', 'row_idx_auged.txt')
p(output_row_idx_auged_file).write_text('\n'.join(map(str, row_idx_will_aug)))



    



