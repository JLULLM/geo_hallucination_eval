
import json
from pathlib import Path as p
import re
import pandas as pd
import random
random.seed(42)

import sys
self_path = '/workspace/linjh'
sys.path.append(self_path)
from utils import nlp_tools as nlp
import os


# ==============================================================================
# Aim: parse cot data in subject set
# input: file with cot_result
# output: merge 'parsed_cot'
# ==============================================================================

class Parser:
    @staticmethod
    def parse_response(content):
        if not content: return ''
        content = content.strip().strip('`')
        # 将开头的markdown
        content = re.sub(r'^markdown', '', content).strip()
        if '第一道小题' not in content:
            return content
        else:
            sub_answers = Parser.get_sub_answers(content)
            return dict(content = content, sub_answers = sub_answers)
    
    @staticmethod
    def get_sub_answers(content) -> list:
        sub_answers_list = ['']
        lines = content.split('\n')
        for line in lines:
            if line.startswith('## 第'):
                sub_answers_list.append(line)
            else:
                sub_answers_list[-1] += '\n' + line
        # filter. Each item should contain '### 题目解析', '### 知识点回忆', '### 逐步推理', '### 最终答案' 
        keys = ['### 题目解析', '### 知识点回忆', '### 逐步推理', '### 最终答案']
        is_qualify = lambda x: all([key in x for key in keys])
        sub_answers_list = [x for x in sub_answers_list if is_qualify(x)]
        return sub_answers_list


template_pool = dict(
    sub_questions = ['请解答第{x}小题。','请回答第{x}小题。', '请解答第{x}题。', '请回答第{x}题。', '看看第{x}小题。'],
    ori_question = ['请解答问题。', '请回答问题。', '请解答这个问题。', '请回答这个问题。', '请回答问题。']
)

get_filled_sub_questions = lambda n_sub_answers: [t.format(x=i+1) for i, t in enumerate(
        # random.sample(template_pool['sub_questions'], n_sub_answers)
        # Sample larger than population or is negative
        random.choices(template_pool['sub_questions'], k=n_sub_answers)
    )]
    
def parse_one_file(json_data_path, save_file):
    if os.path.exists(save_file):
        print(f'File {save_path} already exists, skip the parsing.')
        return
    with open(json_data_path, 'r') as f:
        data = json.load(f)
    for item in data:
        cot_result = item['cot_result']
        if not cot_result or cot_result == '':
            continue
        parsed_cot = Parser.parse_response(cot_result)
        
        parsed_obj = dict(f='', original_answer='',
            sub_questions=[], sub_answers=[])
        # 加入大问题
        original_answer = parsed_cot if isinstance(parsed_cot, str) else parsed_cot['content']
        original_question = random.choice(template_pool['ori_question'])
        parsed_obj.update(original_question=original_question, original_answer=original_answer)
        # 加入子问题
        if isinstance(parsed_cot, dict):
            sub_answers = parsed_cot['sub_answers']
            sub_questions = get_filled_sub_questions(len(sub_answers))
            parsed_obj.update(sub_questions=sub_questions, sub_answers=sub_answers)
        item['parsed_cot'] = parsed_obj
        # 后处理TODO：1. gpt会偷偷不回答后边的小题目（说同上可以推理得出） 2. 错误识别成多道题目

    with open(save_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'Save parsed cot to {save_file}')
    return data


args = nlp.Getter.get_args(
    # input_dir='/workspace/linjh/CoT_Factory/assets/output/cot_subject_1024/normal_and_multi',
    # do_parsed_whole_dir=True,
    # input_file='/workspace/linjh/CoT_Factory/assets/output/cot_subject_1024/normal_and_multi/xueke_000000_gpt-4o-2024-08-06_cot.json',
    do_parsed_whole_dir=False,
    modify_file_suffix='_cot.json',  # will generate new_file_suffix base on modify_dir/modify_file_suffix
    new_file_suffix='_cot_parsed.json'
)

if args.do_parsed_whole_dir:
    # i don't need data here
    for one_file in p(args.input_dir).glob('*' + args.modify_file_suffix):
        save_path = str(one_file).replace(args.modify_file_suffix, args.new_file_suffix)
        parse_one_file(one_file, save_path)
else:
    assert args.input_file != 'input_file', 'input_file should be specified when do_verify_whole_dir=False'
    save_path = args.input_file.replace(args.modify_file_suffix, args.new_file_suffix)
    parse_one_file(args.input_file, save_path)
    
# get a html
if args.do_show_html:
    from to_html import show_gen_cot_data
    show_gen_cot_data(
        input_file_or_dir_or_df=save_path,
        textual_cols = ['q & a', 'cot_result', 'original_question', 'original_answer', "sub_questions", "sub_answers"],
        sample_size=100,
        output_html_path=save_path.replace('.json', '.html'),
        data_root_dir = '/workspace/image_sft/datav20240920/SFT/Subject/xueke_0927',
    )




                         