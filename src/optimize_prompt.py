from copy import deepcopy
import random
import time

from tqdm import tqdm
random.seed(42)
import sys
self_path = '/workspace/linjh'
sys.path.append(self_path)
from utils import nlp_tools as nlp
import os
import pandas as pd
import re
from pathlib import Path as p
from chatapi_server import ChatAPI
from template import get_cur_template
from collections import Counter
from prettytable import PrettyTable
from cot_2_verify import get_verify_succ_rate
from tarutils import get_orig_sample, get_image_file


global_prompt_id = 1
class Prompt:
    def __init__(self, template, score=None, evaluation_record_df=None, prompt_id=None):
        self.template = template
        self.evaluation_result = dict(
            score = score,
            evaluation_record_df = evaluation_record_df,
        )
        if prompt_id is None:
            global global_prompt_id
            prompt_id = global_prompt_id
            global_prompt_id += 1
        self.prompt_id = prompt_id
        
        
        
        
args = nlp.Args.get_args(optimize_step=3, 
                         init_prompt_template='./assets/output/optim_prompt/init_prompt.txt', 
                         init_prompt_evaluation_record='./assets/output/cot/only_q/v1017/benchmark_geo3k_000000000_gpt-4o-2024-08-06_cot_verify.json',
                         model_name='gpt-4o-2024-08-06',
                         max_workers=30,
                         sample_test_size=200,
                         beam_seach_size=2,
                         bfs_size=2,)
cur_prompt_template = p(args.init_prompt_template).read_text()
cur_prompt_evaluation_record_df = pd.read_json(args.init_prompt_evaluation_record).sample(args.sample_test_size, random_state=42).to_dict(orient='records')  #  有抖动，尽可以越大越好
cur_prompt = Prompt(cur_prompt_template, None, cur_prompt_evaluation_record_df, prompt_id=args.init_prompt_template)

meta_jsonl_data = []
rm_cols = ["cot_status", "cot_result", "cot_prompt", "verify_status", "verify_result", "verify_tag"]
for item in cur_prompt_evaluation_record_df:
    item_copy = deepcopy(item)
    for col in rm_cols:
        item_copy.pop(col, None)
    meta_jsonl_data.append(item_copy)



chatapi = ChatAPI(model_name=args.model_name)
o1_chatapi = ChatAPI(model_name='o1-mini-2024-09-12')

def cot_reason_and_verify(prompt, meta_jsonl_data):
    cur_df_data = deepcopy(meta_jsonl_data)
    gen_cot_template = prompt.template
    verify_template = get_cur_template(prompt_class=args.prompt_class, prompt_version=args.prompt_version)
    
    def merge_cot_result(item, log=False):
        assert item['conversations'][0]['role'] == 'user' and item['conversations'][1]['role'] == 'assistant'
        question, answer = item['conversations'][0]['text'], item['conversations'][1]['text']
        input_info = {'question': question, 'answer': answer, **item}
        prompt = gen_cot_template.format(**input_info)
        if log and random.random() < 0.1:
            print(f'prompt: {prompt}')
            print(f'img_file_io_param: {item["img_file_io_param"]}')
        img_bytes_io_arr = get_image_file(**item['img_file_io_param'])
        status, result = chatapi.get_gpt_response("chatgpt", prompt=prompt, image=img_bytes_io_arr, model=args.model_name)
        
        # save
        item.update({'cot_status': status, 'cot_result': result, 'cot_prompt': prompt})
        return '' # no_need_return_in_this_case
    
    nlp.Parrallel.process_parallel_in_all_case(item_list = cur_df_data, max_workers=args.max_workers, process_one=merge_cot_result, model_name=args.model_name)
    
    def merge_verify_result(item):
        cot_result = item['cot_result']
        # cot_result = parse_conclusion(cot_result)
        question, answer = item['conversations'][0]['text'], item['conversations'][1]['text']
        verify_prompt = verify_template.format(cot_result=cot_result, question=question, answer=answer)
        verify_status, verify_result = chatapi.get_gpt_response("chatgpt", prompt=verify_prompt, model=args.model_name)
        item.update({'verify_status': verify_status, 'verify_result': verify_result})
        return ''

    nlp.Parrallel.process_parallel_in_all_case(cur_df_data, max_workers=args.max_workers, process_one=merge_verify_result, model_name=args.model_name)
    prompt.evaluation_result['evaluation_record_df'] = cur_df_data
    print(f'Finish cot_reason_and_verify for a prompt')
    
    
def eval_prompt(prompt):
    if prompt.evaluation_result['evaluation_record_df'] is None:
        cot_reason_and_verify(prompt, meta_jsonl_data)
    if prompt.evaluation_result['score'] is None:
        counter, table = get_verify_succ_rate(verify_file_path = None, verify_df = prompt.evaluation_result['evaluation_record_df'],
                            prompt_id = prompt.prompt_id, return_table=True)
        prompt.counter, prompt.table = counter, table
        prompt.evaluation_result['score'] = counter['True'] / sum(counter.values())





# log the initial prompt
eval_prompt(cur_prompt)

beam_seach_size = args.beam_seach_size
bfs_size = args.bfs_size
candicate_prompts = [cur_prompt]
optimize_template = get_cur_template(prompt_class='optimize_prompt', prompt_version='v1018')


def generate_new_optimize_prompts(prompt, bfs_size):
    """初步写成是badcase驱动优化，后期再加上goodcase也驱动优化，这样可以更好的保证优化的方向是对的。"""
    # 1. get the bad case
    n_bad_case = 1
    bad_case = random.sample([x for x in prompt.evaluation_result['evaluation_record_df']
                              if x['verify_tag'] == 'False'], n_bad_case)
    # 2. generate new optimize prompts based on the bad case
    bad_case[0]['question'] = bad_case[0]['conversations'][0]['text']
    bad_case[0]['CoT_prompt_template'] = prompt.template
    optimize_prompt = optimize_template.format(**bad_case[0])
    # status, new_prompt_template_response = chatapi.get_gpt_response("chatgpt", prompt=optimize_prompt, model=args.model_name)
    # add retry 5 times, each time sleep 1s
    cur_new_optimize_prompts = []
    
    for i in tqdm(range(bfs_size), desc='generate_new_optimize_prompts for a prompt'):
        status = False
        for i in range(5):
            status, new_prompt_template_response = o1_chatapi.get_gpt_response("chatgpt", prompt=optimize_prompt, model='o1-mini-2024-09-12')
            if status:
                cur_new_optimize_prompts.append(new_prompt_template_response)
                break
            time.sleep(1)
        else:
            print(f'Failed to get response for optimize_prompt')
        
    
    # new_prompt_template_response like ```markdown\n{{new CoT_prompt}}```
    def decorate_new_prompt_template(new_prompt_template_response):
        new_prompt_template = new_prompt_template_response.strip('`').strip('markdown').strip()
        new_prompt = Prompt(new_prompt_template, None, None)
        return new_prompt
    cur_new_optimize_prompts = [decorate_new_prompt_template(x) for x in cur_new_optimize_prompts]
    print(f'### Finish generate_new_optimize_prompts for a prompt')
    print(f'original prompt: \n{prompt.template}')
    for cur_new_prompt in cur_new_optimize_prompts:
        print(f'##### new prompt {cur_new_prompt.prompt_id}: \n{cur_new_prompt.template}')
    return cur_new_optimize_prompts
    
    
    
    
for i in range(args.optimize_step):
    print(f'########## Optimize step {i+1}\n' * 10)
    new_optimize_prompts = []
    # 1. generate 10 new optimize prompts based on the cur prompt and its evaluation result
    for prompt in candicate_prompts:
        new_optimize_prompts += generate_new_optimize_prompts(prompt, bfs_size)  # cur_prompt should be a class including prompt, evaluation result, etc.
    
    # 2. evaluate the 10 new optimize prompts
    for new_optimize_prompt in new_optimize_prompts:
        eval_prompt(new_optimize_prompt)
    
    # 3. choose the best prompt as `cur prompt`` for next iteration
    new_optimize_prompts += candicate_prompts
    new_optimize_prompts.sort(key=lambda x: x.evaluation_result['score'], reverse=True)
    candicate_prompts = new_optimize_prompts[:beam_seach_size]

output_text = ''
for top, candi_prompt in enumerate(candicate_prompts, start=1):
    output_text += f'### prompt Top{top}\n'
    output_text += f'prompt: \m{candi_prompt.template}\n'
    output_text += f'score: {candi_prompt.evaluation_result["score"]}\n'
    output_text += f'{candi_prompt.table}\n'
p(args.init_prompt_template.replace('init_prompt.txt', 'optim_prompt.txt')).write_text(output_text)
    