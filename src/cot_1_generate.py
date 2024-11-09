import re
from chatapi_server import ChatAPI
from selector import Selector
from tarutils import get_orig_sample, get_image_file
from pathlib import Path as p
import pandas as pd
import random
random.seed(42)
from template import get_cur_template

import sys
self_path = '/workspace/linjh'
sys.path.append(self_path)
from utils import nlp_tools as nlp
import os
from to_html import get_meta_jsonl_data, show_gen_cot_data

print = nlp.richf

# ==============================================================================
# Aim: generate cot reasoning process for the original short answer data
# input: tar json file with short answer data and image offset
# output: merge cot reasoning process 'result' into the original short answer data. TODO: 原始的多图情况时，不能很方便地还原对话回去(只能通过uuid)。之后打算在get_orig_sample中，将原始的对话信息等各种需要的信息也存储下来
# ==============================================================================

args = nlp.Args.get_args(prompt_class='only_q', prompt_version='v1017', sample_size=-1)
cur_template = get_cur_template(args.prompt_class, args.prompt_version)

# === for tar format, can't just pd.read_json()
if args.input_file != 'input_file':
    meta_jsonl_data = get_orig_sample(args.input_file, show_change=False)
else:
    # if from a file, --input_file x or --input_dir x is both ok, it will conveniently for shell, you just need to --input_dir
    meta_jsonl_data = get_meta_jsonl_data(args.input_dir, sample_size=args.sample_size)  


if args.is_debug or \
    (args.sample_size != -1 and args.sample_size != len(meta_jsonl_data)):
    print(f'Do sampling with sample_size={args.sample_size}')
    meta_jsonl_data = random.sample(meta_jsonl_data, args.sample_size)

selector = Selector()
meta_jsonl_data = selector(meta_jsonl_data, do_filter_multi_img=True, do_filter_single_question=False, do_filter_img2img=False)

chatapi = ChatAPI(model_name=args.model_name)


def process_one(item, log=False):
    assert item['conversations'][0]['role'] == 'user' and item['conversations'][1]['role'] == 'assistant'
    question, answer = item['conversations'][0]['text'], item['conversations'][1]['text']
    # re.sub  "Short answer."
    question = re.sub(r'\s*Short answer\.\s*', '', question)
    input_info = {'question': question, 'answer': answer, **item}
    prompt = cur_template.format(**input_info)
    if log and random.random() < 0.1:
        print(f'prompt: {prompt}')
        print(f'img_file_io_param: {item["img_file_io_param"]}')
    img_bytes_io_arr = get_image_file(**item['img_file_io_param'])
    status, result = chatapi.get_gpt_response("chatgpt", prompt=prompt, image=img_bytes_io_arr, model=args.model_name)
    
    # save
    item.update({'cot_status': status, 'cot_result': result, 'cot_prompt': prompt})
    return '' # no_need_return_in_this_case


if args.output_file != 'output_file' and not os.path.exists(args.output_file):
    nlp.Parrallel.process_parallel_in_all_case(item_list = meta_jsonl_data, max_workers=args.max_workers, process_one=process_one, model_name=args.model_name)
    pd.DataFrame(meta_jsonl_data).to_json(args.output_file, orient='records', indent=2, force_ascii=False)
    print(f'File {str(args.output_file)} generated.')
else:
    print(f'File {str(args.output_file)} already exists, skip the generation.')
    

if args.do_show_html:
    from to_html import show_gen_cot_data
    # 
    show_gen_cot_data(
        input_file_or_dir_or_df=args.output_file,
        textual_cols = ['q & a', 'cot_result'],
        sample_size=30,
        data_root_dir = p(args.input_file).parent.parent if args.input_file != 'input_file' else args.input_dir,
    )

    



