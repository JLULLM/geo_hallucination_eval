import os
import pandas as pd
import sys
self_path = '/workspace/linjh'
sys.path.append(self_path)
from utils import nlp_tools as nlp
import re
from pathlib import Path as p
from chatapi_server import ChatAPI
from template import get_cur_template
from collections import Counter
from prettytable import PrettyTable


# ==============================================================================
# Aim: verify which cot is right and calculate succ rate of cot-data pipeline
# input: json file with llm response and right ans
# output: merge `verify_result`(long response), `verify_tag`(True/False/NoAnswer), and show the succ rate
# ==============================================================================

def get_verify_succ_rate(verify_file_path=None, verify_df=None, prompt_id = None, return_table=True):
    """need input_df have columns: verify_result, verify_status, 
    verify_tag will be merged into the input_df and save it into verify_file_path.
    At last, the succ rate of the verify_tag will be calculated and printed.
    """
    metric_file = None
    if verify_file_path is not None and verify_df is None:
        metric_file = verify_file_path.replace('.json', '.metric')
        if os.path.exists(metric_file):
            print(f'File {metric_file} already exists, skip the calculate succ rate.')
            content = p(metric_file).read_text()
            print(f'its content is:\n{content}')
        print(f'calculate succ rate in {verify_file_path}')
        verify_df = pd.read_json(verify_file_path).to_dict(orient='records')
    else:
        assert verify_df is not None, 'verify_file_path and verify_df can not be None at the same time.'
    
    counter = Counter()
    for item in verify_df:
        verify_result = item['verify_result']
        # \n\n -> \n.  print(verify_result)
        # verify_result, verify_reason = re.sub(r'\n+', '\n', verify_result).split('\n', 1)
        # '1. Output: xyz' => 'xyz'
        # verify_result = re.sub('\d.\s*Output:\s*', '', verify_result)
        
        parse_verify_result = nlp.Parser.match_item_in_paragraph(verify_result, prefix_str='[Oo]utput')
        if parse_verify_result in ['True', 'False', 'NoAnswer']:
            counter[parse_verify_result] += 1
        elif parse_verify_result == '':
            counter[f'verify_status_{item["verify_status"]}'] += 1
        else:
            counter['parse_wrong'] += 1
        item['verify_tag'] = parse_verify_result
        
    if verify_file_path is not None:
        if not os.path.exists(verify_file_path):
            pd.DataFrame(verify_df).to_json(verify_file_path, orient='records', indent=2, force_ascii=False)
            print(f'Save verify tag into {verify_file_path}')
        else:
            print(f'File {verify_file_path} exists, skip the save verify tag.')
    
    # ==== print
    table = PrettyTable()
    table.field_names = ["state", "num", 'precent(%)', 'num / num_True (%)']
    total = sum(counter.values())
    base = counter['True']
    tmp_list = []
    for k, v in counter.items():
        precent = f'{v / total * 100: .2f}'
        how_many_base = f'{v / base * 100: .2f}' if base != 0 else '0'
        tmp_list += [(k, v, precent, how_many_base)]
    tmp_list = sorted(tmp_list, key=lambda x: x[0], reverse=True)
    for k, v, precent, how_many_base in tmp_list:
        table.add_row([k, v, precent, how_many_base])
    
        
    print(f'#### prompt_id: {verify_file_path if verify_file_path is not None else prompt_id}\n' * 3)
    print(table)
    if verify_file_path is not None:
        if not os.path.exists(metric_file):
            with open(metric_file, 'w') as f:
                f.write(f'## {verify_file_path}' + '\n' + str(table))
            print(f'Save metric to {metric_file}')
        else:
            print(f'File {metric_file} already exists, skip the save metric.')
    if return_table:
        return counter, str(table)
    return counter



if __name__ == '__main__':
    args = nlp.Args.get_args(
            cot_result_col='cot_result', 
            model_name = "gpt-4",        
            modify_dir='./assets/output/cot/only_q/v1017',
            modify_file_suffix='_cot.json',  # will generate new_file_suffix base on modify_dir/modify_file_suffix
            new_file_suffix='_cot_verify.json',
            do_verify_whole_dir=False
    )


    chatapi = ChatAPI(model_name=args.model_name)
    cur_template = get_cur_template(args.prompt_class, args.prompt_version)

    def process_one(item):
        cot_result = item[args.cot_result_col]
        question, answer = item['conversations'][0]['text'], item['conversations'][1]['text']
        verify_prompt = cur_template.format(cot_result=cot_result, question=question, answer=answer)
        verify_status, verify_result = chatapi.get_gpt_response("chatgpt", prompt=verify_prompt, model=args.model_name)
        item.update({'verify_status': verify_status, 'verify_result': verify_result})
        return ''

    ## ==== start to verify
    def verify_each_file(json_data_path):
        verify_file_path = json_data_path.replace(args.modify_file_suffix, args.new_file_suffix)
        if not os.path.exists(verify_file_path):
            print(f'Verify {json_data_path}')
            json_data = pd.read_json(json_data_path).to_dict(orient='records')
            nlp.Parrallel.process_parallel_in_all_case(json_data, max_workers=args.max_workers, process_one=process_one, model_name=args.model_name)        
            get_verify_succ_rate(verify_file_path=verify_file_path, verify_df=json_data)  # will merge verify_tag into file
            pd.DataFrame(json_data).to_json(verify_file_path, orient='records', indent=2, force_ascii=False)
            print(f'Save to {verify_file_path}')
        else:
            print(f'File {verify_file_path} already exists, skip the Verify.')
            get_verify_succ_rate(verify_file_path)
        return
    
    if not args.do_verify_whole_dir:
        for item in p(args.modify_dir).glob('*' + args.modify_file_suffix):
            json_data_path = str(item)
            verify_each_file(json_data_path)
    else:
        assert args.input_file != 'input_file', 'input_file should be specified when do_verify_whole_dir=False'
        verify_each_file(args.input_file)        
    
    
    

    
