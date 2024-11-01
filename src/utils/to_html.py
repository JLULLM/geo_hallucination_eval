from collections import Counter
import re
import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm
from tarutils import get_orig_sample, get_image_file
# import sys
# self_path = '/workspace/linjh'
# sys.path.append(self_path)
from utils import nlp_tools as nlp
from pathlib import Path as p
import os

# ==============================================================================
# Aim: generate visiable html for original dataset or cot dataset
# input: original dataset jsonl file or cot dataset json file
# output: a html file with image, q & a, result
# ==============================================================================

def show_cot_result(res_file):
    show_df = pd.read_json(res_file)
    
    # just get image io, i do the rest in nlp.tools
    show_df['image'] = show_df['img_file_io_param'].apply(lambda x: get_image_file(**x))
    
    # 'conversations'
    # get qa
    show_df['q & a'] = show_df['conversations'].apply(lambda x: '<hr>'.join([item['text'] for item in x]))
    # drop some columns
    show_df = show_df.drop(columns=['uuid', 'media_map', 'metadata', 'img_file_io_param', 'conversations', 'status', 'parse_status'])
    show_df = show_df[['image', 'q & a', 'result', 'parse_choice_all']]
    nlp.TencentViz.to_html(show_df=show_df, output_html_path=res_file.replace('.json', '.html'), textual_cols = ['q & a', 'result', 'parse_choice_all'])


def test_show_cot_result(
    base_dir = '/workspace/linjh/cot_data_search/bak_only_q',
    pattern = '*_parse_choice.json'
):
    """Aready have img_file_io_param, conversations, result, parse_choice_all"""
    for item in p(base_dir).glob(pattern):
        if not os.path.exists(str(item).replace('.json', '.html')):
            show_cot_result(res_file=str(item))
            
def cnt_2_table(counter, col_name='state'):
    total = sum(counter.values())
    counter = counter.most_common(10)
    counter = {k:v for k, v in counter}
    table = PrettyTable()
    table.field_names = [col_name, "num", 'precent(%)']
    for k, v in counter.items():
        precent = f'{v / total * 100: .2f}'
        table.add_row([k, v, precent])
    # add a total row
    table.add_row(['total', total, '100'])
    print(table)
    

def analyze_meta_jsonl_data(meta_jsonl_data):
    # image count in each sample
    cnt = Counter([len(item['img_file_io_param']) for item in meta_jsonl_data])
    cnt_2_table(cnt, col_name='image count')
    ## user/assistant 的text中含有的<image>数量
    def get_user_img_count(role='user'):
        user_cnt = Counter()
        for item in meta_jsonl_data:
            user_conv = [x['text'] for x in item['conversations'] if x['role'] == role]
            img_count = sum([x.count('<image>') for x in user_conv])
            user_cnt[img_count] += 1
        cnt_2_table(user_cnt, col_name=f'{role} img count')
    get_user_img_count('user')
    get_user_img_count('assistant')
    
    ## how many answer with <image>, then ignore 解析
    meta_jsonl_data_with_img_in_assistant = \
        [item for item in meta_jsonl_data if '<image>' in item['conversations'][1]['text']]
    response_list = [item['conversations'][1]['text'].replace('<image>', '') for item in meta_jsonl_data_with_img_in_assistant]
    len_list = [nlp.Paragraph.count_words_mixed_language(x) for x in response_list]
    # 小于10个字的比例, 用table展示
    print(f'小于10个字的比例: {sum([x < 10 for x in len_list]) / len(len_list)}')
    cnt = Counter(
        dict(len_smaller_then_five = sum([x < 5 for x in len_list]),
            len_large_then_five = sum([x >= 5 for x in len_list]) )
                  )
    cnt_2_table(cnt, col_name='len of ans(w img)')
    
    # plot a hist and save it 
    import matplotlib.pyplot as plt
    plt.hist(len_list, bins=100)
    plt.savefig('response_len_hist.png')
    
    return 
        
    

def get_meta_jsonl_data(input_file_or_dir, sample_size=200, anal=False, pattern='*.jsonl', data_root_dir=None):
    """if your file is not original jsonl, you need to set data_root_dir, cause i will use it to get tarfile with [fileid in your jsonl file ===> secret_sub_gpt-4o-2024-08-06_cot_parsed """
    if os.path.isdir(input_file_or_dir):
        meta_jsonl_data = []
        file_list = list(p(input_file_or_dir).glob(pattern))
        
        if sample_size != -1:
            import random; random.seed(66); random.shuffle(file_list)
            file_list = file_list[:min(10, len(file_list))]
            
        for item in tqdm(file_list, desc=f'loading jsonl in {input_file_or_dir}'):
            meta_jsonl_data.extend(get_orig_sample(str(item), show_change=False, data_root_dir=data_root_dir))
        # don't work
        # each_one_func = lambda x: get_orig_sample(str(x), show_change=False, data_root_dir=data_root_dir)
        # meta_jsonl_data = nlp.Parrallel.process_parallel_in_all_case(file_list, max_workers=10, process_one=each_one_func, desc='loading jsonl...')
        if anal:
            analyze_meta_jsonl_data(meta_jsonl_data)
    else:
        meta_jsonl_data = get_orig_sample(input_file_or_dir, show_change=True, data_root_dir=data_root_dir)
        
    import random; random.seed(66); random.shuffle(meta_jsonl_data)
    if sample_size != -1:
        sample_size = min(sample_size, len(meta_jsonl_data))
        print(f'sample {sample_size} from {len(meta_jsonl_data)} in {input_file_or_dir}{f"/{pattern}" if os.path.isdir(input_file_or_dir) else ""}')
        meta_jsonl_data = meta_jsonl_data[:sample_size]
    return meta_jsonl_data

def show_html(meta_jsonl_data, output_html_path, textual_cols = ['q & a'], sample_size = 200, just_need_single_graph=True):
    # prepare something in html
    if os.path.exists(output_html_path):
        print(f'{output_html_path} already exists, skip')
        return
    if just_need_single_graph:
        meta_jsonl_data = [item for item in meta_jsonl_data \
            if isinstance(item['img_file_io_param'], dict) or len(item['img_file_io_param']) == 1]
        for x in meta_jsonl_data:
            x['img_file_io_param'] = x['img_file_io_param'][0] if isinstance(x['img_file_io_param'], list) else x['img_file_io_param']
    
    show_df = pd.DataFrame(meta_jsonl_data) if isinstance(meta_jsonl_data, list) else meta_jsonl_data
    show_df = show_df.sample(sample_size, random_state=42) if len(show_df) > sample_size else show_df
    
    show_df['q & a'] = show_df['conversations'].apply(lambda x: '<hr>'.join([item['text'] for item in x]))
    # if any img_file_io_param is not a list
    if not any([isinstance(x, list) for x in show_df['img_file_io_param']]):
        show_df['image'] = show_df['img_file_io_param'].apply(lambda x: get_image_file(**x))
    else:
        # 这批里有一个多图，就会转成多图处理逻辑
        # 混合后，img_file_io_param可能有list(dict),也有dict
        show_df['img_file_io_param'] = show_df['img_file_io_param'].apply(lambda x: [x] if not isinstance(x, list) else x)
        max_image = min(3, max(show_df['img_file_io_param'].apply(len)))
        for i in range(max_image):
            show_df[f'image_{i}'] = show_df['img_file_io_param'].apply(lambda x: get_image_file(**x[i]) if i < len(x) else None)
    # drop some columns not for html
    
    final_show_df = show_df.drop(columns=['uuid', 'media_map', 'metadata', 'img_file_io_param', 'conversations'])
    for x in ['cot_prompt', 'cot_status', 'ori_conversations', 'media_size', 'refine_prompt']:
        if x in final_show_df.columns:
            final_show_df = final_show_df.drop(columns=[x])
    nlp.TencentViz.to_html(show_df=final_show_df, 
        output_html_path = output_html_path,
        textual_cols = textual_cols,
        merge_cols=['image', 'q & a']
    )
    return show_df


def show_open_source_data(dataset_name='scalequest'):
    name2path = {
        'scalequest' : '/workspace/linjh/MLLM-from-scratch/assets/hf/ScaleQuest-Math/train.json',
        # /workspace/linjh/CoT_Factory/MLLM-from-scratch/assets/hf/R-CoT/GeoMM.json
        'R-CoT' : '/workspace/linjh/CoT_Factory/MLLM-from-scratch/assets/hf/R-CoT/GeoMM.json'
    }
    if dataset_name not in name2path:
        raise ValueError(f'{dataset_name} not in {name2path.keys()}')
    if dataset_name == 'scalequest':
        df = pd.read_json(name2path[dataset_name], lines=True)
        print(df.shape)
        # 去掉response按照'\n'切分最大的行长度>100的样本
        df = df[df['response'].apply(lambda x: max([len(y) for y in x.split('\n')]) < 100)]
        opendata_to_html(df, output_html_path=f'/workspace/linjh/MLLM-from-scratch/assets/hf/ScaleQuest-Math/train.html', textual_cols=['query', 'response'])
    elif dataset_name == 'R-CoT':
        df = pd.read_json(name2path[dataset_name])
        print(df.shape)
        path_corr = lambda x: '/workspace/linjh/MLLM-from-scratch/assets/hf/R-CoT/Geometry_generation/' + re.sub('^Your_path/R-CoT-main/GeoMM/', '', x[0])  # first 
        df['image'] = df['image'].apply(path_corr)
        # check 有多少图片不在
        print(f'有多少图片不在: {sum([not os.path.exists(x) for x in df["image"]])}')
        # get q & a from conversations
        assert all([x[0]['from'] == 'user' and x[1]['from'] == 'assistant' for x in df['conversations']])
        df['q & a'] = df['conversations'].apply(lambda x: '<hr>'.join([item['value'] for item in x]))
        df.drop(columns=['conversations'], inplace=True)
        opendata_to_html(df, output_html_path=f'/workspace/linjh/MLLM-from-scratch/assets/hf/R-CoT/GeoMM_global_sample_100.html', textual_cols=['q & a'])

        
        
    
def opendata_to_html(show_df, output_html_path, textual_cols = ['q & a'], sample_size = 100):
    # 
    # sample
    show_df = show_df.sample(sample_size, random_state=66) if len(show_df) > sample_size else show_df
    nlp.TencentViz.to_html(show_df=show_df,
        output_html_path = output_html_path,
        textual_cols = textual_cols,
    )
    return show_df
    
    
 
def show_train_data(input_file_or_dir, output_dir, sample_size=30):
    
    if os.path.isdir(input_file_or_dir):
        cur_stem = p(input_file_or_dir).parent.name
        if cur_stem == 'MetaFiles':  cur_stem = p(input_file_or_dir).parent.parent.name
        output_html_path = p(output_dir) / f'global_random_{sample_size}_{cur_stem}.html'
    else:
        output_html_path=p(output_dir) / (p(input_file_or_dir).stem + '.html')
        
    if os.path.exists(output_html_path):
        print(f'{output_html_path} already exists, skip')
        return
    else:
        nlp.richf(f'Loading {sample_size} samples from {input_file_or_dir}')
        meta_jsonl_data = get_meta_jsonl_data(input_file_or_dir, sample_size=sample_size)
        nlp.richf(f'Done loading {len(meta_jsonl_data)} samples from {input_file_or_dir}')
        return show_html(meta_jsonl_data, output_html_path)


def process_func(meta_jsonl_data):
    for item in meta_jsonl_data:
        if 'parsed_cot' in item:
            item.update(
                original_question = None, original_answer = None, sub_questions = None, sub_answers = None
            )
        try:
            item.update(**item['parsed_cot'])
            item.pop('parsed_cot') 
        except:
            pass
        

def show_gen_cot_data(input_file_or_dir_or_df, output_html_path=None, textual_cols=['q & a', 'cot_result', 'verify_result'], sample_size=200, pattern='*.jsonl', process_func = process_func, 
                      data_root_dir = None):
    """after we merge `cot_result`, `verify_result`, we want to show this two"""
    if isinstance(input_file_or_dir_or_df, str):
        input_file_or_dir = input_file_or_dir_or_df
        meta_jsonl_data = get_meta_jsonl_data(input_file_or_dir, sample_size=sample_size, pattern=pattern, data_root_dir=data_root_dir)
        input_df = meta_jsonl_data
        if output_html_path is None:
            output_html_path = input_file_or_dir.replace('.json', '.html') if input_file_or_dir.endswith('.json') \
                else str(p(input_file_or_dir) / f'global_sample_{sample_size}.html')
    else:
        input_df = input_file_or_dir_or_df
        assert output_html_path.endswith('.html'), 'when input is a df, output_html_path must end with .html'
    
    # 支持动态生成一些列
    if process_func: 
        process_func(input_df)
    return show_html(input_df, output_html_path, textual_cols=textual_cols, sample_size=sample_size)
        



if __name__ == '__main__':
    args = nlp.Getter.get_args(
        input_file_or_dir='/secret/secret_sub/MetaFiles/secret_sub.jsonl',
        output_dir='/workspace/linjh/CoT_Factory/assets/anal/subject'
    )
    
    # 测试三个开源数据集
    # show_open_source_data(
    #     # dataset_name='scalequest'
    #     dataset_name='R-CoT'
    # )
    
    """展示训练数据"""
    show_train_data(
        input_file_or_dir=args.input_file_or_dir,
        # input_file_or_dir='/secret/secret_sub/MetaFiles/', 
        output_dir=args.output_dir
    )
    
    """展示学科训练数据 进一步精化 生成的cot数据"""
    # to use this, you need to register your col into textual_cols
    # show_gen_cot_data(
    #     input_file_or_dir_or_df='s',
    #     textual_cols = ['q & a', 'cot_result', 'original_question', 'original_answer', "sub_questions", "sub_answers"],
    #     output_html_path=None,
    #     pattern='*_cot_parsed.json',
    #     data_root_dir = '/secret/secret_sub',
    # )

    
