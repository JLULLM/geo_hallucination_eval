import markdown
from rich.traceback import install
from rich.console import Console
from rich.table import Table
install(show_locals=True)
from rich import print
from pathlib import Path
from typing import List, Tuple, Dict
import concurrent
import warnings
import jieba
import os
import re
from collections import Counter, defaultdict
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1024)
from tqdm import trange, tqdm
import yaml
from pathlib import Path as p
######
import_docstring = """
import pandas as pd
import sys
sys.path.append('/home/MuseLLM/analyse_data')
from utils import nlp_tools as nlp
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
import pickle
import random
from pathlib import Path as p
"""
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

import logging
import argparse

emojis = [
    ":fire:", ":star:", ":heart:", ":smile:", ":tada:", ":sparkles:", ":confetti_ball:", ":balloon:", ":cake:", ":gift:",
    ":sun_with_face:", ":cloud_with_lightning_and_rain:", ":snowman:", ":umbrella:", ":leaves:", ":rose:", ":tulip:",
    ":four_leaf_clover:", ":cherry_blossom:", ":bouquet:"
]

console = Console()
def richf(message):
    random_emoji = random.choice(emojis)
    width = 100
    panel = Panel(f"{random_emoji} {message}".center(width),
                  title="linjh's message",
        title_align="left",
        border_style="yellow",
        padding=(1, 2),
        width=width,
        style=Style(color="green", bold=True, italic=True))
    console.print(panel)

class SuperLogger:
    def __init__(self, output_log_file, exclude_keywords=None):
        self.output_log_file = output_log_file
        self.exclude_keywords = exclude_keywords

        self.logger = logging.getLogger('linjh_logger')
        self.logger.setLevel(logging.INFO)

        # 防止重复添加处理器
        if not self.logger.hasHandlers():
            # 创建文件处理器
            file_handler = logging.FileHandler(output_log_file, mode='a', encoding='utf-8')
            # 创建控制台处理器
            stream_handler = logging.StreamHandler()

            # 创建格式器并设置格式
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'  # 设置简短的时间格式
            )
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            # 添加过滤器
            if self.exclude_keywords:
                keyword_filter = self.ExcludeKeywordFilter(self.exclude_keywords)
                file_handler.addFilter(keyword_filter)
                stream_handler.addFilter(keyword_filter)

            # 将处理器添加到 Logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

        self.print(f'Log file: {output_log_file}')
    
    def print(self, message):
        print(message)  
        self.logger.info(message)
    
    class ExcludeKeywordFilter(logging.Filter):
        def __init__(self, exclude_keywords):
            super().__init__()
            self.exclude_keywords = exclude_keywords

        def filter(self, record):
            return not any(keyword in record.getMessage() for keyword in self.exclude_keywords)
    
    
class Getter:
    """You don't need to remember sth or import sth, just use Getter.get_sth()"""
    @staticmethod
    def get_logger(output_log_file):
        return SuperLogger(output_log_file)
    
    @staticmethod
    def get_args(**kwargs):
        parser = SuperArgumentParser(**kwargs)
        args = parser.parse_args()
        return args
        
    
    
    
class SuperArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__()
        args_dict = dict(
            # ==== input ==== # 
            input_file='input_file',
            input_dir='input_dir',
            modify_dir='modify_dir',
            modify_file_suffix='_cot.json',  # will generate new_file_suffix base on modify_dir/modify_file_suffix
            new_file_suffix='_cot_verify.json',
            # ==== llm api ==== #
            model_name='gpt-4o-2024-08-06',
            max_workers=os.cpu_count(),
            sample_size=-1,
            prompt_class='only_q',
            prompt_version='v1017',
            # ==== output ==== #
            output_file= 'output_file',
            output_dir='output_dir',
            log_file='log_file',
            do_show_html=False,  # 【bug记录：不在这里的bool，只能默认值设定为0，命令行只能修改为True】
            # output_file='stick'  will take input_file as input
            output_suffix='.json',
        ) if 'brand_new' not in kwargs else {}
        if 'brand_new' in kwargs:
            kwargs.pop('brand_new')
        args_dict.update(kwargs)
        for key, value in args_dict.items():
            # if exist 'size' / 'num' / 'n_' in key, set type=int
            dtype = self.get_dtype(key)
            self.add_argument(f'--{key}', default=value, type = dtype)  # 有bug：如果新定义一个bool，那么在命令行中无法修改其值，只能把他加到定理这里
        self.add_argument('--is_debug', action='store_true', default=False)
        args_dict['is_debug'] = False
        self.args_dict = args_dict
        # {output_dir}/{input_file}{model_name}{output_suffix} or {output_dir}/{time}{output_suffix}
        # this will convert modify, but for easy reading, i keep modify
        self.output_speical_tag = ['stick', 'stick_add_model', 'output_file']

    def set_output_file(self, args):
        if args.output_file.startswith('stick'):
            assert args.input_file != 'input_file'
            args.output_file = p(args.input_file).stem + '_' \
                    + (args.model_name if args.output_file == 'stick_add_model' else '') \
                    +  args.output_suffix
            args.output_file = str(p(args.output_dir) / args.output_file)
        elif args.input_file == 'input_file' and args.input_dir != 'input_dir':
            from datetime import datetime, timedelta
            formatted_time = (datetime.now() + timedelta(hours=8)).strftime("%m_%d_%H%M_%S")
            args.output_file = args.output_dir + formatted_time + args.suffix
            print(f'output_file: {args.output_file}')
            assert not os.path.exists(args.output_file)
            
        return args
    
    def get_dtype(self, key):
        if re.search(r'size|num|^n_|max|min|average|step', key):
            dtype = int
        elif re.search(r'is|do', key):
            dtype = bool
        elif re.search(r'list', key):
            dtype = list
        else:
            dtype = str
        return dtype
    
    def post_process(self, args):
        if args.output_file in self.output_speical_tag and args.output_dir != 'output_dir':
            args = self.set_output_file(args)
        return args
    
    def decor(self, path):
        if not isinstance(path, str) or len(path) < 40:
            return path
        else:
            each_dir = path.split('/')
            shorten_func = lambda x: x[:3] + '.' + x[-3:] if len(x) > 6 else x
            # unless the last one
            each_dir = [shorten_func(x) for x in each_dir[:-1]] + [each_dir[-1]]
            return '/'.join(each_dir)
        

        
    def parse_args(self, *args, **kwargs):
        parsed_args = super().parse_args(*args, **kwargs)
        
        parsed_args = self.post_process(parsed_args)
        if parsed_args.is_debug:
            parsed_args.sample_size = 10
        # import prettytable as pt
        # tb = pt.PrettyTable()
        # tb.field_names = ['Keys in Args', 'Value', 'Default Value']
        # for key, value in vars(parsed_args).items():
        #     tb.add_row([key if value == self.args_dict[key] else f'* {key} *',
        #                 self.decor(value), self.decor(self.args_dict[key])])
        # print(tb)
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        
        table.add_column("Keys in Args", justify="center", style="dim")
        table.add_column("Value", justify="center")
        table.add_column("Default Value", justify="center")
        
        for key, value in vars(parsed_args).items():
            display_key = key if value == self.args_dict[key] else f"* {key} *"
            table.add_row(display_key, str(self.decor(value)), str(self.decor(self.args_dict[key])))
        
        console.print(table)
        return parsed_args

class Args:
    @staticmethod
    def get_args(**kwargs):
        warnings.warn(
            "`Args.get_args` is deprecated and will be removed in future versions. Use `Getter.get_args` instead.",
            DeprecationWarning
        )
        parser = SuperArgumentParser(**kwargs)
        args = parser.parse_args()
        return args
    

"""
# 极致优化
That's all you need to do: `args = nlp.Args.get_args(cot_result_col='result'))`
# support modify default value
args = nlp.Args.get_args(model_name='gpt-4o-2024-08-01')
"""

import concurrent.futures
# 有时候报错：AttributeError: module 'concurrent' has no attribute 'futures'， 解决办法：将上述行加到目标python文件中，估计和rich冲突了？
class Parrallel:
    @staticmethod
    def process_parallel_in_all_case(item_list, max_workers, process_one, model_name='gpt', desc=None):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=len(item_list), desc=f"'{model_name}' 正在推理数据" if desc is None else desc)
            results_and_idx = []
            for i, item in enumerate(item_list):
                future = executor.submit(process_one, item)
                results_and_idx.append(future)
            for _ in concurrent.futures.as_completed(results_and_idx):
                pbar.update(1)
        print(":smiley: :vampire: :thumbs_up:")
        return results_and_idx

    @staticmethod
    def process_parallel_in_all_case_new(item_list, max_workers, process_one, model_name='gpt', desc=None):
        # process_one返回的是东西会被包成future，调用as_completed，会让完成的逐渐吐出来
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=len(item_list), desc=f"'{model_name}' 正在推理数据" if desc is None else desc)
            futures = [executor.submit(process_one, item) for item in item_list]
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)
            pbar.close()  # 确保进度条在完成时关闭
        print(":smiley: :vampire: :thumbs_up:")
        return results

######
info_dict = {}
class Info:
    @staticmethod
    def save_import_df_json(df, target_save_path, in_config_name, in_config_comment='0731: qwen对5500中的4400都不熟悉，现在打算用pt对这些歌打乐器场景等标签，进而qwen再总结，形成一个两步的CoT数据'):
        Info.log_info(aidj_info=None, name=in_config_name, value=target_save_path, comment=in_config_comment)
        df.to_json(target_save_path, orient='records', force_ascii=False, indent=4)
        print('\n###', in_config_comment)
        print(f'{target_save_path} saved and log successfully.')

    @staticmethod
    def get_info(info_path = '/workspace/linjh/CoT_Factory/data.yaml'):
    # def get_info(info_path = '/cfs-datasets/users/jinghaolin/codes/MusicLLM-Qwen/aidj_info.yaml'):
        if 'aidj_info' not in info_dict:
            with open(info_path, 'r') as f:  # 再定义各种base_path
                aidj_info = yaml.load(f, Loader=yaml.FullLoader)
            info_dict['aidj_info'] = aidj_info
            
        return info_dict['aidj_info']
    
    @staticmethod
    def log_info(aidj_info, name, value, comment):
        if aidj_info is None: aidj_info = Info.get_info()
        aidj_info_itself_path = aidj_info['aidj_info_itself_path']
        with open(aidj_info_itself_path, 'a') as f:
            f.write(f'\n### {comment}\n{name}: {value}\n')
        with open(aidj_info_itself_path, 'r') as f:  # 再定义各种base_path
            aidj_info = yaml.load(f, Loader=yaml.FullLoader)
        return aidj_info
    
    @staticmethod
    def get_value(key):
        info = Info.get_info()
        if key in info:
            return info[key]
        else:
            print(f'we have keys: {info.keys()}, but {key} not found.')
            return 'linjh1118'
    
    @staticmethod
    def get_df(key):
        path = Info.get_info()[key]
        if path.endswith('.json') or path.endswith('.jsonl'):
            return pd.read_json(path, lines=path.endswith('.jsonl'))
        elif path.endswith('.csv'):
            try: 
                df = pd.read_csv(path)
            except:
                df = pd.read_csv(path, sep='\t')
            return df
        return None
    
    @staticmethod
    def get_align_func():
        align_id = lambda x: int(x.strip('qqmusic_')) if isinstance(x, str) else int(x)
        return align_id
    
    @staticmethod
    def get_songid2meta():
        songid2meta_pure_df = pd.read_csv(Info.get_value('songid2meta'), sep='\t')
        songid2meta_pop_df = pd.read_csv(Info.get_value('songid2meta_pop'), sep='\t')
        align_id = lambda x: int(x.strip('qqmusic_')) if isinstance(x, str) else int(x)

        songid2_singer_name_song_name = defaultdict(dict)
        for i, row in songid2meta_pop_df.iterrows():
            song_id = align_id(row['track_id'])
            songid2_singer_name_song_name[song_id]['singer_name'] = row['singer_name1']
            songid2_singer_name_song_name[song_id]['song_name'] = row['track_name']

        for i, row in songid2meta_pure_df.iterrows():
            song_id = align_id(row['songid'])
            songid2_singer_name_song_name[song_id]['singer_name'] = row['singer_name']
            songid2_singer_name_song_name[song_id]['song_name'] = row['song_name']
        
        return songid2_singer_name_song_name
        

# nlp.log_info(aidj_info, name='unfamiliar_4400_song_pt_tag_qa', value='/home/MuseLLM/AI_DJ/notebook/compare_data.json', comment='0731: qwen对5500中的4400都不熟悉，现在打算用pt对这些歌打乐器场景等标签，进而qwen再总结，形成一个两步的CoT数据')


class Paragraph:
    
    stop_words = set()
    
    @staticmethod
    def get_stopwords():
        """
        Get the stopwords
        """
        global stop_words
        if len(stop_words) != 0:
            print(f'get {len(stop_words)} stopwords successfully')
            return stop_words
        stopword_dir = Path(os.path.dirname(__file__)) / 'stopwords'
        file_list = os.listdir(stopword_dir)
        file_list = [stopword_dir / file for file in file_list]
        for x in file_list:
            with open(x, 'r', encoding='utf-8') as file:
                for line in file:
                    stop_words.add(line.strip())
        print(f'load {len(stop_words)} stopwords successfully for the first time by reading {file_list} files')
        return stop_words.copy()

    @staticmethod
    def count_words_mixed_language(text):
        # 初始化英文单词和中文词的计数器
        word_count = 0
        
        # 按空格分割获取英文单词
        english_words = text.split()
        word_count += len(english_words)
        
        # 遍历文本中的每个字符，检查是否为中文字符
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 判断是否是中文字符
                word_count += 1
        
        return word_count
    
    @staticmethod
    def cut_text(text, max_length=1000):
        """
        Cut the text into pieces of max_length
        """
        # 2. 分词
        words = jieba.lcut(text)

        # 3. 去除停用词
        stop_words = set()
        with open('stopwords.txt', 'r', encoding='utf-8') as file:
            for line in file:
                stop_words.add(line.strip())

        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    @staticmethod    
    def is_pure_chinese(sentence):
        chinese_characters_pattern = re.compile(r'^[\u4e00-\u9fff。，、！？“”‘’：；（）《》【】……]*$')
        return bool(chinese_characters_pattern.match(sentence))
    
    @staticmethod
    def show_LDA(addtional_stopwords = ['首歌', '歌曲', '适合', '非常适合', '主题', "时", '做', '听', '歌', '音乐', '歌曲', '听歌', '听歌曲', '\t', '推荐', '节奏', '带有']):
        from gensim.models import LdaModel  # 4.3.2
        from gensim.corpora import Dictionary
        import jieba # 0.42.1
        
        # 准备数据
        qa_list_file = "/home/MuseLLM/analyse_data/data/train_interpret_m2v_0527_qa.txt"
        
        qa_list = open(qa_list_file, encoding = 'utf-8',errors = 'ignore').read().split('\n')[:100]
        wordlist_list = [] 
        st_words = get_stopwords()
        if addtional_stopwords:
            st_words.update(addtional_stopwords)
        for qa in qa_list:
            wordlist = list(jieba.cut(qa))
            # print(list(wordlist))
            wordlist = [word for word in wordlist if word not in st_words and len(word) > 1]
            # print(list(wordlist)); break
            wordlist_list.append(wordlist)

        dictionary = Dictionary(wordlist_list)  # 构建 document-term matrix
        corpus = [dictionary.doc2bow(text) for text in wordlist_list]
        
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=30, passes = 30, random_state=1)
        topic_list=lda.print_topics()
        for topic in topic_list:
            print(topic)
    
    @staticmethod
    def draw_word_cloud(word_list: List[str], font_path=None):
        """Bro, input a list of words, and I will draw a word cloud for you."""
        # Libraries
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        if font_path is None:
            font_path = '/home/MuseLLM/analyse_data/utils/庞门正道粗书体.ttf'

        # Create a list of word
        if isinstance(word_list[0], list):
            # now word_list is a list of sentences
            word_list = [' '.join(x) for x in word_list]
        text = ' '.join(word_list)

        # Create the wordcloud object
        wordcloud = WordCloud(
            width=480, height=480,
            max_font_size=100, min_font_size=10,
            background_color='white',
            font_path=font_path
        ).generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.show()


class Parser:

    
    @staticmethod
    def match_item_in_line(line, prefix_pattern=None, prefix_str=None, no_digit=False) -> Tuple[bool, str]:
        if prefix_pattern is None:
            assert prefix_str is not None
            # digit_matcher = "" if no_digit else "\d+"
            # prefix_pattern = f'^{prefix_str}\s*{digit_matcher}\s*[：.、]\s*'
            digit_matcher = "" if no_digit else "\d*"
            prefix_pattern = f'^\s*{digit_matcher}\s*[：:.、]*\s*{prefix_str}\s*{digit_matcher}\s*[：:.、]\s*'
        # 去掉两侧的特殊符号 ' " `等
        line = re.sub(r'^[\'"`]+|[\'"`]+$', '', line)
        line = line.replace("*", "").strip()
        if re.search(prefix_pattern, line) is not None:
            return True, re.sub(prefix_pattern, '', line)
        return False, None
    
    @staticmethod
    def match_item_in_paragraph(content, prefix_pattern=None, prefix_str=None, no_digit=False) -> str:
        if content is None:
            content = ''
        for line in content.split('\n'):
            flag, item = Parser.match_item_in_line(line, prefix_pattern=prefix_pattern, prefix_str=prefix_str, no_digit=no_digit)
            if flag:
                return item
        return None

    @staticmethod
    def match_several_item(content, prefix_pattern_set={'音乐专业鉴赏总结', '推荐理由'}, need_clean_content=False):
        """建立先抽取，然后自己分析异常值并洗掉的流程，而不是在parse中直接洗掉所有异常值。这样多方面parse会混乱"""
        NOTHING_STR = 'Lin: Nothing'
        prefix2res = {k: NOTHING_STR for k in prefix_pattern_set}
        if need_clean_content:
            # 将两个回车去掉
            content = re.sub('\n{2,}', '\n', content).strip()
        for line in content.split('\n'):
            for prefix in prefix_pattern_set:
                flag, item = Parser.match_item_in_line(line, prefix_str=prefix, no_digit=True)
                if flag: 
                    prefix2res[prefix] = item
                    prefix_pattern_set.remove(prefix)
                    break
        return prefix2res
    
    @staticmethod
    def match_pair_item(content, pattern=r"问题\s*\d+[：:](.+)?\n答案\s*\d+[：:](.+)?", n_pairs=4) -> dict:
        NOTHING_STR = 'Lin: Nothing'
        # 将两个回车去掉
        content = re.sub('\n{2,}', '\n', content)
        parsed_ans = {**{f'问题{i}': NOTHING_STR for i in range(1, n_pairs+1)}, 
                    **{f'答案{i}': NOTHING_STR for i in range(1, n_pairs+1)}
                    }
        
        matches = re.finditer(pattern, content)
        for i, match in enumerate(matches, start=1):
            if i > n_pairs:
                continue
            question = match.group(1)
            answer = match.group(2)
            parsed_ans[f'问题{i}'] = question
            parsed_ans[f'答案{i}'] = answer
            

        return parsed_ans

    

        
    @staticmethod
    def parse_score_and_explanation(content, 
                                key2score_prefix = {
                                                '曲风流派': '曲风流派一致性分数',
                                                '情感氛围': '情感氛围一致性分数',
                                                '场景主题': '场景主题一致性分数',
                                                '乐器': '乐器一致性分数',
                                                '节奏': '节奏一致性分数',
                                                },
                                key2explanation_prefix = {
                                                '曲风流派': '曲风流派(一致性)?解释',
                                                '情感氛围': '情感氛围(一致性)?解释',
                                                '场景主题': '场景主题(一致性)?解释',
                                                '乐器': '乐器(一致性)?解释',
                                                '节奏': '节奏(一致性)?解释',
                                }
                                ):
        parse_res = {k: {'score':'', 'explanation': ''} for k in key2score_prefix}
        import re
        content = re.sub('\n{2,}', '\n', content)
        # set suffix 
        suffix = '分'
        # parse score
        for line in content.split('\n'):
            for key, prefix_str in key2score_prefix.items():
                # 场景主题一致性分数：1分
                # 将1抽取出来
                if parse_res[key]['score'] != '':
                    continue
                prefix_pattern = re.compile(f'{prefix_str}\s*[：.、]\s*(\d){suffix}')
                prefix_res = prefix_pattern.search(line)
                if prefix_res:
                    parse_res[key]['score'] = prefix_res.group(1)
        # parse explantion
        for line in content.split('\n'):
            for key, prefix_str in key2explanation_prefix.items():
                
                # 场景主题一致性解释：xxxx
                if parse_res[key]['explanation'] != '':
                    continue
                prefix_pattern = re.compile(f'{prefix_str}\s*[：.、]\s*')
                if prefix_pattern.search(line):
                    line = prefix_pattern.sub('', line)
                    parse_res[key]['explanation'] = line.strip()
        return parse_res

    @staticmethod
    def parse_col_in(df=None, ans_col_name='ans', wrong_ans_word='抱歉', res_jsonl_file=None, match_func=match_pair_item):
        """请用下边那个函数"""
        NOTHING_STR = 'Lin: Nothing'
        if df is None:
            assert res_jsonl_file is not None
            df = pd.read_json(res_jsonl_file, lines=True)
            print(f'{df.shape} ==> df.dropna(subset=["track_id"]) ==> ', end='')
            df = df.dropna(subset=['track_id']); print(df.shape)
        # 过滤ans_col中含有wrong_ans_word的行
        print(f'filter wrong ans "{wrong_ans_word}": ', df.shape, end='->')
        df = df[~df[ans_col_name].str.contains(wrong_ans_word)]
        print(df.shape)
        
        # 传入带有ans列的df，遍历该列，自动生成新的几列， 返回新的col
        parsed_ans_list: list[dict] = []
        for ans in df[ans_col_name]:
            parsed_ans = match_func(ans)
            parsed_ans_list.append(parsed_ans)
        
        # 将parsed_ans_list转为df
        parsed_ans_df = pd.DataFrame(parsed_ans_list)
        print(f'parse new cols: {parsed_ans_df.columns}')
        # 将parsed_ans_df(1828, 8)与原df (1828, 15) 合并
        # df = pd.concat([df, parsed_ans_df], axis=1) 有问题。。
        pd.options.mode.chained_assignment = None
        for key in parsed_ans_df.columns:
            df[key] = parsed_ans_df[key].tolist()  # 不用tolist也会出问题，会出来一些nan，真是纳闷了。。
        
        # 清洗格式正确，但是内容垃圾的行
        print('before check clean: ', df.shape)
        for col in parsed_ans_df.columns:
            print(f'### clean {col} ###')
            # show something
            # not_parse_df = df[df[col].apply(lambda x: not isinstance(x, str))]
            # print(not_parse_df[['ans', col]].head(2))
            # res = match_pair_item(not_parse_df['ans'].iloc[0])
            # print(res)
            # df = df[df[col].apply(lambda x: isinstance(x, str))]
            # show something
            df = df[df[col] != NOTHING_STR]
            df = df[df[col].apply(lambda x: len(x) > 3)]
            print(f'clean {col}: -> {df.shape}')
        print('convert done')
        return df
    
    @staticmethod
    def parse_response_col_of_LLM_in_DF(res_df=None, res_jsonl_file=None, 
                                        ans_col_name='ans', func_parse_row=None, preprocess_func=None, cut_len=10):
        """All you need to do is to write a func_parse_row(row) -> {'问题1': ans, '问题2': None}, 
        要确保func_parse_row(row)输出包含所有keys，并且确实的value为None
        """
        NOTHING_STR = 'Lin: Nothing'
        if res_df is None:
            res_df = pd.read_json(res_jsonl_file, lines=res_jsonl_file.endswith('.jsonl'))
            print(f"#### We're parsing this file: {res_jsonl_file}")
        if preprocess_func:
            # 自己可以写一个去重 & 过滤'拒绝回答词'
            res_df = preprocess_func(res_df)
        res_df['tmp_parse_ans_dict'] = res_df[ans_col_name].apply(func_parse_row).tolist()
        for key in res_df['tmp_parse_ans_dict'].iloc[0].keys():
            res_df[key] = res_df['tmp_parse_ans_dict'].apply(lambda x: x[key])
        res_df.info()
        # 去掉解析空值
        parse_keys = list(res_df['tmp_parse_ans_dict'].iloc[0].keys())
        print('we parse keys: ', parse_keys); print('before clean: ', res_df.shape)
        for key in parse_keys:
            res_df = res_df[res_df[key] != NOTHING_STR]
            print(f'clean ({key} == nan): -> {res_df.shape}')
            # 将长度小于10的去掉
            res_df = res_df[res_df[key].apply(lambda x: len(x) > cut_len)]
            print(f'clean len({key}) < {cut_len}: -> {res_df.shape}')
        # res_df.drop('tmp_parse_ans_dict', inplace=True)
        return res_df
        
    
    @staticmethod
    def judge_tag(question, answer):
        for s in [question, answer]:
            for item in ['旋律', '和声', '乐器', '歌词']:
                if item in s:
                    return item
        return '其他'


class LLM_Data_Builder:

    @staticmethod
    def construct_ft_format(input_info_list, output_file):
        
        output_json_data_list = []

        assert input_info_list is not None
        for info in input_info_list:
            track_id, audio_path, question, answer, m2v_feature_path, tianqin_feature_path = \
            info['track_id'], info['audio_path'], info['question'], info['answer'], info['m2v_feature_path'], info['tianqin_feature_path']
            cur_sample = {
                "id": track_id,
                "audio": audio_path,
                "conversations": [
                    {
                        "question": question,
                        "answer": answer,
                        "type": judge_tag(question, answer)
                    }
                ],
                "m2v_feature": m2v_feature_path,
                "tianqin_feature": tianqin_feature_path
            }
            output_json_data_list.append(cur_sample)
        
        # 保存成json文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_json_data_list, f, ensure_ascii=False, indent=4)
        print(f'{output_file} saved.')

    @staticmethod
    def res_file_to_ft_format(res_jsonl_file, output_file, match_func=Parser.match_pair_item, n_pairs=4):
        if os.path.exists(output_file):
            print(f'{output_file} already exists, return.')
            return
        df = pd.read_json(res_jsonl_file, lines=True)
        df = parse_col_in(df, match_func=match_func)
        # prepare input info list
        
        ## 读取songid2feas
        BIG_TRAIN_DATA = '/cfs-datasets/users/yuuhong/MusicLLM/traindata/pretrain_train_0517_tianqin_emb.json'
        songid2feas = dict()
        with open(BIG_TRAIN_DATA, 'r') as f:
            train_data_before = json.load(f)
        for x in train_data_before:
            songid = int(x['id'].split('_')[1])
            songid2feas[songid] = {'audio': x['audio'], 'm2v_feature': x['m2v_feature'], 'tianqin_feature': x['tianqin_feature']}
        
        ## QA <== emb_feature
        input_info_list = []
        n_loss_fea = 0
        # for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        for index, row in df.iterrows():
            songid = int(row['track_id'])
            if songid not in songid2feas:
                n_loss_fea += 1; continue
            cur_info = dict(track_id=songid,
                    audio_path=songid2feas[songid]['audio'],
                    m2v_feature_path=songid2feas[songid]['m2v_feature'],
                    tianqin_feature_path=songid2feas[songid]['tianqin_feature']
                )
            
            for i in range(1, n_pairs+1):
                input_info_list.append(
                    {**cur_info, **dict(question=row[f'问题{i}'], answer=row[f'答案{i}'])}
                )
        print(f'{n_loss_fea} examples missing emb features, So we got {len(input_info_list)} examples.')
        construct_ft_format(input_info_list, output_file)
        
    @staticmethod
    def train_test_eval_split(all_train_data_json):
        with open(all_train_data_json, 'r') as f:
            all_train_data = json.load(f)
        # shuffle and split
        import random
        random.seed(1024666)
        random.shuffle(all_train_data)
        N_TEST = 50
        N_EVAL = 100
        test_data = all_train_data[:N_TEST]
        eval_data = all_train_data[N_TEST:N_TEST+N_EVAL]
        train_data = all_train_data[N_TEST+N_EVAL:]
        # save
        train_test_eval_file_list = [all_train_data_json.replace('.json', x) 
                                    for x in ['.train.json', '.test.json', '.eval.json']]
        with open(train_test_eval_file_list[0], 'w') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        with open(train_test_eval_file_list[1], 'w') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
        with open(train_test_eval_file_list[2], 'w') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=4)
        train_test_eval_file_list = '\n'.join(train_test_eval_file_list)
        print(f'Done Saving to {train_test_eval_file_list}, \nsize = [{len(train_data)}, {len(test_data)}, {len(eval_data)}]')
        
    
class SuperDF:

    def get_df_dict_of_suffix(suffix='json', file_dir='/cfs-datasets/users/jinghaolin/codes/MusicLLM-Qwen/results/0805_ft_task_extension_0805_fuse_0801_cot_r12_MIXMIX'):
        df_dict = {}
        for item in p(file_dir).glob(f'*.{suffix}'):
            df_dict[item.stem] = pd.read_json(item)
        return df_dict
    
    # really popular 
    # nlp.SuperDF.read_and_anal_csv(checked_df=parse_ft_ans_df, target_col=['独特卖点', '音乐特色'])
    # 加上一个分析df各列的函数
    
    @staticmethod
    def show_dist(df, key):
        fig, ax1 = plt.subplots()

        # 绘制频数直方图，使用左侧的Y轴
        # 选择一种较淡的颜色，例如淡蓝色
        df[key].hist(bins=100, ax=ax1, alpha=0.75, color='skyblue')
        ax1.set_ylabel('count', color='black')

        # 创建第二个Y轴
        ax2 = ax1.twinx()

        # 计算频率并绘制频率直方图
        # 选择一种更鲜艳的颜色，例如橙色
        weights = (1 / df.shape[0]) * np.ones_like(df[key])
        df[key].hist(bins=100, ax=ax2, alpha=0.75, color='blue', weights=weights)
        ax2.set_ylabel('proportion', color='grey')

        # 为了图表更加清晰，可以设置一下背景颜色
        ax1.set_facecolor('whitesmoke')  # 设置ax1的背景颜色
        ax2.set_facecolor('whitesmoke')  # 设置ax2的背景颜色

        # 显示图表
        plt.show()
                
    @staticmethod
    def read_and_anal_csv(file_path=None, checked_df=None, target_col: List[str] = None):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        if checked_df is None:
            df = pd.read_csv(file_path)
        else:
            df = checked_df[target_col]
        
        print('#################### info of whole csv ####################')
        info = df.info()
        if not target_col:
            return df
        for col in target_col:
            # 如果此列每元素是数字
            print(f'#################### info of {col} ####################')
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                print(df[col].describe())
                continue
            # 如果此列每元素是句子
            # 展示10个
            print('########## sample 10 ##########')
            value_counts = df[col].value_counts()
            print(value_counts.head(10))
            print('########## show cloud and length dist ##########')
            # 画出词云，统计句子长度
            Paragraph.draw_word_cloud(df[col].tolist())
            # 按照中文方式统计句子长度，画出直方图
            length = df[col].apply(lambda x: Paragraph.count_words_mixed_language(x))
            df[f'{col}_len'] = length
            
            # 直方图
            SuperDF.show_dist(df, f'{col}_len')
            

    
    @staticmethod
    def convert_muselm_train_data_2_df(file_path):
        df = pd.read_json(file_path)
        # conversation列，每个元素是dict，将其key拿出来
        df['question'] = df['conversations'].apply(lambda x: x[0]['question'])
        df['answer'] = df['conversations'].apply(lambda x: x[0]['answer'])
        if all(['attr' in x[0] for x in df['conversations']]):
            df['attr'] = df['conversations'].apply(lambda x: x[0]['attr'])
        return df

    @staticmethod
    def convert_muselm_train_data_2_df_backward(df, save_json_path):
        # 将df转为muselm的train_data格式
        # 去掉conversations列
        df.drop(columns=['conversations'], inplace=True)
        train_data = []
        for idx, row in df.iterrows():
            cur_sample = {
                'id': row['id'],
                'audio': row['audio'],
                'conversations': [
                    {
                        'question': row['question'],
                        'answer': row['answer'],
                        'attr': row['attr']
                    },
                ]
            }
            # 将其他列也加入
            for col in df.columns:
                if col not in ['id', 'audio', 'question', 'answer', 'attr', 'conversations']:
                    cur_sample[col] = row[col]
            train_data.append(cur_sample)
        with open(save_json_path, 'w') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        return

    @staticmethod
    def show_col_pie(df, col_name):
        import plotly.express as px
        attr_counts = df[col_name].value_counts()
        fig = px.pie(names=attr_counts.index, values=attr_counts.values)
        # fig.update_layout(width=500, height=500)
        fig.update_layout(width=500, height=500, showlegend=False, annotations=[dict(text=f"{attr_counts.values[i]}", x=attr_counts.index[i], y=attr_counts.values[i], font_size=12, showarrow=False) for i in range(len(attr_counts))])
        fig.show()
    
    @staticmethod 
    def show_col_pie_score(df, col_name, zh_name='曲风'):
        import numpy as np
        import plotly.express as px
        attr_counts = df[col_name].value_counts()
        fig = px.pie(names=attr_counts.index, values=attr_counts.values)
        # fig.update_layout(width=500, height=500,font=dict(size=14), title=f"曲风评分均值: {np.mean(df[col_name]):.2f}<br>2:1:0 = {attr_counts[2]}: {attr_counts[1]}: {attr_counts[0]}")
        fig.update_layout(width=500, height=500,font=dict(size=14), title=f"{zh_name}评分均值: {np.mean(df[col_name]):.2f}")
        customdata = [(attr_counts.values[i], attr_counts.values[i]/sum(attr_counts.values) * 100) for i in range(len(attr_counts))]
        fig.update_traces(textinfo='value+percent', 
                        hovertemplate='%{label}分出现数量: %{customdata[0][0]}, 占比: %{customdata[0][1]:.2f}%',
                        customdata=customdata,
                        texttemplate='%{label}分数量: %{value}<br>占比: %{percent}')
        fig.show()
    
    @staticmethod
    def show_condition_dist(df, condition = lambda row: any([ch in row['question'] for ch in '《》']), 
                            condition_str_list=None, size=500):
        # 统计符合condition的有多少行
        count = df.apply(condition, axis=1).sum()
        if condition_str_list is None:
            import inspect
            condition_str = inspect.getsource(condition)
            condition_str = condition_str.split(':')[1].strip()
            condition_str_list = [condition_str, 'not ' + condition_str]
            
        import plotly.express as px
        fig = px.pie(names=condition_str_list, values=[count, len(df) - count])
        
        fig.update_layout(width=size, height=size)
        fig.show()

        """show_condition_dist(train_df, condition=lambda row: any([ch in row['question'] for ch in '《》']), 
                        # condition_str_list=['question contains 《》', 'question not contains 《》'], size=450)
                        condition_str_list=['问题中含有歌名', '问题中不含歌名'], size=450)"""
    
    @staticmethod
    def show_some_qa(df, q_col_name='question', a_col_name='answer', tag_col_name='id', sample_size=10):
        for idx, row in df.sample(sample_size).iterrows():
            print('###### {} ######'.format(row[tag_col_name] if tag_col_name in row else idx))
            if q_col_name in row:
                print("{} ~~> \n{}\n".format(row[q_col_name], row[a_col_name]))
            else:
                for i in range(len(row['conversations'])):
                    print("{} ~~> \n{}\n".format(row['conversations'][i]['question'], row['conversations'][i]['answer']))
            print()
     
    @staticmethod    
    def show_some_col(df, col_names, sample_size=10, for_gpt_read=False):
        for idx, row in df.sample(sample_size, random_state=1024).iterrows():
            if not for_gpt_read:
                print('\n################## {} ##################'.format(idx))
                for col_name in col_names:
                    print(f'~~>~~>~~> {col_name} <~~<~~<~~')
                    print(row[col_name], end='\n\n')
            else:
                print(f'\nid: {idx}')
                print('# data')
                for i, col_name in enumerate(col_names, start=1):
                    print(f'### {i}. {col_name}')
                    print(row[col_name].replace('\t', '\n'), end='\n\n')

class HfDataset:
    """just for memory, and do not import some many dataset"""
    @staticmethod
    def load_local(path='/home/MuseLLM/Fuse_data/m-a-p/MusicPile-sft/data'):  # 只有一个split
        # 2.18.0
        from datasets import load_dataset
        dataset = load_dataset(path=path)
        return dataset
    
    @staticmethod
    def to_df(dataset=None, path=None):
        assert dataset or path
        if dataset is None:
            dataset = HfDataset.load_local(path)
            print('finish loading dataset.')
        df_split = {}
        for split in ['train', 'test', 'dev']:
            print(f'converting dataset {split} to df...')
            if split in dataset:
                data_list = [dataset[split][i] for i in trange(len(dataset['train']))]
                df_split[split] = pd.DataFrame(data_list)
        return df_split
    
    @staticmethod
    def load_serval_split(path = '/root/data/linjh/assets/MMMU/MMMU', split=None, show=True) -> Dict:  # 有多个split，得通过split = name 进行读取
        from datasets import load_dataset
        # 子文件名字
        data_dir = p(path)
        item_list = [item.stem for item in p(data_dir).iterdir() if item.is_dir()]
        ds_dict = {}
        for item in item_list:
            if split is None:
                ds = load_dataset(path=path, name=item)
            else:
                ds = load_dataset(path=path, name=item, split=split)
            ds_dict[item] = ds
            
        if show:
            from prettytable import PrettyTable
            table = PrettyTable()
            if split is None:
                table.field_names = ["Dataset", "dev", 'test', 'validation']
                for item, ds in ds_dict.items():
                    table.add_row([
                        item,
                        ds['dev'].num_rows, ds['test'].num_rows, ds['validation'].num_rows,
                    ])
            else:
                table.field_names = ["Dataset", split]
                for item, ds in ds_dict.items():
                    table.add_row([item, ds.num_rows])
            print(table)
        return ds_dict
        
        

import plotly.express as px
class Viz:
    
    @staticmethod
    def plot_loss_jsonl(file_path = 'l3_loss.json', keys: List[str] = ["loss"], 
                save_dictionary='/ssd2/linjinghao01/rrhf/rrhf_418_c2/output_ill_c2', 
                train_id='MuseLM-FT', start_step=None, end_step=None) -> None:
        """input ckpt/training.state
        
        `nlp_tools.plot_loss_jsonl("/cfs-datasets/users/jinghaolin/codes/MusicLLM-Qwen/output/0701_pop_ft/checkpoint-6600/trainer_state.json")`
        """
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            data_list = []
            for l in lines:
                data = json.loads(l)
                data_list.append(data)
        else:
            with open(file_path, 'r') as file:
                data = json.load(file)
            data_list = data['log_history']
        

        def smooth(scalars: List[float]) -> List[float]:
            r"""
            EMA implementation according to TensorBoard.
            """
            last = scalars[0]
            smoothed = list()
            weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
            for next_val in scalars:
                smoothed_val = last * weight + (1 - weight) * next_val
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed
        
        eval_steps, eval_losses = [], []
        for i in range(len(data_list)):
            if 'eval_loss' in data_list[i]:
                eval_steps.append(data_list[i]["step"])
                eval_losses.append(data_list[i]['eval_loss'])
        
        for key in keys:
            steps, metrics = [], []
            for i in range(len(data_list)):
                if key in data_list[i]:
                    steps.append(data_list[i]["step"])
                    metrics.append(data_list[i][key])

            if start_step is not None:
                metrics = [metric for step, metric in zip(steps, metrics) if step >= start_step]
                steps = [step for step in steps if step >= start_step]
                eval_losses = [loss for step, loss in zip(eval_steps, eval_losses) if step >= start_step]
                eval_steps = [step for step in eval_steps if step >= start_step]
                
                
            if end_step is not None:
                metrics = [metric for step, metric in zip(steps, metrics) if step <= end_step]
                steps = [step for step in steps if step <= end_step]
                eval_losses = [loss for step, loss in zip(eval_steps, eval_losses) if step <= end_step]
                eval_steps = [step for step in eval_steps if step <= end_step]
                

            plt.figure()
            plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original-train-loss")
            plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed-train-loss")
            if key == "loss":
                plt.plot(eval_steps, eval_losses, color="#FF0000", label="eval-loss")
            # plt.title("training {} of {}".format(key, save_dictionary))
            plt.title("training {} of {}".format(key, train_id))
            plt.xlabel("step")
            plt.ylabel(key)
            plt.legend()
            # figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
            # plt.savefig(figure_path, format="png", dpi=100)
            # print("Figure saved at:", figure_path)
        
    @staticmethod
    def refer_tsne():
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.decomposition import PCA

        # 读取数据
        with open('m2v_feature.json', 'r') as f:
            m2v_feature = json.load(f)
        m2v_feature = np.array(m2v_feature)
        print(m2v_feature.shape)
        # PCA降维
        pca = PCA(n_components=50)
        m2v_feature = pca.fit_transform(m2v_feature)
        print(m2v_feature.shape)
        # TSNE降维
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        m2v_feature = tsne.fit_transform(m2v_feature)
        print(m2v_feature.shape)
        # 画图
        plt.figure(figsize=(10, 10))
        plt.scatter(m2v_feature[:, 0], m2v_feature[:, 1])
        plt.show()
    
    @staticmethod
    def show_m2v_dist(muselm_train_data_json_file, label_col='source', max_sample_size_each_label=100, save_html_path=f'/cfs-datasets/users/jinghaolin/codes/utils/{random.randint(0, 10000000)}.html'):
        from sklearn.manifold import TSNE
        ## prepare
        # get song2id 2 feature
        train_df = SuperDF.convert_muselm_train_data_2_df(muselm_train_data_json_file)
        songid2feas = defaultdict(dict)
        label2songid_list = defaultdict(list)
        train_df['id'] = train_df['id'].apply(lambda x: str(x)) # 有时候会是一个int
        for idx, row in train_df.iterrows():
            songid = int(row['id'].split('_')[1]) if '_' in row['id'] else int(row['id'])
            songid2feas[songid]['audio'] = row['audio']
            songid2feas[songid]['m2v_feature'] = row['m2v_feature']
            songid2feas[songid]['tianqin_feature'] = row['tianqin_feature']
            if label_col in row:
                label2songid_list[row[label_col]].append(songid)
        
        label2songid_list = {k: list(set(v)) for k, v in label2songid_list.items()}
        # sample
        import random
        random.seed(1024)
        for label in label2songid_list:
            if len(label2songid_list[label]) > max_sample_size_each_label:
                label2songid_list[label] = random.sample(label2songid_list[label], max_sample_size_each_label)
            print(f'n of "{label}": {len(label2songid_list[label])}')
        
        # print(f'all_qa: {train_df.shape}')
        # all_sample_song_list = [x for v_list in label2songid_list.values() for x in v_list]
        # train_df['songid'] = train_df['id'].apply(lambda x: int(x.split('_')[1]) if '_' in x else int(x))
        # train_df = train_df[train_df['songid'].isin(all_sample_song_list)]
        # print(f'cover_qa: {train_df.shape}')
        
        # get record
        info_list = []
        for label, songid_list in label2songid_list.items():
            for songid in songid_list:
                info_list.append({
                    label_col: label,
                    'songid': songid,
                    'audio': songid2feas[songid]['audio'],
                    'm2v_feature': songid2feas[songid]['m2v_feature'],
                    'tianqin_feature': songid2feas[songid]['tianqin_feature'],
                })
        show_df = pd.DataFrame(info_list)
            
        # get m2v feature
        m2v_feature_list = show_df['m2v_feature'].apply(lambda feature_path: np.expand_dims(np.load(feature_path), axis=0)).tolist()
        m2v_feature_matrix = np.concatenate(m2v_feature_list, axis=0)
        m2v_feature_matrix_tsne_dim = 3
        print('getting tsne...')
        m2v_feature_matrix_tsne = TSNE(
            n_components=m2v_feature_matrix_tsne_dim,
            perplexity=30,
            learning_rate='auto', 
            n_iter=2000,
            init='pca',
            random_state=1024,
            n_jobs=-1,
        ).fit_transform(m2v_feature_matrix)
        print('done getting...')
        for xyz, i in [('x', 0), ('y', 1), ('z', 2)]:
            show_df[f'tsne_{xyz}'] = m2v_feature_matrix_tsne[:, i]
        fig = px.scatter_3d(show_df, x='tsne_x', y='tsne_y', z='tsne_z', color=label_col)
        fig.show()
        fig.write_html(save_html_path)
        
import sys
import time
import functools
def time_tiktok(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 打印函数名和参数
        print(f"\n{'='*10} Time_TikTok {'='*10}", file=sys.stderr)
        print(f"Calling function '{func.__name__}' with args: {args} and kwargs: {kwargs}", file=sys.stderr)
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 调用被装饰的函数
        result = func(*args, **kwargs)
        
        # 记录结束时间
        end_time = time.perf_counter()
        
        # 计算并打印执行时间
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds", file=sys.stderr)
        
        return result
    return wrapper
        
class Wget:
    @staticmethod
    def get_ms(resource_id, cache_dir='/home/MuseLLM/Fuse_data', is_dataset=False):
        from modelscope import snapshot_download
        from modelscope.msdatasets import MsDataset
        assert resource_id is not None; assert cache_dir is not None
        if is_dataset:
            # ds =  MsDataset.load('m-a-p/Music-Instruct', cache_dir='/home/MuseLLM/Fuse_data')
            ds = MsDataset.load(resource_id, cache_dir=cache_dir)
            return ds
        else:
            # model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='/cfs-datasets/public_models/')
            model_dir = snapshot_download(resource_id, cache_dir=cache_dir)
            return model_dir
        return
    # https://github.com/modelscope/modelscope/issues/836 modelscope其实还是走hf的datasets下载，所以hf大改的时候，会有版本问题，难以修复。
    # 此时直接走下边的hf即可
    
    
    @staticmethod
    @time_tiktok
    def get_hf(resource_id, cache_dir='/home/MuseLLM/Fuse_data', is_dataset=False):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        addi_str = '--repo-type dataset '
        command_str = f'huggingface-cli download {addi_str}--resume-download {resource_id} --local-dir {cache_dir}'
        os.system(command_str)
    """for cur_id in ['m-a-p/MusicTheoryBench', 'm-a-p/Music-Instruct', 'm-a-p/MusicPile-sft']:
    Wget.get_hf(resource_id=cur_id, is_dataset=True, cache_dir=f'/home/MuseLLM/Fuse_data/{cur_id}')"""
           
import base64
import io

def image_to_base64(image_path_or_sth):
    if isinstance(image_path_or_sth, str):
        image_path = image_path_or_sth
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        
    elif isinstance(image_path_or_sth, io.BytesIO):
        byte_arr = image_path_or_sth
        byte_arr = byte_arr.getvalue()
        encoded_string = base64.b64encode(byte_arr)
    else:
        # 当做是一个png Image对象
        byte_arr = io.BytesIO()
        image_path_or_sth.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        encoded_string = base64.b64encode(byte_arr)
    
    return encoded_string

def image_to_show(image_path_or_sth):
    encoded_image = image_to_base64(image_path_or_sth).decode('ascii')
    template = '<img src="data:image/png;base64,{encoded_image}" width="320" alt="Embedded Image">'

    return template.format(encoded_image=encoded_image)

latex = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MathJax Example</title>
    <script>
      MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']]
        }
      };
    </script>
    <script id="MathJax-script" async src="https://cdn.bootcdn.net/ajax/libs/mathjax/3.2.2/es5/tex-chtml.js"></script>   # 这几个没有加入
</head>
<body>
    <h1>MathJax 示例</h1>
    <p>这是一个内嵌的公式 $a_2 = \sin \theta$。</p>
    <p>这是另外一个公式 \( e^{i\pi} + 1 = 0 \)。</p>
</body>
</html>
"""

def markdown_to_html(content):
    content = content.strip().strip('`')
    # 将开头的markdown
    content = re.sub(r'^markdown', '', content).strip()
    return markdown.markdown(content)
                
class TencentViz:

    @staticmethod
    def to_html(show_df=None, output_html_path = 'hello_html', textual_cols=['q & a', 'result'], merge_cols=None):
        for col in show_df.columns:
            if 'image' in col.lower() or 'graph' in col.lower():
                show_df[col] = show_df[col].apply(lambda x: image_to_show(x))
        for col in show_df.columns:
            if col.lower() in textual_cols:
                # markdown
                # show_df[col] = show_df[col].apply(lambda x: markdown_to_html(x) if isinstance(x, str) else x)  # 这个不太好，显示很怪，就这样吧，不管了
                # convert list to str join by <br>
                show_df[col] = show_df[col].apply(lambda x: '<hr>'.join(x) if isinstance(x, list) else x)
                show_df[col] = show_df[col].apply(lambda x: x.strip().replace('\n', '<br>') if isinstance(x, str) else x)
                # use replace <image> ==> &lt;image&gt;
                show_df[col] = show_df[col].apply(lambda x: x.replace('<image>', '&lt;image&gt;') if isinstance(x, str) else x)
                # replace $$ -> $
                show_df[col] = show_df[col].apply(lambda x: x.replace('$$', '$') if isinstance(x, str) else x)
                show_df[col] = show_df[col].apply(lambda x: re.sub(r'\$(.*?)\$', r'\( \1 \)', x) if isinstance(x, str) else x)
                # 
        
        if merge_cols:
            show_df[' & '.join(merge_cols)] = show_df.apply(lambda x: '<br>'.join([x[col] for col in merge_cols]), axis=1)
            # drop
            show_df.drop(columns=merge_cols, inplace=True)
        df_html = show_df.to_html(render_links=True,escape=False)
        
        
        # 如果展示少行，或者没有stick，那么可能就是这里被误触加了一个字母啥的（不会崩溃，但是就是很怪）。。。
        css_styles = """<style>
            th {
                background-color: #fff;
                position: sticky;
                top: 0;
                z-index: 1;
            }
            th:before {
                content: "";
                position: absolute;
                top: -1px;
                bottom: -1px;
                left: -1px;
                right: -1px;
                border: 1px solid LightGrey;
                z-index: -1;
            }
        </style>
        """
        latex_styles = """<script id="MathJax-script" async src="https://cdn.bootcdn.net/ajax/libs/mathjax/3.2.2/es5/tex-chtml.js"></script>"""
        md_styles = """<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>"""
        

        df_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Sticky Table Headers and First Column</title>
            {css_styles}
            {latex_styles}
            </head>
            <body>
                <div class="div1">
            {df_html}
                </div>
            </body>
            </html>
        """

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.writelines('<meta charset="UTF-8">\n')
            f.write(df_html)
        url = output_html_path
        print(f"{url} saved.")
        return url
    
    @staticmethod
    def json_to_html(
            # json & df
            json_path, sep, 
            function_list = [lambda df: df.rename(columns={'singer_name1': 'singer_name'}),
                            lambda df: df.assign(song_name=lambda x: x['song_name'] + ' - ' + x['singer_name']),
                            lambda df: df.drop(columns=['track_id', 'ans', 'singer_name'])
            ],
            # audio / img
            audio_dir=None, img_dir=None, 
            # output         
            output_html_path=None,
            hang_cfs_datasets = lambda s: s.replace('/cfs-datasets/', 'http://10.101.133.93/v1/media/innovation_cfs/')
            ):
        """展示一个结果json文件，主要封装了三个功能，【1. 读取df并改名+修改/加列+删列; 2. 处理各种语音和图片装成html元素; 3. 最后输出html文件】"""
        # 
        df = pd.read_json(json_path, sep)
        for func in function_list:
            df = func(df)
        
        # 
        df['audio'] = df['audio'].apply(lambda idx: f'<audio controls src="{Path(audio_dir) / str(idx)}.mp3">')
        df['audio'] = df['audio'].apply(hang_cfs_datasets)
        
        # 
        df_html = df.to_html(render_links=True,escape=False)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.writelines('<meta charset="UTF-8">\n')
            f.write(df_html)
        url = hang_cfs_datasets(output_html_path)
        print(f"{url} saved.")
        return url
    
    @staticmethod
    def show_df_to_html(
        # json
        show_df=None, show_json_file=None,
        hang_cfs_datasets = lambda s: s.replace('/cfs-datasets/', 'http://10.101.133.93/v1/media/innovation_cfs/'),
        output_html_path=None
        ):
        """展示一个结果json文件，主要封装了三个功能，【1. 读取df并改名+修改/加列+删列; 2. 处理各种语音和图片装成html元素; 3. 最后输出html文件】"""
        if show_df is None:
            with open(show_json_file, 'r') as f:
                show_df = pd.read_json(f)

        # show_df = show_df.sample(frac=1, random_state=8888).reset_index(drop=True)
        
        # 更换语音
        for col in show_df.columns:
            if 'audio' in col.lower() and 'qwen' not in col.lower():
                show_df.rename(columns={col: 'Audio'}, inplace=True)
            
        if 'Audio' in show_df.columns:
            show_df['Audio'] = show_df['Audio'].apply(lambda src: 
                f'<audio controls src="{src}">' 
                if src is not None and '<audio controls src=' not in src and src != '👆'
                else str(src))
            show_df['Audio'] = show_df['Audio'].apply(hang_cfs_datasets)
        
        for col in show_df.columns:
            if 'graph' in col.lower():
                show_df[col] = show_df[col].apply(hang_cfs_datasets)
            if isinstance(show_df[col].tolist()[0], str) and '.jpg' in show_df[col].tolist()[0]:
                show_df[col] = show_df[col].apply(hang_cfs_datasets)
        
        # 将每一列中的元素的\n替换为<br>
        for col in show_df.columns:
            # 如果是str类型才替换
            # if not ('int' in show_df[col].dtype.name or 'float' in show_df[col].dtype.name):
            show_df[col] = show_df[col].apply(lambda x: x.strip().replace('\n', '<br>') if isinstance(x, str) else x)
        
        if 'PT结果拼接' in show_df.columns:
            show_df['PT结果拼接'] = show_df['PT结果拼接'].apply(lambda x: x.replace('<br>', '</b><br><hr>').replace('：', '<br><b>') + '<hr><hr>'
                                                        if '</b><br><hr>' not in x else x)
        
        df_html = show_df.to_html(render_links=True,escape=False)
        
        css_styles = """<style>
            th {
                background-color: #fff;
                position: sticky;
                top: 0;
                z-index: 1;
            }
            th:before {
                content: "";
                position: absolute;
                top: -1px;
                bottom: -1px;
                left: -1px;
                right: -1px;
                border: 1px solid LightGrey;
                z-index: -1;
            }
        </style>
        """
        
        df_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Sticky Table Headers and First Column</title>
            {css_styles}
            </head>
            <body>
                <div class="div1">
            {df_html}
                </div>
            </body>
            </html>
        """
        
        
        if output_html_path is None and show_json_file is not None:
            output_html_path = show_json_file.replace('.json', '.html')
        if '/' not in output_html_path: # it's a file name
            output_html_path = f'/cfs-datasets/users/jinghaolin/codes/MusicLLM-Qwen/assets/show/{output_html_path}.html'
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.writelines('<meta charset="UTF-8">\n')
            f.write(df_html)
        url = hang_cfs_datasets(str(output_html_path))
        print(f"{url} saved.")
        return url
    
    @staticmethod
    def show_exp_id_all_step(exp_id='0805_ft_task_extension_0805_fuse_0801_cot_r12', output_html_name=None):
        # if 'cfs-datasets' not in exp_id:
        #     df_dict = SuperDF.get_df_dict_of_suffix(file_dir=f'/cfs-datasets/users/jinghaolin/codes/MusicLLM-Qwen/results/{exp_id}')
        
        file_dir = f'/cfs-datasets/users/jinghaolin/codes/MusicLLM-Qwen/results/{exp_id}' if 'cfs-datasets' not in exp_id else exp_id
        df_dict = SuperDF.get_df_dict_of_suffix(file_dir=file_dir)
        key_list = list(df_dict.keys());  assert len(key_list) > 0
        final_df = df_dict[key_list[0]]
        final_df.rename(columns={'prediction': f'prediction<hr>step-{key_list[0]}'}, inplace=True)
        for key in key_list[1:]:
            final_df[f'prediction<hr>step-{key}'] = df_dict[key]['prediction'].tolist()

        def polish(content):
            import re
            lines = content.split('\n')
            final_lines = []
            for line in lines:
                if '音乐鉴赏总结：' in line:
                    re.sub(r'音乐鉴赏总结：', '', line)
                    line = f'<b>{line}</b>'
                # re.sub(r'^问题\s*\d\s*的答案：', '- ', line)
                # if re.search(r'^问题\s*\d\s*的答案：', line):
                #     line = re.sub(r'^问题\s*\d\s*的答案：', '', line)
                #     # line = f'<li>{line}</li>'
                #     line = line.strip('"')
                final_lines.append(line)
            return '\n'.join(final_lines)

        for col in final_df.columns:
            if 'prediction' in col:
                final_df[col] = final_df[col].apply(polish)
        if output_html_name is None:  output_html_name = exp_id
        TencentViz.show_df_to_html(final_df, output_html_path=output_html_name)
        return final_df
        

    """
    nlp.TencentViz.show_exp_id_all_step(exp_id='0805_ft_task_extension_0805_fuse_0801_cot_r12_MIXMIX', output_html_name='cot_r12_task_extension_MIXMIX')
    nlp.TencentViz.show_exp_id_all_step(exp_id='0805_ft_task_extension_0805_fuse_0801_cot_r12', output_html_name='cot_r12_task_extension')
    """

if __name__ == '__main__':
    print("~~> Linjh's code is really awesome! <~~")