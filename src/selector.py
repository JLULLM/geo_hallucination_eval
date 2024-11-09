import re
from typing import Any

# import logging
# logger = logging.getLogger('linjh_logger')


class Selector:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        
    def do_filter_multi_img(self, jsonl_data: Any) -> Any:
        ori_len = len(jsonl_data)
        
        # if item['img_file_io_param'] is a dict, turn it into a list
        for item in jsonl_data:
            if isinstance(item['img_file_io_param'], dict):
                item['img_file_io_param'] = [item['img_file_io_param']]
        jsonl_data = [item for item in jsonl_data if len(item['img_file_io_param']) == 1]
        # convert list to just one
        for item in jsonl_data:
            item['img_file_io_param'] = item['img_file_io_param'][0]
        print(f'Filter multi-image samples: {ori_len} -> {len(jsonl_data)}')
        return jsonl_data
    
    def do_filter_single_question(self, jsonl_data: Any) -> Any:
        ori_len = len(jsonl_data)
        pattern = r'[(（]\s*4\s*[)）]'
        jsonl_data = [item for item in jsonl_data 
                      if re.search(pattern, item['conversations'][1]['text'])]
        pattern = r'解析'
        jsonl_data = [item for item in jsonl_data 
                      if re.search(pattern, item['conversations'][1]['text'])] 
        
        print(f'Filter single-question samples: {ori_len} -> {len(jsonl_data)}')
        return jsonl_data
    
    def do_filter_img2img(self, jsonl_data: Any) -> Any:
        ori_len = len(jsonl_data)
        jsonl_data = [item for item in jsonl_data if item['conversations'][1]['text'].strip() == '<image>']
        # jsonl_data = [item for item in jsonl_data if '<image>' in item['conversations'][1]['text'].strip()]
        print(f'Filter img2img samples: {ori_len} -> {len(jsonl_data)}')
        return jsonl_data
    
    def __call__(self, jsonl_data, **kwgs):
        self.kwargs.update(kwgs)
        func_list = [getattr(self, k) for k, v in self.kwargs.items() if v and hasattr(self, k)]
        for func in func_list:
            jsonl_data = func(jsonl_data)
        return jsonl_data

def test():
    s = Selector()
    from to_html import get_meta_jsonl_data
    data = get_meta_jsonl_data('/workspace/image_sft/datav20240920/SFT/Subject/xueke_0927/MetaFiles/xueke_000000.jsonl')
    s(data, do_sample=True, do_filter_multi_img=True, do_filter_single_question=False, do_filter_img2img=True)
    

            
        
        
        