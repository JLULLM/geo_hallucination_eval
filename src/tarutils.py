from copy import deepcopy
import io
import mmap
import struct
import traceback
import collections
from PIL import Image
from wids import wids
import re
import os
import random
random.seed(66)
import pickle as pkl
import matplotlib.pyplot as plt

# from vldata.utils.tarutils import extract_data_from_tarfile
import pandas as pd
from pathlib import Path as p

TarHeader = collections.namedtuple(
    "TarHeader",
    [
        "name",
        "mode",
        "uid",
        "gid",
        "size",
        "mtime",
        "chksum",
        "typeflag",
        "linkname",
        "magic",
        "version",
        "uname",
        "gname",
        "devmajor",
        "devminor",
        "prefix",
    ],
)


def get_image_file(media_name, media_indexdata=None, media_tarpath=None, media_offset=None) -> io.BytesIO:
    """way1. if your have img_file_io_param, then i will use its media_tarpath and media_offset to get image
    way2. use media_indexdata[media_name] to get offset, then read like upper. It's a old way."""
    if media_offset is None:
        media_offset, _ = media_indexdata[media_name]  # to get offset
    img_name, img_data = extract_data_from_tarfile(media_tarpath, media_offset)
    assert img_name == media_name, f"{img_name} != {media_name}"
    return io.BytesIO(img_data)
            
def get_orig_sample(jsonl_file, media_tarpath=None, media_indexpath=None, show_change=False, data_root_dir = None) -> list[dict]:
    #### need: jsonl_data, media_tarpath(don't need to read), media_indexdata(media_name -> (offset, hash(暂时用不到，这个应该是minhash用于去重的？)))
    meta_jsonl_data = pd.read_json(jsonl_file, lines=jsonl_file.endswith('jsonl'))
    if data_root_dir is None:
        """此时默认你传入的是 jsonl文件是 /workspace/image_sft/datav20240920/SFT/Subject/xueke_0927/MetaFiles/xueke_000000.jsonl """
        if media_tarpath is None:
            media_tarpath = str( p(jsonl_file).parent.parent / 'TarFiles' / (p(jsonl_file).stem + '.tar') ) # MetaFiles/a.jsonl -> TarFiles/a.tar
        if media_indexpath is None:
            media_indexpath = media_tarpath.replace(".tar", ".index") # TarFiles/a.tar -> TarFiles/a.index
    else:
        """默认你传入的是jsonl文件是后生成的例如 /workspace/linjh/CoT_Factory/assets/output/cot_subject_1024/normal_and_multi/xueke_000034_gpt-4o-2024-08-06_cot.json
        你需要传入 data_root_dir = /workspace/image_sft/datav20240920/SFT/Subject/xueke_0927
        """
        file_id = []
        for item in p(jsonl_file).stem.split('_'):
            file_id.append(item)
            if item.isdigit():
                break
        file_id = '_'.join(file_id)
        if media_tarpath is None:
            media_tarpath = str( p(data_root_dir) / 'TarFiles' / (file_id + '.tar') )
        if media_indexpath is None:
            media_indexpath = media_tarpath.replace(".tar", ".index")
            
    with open(media_indexpath, "rb") as fp:
        media_indexdata = pkl.load(fp)
    
    #### 分析：推理需要conv和image，观察这里发现，1. 没有imageurl之类的 2. conv中的media_tag不是常见的<image>, 而是<|ZP_MM_PLH=xxx|>
    # 而对于单图来说，xxx=default。所以单图样本推理，1. conv直接replace一个pattern即可。 2. find img，也就是 我需要一个path，或者io.BytesIO(img_data)从而给gpt
    meta_jsonl_data = meta_jsonl_data.to_dict(orient='records')
    
    show_conversations = lambda conv: print('\n'.join([f'## {item["role"]}:\n {item["text"]}' for item in conv]))
    
    #### 转换成普通样本（更换conv，保存获取img的关键小参数，后续可以通过get_image_file直接获取
    if 'img_file_io_param' in meta_jsonl_data[0]:
        pass
    else:
        if show_change:
            print('--->before')
            lucky_int = random.randint(0, len(meta_jsonl_data))
            show_conversations(meta_jsonl_data[lucky_int]['conversations'])
    
        print(f'start to change text and get img_file_io_param for {jsonl_file}')
        for item in meta_jsonl_data:
            #### ===============
            #### for single image
            #### ===============
            media_map = item['media_map'] # {'default_media_tag': ('tarid', 'media_name')}
            item['ori_conversations'] = deepcopy(item['conversations'])
            if len(media_map) == 1:
                media_tag = list(media_map.keys())[0]  # it will get 'default' or 'image1'
                # 1. change conv
                for conv in item['conversations']:
                    # <|ZP_MM_PLH=default|> -> <image>
                    conv['text'] = re.sub(rf'<\|ZP_MM_PLH={media_tag}\|>', '<image>', conv['text'])
                # 2. get image
                _, media_name = media_map[media_tag]
                #### a. get it now
                # img_file_io = get_image_file(media_name, media_indexdata, media_tarpath)
                #### b. get it later
                item['img_file_io_param'] = dict(media_name=media_name, media_tarpath=media_tarpath, media_offset=media_indexdata[media_name][0])
            else:
                # implement is easy, just replace media_tag and save image [one by one] ==> media_keys: img_0, img_1...
                # get m_tag set => (replace; save) one by one
                # 一般VLM的调用都是：{"query": "eeeee<image>eeeee<image>eee<image>ee", "response": "fffff", "history": [], "images": ["image_path1", "image_path2", "image_path1"]} 对话中，如果第三张和第一张都是第一张
                item['img_file_io_param'] = []
                all_content = ''.join([conv['text'] for conv in item['conversations']])
                pattern = r'<\|ZP_MM_PLH=([^|]+)\|>'
                matches = re.findall(pattern, all_content)
                # '<|ZP_MM_PLH=img_1|>你好你好\n\n\n<|ZP_MM_PLH=img_0|>你好你好<|ZP_MM_PLH=img_2|>你好<|ZP_MM_PLH=img_1|>' 
                # 1. ==> images: [img1, img0, img2, img1], item['img_file_io_param'] is a list like left.
                
                # 基本上大多数的media_tag都有几个，所以都走了这里，但这里并不是全都是多图的。很多这里的样本中 文本里其实只有单图，所以我之后的过滤逻辑也完全适用，不会遗漏任何。
                for media_tag in matches:
                    _, media_name = media_map[media_tag]
                    item['img_file_io_param'].append(
                        dict(media_name=media_name, media_tarpath=media_tarpath, media_offset=media_indexdata[media_name][0])
                    )
                # 2. all <|ZP_MM_PLH=img_1|> ==> <image>
                for conv in item['conversations']:
                    conv['text'] = re.sub(pattern, '<image>', conv['text'])
                
        if show_change:
            print('-->after')
            show_conversations(meta_jsonl_data[lucky_int]['conversations'])
            if isinstance(info:=meta_jsonl_data[lucky_int]['img_file_io_param'], list):
                param = info[0]
            else:
                param = info
            img = Image.open(get_image_file(**param)).convert('RGB')
            _, ax = plt.subplots()
            ax.imshow(img)
    return meta_jsonl_data

def parse_tar_header(header_bytes):
    """解析 tar 格式的文件头信息
    Args:
        header_bytes (bytes): header bytes, less than 500
    Returns:
        tar header info
    """
    assert len(header_bytes) <= 500
    header = struct.unpack("!100s8s8s8s12s12s8s1s100s6s2s32s32s8s8s155s", header_bytes)
    return TarHeader(*header)

def show_img(img, item=None):
    """draw graph and draw its boxes"""
    print(img.size)
    _, ax = plt.subplots()
    ax.imshow(img)
    if item:
        for ci in item["json"]:
            # find grounding boxes
            pattern = r"<ph_st>(.*?)<ph_ed>"
            matches = re.findall(pattern, ci["metadata"]["response"])
            print(matches)
            if len(matches) > 0:
                for boxes in matches:
                    for box in eval(boxes):
                        width, height = box[2] - box[0], box[3] - box[1]
                        rect = plt.Rectangle(box[:2], width, height, fill=False, edgecolor='red')
                        ax.add_patch(rect)
            break
    plt.show()
    
def find_all_medias(s, pattern=r'<\|ZP_MM_PLH=(\w+)\|>'):
    """找到所有媒体标记及其位置
    Args:
        s (str): 输入字符串
        pattern (str): 匹配模式
    Returns:
        result (list): [(media_id, start_pos, end_pos), ...]
    """
    matches = re.finditer(pattern, s)
    result = []
    for match in matches:
        start = match.start()
        end = match.end()
        result.append((match.group(1), start, end))
    return result

def extract_data_from_tarfile(tar_path, offset):
    """根据偏移量从tar流中获取数据
    Args:
        tar_path (str): tar path
        offset (int): offset
    Returns:
        name, 
        data bytes
    """
    try:
        with open(tar_path, "rb") as stream:
            stream = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            header = parse_tar_header(stream[offset: offset + 500])
            name = header.name.decode("utf-8").strip("\x00")
            start = offset + 512
            end = start + int(header.size.decode("utf-8")[:-1], 8)
            return name, stream[start: end]
    except:
        print(f"Failed: {tar_path}, offset: {offset}")
        print(traceback.format_exc())


def build_tar_index_func(tar_path):
    """获取 tar 中各文件的文件名和偏移量
    Args:
        tar_path (str): path
    Returns:
        indices (dict(str, str)): {name: (offset, size)}
    """
    try:
        ds = wids.IndexedTarSamples(str(tar_path), use_mmap=True)
        mmap_reader = ds.reader
        indices = mmap_reader.by_index
        indices = {v[0]: (v[1], v[2]) for v in indices}
        return indices
    except:
        print("Failed:", tar_path)
        print(traceback.format_exc())


def save_img(img_bytes, save_path):
    """保存图片
    Args:
        img_bytes (bytes): bytes of image
        save_path (str): path of saving image
    """
    img = Image.open(io.BytesIO(img_bytes))
    img.save(save_path)
