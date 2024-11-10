import os
import json
import openai
from openai import OpenAI
import pandas as pd
from langchain.prompts import PromptTemplate
import concurrent
import concurrent.futures
from tqdm import tqdm

from chatapi_server import ChatAPI

# 读取配置
cfg_path = "/root/autodl-tmp/wutr/geo_hallucination_eval/config/benchmark_cfg.json"
with open(cfg_path, 'r') as json_file:
    cfg_data: dict = json.load(json_file)

# API 设置
api_set = cfg_data["generator"]
api_key = api_set["api_key"]
base_url = api_set["base_url"]
model = api_set["model"]
temperature = int(api_set["temperature"])

class PipelineAPI(ChatAPI):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
        self.api_config["chatgpt"] = {
                "url": base_url,
                "api_key": api_key,
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": 2048,
                    "top_p": 1.0
                },
                "post_processor": self._post_chatgpt,
                "pre_processor": self._pre_chatgpt
            }

chatapi = PipelineAPI(model_name=model)


# Prompt 设置
# 1. 存在、属性、关系
temp1 = """You are a math teacher, now given some known image caption information, you need to provide three types of true or false questions based on the description:
1. Existence Type Questions
2. Attribute Type Questions
3. Relationship Type Questions

Output format requirements:
1. Use ### to mark before each category
2. After each question, give the answer directly using Yes or No, no need to provide reasons.
3. There should be a blank line separating the three categories of questions.
4. Give a direct description of each question you designed, without summarizing them at the end.

Here are two examples:
Example1:
caption:
Firstly, draw a Rectangle ABCD, creating a semicircle outside with CD as the diameter, and then eliminate the side CD. Secondly, aligned with edge AD, illustrate a Rectangle ADEF, creating a semicircle inside with EF as the diameter, and then eliminate the side CD. Thirdly, joined to edge ED, draft a Square EDGH. Angle BAD measures 90 degrees. Angle DAF has a measure of 90 degrees. The angle DEH is 90 degrees. Side AB measures 10.0 units in length. The side AD length equals 9.0 units. Side ED measures 12.0 units in length.

response:
### Existence Type Questions:
1. Does point A exist? Yes
2. Does point C exist? Yes
3. Does point E exist? Yes
4. Does point G exist? Yes
5. Does point F exist? Yes
6. Does point H exist? Yes
7. Does point D exist on AB? No
8. Does point B exist on AD? No

### Attribute Type Questions:
1. Is the length of AB 10.0 units? Yes
2. Is the length of AD 9.0 units? Yes
3. Is the length of ED 12.0 units? Yes
4. Is the angle BAD 90 degrees? Yes
5. Is the angle DAF 90 degrees? Yes
6. Is the angle DEH 90 degrees? Yes
7. Is the length of CD 9.0 units? Yes
8. Is the length of EF 12.0 units? Yes

### Relationship Type Questions:
1. Is AB perpendicular to AD? Yes
2. Is AD perpendicular to DE? Yes
3. Is DE perpendicular to GH? Yes
4. Is AB parallel to CD? Yes
5. Is AD parallel to EF? Yes
6. Is EF parallel to GH? Yes
7. Is AB perpendicular to CD? No
8. Is AD perpendicular to CD? No
9. Is EF perpendicular to CD? No
10. Is the semicircle with diameter CD outside the rectangle ABCD? Yes
11. Is the semicircle with diameter EF inside the rectangle ADEF? Yes

Example2:
caption:
The function can be described by the equation y = - 2*x^4 - x^3 - 3*x^2 - 3. In the x range [-6.0, 4.0], zero points do not exist. The range of x is [-6.0, 4.0]. Within this range, the function peaks at 0.0 with a maximum value of -3.0, but no minimum value point exists. There are zero asymptotes associated with the functions.

response:
### Existence Type Questions:
1. Does a zero point exist for the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) in the range \([-6.0, 4.0]\)? No
2. Does a maximum value point exist for the function within the range \([-6.0, 4.0]\)? Yes
3. Does a minimum value point exist for the function within the range \([-6.0, 4.0]\)? No
4. Do any asymptotes exist for the function \( y = -2x^4 - x^3 - 3x^2 - 3 \)? No

### Attribute Type Questions:
1. Is the maximum value of the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) within the range \([-6.0, 4.0]\) equal to -3.0? Yes
2. Is the x-coordinate of the maximum value point of the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) within the range \([-6.0, 4.0]\) equal to 0.0? Yes
3. Is the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) continuous within the range \([-6.0, 4.0]\)? Yes

### Relationship Type Questions:
1. Is the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) increasing at \( x = 0.0 \) within the range \([-6.0, 4.0]\)? No
2. Is the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) decreasing at \( x = 0.0 \) within the range \([-6.0, 4.0]\)? Yes
3. Is the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) concave up at \( x = 0.0 \) within the range \([-6.0, 4.0]\)? No
4. Is the function \( y = -2x^4 - x^3 - 3x^2 - 3 \) concave down at \( x = 0.0 \) within the range \([-6.0, 4.0]\)? Yes

Now, according to the following image caption, provide these three types of questions in the output format:
caption: 
{caption}"""
temp1 = PromptTemplate(input_variables=['caption'], template=temp1)

# 2. 数学标记类（直角）
temp2 = """You are a math teacher, now given some known image caption information, you need to provide two types of problems involving right angles based on the description:
1. Attribute Type Questions
2. Relationship Type Questions

You should keep in mind that the questions you ask must be about right angles or about the sides and points that make up a right angle. 
If a right angle appears in a triangle, you can ask questions based on the properties of a right triangle.This includes the length of each side and the angle of each corner.
For sides and points that are not related to right angles, no questions need to be raised.

Output format requirements:
1. Use ### to mark before each category
2. After each question, give the answer directly using Yes or No, no need to provide reasons.
3. There should be a blank line separating the three categories of questions.

Here is an example:
caption: 
The triangle has a right angle at vertex B. The measure of angle ACB is 37°. The length of side AB is 12.0 units. The sketch may not show all the edges of the triangle.

response: 
### Attribute Type Questions:
1. Is the length of AB 12.0 units? Yes
2. Is the measure of angle ACB 37°? Yes
3. Is the measure of angle ABC 90°? Yes
4. Is the length of BC 15.0 units? No
5. Is the measure of angle BAC 53°? Yes

### Relationship Type Questions:
1. Is AB perpendicular to BC? Yes
2. Is AB parallel to AC? No
3. Is AC the hypotenuse of the triangle? Yes
4. Is the triangle ABC a right-angled triangle? Yes
5. Is the triangle ABC an isosceles triangle? No

Now, according to the following image caption, provide these two types of questions in the output format:
caption:
{caption}"""
temp2 = PromptTemplate(input_variables=['caption'], template=temp2)

# 3. 不确定性回答问题
temp3 = """You are a math teacher, now given some known image caption information, you need to provide two types of problems about the uncertainty of the image based on the description:
1. Attribute Type Questions
2. Relationship Type Questions

Definition of the uncertainty of the image:
The shapes in the graphic appear to have certain special properties (such as right angles, rhombuses, squares, regular polygons, isosceles triangles, and equilateral triangles, etc.), but they are not mentioned in the caption.

Follow these steps to design questions and provide answers:
1. Observe the image and determine if there are any special shapes (such as right angles, rhombuses, squares, regular polygons, isosceles triangles, and equilateral triangles, etc.). Remember their letter descriptions.
2. Look at the caption to see if it mentions any of these shapes.
3. Determine which shapes are part of the uncertainty of the image.
4. Design questions and provide answers. For the uncertainty of the image, the answer must be Uncertain.

You need to keep in mind that your questions should be related to the uncertainty of the image, and any questions unrelated to this should not be asked.

Output format requirements:
1. Use ### to mark before each category
2. After each question, give the answer directly using Yes or No or Uncertain, no need to provide reasons.
3. There should be a blank line separating the three categories of questions.

Here is an example of output format:
### Attribute Type Questions:
1. Is the shape of ... a square? Uncertain
2. Is the shape of ... a right triangle? Uncertain
3. Is the shape of ... a isosceles triangle? Uncertain
4. Is the shape of ... a equilateral triangle? Uncertain

### Relationship Type Questions:
1. Is ... perpendicular to ...? Uncertain
2. Is ... parallel to ...? Uncertain
3. Is ... equal to ...? Uncertain

Now, according to the following image and image caption, provide these two types of questions in the output format:
caption:
{caption}"""
temp3 = PromptTemplate(input_variables=['caption'], template=temp3)

# 4. 自以为然，多加条件（针对remove数据，然后询问 这位虚假的边 有没有 某种属性，模型应该回答不确定，并给出好的理由）
temp4 = """You are a math teacher, now given some known image caption information, you need to provide three types of problems including the removed edges based on the description:
1. Existence Type Questions
2. Attribute Type Questions
3. Relationship Type Questions

You should keep in mind that the questions you ask must be related to the edge that has been removed. 
For questions that are unrelated to the removed edge, you do not need to ask. 

You need to follow these steps:
1. Analyze which edges have been removed or eliminated through the image and caption.If an edge is removed in caption but still exist in image, the edge exists.
2. Ask questions around the removed edges and give the answer: Yes or No or Uncertain.

When an edge is removed or eliminated, the following occurs:
1. The edge no longer exists, but the two points does exist. When asked if the removed edge exists, you should respond "No".
2. The length of the edge no longer exists, but the distance between the two points does exist. When asked about the length of the removed edge, you should respond "Uncertain".
3. The edge cannot be parallel or perpendicular to any other edge, because it no longer exists. When asked about the relationship between the removed edge and the other sides, you should respond "Uncertain".
4. The edge does not form a corner. When asked whether the angle formed by this side exists or what its degree measure is, you should respond "Uncertain".

Output format requirements:
1. Use ### to mark before each category
2. After each question, give the answer directly using Yes or No or Uncertain, no need to provide reasons.
3. There should be a blank line separating the three categories of questions.

Here is an example:
caption:
A rectangle ABCD with a length of 8.0 units for side AB and 6.0 units for side CB has been designed. Side CD has been extended inward to form an equilateral triangle, then removed. A rectangle CBFG has been joined to edge CB, and side FG has been eliminated. A sector CGH has been designed adjacent to edge CG. Angle BAD measures 90 degrees, angle BCG measures 90 degrees, and angle HGC measures 30 degrees. Side CG is 5.0 units long.

response:
### Existence Type Questions:
1. Does edge CD exist? No
2. Dose edge FG exist? No
3. Does point C exist? Yes
4. Does point C exist on AB? No
5. Does point F exist? Yes
6. Does point G exist? Yes
7. Does point H exist? Yes


### Attribute Type Questions:
1. Is the length of CD 8.0 units? Uncertain
2. Is the length of FG 6.0 units? Uncertain
4. Is the angle of angle ADC 90 degrees? Uncertain
5. Is the angle of angle CGF 90 degrees? Uncertain
6. Is the angle of angle FGC 90 degrees? Uncertain
7. Is the the distance from point C to point D 8.0 units? Yes
8. Is the the distance from point F to point G 6.0 units? Yes

### Relationship Type Questions:
1. Is AB perpendicular to CD? Uncertain
2. Is AB parallel to CD? Uncertain
3. Is CG perpendicular to FG? Uncertain
4. Is CG parallel to FG? Uncertain
5. Is the equilateral triangle formed by extending CD still present in the final design? No
6. Is the rectangle CBFG still intact after removing side FG? No

Now, according to the following image caption and image, provide these two types of questions in the output format:
caption:
{caption}"""
temp4 = PromptTemplate(input_variables=['caption'], template=temp4)


# 功能函数
# check prompt
def check_temp():
    tmp_caption = {'caption': 11111111}
    for i, t in enumerate([temp1, temp2, temp3, temp4]):
        print(f"\nprompt {i+1}:")
        print(t.format(**tmp_caption))

# 处理mavis数据
def get_mavis_data(mavis_path: str, select: str="all")->list[dict]:
    """
    select: all, Geo_Caption, function, Aynaltic_Geo
    return: [{'id': ..., 'image': ..., 'caption': ... }, ...]
    """
    assert select in ["all", "Geo_Caption", "function", "Aynaltic_Geo", "remove"], \
        "select must in [all, Geo_Caption, function, Aynaltic_Geo, remove]"
    with open(mavis_path, 'r') as json_file:
        mavis_data: list[dict] = json.load(json_file)
    print(f"MAVIS: {len(mavis_data)}")
    res_list = []
    for item in mavis_data:
        c_id = item["id"]
        img_path = os.path.join("MAVIS_Caption", item["image"].split("MAVIS_Caption/")[-1])
        cap = item["conversations"][1]["value"]
        if select != "all":
            if select == "remove":
                if not ("eliminate" in cap or "remove" in cap):
                    continue
            elif select not in img_path:
                continue
        res_list.append({"id": c_id, "image": img_path, "caption": cap})

    return res_list

def dict2csv(d: dict, path: str):
    df = pd.DataFrame(d)
    df.to_csv(path, index=False)
    pass

def dict2markdown(d: dict):
    df = pd.DataFrame(d)
    for i in range(df.shape[0]):
        data = df.iloc[i, :]
        c_id = data["id"]
        img_path = data["image"]
        cap = data["caption"]
        q_status = data["q_status"]
        q = data["raw_chat"]
        if str(q_status) == "-1":
            continue
        print(f"## id {c_id}")
        print(f"![]({img_path})")
        print(f"caption: \n{cap}")
        print(f"question: \n{q}")

def csv2dict(csv_path: str)->list[dict]:
    res = []
    df = pd.read_csv(csv_path)
    for i in range(df.shape[0]):
        data = {}
        cols = df.columns.tolist()
        for j in cols:
            data[j] = df.iloc[i, :][j]
        res.append(data)
    return res


# 访问API得到未分割问题
def process_one_caption(item: dict, temp: PromptTemplate, use_img: bool, mavis_root: str):
    """
    temp默认为: 存在、关系、属性
    item: {'id': ..., 'image': ..., 'caption': ... }
    回答后：
    item: {'id': ..., 'image': ..., 'caption': ..., 'q_status': ..., 'raw_chat': ...(一次生成多个问题) }
    """
    cap = {'caption': item['caption']}
    prompt = temp.format(**cap)
    if use_img:
        img_path = os.path.join(mavis_root, item["image"].split("MAVIS_Caption/")[-1])
        status, result = chatapi.get_gpt_response("chatgpt", prompt=prompt, image=img_path, model=model)
    else:
        status, result = chatapi.get_gpt_response("chatgpt", prompt=prompt, model=model)
    item.update({'q_status': status, 'raw_chat': result})
    pass

# 并发
def process_parallel_in_all_case(item_list: list, curr_temp: str, use_img: bool, mavis_root: str, max_workers: int, process_one, model_name='gpt', desc=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pbar = tqdm(total=len(item_list), desc=f"'{model_name}' 正在推理数据" if desc is None else desc)
        results_and_idx = []
        for i, item in enumerate(item_list):
            future = executor.submit(process_one, item, curr_temp, use_img, mavis_root)
            results_and_idx.append(future)
        for _ in concurrent.futures.as_completed(results_and_idx):
            pbar.update(1)
    print("done")
    return results_and_idx

kwd_dict = {"temp1": ['### Existence Type Questions:',\
                       '### Attribute Type Questions:',\
                       '### Relationship Type Questions:'],
            "temp2": ['### Attribute Type Questions:',\
                       '### Relationship Type Questions:'],
            "temp3": ['### Attribute Type Questions:',\
                       '### Relationship Type Questions:'],
            "temp4": ['### Existence Type Questions:',\
                       '### Attribute Type Questions:',\
                       '### Relationship Type Questions:'],
            }

def split_qa(qa: str):
    ans_list = ["Yes", "No", "Uncertain"]
    for ans in ans_list:
        if ans in qa:
            q = qa.split(ans)[0]
            a = ans
            return q, a
    return '', ''
        
def split_chat(qs: str):
    qs_list = qs.split("\n")
    res_list = []
    for x in qs_list:
        if x.strip()=="":
            continue
        else:
            res_list.append(x)
    return res_list

# split 不同模型可能需要调整
def split_items(res: list[dict], format_idx: str="temp1")->list[dict]:
    res_split = []
    kwd_list = kwd_dict[format_idx]
    for data in res:
        stat = data['q_status']
        if str(stat) == "-1":
            continue
        raw_chat = data['raw_chat']
        # print(repr(raw_chat))
        chats = raw_chat.split("\n\n")[:len(kwd_list)] # 调整
        for chat in chats:
            qa_list = []
            for kwd in kwd_list:
                if kwd in chat:
                    raw_qas = chat.split(kwd)[-1]
                    qa_list += split_chat(raw_qas)
                    break
            # print(qa_list)
            # assert 0
            for qa in qa_list:
                q, a = split_qa(qa)
                if q == '':
                    continue
                data['q'] = q
                data['gt'] = a
                tmp_data = data.copy()
                res_split.append(tmp_data)
    return res_split


if __name__ == "__main__":
    mavis_path = cfg_data["input_path"]
    mavis_root = cfg_data["input_root"]
    output_file = cfg_data["output_file"]
    temp_choice = cfg_data["temp_choice"] # 1, 2, 3, 4
    print("start benchmark pipeline")
    
    for k, v in cfg_data.items():
        print(f"{k}: {v}")

    # 切换template 
    # 1 通用, 
    if temp_choice == 1:
        curr_temp = temp1
        format_idx = "temp1"
        use_img = False
        select = "all"
    # 2 直角标记, 
    elif temp_choice == 2:
        curr_temp = temp2
        format_idx = "temp2"
        use_img = False
        select = "Geo_Caption"
    # 3 不确定, 
    elif temp_choice == 3:
        curr_temp = temp3
        format_idx = "temp3"
        use_img = True
        select = "Geo_Caption"
    # 4 针对remove
    elif temp_choice == 4:
        curr_temp = temp4
        format_idx = "temp4"
        use_img = True
        select = "remove"

    # ["all", "Geo_Caption", "function", "Aynaltic_Geo", "remove"]
    res_dict = get_mavis_data(mavis_path, select)
    process_parallel_in_all_case(
        item_list=res_dict,
        curr_temp=curr_temp,
        use_img=use_img,
        mavis_root= mavis_root,
        max_workers=100,
        process_one=process_one_caption,
        model_name=model,
    )
    dict2csv(res_dict, output_file) # 最好在split前先保存一次，避免split失败浪费token

    # res_dict = csv2dict(output_file) 
    res_dict = split_items(res_dict, format_idx=format_idx)
    dict2csv(res_dict, output_file)
    # dict2markdown(res_dict)
    # res_dict = csv2dict(output_file)
    # res_dict = split_items(res_dict)
    # dict2csv(res_dict, "test1.csv")
