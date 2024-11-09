import functools
import json
import threading
import requests
requests.packages.urllib3.disable_warnings()
from openai import OpenAI
import base64
import re
from tqdm import tqdm
from typing import List, Dict, Any
import io
import base64
from collections import defaultdict
from io import BytesIO

"""
chatglm4-32b, chatglm4-32b-public在火山可用，chatglm-embedding不可用
chatgpt在火山可用，支持模型为https://zhipu-ai.feishu.cn/docx/PnNadsGiUovHovxX7ofcGeAjnAc?from=from_copylink
"""


# ==========================================
#  师弟，这个需要自己写一个api server，下边的仅GLM组的内网可用
# ===========================================
# 定义装饰器，超时时间为10秒
def timeout_after_80s(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = [(-1, "too long time")]
        def target():
            result[0] = func(*args, **kwargs)
        
        thread = threading.Thread(target=target)
        thread.start()
        time = 300
        thread.join(time)  # 超时时间设为10秒
        
        if thread.is_alive():
            return (-1, f'Timeout after {time}s')
        return result[0]
    
    return wrapper


class ChatAPI():
    def __init__(self, model_name) -> None:
        self.api_config = {
            "chatglm4-32b": {
                "url": "https://api.chatglm.cn/v1/chat/completions",
                "headers": {
                    "Authorization": "Bearer PbWF8a9tDNHcntV0ie8VERQ8WDaiFfu8vcLM7fB8XMmiQJeGkUCL2xiPHckt2DcF"
                },
                "parameters": {
                    "model": "chatglm-qingyan",
                    # "do_sample": False,
                    "max_tokens": 2048,
                    "stream": False,
                    # "seed": 1234
                },
                "post_processor": self._post_chatglm4_32b,
                "pre_processor": self._pre_chatgpt
            },
            "chatglm4-32b-public": {
                "url": "https://api.chatglm.cn/v1/chat/completions",
                "headers": {
                    "Authorization": "Bearer PbWF8a9tDNHcntV0ie8VERQ8WDaiFfu8vcLM7fB8XMmiQJeGkUCL2xiPHckt2DcF"
                },
                "parameters": {
                    "model": "glm-4-public",
                    # "do_sample": False,
                    "max_tokens": 2048,
                    "stream": False,
                    # "seed": 1234
                },
                "post_processor": self._post_chatglm4_32b,
                "pre_processor": self._pre_chatgpt
            },
            "chatglm-embedding_single": {
                "url": "https://117.161.233.25:8443/v3/embeddings",
                "headers": {
                    "Content-Type": "application/json",
                    "Host": "embedding-api.glm.ai",    
                    "charset": "utf-8"
                },
                "parameters": {
                    "taskId": 0,
                    'model': 'large-v2',
                    "max_length": 2048,
                },
                "post_processor": self._post_chatglm_embedding,
                "pre_processor": self._pre_chatglm_embedding
            },
            "chatglm-embedding": {
                "url": "https://117.161.233.25:8443/v3/embeddings",
                "headers": {
                    "Content-Type": "application/json",
                    "Host": "embedding-api.glm.ai",    
                    "charset": "utf-8"
                },
                "parameters": {
                    "taskId": 0,
                    'model': 'large-v2',
                    "max_length": 2048,
                },
                "post_processor": self._post_chatglm_embedding2,
                "pre_processor": self._pre_chatglm_embedding2
            },
            "chatgpt": {
                "url": "https://one-api.glm.ai/v1",
                # "api_key": "sk-...",  # no money 
                "api_key": "sk-...",  # 1015 switch
                "parameters": {
                    "temperature": 1.0,
                    "max_tokens": 2048,
                    "top_p": 1.0
                },
                "post_processor": self._post_chatgpt,
                "pre_processor": self._pre_chatgpt
            }
        }
        self.failed = "Request Failed"
        ## Fix: Error code: 400 - {'error': {'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead. 
        if 'o1' in model_name:
            print(f'o1 model, remove max_tokens')
            del self.api_config['chatgpt']['parameters']['max_tokens']


    def image_to_base64(self, image_path_or_sth):
        if isinstance(image_path_or_sth, str):
            image_path = image_path_or_sth
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            
        elif isinstance(image_path_or_sth, BytesIO):
            byte_arr = image_path_or_sth
            byte_arr = byte_arr.getvalue()
            encoded_string = base64.b64encode(byte_arr)
        else:
            # 当做是一个png Image对象
            byte_arr = io.BytesIO()
            image_path_or_sth.save(byte_arr, format='PNG')
            byte_arr = byte_arr.getvalue()
            encoded_string = base64.b64encode(byte_arr)
        
        return encoded_string.decode('utf-8')

    def get_api_servers(self):
        return list(self.api_config.keys())
    
    def png_image_file_obj_to_base64(self, png_image_file_obj):
        if isinstance(png_image_file_obj, BytesIO):
            byte_arr = png_image_file_obj
            byte_arr = byte_arr.getvalue()
        else:
            # png对象
            # 创建一个BytesIO对象来保存图像数据
            byte_arr = io.BytesIO()
            png_image_file_obj.save(byte_arr, format='PNG')
            # 获取字节数据
            byte_arr = byte_arr.getvalue()
        
        # 使用base64对字节数据进行编码
        encoded_string = base64.b64encode(byte_arr)
        
        # 将字节转换成字符串并返回
        return encoded_string.decode('utf-8')

    def _pre_chatgpt(self, prompt, **kwargs):
        if "image" not in kwargs:
            update_parameters = {"messages": [{"role": "user", "content": prompt}], **kwargs}
        else:
            image_path_or_obj = kwargs.pop("image")
            if not isinstance(image_path_or_obj, list):
                update_parameters = {"messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": 
                                f"data:image/png;base64,{self.image_to_base64(image_path_or_obj)}"}
                            }
                        ]
                    }
                ], **kwargs}
            else:
                image_item_list = []
                for each in image_path_or_obj:
                    image_url = dict(url=f"data:image/png;base64,{self.image_to_base64(each)}")
                    image_item_list.append(
                        dict(type="image_url", image_url=image_url)
                    )
                update_parameters = kwargs
                update_parameters["messages"] = [
                    dict(role="user", content=[
                            {"type": "text", "text": prompt},
                            *image_item_list
                        ])
                ]
                  
        return update_parameters
    
    def _post_chatgpt(self, response):
        return 200, response.choices[0].message.content

    def _pre_chatglm_embedding(self, prompt):
        update_parameters = {"input": [{"request": {"prompt": prompt}, "callBackId": 0}]}
        return update_parameters

    def _pre_chatglm_embedding2(self, prompt):
        update_parameters = {"input": [{"request": {"prompt": prompt[0]}, "callBackId": 0}, 
                                       {"request": {"prompt": prompt[1]}, "callBackId": 1}]}
        return update_parameters

    def _post_chatglm4_32b(self, response):
        status, result = response.status_code, self.failed
        if status == 200:
            result = json.loads(response.text)["choices"][0]["message"]["content"].strip()
        return status, result

    def _post_chatglm_embedding(self, response):
        status, result = response.status_code, self.failed
        if status == 200:
            output = json.loads(response.text)['data']['outputText']
            result = output['0']['outputText']
        return status, result

    def _post_chatglm_embedding2(self, response):
        status, result = response.status_code, self.failed
        if status == 200:
            output = json.loads(response.text)['data']['outputText']
            result = [output['0']['outputText'], output['1']['outputText']]
        return status, result

    


    def get_response(self, api_server, prompt, **kwargs) -> str:
        config = self.api_config[api_server]
        parameters = config["parameters"]
        parameters.update(config["pre_processor"](prompt, **kwargs))
        # print(parameters)
        status, result = -1, ""
        try:
            with requests.post(config["url"], 
                               headers=config["headers"],
                               json=parameters,
                               verify=False,
                               timeout=50) as response:
                # print(response.text)
                status, result = config["post_processor"](response)
        except Exception as e:
            print(str(e))
        return status, result

    @timeout_after_80s
    def get_gpt_response(self, api_server, prompt, **kwargs) -> str:
        config = self.api_config[api_server]
        parameters = config["parameters"]
        client = OpenAI(api_key=config['api_key'], base_url=config['url'])
        parameters.update(config["pre_processor"](prompt, **kwargs))
        status, result = -1, ""
        try:
            response = client.chat.completions.create(**parameters)
            status, result = config["post_processor"](response)
        except Exception as e:
            print(str(e))
        return status, result



    
if __name__ == "__main__":
    model_name = "gpt-4o-2024-08-06"
    # model_name = "o1-preview-2024-09-12"
    
    # chatapi = ChatAPI(model_name=model_name)
    # prompt, image = "描述这张图片", "/workspace/linjh/self/assets/lin.jpeg"
    # status, result = chatapi.get_gpt_response("chatgpt", prompt=prompt, image=image, model="gpt-4o-2024-05-13")
    # print(status, result, len(result))
    
    
    
