import os
import json
import base64
import pandas as pd
from openai import OpenAI
from datetime import datetime

class Evaluator:
    def __init__(self, config_path) -> None:
        self.benchmark = None
        self.config = None
        with open(config_path, "r") as f:
            self.config = json.load(f)
        pass

    def _encode_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def load_benchmark(self, json_path):
        with open(json_path, "r") as f:
            self.benchmark = json.load(f)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
    def model_response(self, api_key, model_name, image_path, question):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        image_base64 = self._encode_base64(image_path)
        messages = [{"role": "user","content": [
            {"type": "text","text": question + "\nAnswer format:\nYes\n\nyour reason\n"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}]
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(model=model_name, messages=json.loads(json.dumps(messages)))
        return response.choices[0].message.content

    def discriminate(self, api_key, model_name, answer, ground_truth):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': f'This is the model answer:\n {answer}\nThis is the ground truth:\n{ground_truth}\nPlease judge if the answer is right. Answer format:\nYes or No.\n\nyour judugement reason'}]
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content
    
    def run(self):
        results = {"model": [], "result": []}
        self.load_benchmark(self.config["input_path"])
        discriminator_name = self.config["discriminator"]["name"]
        discriminator_api_key = self.config["discriminator"]["api_key"]
        for model in self.config["evaluator"]:
            model_api_key = model["api_key"]
            model_name = model["name"]
            num_correct = 0
            for item in self.benchmark:
                image_path = item["image_path"]
                question = item["question"]
                ground_truth = item["ground_truth"]
                answer = self.model_response(model_api_key, model_name, image_path, question)
                judgement = self.discriminate(discriminator_api_key, discriminator_name, answer, ground_truth)
                if "Yes" in judgement.split("\n\n")[0]:
                    num_correct += 1
            results["model"].append(model_name)
            results["result"].append(num_correct/len(self.benchmark))
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pd.DataFrame(results).to_csv(os.path.join(self.config["output_path"], now_time), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='evaluator',
        description='Pipline of evaluation'
    )
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    evaluator = Evaluator(args.config)
    evaluator.run()
    
    