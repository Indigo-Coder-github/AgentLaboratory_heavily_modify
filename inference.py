import json
import os
import time

import anthropic
import openai
import tiktoken
from openai import OpenAI
from transformers import pipeline

encoding = tiktoken.get_encoding("cl100k_base")

class Inference:
    
    TOKENS_IN, TOKENS_OUT = dict(), dict()

    @classmethod
    def curr_cost_est(cls) -> float:
        costmap_in = {
            "gpt-4o": 2.50 / 1000000,
            "gpt-4o-mini": 0.150 / 1000000,
            "o1-preview": 15.00 / 1000000,
            "o1-mini": 3.00 / 1000000,
            "claude-3-5-sonnet": 3.00 / 1000000,
            "deepseek-chat": 1.00 / 1000000,
            "o1": 15.00 / 1000000,
        }
        costmap_out = {
            "gpt-4o": 10.00 / 1000000,
            "gpt-4o-mini": 0.6 / 1000000,
            "o1-preview": 60.00 / 1000000,
            "o1-mini": 12.00 / 1000000,
            "claude-3-5-sonnet": 12.00 / 1000000,
            "deepseek-chat": 5.00 / 1000000,
            "o1": 60.00 / 1000000,
        }
        return sum([costmap_in[tokens]*cls.TOKENS_IN[tokens] for tokens in cls.TOKENS_IN]) + sum([costmap_out[tokens]*cls.TOKENS_OUT[tokens] for tokens in cls.TOKENS_OUT])

    @classmethod
    def query_model(cls, model_str, prompt, system_prompt, openai_api_key=None, anthropic_api_key=None, hf_model_path=None, tries=5, timeout=5.0,
                    temperature=None, print_cost=True,) -> str:
        preloaded_api = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None and preloaded_api is not None:
            openai_api_key = preloaded_api
        if openai_api_key is None and anthropic_api_key is None:
            raise Exception("No API key provided in query_model function")
        if openai_api_key is not None:
            openai.api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if anthropic_api_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        for _ in range(tries):
            try:
                match model_str:
                    case ["gpt-4o-mini", "gpt4o-mini", "gpt4omini", "gpt-4omini", "o1-mini"]:
                        model_str = "gpt-4o-mini"
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}]
                        client = OpenAI()
                        completion = client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temperature)
                        answer = completion.choices[0].message.content
                    case "claude-3.5-sonnet":
                        client = anthropic.Anthropic(
                            api_key=os.environ["ANTHROPIC_API_KEY"])
                        message = client.messages.create(
                            model="claude-3-5-sonnet-latest",
                            system=system_prompt,
                            messages=[{"role": "user", "content": prompt}])
                        answer = json.loads(message.to_json())["content"][0]["text"]
                    case ["gpt4o", "gpt-4o"]:
                        model_str = "gpt-4o"
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}]
                        client = OpenAI()
                        completion = client.chat.completions.create(model="gpt-4o-2024-08-06", messages=messages, temperature=temperature)
                        answer = completion.choices[0].message.content
                    case "deepseek-chat":
                        model_str = "deepseek-chat"
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}]
                        deepseek_client = OpenAI(
                            api_key=os.getenv('DEEPSEEK_API_KEY'),
                            base_url="https://api.deepseek.com/v1"
                        )
                        completion = deepseek_client.chat.completions.create(model="deepseek-chat", messages=messages,  temperature=temperature)
                        answer = completion.choices[0].message.content
                    case "o1":
                        model_str = "o1"
                        messages = [
                            {"role": "user", "content": system_prompt + prompt}]
                        client = OpenAI()
                        completion = client.chat.completions.create(
                            model="o1-2024-12-17", messages=messages)
                        answer = completion.choices[0].message.content
                    case "o1-preview":
                        model_str = "o1-preview"
                        messages = [
                            {"role": "user", "content": system_prompt + prompt}]
                        client = OpenAI()
                        completion = client.chat.completions.create(
                            model="o1-preview", messages=messages)
                        answer = completion.choices[0].message.content
                    case "huggningface":
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}]
                        client = pipeline("text-generation", model=hf_model_path, trust_remote_code=True)
                        answer = client(messages)[0]["generated_text"]

                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                elif model_str in ["deepseek-chat"]:
                    encoding = tiktoken.encoding_for_model("cl100k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_str)
                if model_str not in cls.TOKENS_IN:
                    cls.TOKENS_IN[model_str] = 0
                    cls.TOKENS_OUT[model_str] = 0
                cls.TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                cls.TOKENS_OUT[model_str] += len(encoding.encode(answer))
                if print_cost:
                    print(f"Current experiment cost = ${
                          cls.curr_cost_est()}, ** Approximate values, may not reflect true cost")
                return answer
            except Exception as e:
                print("Inference Exception:", e)
                time.sleep(timeout)
                continue
        raise Exception("Max retries: timeout")

curr_cost_est = Inference.curr_cost_est
query_model = Inference.query_model
# print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
