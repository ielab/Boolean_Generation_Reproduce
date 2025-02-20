from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from openai import OpenAI
import os
from mistralai import Mistral
import time
import re
import json
from pydantic import BaseModel

class  Boolean_Output(BaseModel):
    boolean_query: str

class Generator(ABC):
    def __init__(self, model_name=None, max_new_tokens=2048, max_length=None, quantization=None, **kwargs):
        if not model_name:
            raise ValueError("A model_name must be provided.")
        self.max_length = max_length
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.json_output = kwargs.get('json_output', False)


    @abstractmethod
    def generate_batch(self, batched_instructions):
        pass

    @abstractmethod
    def compile_prompt(self, prompt, existing_prompt_list=None, **kwargs):
        pass


import re

class Llama(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attn = "sdpa" if "tinyllama" in self.model_name.lower() else "flash_attention_2"

        if self.quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                attn_implementation=attn,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                attn_implementation=attn,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.temperature = kwargs.get('temperature', 1.0)
        self.seed = kwargs.get('seed', 42)

    def set_temperature(self, temperature):
        self.model.generation_config.temperature = temperature

    def set_seed(self, seed):
        self.model.generation_config.seed = seed
        self.seed = seed

    def generate_batch(self, batched_instructions, target="boolean_query"):
        instructions_tokenized = self.tokenizer(
            batched_instructions,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to('cuda')

        input_len = instructions_tokenized['input_ids'].shape[1]
        with torch.no_grad():
            generated_responses = self.model.generate(
                **instructions_tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                #seed = self.seed
            )
            generated_responses = generated_responses[:, input_len:]
        final_generated =  self.tokenizer.batch_decode(generated_responses, skip_special_tokens=True)
        for i in range(len(final_generated)):
            if self.json_output:
                try:
                    final_generated[i] = json.loads(final_generated[i])[target]
                except:
                    final_generated[i] = None
        return final_generated


    def compile_prompt(self, prompt, existing_prompt_dict=None, **kwargs):
        """
        Dynamically fills placeholders found in prompt['user'] using kwargs.
        """
        extracted_placeholders = re.findall(r"{(.*?)}", prompt["user"])  # Find all placeholders
        replacement_dict = {key: kwargs.get(key, f"Default {key}") for key in extracted_placeholders}

        def format_prompt(text, replace_dict):
            return text.format(**replace_dict)

        # Initialize prompt dictionary if needed
        if not existing_prompt_dict or not existing_prompt_dict.get("system"):
            existing_prompt_dict = {
                "system": [{"role": "system", "content": prompt["system"]}],
                "example": [],
                "user": []
            }

        # Process examples
        if prompt.get("example"):
            for example in prompt["example"]:
                example_replacements = {key: example.get(key, f"Default {key}") for key in extracted_placeholders}
                existing_prompt_dict["example"].extend([
                    {"role": "user", "content": format_prompt(prompt["user"], example_replacements)},
                    {"role": "assistant", "content": example["response"]}
                ])

        # Process user input
        existing_prompt_dict["user"].append(
            {"role": "user", "content": format_prompt(prompt["user"], replacement_dict)}
        )

        existing_prompt_list = (
            existing_prompt_dict["system"] +
            existing_prompt_dict["example"] +
            existing_prompt_dict["user"]
        )
        return existing_prompt_dict, self.tokenizer.apply_chat_template(
            existing_prompt_list, add_generation_prompt=True, tokenize=False
        )

class APIModel(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure we have the json_output flag from kwargs
        self.json_output = kwargs.get('json_output', False)
        if "gpt" in self.model_name or "o3" in self.model_name or "o1" in self.model_name:
            api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY'))
            self.model = OpenAI(api_key=api_key)
        elif "mistral" in self.model_name or "mixtral" in self.model_name:
            api_key = kwargs.get('api_key', os.environ.get('MISTRAL_API_KEY'))
            self.model = Mistral(api_key=api_key)
        self.temperature = kwargs.get('temperature', 1.0)
        self.seed = kwargs.get('seed', 42)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_seed(self, seed):
        self.seed = seed

    def generate_batch(self, batched_instructions, target="boolean_query"):
        generated_responses = []
        for instruction in batched_instructions:
            if "gpt" in self.model_name or "o3" in self.model_name or "o1" in self.model_name:
                if self.json_output:
                    completion = self.model.chat.completions.create(
                        model=self.model_name,
                        messages=instruction,
                        temperature=self.temperature,
                        seed=self.seed,
                        response_format={"type": "json_object"},
                    )
                    try:
                        response = json.loads(completion.choices[0].message.content)[target]
                    except:
                        response = None
                else:
                    completion = self.model.chat.completions.create(
                        model=self.model_name,
                        messages=instruction,
                        temperature=self.temperature,
                        seed=self.seed
                    )
                    response = completion.choices[0].message.content

            elif "mistral" in self.model_name or "mixtral" in self.model_name:
                if self.json_output:
                    completion = self.model.chat.complete(
                        model=self.model_name,
                        messages=instruction,
                        temperature=self.temperature,
                        random_seed=self.seed,
                        response_format={"type": "json_object"}
                    )
                    try:
                        response = json.loads(completion.choices[0].message.content)[target]
                        print(response)
                    except:
                        response = None
                else:
                    completion = self.model.chat.complete(
                        model=self.model_name,
                        messages=instruction,
                        temperature=self.temperature,
                        random_seed=self.seed
                    )
                    response = completion.choices[0].message.content
                time.sleep(0.5)

            generated_responses.append(response)
        return generated_responses

    def compile_prompt(self, prompt, existing_prompt_dict=None, **kwargs):
        """
        Dynamically fills placeholders found in prompt['user'] using kwargs.
        This version does NOT use tokenization. If json_output is True,
        returns the prompt as a JSON-formatted string.
        """
        extracted_placeholders = re.findall(r"{(.*?)}", prompt["user"])
        replacement_dict = {key: kwargs.get(key, f"Default {key}") for key in extracted_placeholders}

        def format_prompt(text, replace_dict):
            return text.format(**replace_dict)

        if not existing_prompt_dict or not existing_prompt_dict.get("system"):
            existing_prompt_dict = {
                "system": [{"role": "system", "content": prompt["system"]}],
                "example": [],
                "user": []
            }

        if prompt.get("example"):
            for example in prompt["example"]:
                example_replacements = {key: example.get(key, f"Default {key}") for key in extracted_placeholders}
                existing_prompt_dict["example"].extend([
                    {"role": "user", "content": format_prompt(prompt["user"], example_replacements)},
                    {"role": "assistant", "content": example["response"]}
                ])

        formatted_user_prompt = format_prompt(prompt["user"], replacement_dict)

        # For Mistral/Mixtral, prepend the system message to the user prompt
        if "mistral" in self.model_name or "mixtral" in self.model_name:
            if "system" in prompt:
                formatted_user_prompt = f"{prompt['system']}\n{formatted_user_prompt}"

        existing_prompt_dict["user"].append({"role": "user", "content": formatted_user_prompt})

        existing_prompt_list = (
            existing_prompt_dict["system"] +
            existing_prompt_dict["example"] +
            existing_prompt_dict["user"]
        )

        return existing_prompt_dict, existing_prompt_list

