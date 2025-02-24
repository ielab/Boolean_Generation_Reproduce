# Boolean_Generation_Reproduce
Boolean Query Generation Reproduce
---

## 1. Prerequisites

1. **Python 3.10**  
   Code are run and tested on Python 3.10

2. **Required Python Libraries**  
    The following Python libraries are required to run the code:
    - `torch`
    - `transformers`
    - `openai`
    - `mistralai`
    - `biopython`

## 2. Formulation or Refinement of Boolean Queries (q1-q7)

Below is the script that runs the reproduction of the Boolean Query Formulation/Refinement Prompts
    
```bash
python3 process_reproduce.py
  --input_queries data/topics/CLEF-2018.jsonl # input queries, could be either CLEF-2018.jsonl or seed_collection.jsonl
  --method api_based # either api_based or l
  --llm_prompt_file ${prompt} # prompt file, should store in generation_prompts folder
  --model ${model} # model name, such as gpt-3.5-turbo-0125 for api_based and mistralai/Mistral-7B-Instruct-v0.2 for llm_based
  --date_file data/topics/combined_pubdates # date file, needed for CLEF, not Seed Collection
  --output_folder CLEF-2018_reproduce/${model}/${prompt}_v${time} # output folder
  --no_retrieval False # if True, formulated Boolean query will not be retrieved
  --json_output False # if True, output will be enforced to json
```


## 3. Evaluation of Boolean Queries

Below is the script that runs the reproduction of the Boolean Query Evaluation

```bash
python3 eval_all.py \
  --input_trec_folder CLEF-2018_reproduce/${model}/${prompt}_v${time}/final_trec_result \ # trec result folder
  --input_qrels data/topics/qrels_2018.txt # qrels file
```




