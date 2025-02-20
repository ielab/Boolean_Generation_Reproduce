import time
from argparse import ArgumentParser
import json
import yaml
import os
from generator import Llama, APIModel
from tqdm import tqdm
import glob
from check_bool_query import check_correct, check_response
from pubmed_submission import pubmed_submission, dates_check
import random

max_retries = 20
default_seed = 42
default_extractor = APIModel(model_name="gpt-3.5-turbo-0125", temperature=0)

default_dates = {
    "mindate": "1975/01/01",
    "maxdate": "2023/12/31"
}


topic_id_not_in_irj = ["CD010771", "CD011145", "CD010772", "CD010775", "CD010783", "CD010896", "CD007431", "CD010860"]


def boolean_generation(queries_dicts, date_dict, method, model_name, llm_prompt_file, quantization, output_folder, temperature, extractor=default_extractor, extraction_prompt_file=None, no_retrieval=False, json_output=False):
    retry_file = os.path.join(output_folder, "retry_list.jsonl")
    if method == "llm_based":
        # if "llama" not in model_name:
        #     raise NotImplementedError("Only support llama model")
        # model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        model = Llama(model_name=model_name, quantization=quantization, temperature=temperature, json_output=json_output)
    elif method == "api_based":
        # model_name = "gpt3.5"
        model = APIModel(model_name=model_name, temperature=temperature, json_output=json_output)

    elif method == "no_loading":
        model = None

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    generation_output_folder = os.path.join(output_folder, "generation_output")
    if not os.path.exists(generation_output_folder):
        os.makedirs(generation_output_folder)

    llm_prompt_file = os.path.join("generation_prompts", llm_prompt_file + ".yaml")
    with open(llm_prompt_file, "r") as f:
        generation_steps = yaml.load(f, Loader=yaml.FullLoader)["Steps"]
    extraction_prompt_file = os.path.join("generation_prompts", extraction_prompt_file + ".yaml")
    if extraction_prompt_file is not None:
        with open(extraction_prompt_file, "r") as f:
            extraction_step = yaml.load(f, Loader=yaml.FullLoader)["Steps"][0]
    else:
        extraction_step = None


    # Prepare existing prompt lists (one per query in queries_list)
    existing_prompt_dicts = {
        qid:
            {
                "system": [],
                "user": [],
                "example": []
            } for qid in queries_dicts.keys()}
    qid_list = list(queries_dicts.keys())

    # --------------------------------------------
    # Main Generation Loop
    # --------------------------------------------
    for step_i, step in enumerate(generation_steps):
        print("Now processing step", step_i)
        current_generation_output_file = os.path.join(generation_output_folder, f"step_{step_i}.jsonl")

        # ---------------------------------------------------------------------
        # 2.1. Read existing partial results (if any), populate skip set
        # ---------------------------------------------------------------------
        skip_qids = set()

        if os.path.exists(current_generation_output_file):
            with open(current_generation_output_file, "r") as f:
                for line in f:
                    current_existing_dict = json.loads(line)
                    qid = current_existing_dict["id"]
                    skip_qids.add(qid)
                    if qid in qid_list:
                        existing_prompt_dicts[qid] = current_existing_dict["existing_prompts"]


        # We open in append mode so that if we generate new results,
        # we can write them line by line to preserve partial progress.
        f_out = open(current_generation_output_file, "a")

        # ---------------------------------------------------------------------
        # 2.2. Batching
        # ---------------------------------------------------------------------
        batch_size = step.get("batch_size", 1)
        bool_extract = step.get("bool_extract", False)

        for i in tqdm(range(0, len(qid_list), batch_size)):
            batched_qids = qid_list[i:i + batch_size]
            # -----------------------------------------------------------------
            # 2.3. Prompt Compilation
            # -----------------------------------------------------------------
            batched_instructions = []
            qids_to_generate = []

            for qid in batched_qids:
                if qid in skip_qids:
                    continue
                query = queries_dicts[qid]
                existing_prompt_dict, instruction = model.compile_prompt(
                    step,
                    existing_prompt_dicts[qid],
                    **query
                )
                batched_instructions.append(instruction)
                existing_prompt_dicts[qid] = existing_prompt_dict
                qids_to_generate.append(qid)

            if len(batched_instructions) == 0:
                continue
            # -----------------------------------------------------------------
            # 2.4. Generation
            # -----------------------------------------------------------------
            try:
                responses = model.generate_batch(batched_instructions)
            except Exception as e:
                print(f"Error: {e}")
                # Regenerate with a different seed
                model.set_seed(random.randint(0, 100000))
                responses = model.generate_batch(batched_instructions)
                # Change seed back
                model.set_seed(default_seed)

            # -----------------------------------------------------------------
            # 2.5. Correctness Checks (only if final step), first do extraction and then correct check
            # -----------------------------------------------------------------
            if (step_i == len(generation_steps) - 1) or bool_extract:
                extracted_booleans = []
                # FIRST do extract of boolean query
                if not json_output:
                    for response_content in responses:
                        _, extraction_instruction = extractor.compile_prompt(extraction_step, response_content=response_content)
                        try:
                            extracted_boolean = extractor.generate_batch([extraction_instruction])[0]
                            extracted_booleans.append(extracted_boolean)
                        except Exception as e:
                            print(f"Error: {e}")
                            print(extraction_instruction)
                            extracted_booleans.append(None)
                else:
                    extracted_booleans = responses
                # Extract boolean queries
                for q_idx, qid in enumerate(qids_to_generate):
                    instruction = batched_instructions[q_idx]
                    # Get the appropriate date range
                    if qid in date_dict:
                        mindate = date_dict[qid]["mindate"]
                        maxdate = date_dict[qid]["maxdate"]
                    else:
                        mindate = default_dates["mindate"]
                        maxdate = default_dates["maxdate"]

                    final_boolean = extracted_booleans[q_idx]
                    retry_count = 0
                    already_generated_boolean = [final_boolean]

                    # Try multiple times if not correct, much meet
                    while (not check_correct(final_boolean, mindate, maxdate)) or (not check_response(final_boolean)):
                        time.sleep(1)  # Wait for 1 second before retrying
                        if retry_count >= max_retries:
                            print(
                                f"[WARNING] Query {qid} failed correctness after {max_retries} retries. Skipping.")
                            break

                        # If we've already generated this exact string, skip
                        # to avoid infinite loops on the same generation.
                        if final_boolean in already_generated_boolean and retry_count > 0:
                            print(
                                f"[WARNING] Query {qid} re-generated the same answer after {retry_count} retries. Skipping.")
                            model.set_seed(random.randint(0, 100000))

                        print(
                            f"[INFO] Query {qid} was incorrect. Re-generating last step (attempt {retry_count + 1})..."
                        )
                        try:
                            response_from_model = model.generate_batch([instruction])[0]
                            if not json_output:
                                _, validation_instruction = extractor.compile_prompt(extraction_step,
                                                                                     response_content=response_from_model)
                                response = extractor.generate_batch([validation_instruction])[0]
                            else:
                                response = response_from_model
                        except Exception as e:
                            print(f"Error: {e}")
                            #retry_count += 1
                            continue
                        # Change seed back
                        model.set_seed(default_seed)

                        final_boolean = response

                        already_generated_boolean.append(response)
                        retry_count += 1
                        # Append new response
                    existing_prompt_dicts[qid]["user"].append(
                        {"role": "assistant", "content": final_boolean}
                    )
                    with open(retry_file, "a") as f:
                        f.write(json.dumps({
                            "qid": qid,
                            "step": step_i,
                            "count": retry_count
                        }) + "\n")
            else:

                for q_idx, qid in enumerate(qids_to_generate):
                    existing_prompt_dicts[qid]["user"].append(
                        {"role": "assistant", "content": responses[q_idx]}
                    )
            # -----------------------------------------------------------------
            # 2.7. Write partial progress for this batch to disk
            # -----------------------------------------------------------------

            for q_idx, qid in enumerate(qids_to_generate):
                current_existing_dict = {
                    "id": qid,
                    "topic": queries_dicts[qid]["topic"],
                    "existing_prompts": existing_prompt_dicts[qid]
                }
                # Write line by line in append mode
                f_out.write(json.dumps(current_existing_dict) + "\n")
                f_out.flush()  # ensure immediate write to file

        # Done with all batches for this step
        f_out.close()

    # --------------------------------------------
    # Submission Final Assistant Responses
    # --------------------------------------------
    if no_retrieval:
        print("Skipping final submission to PubMed.")
        return

    qid_already_submitted = set()
    final_trec_folder = os.path.join(output_folder, "final_trec_result")
    if not os.path.exists(final_trec_folder):
        os.makedirs(final_trec_folder)
    else:
        final_trec_files = glob.glob(final_trec_folder + '/*')
        for final_trec_file in final_trec_files:
            with open(final_trec_file) as f:
                for line in f:
                    qid = line.split()[0]
                    qid_already_submitted.add(qid)
                    break

    counter_too_many = 0

    # Now we do the final submission to PubMed
    for qid, item in tqdm(existing_prompt_dicts.items()):
        final_boolean = item["user"][-1]["content"]
        if qid in qid_already_submitted:
            continue

        final_trec_file = os.path.join(final_trec_folder, f"{qid}.trec")
        print("Pmid for topic: " + qid, end=" ")

        # Get date range for submission
        if qid in date_dict:
            dates = date_dict[qid]
        else:
            dates = default_dates

        # If you want to do a final check again here, you can do so:
        if not check_correct(final_boolean, dates["mindate"], dates["maxdate"]):
            print(f"[WARNING] Query {qid} STILL fails correctness at submission time.")
            # Decide what to do: skip or attempt to regenerate again
            # For simplicity, let's just skip at this point
            #continue

        # Submit to PubMed
        pmids, counter_too_many = pubmed_submission(final_boolean, dates, counter_too_many)

        # Write the final TREC file
        with open(final_trec_file, "w") as fw:
            for rank, pmid in enumerate(pmids):
                fw.write(f'{qid} Q0 {pmid} {rank + 1} {1 / (rank + 1)} rank\n')

    print("The number of left queries: ", len(existing_prompt_dicts) - len(qid_already_submitted))
    print("The number of queries that are too many: ", counter_too_many)




def main():
    parser = ArgumentParser()
    parser.add_argument("--input_queries", type=str, help="Path to the input queries")
    parser.add_argument("--method", type=str, help="method", default="llm_based")
    #model name
    parser.add_argument("--model", type=str, help="model name", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--output_folder", type=str, help="llm model", default="sample")
    parser.add_argument("--llm_prompt_file", type=str, help="prompt of the llm model", default="trec_rag_few_shots")
    parser.add_argument("--quantization", type=str, help="quantization type for vllm", default="no")
    parser.add_argument("--date_file", type=str, help="", default="sample")
    parser.add_argument("--temperature", type=float, help="temperature", default=1)
    parser.add_argument("--extraction_prompt_file", type=str, help="prompt of the validation model", default="boolean_extraction")
    parser.add_argument("--no_retrieval", type=bool, help="no retrieval", default=False)
    parser.add_argument("--json_output", type=bool, help="json output", default=False)
    args = parser.parse_args()

    date_dict = {}
    if os.path.exists(args.date_file):
        with open(args.date_file, "r") as f:
            for line in f:
                qid, min_date, max_date = line.split()
                date_dict[qid] = {"mindate": dates_check(min_date), "maxdate": dates_check(max_date)}

    with open(args.input_queries, "r") as f:
        if args.input_queries.endswith(".jsonl"):
            queries_dicts = {}
            for line in f:
                current_dict = json.loads(line)
                if current_dict["topicid"] in topic_id_not_in_irj:
                    continue
                if "Date From" in current_dict and "Date Run" in current_dict:
                    date_dict[current_dict["topicid"]] = {
                        "mindate": dates_check(current_dict["Date From"]),
                        "maxdate": dates_check(current_dict["Date Run"])
                    }
                modified_dict = {
                    "id": current_dict["topicid"],
                    "topic": current_dict["title"],
                    "boolean_query": current_dict["original_query"],
                }
                if "conceptual_query" in current_dict:
                    #print(current_dict["conceptual_query"])
                    modified_dict["conceptual_query"] = current_dict["conceptual_query"]
                if "objective_query" in current_dict:
                    modified_dict["objective_query"] = current_dict["objective_query"]
                queries_dicts[current_dict["topicid"]] = modified_dict
        else:
            raise NotImplementedError("Only support jsonl file")

    boolean_generation(queries_dicts, date_dict, args.method, args.model, args.llm_prompt_file, args.quantization, args.output_folder, args.temperature, extraction_prompt_file=args.extraction_prompt_file, no_retrieval=args.no_retrieval, json_output=args.json_output)



if __name__ == "__main__":
    main()
