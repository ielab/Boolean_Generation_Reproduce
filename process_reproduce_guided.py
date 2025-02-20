import time
from argparse import ArgumentParser
import json
import yaml
import os
from generator import Llama, APIModel
from tqdm import tqdm
import glob
from check_bool_query import check_correct
from pubmed_submission import pubmed_submission, dates_check
import random
from check_bool_query import check_response

max_retries = 20
default_seed = 42

default_dates = {
    "mindate": "1975/01/01",
    "maxdate": "2023/12/31"
}
default_extractor = APIModel(model_name="gpt-3.5-turbo-0125", temperature=0)

topic_id_not_in_irj = ["CD010771", "CD011145", "CD010772", "CD010775", "CD010783", "CD010896", "CD007431", "CD010860"]


def boolean_generation(queries_dict, date_dict, method, model_name, llm_prompt_file, quantization, output_folder, temperature, extractor=default_extractor, extraction_prompt_file=None):
    retry_list = []
    if method == "llm_based":
        # if "llama" not in model_name:
        #     raise NotImplementedError("Only support llama model")
        # model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        model = Llama(model_name=model_name, quantization=quantization, temperature=temperature)
    elif method == "api_based":
        # model_name = "gpt3.5"
        model = APIModel(model_name=model_name, temperature=temperature)

    elif method == "not_loading":
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
    } for qid in queries_dict.keys()}
    qid_list = list(queries_dict.keys())

    # --------------------------------------------
    # Main Generation Loop
    # --------------------------------------------
    for step_i, step in enumerate(generation_steps):
        print("Now processing step", step_i)
        current_generation_output_file = os.path.join(generation_output_folder, f"step_{step_i}.jsonl")

        # ------------------------------------------------------------------------
        # Step 1: Load any partial results from disk for this step (line-by-line)
        # ------------------------------------------------------------------------
        step_already_processed_dict = {}  # {qid -> existing_prompt_dict_from_file}
        if os.path.exists(current_generation_output_file):
            with open(current_generation_output_file, "r") as f:
                for line in f:
                    current_existing_dict = json.loads(line)
                    qid = current_existing_dict["id"]
                    if qid in qid_list:
                        step_already_processed_dict[qid] = current_existing_dict["existing_prompts"]

        # For those queries that were already processed in this step, update the in-memory "existing_prompt_lists"
        # so we can skip re-running them.
        for qid, loaded_prompts in step_already_processed_dict.items():
            existing_prompt_dicts[qid] = loaded_prompts

        # We'll process the queries in batches
        batch_size = step.get("batch_size", 1)
        bool_extract = step.get("bool_extract", False)
        print(f"Batch size: {batch_size}")
        print(f"Boolean extraction: {bool_extract}")

        with open(current_generation_output_file, "a", encoding="utf-8") as step_output_f:
            for i in tqdm(range(0, len(qid_list), batch_size)):
                batched_qids = qid_list[i:i + batch_size]
                batched_instructions = []

                # Collect instructions for those queries not yet processed in this step
                qids_to_generate = []

                for qid in batched_qids:
                    query = queries_dict[qid]
                    if qid in step_already_processed_dict:
                        continue
                    # Not processed yet -> compile prompt
                    existing_prompt_dict, instruction = model.compile_prompt(
                        step,
                        existing_prompt_dict=existing_prompt_dicts[qid],
                        **query
                    )
                    qids_to_generate.append(qid)
                    batched_instructions.append(instruction)
                    existing_prompt_dicts[qid] = existing_prompt_dict

                # Generate a batch of responses
                responses = []
                while len(responses) < len(qids_to_generate):
                    try:
                        generated_responses = model.generate_batch(batched_instructions)
                        if bool_extract:
                            temp_responses = []
                            for generated_response in generated_responses:
                                _, extraction_instruction = extractor.compile_prompt(extraction_step,
                                                                                     response_content=generated_response)
                                extraction_response = extractor.generate_batch([extraction_instruction])[0]
                                temp_responses.append(extraction_response)
                            responses.extend(temp_responses)
                        else:
                            responses.extend(generated_responses)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                # Append the newly generated responses to each query's prompts
                for qid_index, qid in enumerate(qids_to_generate):
                    instruction = batched_instructions[qid_index]
                    final_boolean = responses[qid_index]

                    # -------------------------------------------------------------
                    # If this is the final step, run correctness checks and re-generate if needed
                    # -------------------------------------------------------------
                    retry_count = 0
                    already_generated_boolean = [final_boolean]

                    if bool_extract:
                        original_qid = qid.split('_')[0]
                        # Get the appropriate date range
                        if original_qid in date_dict:
                            mindate = date_dict[original_qid]["mindate"]
                            maxdate = date_dict[original_qid]["maxdate"]
                        else:
                            mindate = default_dates["mindate"]
                            maxdate = default_dates["maxdate"]

                        while (not check_correct(final_boolean, mindate, maxdate)) or (not check_response(final_boolean)):
                            time.sleep(0.2)
                            if retry_count >= max_retries:

                                print(
                                    f"[WARNING] Query {qid} failed correctness after {max_retries} retries. Skipping.")
                                break
                            if final_boolean in already_generated_boolean:
                                print(
                                    f"[WARNING] Query {qid} failed correctness after {retry_count} retries. Skipping.")
                                model.set_seed(random.randint(0, 100000))

                            print(
                                f"[INFO] Query {qid} was incorrect. Re-generating last step (attempt {retry_count + 1})..."
                            )
                            # Generate again
                            try:
                                response = model.generate_batch([instruction])[0]
                                _, extraction_instruction = extractor.compile_prompt(extraction_step,
                                                                                        response_content=response)
                                response = extractor.generate_batch([extraction_instruction])[0]
                            except Exception as e:
                                print(f"Error: {e}")
                                retry_count += 1
                                continue
                            final_boolean = response
                            already_generated_boolean.append(response)
                            model.set_seed(default_seed)
                            retry_count += 1
                    else:
                        while not check_response(final_boolean):
                            time.sleep(0.2)

                            if retry_count >= max_retries*10:
                                print(
                                    f"[WARNING] Query {qid} failed correctness after {max_retries} retries. Skipping.")
                                break
                            print(
                                f"[INFO] Query {qid} was incorrect. Re-generating last step (attempt {retry_count + 1})..."
                            )
                            if final_boolean in already_generated_boolean:
                                print(
                                    f"[WARNING] Query {qid} failed correctness after {retry_count} retries.")
                                model.set_seed(random.randint(0, 100000))
                            # Generate again
                            try:
                                response = model.generate_batch([instruction])[0]
                            except Exception as e:
                                print(f"Error: {e}")
                                retry_count += 1
                                continue
                            final_boolean = response
                            already_generated_boolean.append(response)

                            model.set_seed(default_seed)
                            retry_count += 1
                    # -------------------------------------------------------------
                    # After generation (and possible re-generation), write out to JSONL
                    # for this query immediately (line-by-line).
                    # -------------------------------------------------------------
                    existing_prompt_dicts[qid]["user"].append(
                        {"role": "assistant", "content": final_boolean}
                    )

                    current_existing_dict = {
                        "id": qid,
                        "topic": queries_dict[qid]["topic"],
                        "existing_prompts": existing_prompt_dicts[qid]
                    }
                    step_output_f.write(json.dumps(current_existing_dict) + "\n")
                    step_output_f.flush()
                    retry_list.append(retry_count)



    # --------------------------------------------
    # Submission Final Assistant Responses
    # --------------------------------------------

    #save the retry list
    with open(os.path.join(output_folder, "retry_list.txt"), "a+") as f:
        for retry in retry_list:
            f.write(str(retry) + "\n")

    qid_already_submitted = set()
    final_trec_folder = os.path.join(output_folder, "final_trec_result")
    if not os.path.exists(final_trec_folder):
        os.makedirs(final_trec_folder)
    else:
        final_trec_files = glob.glob(final_trec_folder + '/*')
        for final_trec_file in final_trec_files:
            with open(final_trec_file) as f:
                for line in f:
                    qid_sid = line.split()[0]
                    qid_already_submitted.add(qid_sid)

    counter_too_many = 0

    # Now we do the final submission to PubMed
    for qid, item in tqdm(existing_prompt_dicts.items()):
        if qid in qid_already_submitted:
            continue
        original_qid = qid.split('_')[0]
        if original_qid in topic_id_not_in_irj:
            continue
        try:
            final_boolean = item["user"][-1]["content"]
        except:
            print("No final boolean for topic: " + qid)


        final_trec_file = os.path.join(final_trec_folder, f"{original_qid}.trec")


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
            continue

        # Submit to PubMed
        pmids, counter_too_many = pubmed_submission(final_boolean, dates, counter_too_many)

        # Write the final TREC file
        with open(final_trec_file, "a+") as fw:
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
    parser.add_argument("--seed_collection_file", type=str, help="seed file", default=None)
    parser.add_argument("--output_folder", type=str, help="llm model", default="sample")
    parser.add_argument("--llm_prompt_file", type=str, help="prompt of the llm model", default="trec_rag_few_shots")
    parser.add_argument("--quantization", type=str, help="quantization type for vllm", default="no")
    parser.add_argument("--date_file", type=str, help="", default="sample")
    parser.add_argument("--temperature", type=float, help="temperature", default=1)
    parser.add_argument("--extraction_prompt_file", type=str, help="prompt of the validation model",
                        default="boolean_extraction")
    args = parser.parse_args()

    date_dict = {}
    if os.path.exists(args.date_file):
        with open(args.date_file, "r") as f:
            for line in f:
                qid, min_date, max_date = line.split()
                date_dict[qid] = {"mindate": dates_check(min_date), "maxdate": dates_check(max_date)}

    seed_collection_dict = {}
    with open(args.seed_collection_file, "r") as f:
        for line in f:
            current_dict = json.loads(line)
            seed_collection_dict[str(current_dict["pmid"])] = {
                "title": current_dict["title"],
                "abstract": current_dict["abstract"]
            }
    with open(args.input_queries, "r") as f:
        if args.input_queries.endswith(".jsonl"):
            queries_dict = {}
            for line in f:
                current_dict = json.loads(line)
                if current_dict["topicid"] in topic_id_not_in_irj:
                    continue
                seed_ids = set(current_dict["seed_ids"].split("|"))
                seed_ids = [str(seed_id.strip()) for seed_id in seed_ids]
                for seed_id in seed_ids:
                    if seed_id not in seed_collection_dict:
                        print(f"Seed {seed_id} not found in the seed collection.")
                        continue
                    seed_content = seed_collection_dict[seed_id.strip()]
                    if "Date From" in current_dict and "Date Run" in current_dict:
                        date_dict[current_dict["topicid"]] = {
                            "mindate": dates_check(current_dict["Date From"]),
                            "maxdate": dates_check(current_dict["Date Run"])
                        }
                    modified_dict = {
                        "id": current_dict["topicid"] + "_" + seed_id,
                        "topic": current_dict["title"],
                        "boolean_query": current_dict["original_query"],
                        "seed_content":  seed_content["title"] + " " + seed_content["abstract"]
                    }
                    if current_dict["topicid"] in topic_id_not_in_irj:
                        continue
                    queries_dict[current_dict["topicid"] + "_" + seed_id] = modified_dict
        else:
            raise NotImplementedError("Only support jsonl file")
    print(date_dict)
    boolean_generation(queries_dict, date_dict, args.method, args.model, args.llm_prompt_file, args.quantization, args.output_folder, args.temperature, extraction_prompt_file=args.extraction_prompt_file)


if __name__ == "__main__":
    main()
