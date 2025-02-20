import argparse
import json
import os
from glob import glob
from pubmed_submission import pubmed_submission
from tqdm import tqdm
default_dates = {
    "mindate": "1975/01/01",
    "maxdate": "2023/12/31"
}
def dates_check(date):
    if "/" not in date:
        new_date = date[:4] + "/" + date[4:6] + "/" + date[6:]
    else:
        new_date = date
    return new_date

args = argparse.ArgumentParser()
# add argument to take input json, outputfolder
args.add_argument("--input_json", type=str, required=True)
args.add_argument("--date_file", type=str, default=None)
args.add_argument("--output_folder", type=str, required=True)
args.add_argument("--identifier", type=str, default="original_query")
args = args.parse_args()

input_json = args.input_json
output_folder = args.output_folder
identifier = args.identifier

date_dict = {}

topic_id_not_in_irj = ["CD010771", "CD011145", "CD010772", "CD010775", "CD010783", "CD010896", "CD007431", "CD010860"]


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


date_dict = {}
if args.date_file is not None:
    if os.path.exists(args.date_file):
        with open(args.date_file, "r") as f:
            for line in f:
                qid, min_date, max_date = line.split()
                date_dict[qid] = {"mindate": dates_check(min_date), "maxdate": dates_check(max_date)}


with open(input_json, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        data = json.loads(line)
        id = data['topicid']
        original_query = data[identifier]
        #original_query = data['original_query']
        output_file = os.path.join(output_folder, f"{id}.trec")
        if os.path.exists(output_file):
            # see if it is empty
            if os.stat(output_file).st_size != 0:
                continue
        print(f"Processing {id}")
        if id in topic_id_not_in_irj:
            continue
        if id in date_dict:
            current_min_date = date_dict[id]["mindate"]
            current_max_date = date_dict[id]["maxdate"]
        else:
            if "Date From" in data:
                current_min_date = dates_check(data["Date From"])
                current_max_date = dates_check(data["Date Run"])
                print(current_min_date, current_max_date)
            else:
                current_min_date = default_dates["mindate"]
                current_max_date = default_dates["maxdate"]

        result_list, count = pubmed_submission(original_query, {"mindate": current_min_date, "maxdate": current_max_date}, 0)

        with open(output_file, "w") as f:
            for rank, pubmed_id in enumerate(result_list):
                f.write(f"{id} Q0 {pubmed_id} {rank + 1} {1/(rank+1)} pubmed\n")
        print(f"Finished {id} with {count} results")





