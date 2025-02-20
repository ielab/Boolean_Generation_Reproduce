import argparse
import os
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default="objective")
args = parser.parse_args()

combine_qids = {"51": ["51", "52", "53"], "43": ["43", "96"], "7": ["7", "67"], "8": ["8", "112"]}
inverted_combine_qids = {qid: key for key, qids in combine_qids.items() for qid in qids}

qids_remove = set(["52", "53", "96", "67", "112", "51", "43", "7", "8"])
original_keys = set(combine_qids.keys())

# Define folders
generation_output_folder = os.path.join(args.input_folder, "final_trec_result")
reform_output_folder = os.path.join(args.input_folder, "reformed_final_trec_result")
overall_reform_output_folder = os.path.join(args.input_folder, "reformed_final_trec_result_overall")
reformed_deduplication_output_folder = os.path.join(args.input_folder, "reformed_final_trec_result_deduplication")
overall_reformed_deduplication_output_folder = os.path.join(args.input_folder,
                                                            "reformed_final_trec_result_deduplication_overall")

# Create necessary directories
for folder in [reform_output_folder, overall_reform_output_folder, reformed_deduplication_output_folder,
               overall_reformed_deduplication_output_folder]:
    os.makedirs(folder, exist_ok=True)

all_files = glob.glob(os.path.join(generation_output_folder, '*'))
sid_deduplication_dict = {}
sid_overall_dict = {}

for file in tqdm(all_files):
    file_qid = os.path.basename(file).split('.')[0]
    reform_folder = os.path.join(reform_output_folder, file_qid)
    reform_deduplication_folder = os.path.join(reformed_deduplication_output_folder, file_qid)
    os.makedirs(reform_folder, exist_ok=True)
    if file_qid not in qids_remove:
        os.makedirs(reform_deduplication_folder, exist_ok=True)

    sid_dict = {}
    pubmed_ids = set()

    with open(file, 'r') as f:
        for line in f:
            data = line.split()
            original_qid_sid = data[0]
            only_qid, only_sid = original_qid_sid.split('_')
            pubmed_id = data[2]
            pubmed_ids.add(pubmed_id)
            line_reformed = line.replace(original_qid_sid, only_qid)

            sid_dict.setdefault(only_sid, []).append(line_reformed)

            if only_qid in inverted_combine_qids:
                combined_qid = inverted_combine_qids[only_qid]
                sid_deduplication_dict.setdefault(combined_qid, {}).setdefault(only_sid, {}).setdefault(only_qid,
                                                                                                        []).append(
                    f"{combined_qid} Q0 {pubmed_id} 1 1 STANDARD\n")
                sid_overall_dict.setdefault(combined_qid, set()).add(pubmed_id)

    for sid, lines in sid_dict.items():
        with open(os.path.join(reform_folder, f'{sid}.trec'), 'w') as f:
            f.writelines(lines)

        if file_qid not in qids_remove:
            with open(os.path.join(reform_deduplication_folder, f'{sid}.trec'), 'w') as f:
                f.writelines(lines)

    with open(os.path.join(overall_reform_output_folder, f'{file_qid}.trec'), 'w') as f:
        f.writelines([f'{file_qid} Q0 {pubmed_id} {rank + 1} {1 / (rank + 1)} STANDARD\n' for rank, pubmed_id in
                      enumerate(pubmed_ids)])

    if file_qid not in qids_remove:
        with open(os.path.join(overall_reformed_deduplication_output_folder, f'{file_qid}.trec'), 'w') as f:
            f.writelines([f'{file_qid} Q0 {pubmed_id} {rank + 1} {1 / (rank + 1)} STANDARD\n' for rank, pubmed_id in
                          enumerate(pubmed_ids)])

for qid, sids in sid_deduplication_dict.items():
    deduplication_folder = os.path.join(reformed_deduplication_output_folder, qid)
    os.makedirs(deduplication_folder, exist_ok=True)

    overall_deduplication_file = os.path.join(overall_reformed_deduplication_output_folder, f'{qid}.trec')

    for sid, qid_dict in sids.items():
        picked_key = next((key for key in qid_dict if key in original_keys), next(iter(qid_dict)))
        deduplication_file = os.path.join(deduplication_folder, f'{sid}.trec')

        with open(deduplication_file, 'w') as f:
            f.writelines(qid_dict[picked_key])

    with open(overall_deduplication_file, 'w') as f:
        f.writelines([f'{qid} Q0 {pubmed_id} {rank + 1} {1 / (rank + 1)} STANDARD\n' for rank, pubmed_id in
                      enumerate(sid_overall_dict[qid])])