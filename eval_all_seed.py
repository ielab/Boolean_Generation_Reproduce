import argparse
import os
from glob import glob
from tqdm import tqdm
import json
from eval_all import eval

parser = argparse.ArgumentParser()

parser.add_argument('--input_data', type=str, default="data/topics/chatgpt-result-seed_collection.jsonl")
parser.add_argument('--input_folder', type=str, default="objective")
parser.add_argument('--qrel_folder', type=str, default="data/topics/seed_collection_qrels")
parser.add_argument('--selection_target', type=str, default="set_recall")
parser.add_argument('--trec_eval', type=str, default="trec_eval/trec_eval")
parser.add_argument('--deduplication', type=bool, default=False)
args = parser.parse_args()

treceval = args.trec_eval
all_metrics = ["set_P", "set_F", "set_F_3", "set_recall"]
if args.deduplication:
    folder_seperated = os.path.join(args.input_folder, "reformed_final_trec_result_deduplication")
    folder_overall = os.path.join(args.input_folder, "reformed_final_trec_result_deduplication_overall")
else:
    folder_seperated = os.path.join(args.input_folder, "reformed_final_trec_result")
    folder_overall = os.path.join(args.input_folder, "reformed_final_trec_result_overall")

out_seperated_path = os.path.join(folder_seperated, "results.rel")
out_overall_path = os.path.join(folder_overall, "results.rel")

out_seperated = open(out_seperated_path, 'w')
out_overall = open(out_overall_path, 'w')

folder_list = glob(args.input_folder + '/*/')
input_data = args.input_data

ids = []
seed_idss = []
with open(input_data, 'r') as f:
    for line in f:
        data = json.loads(line)
        ids.append(data['topicid'])
        seed_idss.append(data['seed_ids'])


result_input_dict = {}

selection_target = args.selection_target

max_targets = []
max_others = []
overall_lists = []
overall_targets = []

for id, seed_ids in tqdm(zip(ids, seed_idss), total=len(ids)):
    seed_ids = set(seed_ids.split('|'))
    target = 0
    other = [0]* (len(all_metrics) - 1)
    # first do the seperated evaluation
    for sid in seed_ids:
        input_qrel_file = os.path.join(args.qrel_folder, str(id), sid + '.qrels')
        input_result_trec_file = os.path.join(folder_seperated, str(id), sid + '.trec')
        if not os.path.exists(input_result_trec_file):
            continue
        #print(id, sid)
        trec_result = eval(treceval, input_qrel_file,
                           input_result_trec_file)
        if selection_target in trec_result:
            out_seperated.write(f'{id}_{sid}\t{trec_result["set_P"]}\t{trec_result["set_F"]}\t{trec_result["set_F_3"]}\t{trec_result["set_recall"]}\n')
            #current_best_eval  = [trec_result["set_P"],trec_result["set_F"], trec_result["set_F_3"], trec_result["set_recall"]]

            selection_target_index = all_metrics.index(selection_target)
            current_target = trec_result[selection_target]
            if current_target > target:
                target = current_target
                other = [trec_result[x] for x in all_metrics if x != selection_target]
            elif current_target == target:
                current_other = [trec_result[x] for x in all_metrics if x != selection_target]
                if sum(current_other) > sum(other):
                    other = current_other
        else:
            out_seperated.write(f'{id}_{sid}\t0\t0\t0\t0\n')

            #raise ValueError(f"Selection target {selection_target} not found in trec result")
    max_targets.append(target)
    max_others.append(other)
    #print(f"{id}_best:\tTarget:{selection_target}\t" + str(target) + "\tAll Others:\t" + str(other[0]) + "\t" + str(other[1]) + "\t" + str(other[2] ))


    input_qrel_file_random = os.path.join(args.qrel_folder, str(id), list(seed_ids)[0] + '.qrels')
    input_result_trec_file_overall = os.path.join(folder_overall, str(id) + '.trec')
    if not os.path.exists(input_result_trec_file_overall):
        continue

    trec_result = eval(treceval, input_qrel_file_random, input_result_trec_file_overall)
    if selection_target in trec_result:
        out_overall.write(f'{id}\t{trec_result["set_P"]}\t{trec_result["set_F"]}\t{trec_result["set_F_3"]}\t{trec_result["set_recall"]}\n')
        overall_targets.append(trec_result[selection_target])
        overall_lists.append([trec_result[x] for x in all_metrics if x != selection_target])
    else:
        out_overall.write(f'{id}\t0\t0\t0\t0\n')
        overall_targets.append(0)
        overall_lists.append([0]* (len(all_metrics) - 1))
        #raise ValueError(f"Selection target {selection_target} not found in trec result")
    print(id)

    overall_lists.append([trec_result[x] for x in all_metrics if x != selection_target])

    #print(f"{id}_overall:\tTarget:{selection_target}\t" + str(trec_result[selection_target]) + "\tAll Others:\t" + str(trec_result["set_P"]) + "\t" + str(trec_result["set_F"]) + "\t" + str(trec_result["set_F_3"]))

out_seperated.write(f'All\t{sum([x[0] for x in max_others])/len(max_others)}\t{sum([x[1] for x in max_others])/len(max_others)}\t{sum([x[2] for x in max_others])/len(max_others)}\t{sum(max_targets)/len(max_targets)}\n')
out_overall.write(f'All\t{sum([x[0] for x in overall_lists])/len(overall_lists)}\t{sum([x[1] for x in overall_lists])/len(overall_lists)}\t{sum([x[2] for x in overall_lists])/len(overall_lists)}\t{sum(overall_targets)/len(overall_targets)}\n')

print(f"All_best:\tTarget:{selection_target}\t" + str(sum(max_targets)/len(max_targets)) + "\tAll Others:\t" + str(sum([x[0] for x in max_others])/len(max_others)) + "\t" + str(sum([x[1] for x in max_others])/len(max_others)) + "\t" + str(sum([x[2] for x in max_others])/len(max_others)))
print(f"All_overall:\tTarget:{selection_target}\t" + str(sum(overall_targets)/len(overall_targets)) + "\tAll Others:\t" + str(sum([x[0] for x in overall_lists])/len(overall_lists)) + "\t" + str(sum([x[1] for x in overall_lists])/len(overall_lists)) + "\t" + str(sum([x[2] for x in overall_lists])/len(overall_lists)))
