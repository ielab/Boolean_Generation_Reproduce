import os
from subprocess import Popen, PIPE, STDOUT
import argparse
from tqdm import tqdm

topic_id_not_in_irj = ["CD010438", "CD010771", "CD011145", "CD010772", "CD010775", "CD010783", "CD010896", "CD007431",
                       "CD010860"]


def eval(trec_eval, qrel_path, output_path):
  result_dict = {}
  command = trec_eval + " -m set_recall -m set_P -m set_F " + qrel_path + " " + output_path
  results = Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True).stdout.readlines()
  #print(command)
  #print(results)
  for result in results:
    items = result.split()
    if (len(items) == 3) and (items[1] == "all"):
      result_dict[items[0]] = float(items[-1])

  command = trec_eval + " -m set_F.3 " + qrel_path + " " + output_path
  results = Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True).stdout.readlines()

  for result in results:
    items = result.split()
    if (len(items) == 3) and (items[1] == "all"):
      result_dict[items[0]] = float(items[-1])
  return result_dict

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_trec_folder', type=str, required=True)
  parser.add_argument('--input_qrels', type=str, required=True)
  parser.add_argument('--trec_eval', type=str, default="trec_eval/trec_eval")
  parser.add_argument('--deduplicate', type=str, default=False)

  args = parser.parse_args()
  if not os.path.exists(args.trec_eval):
      raise ValueError("trec_eval not found")

  qid_dict = {}
  with open(args.input_qrels) as f:
    for line in f:
      qid, _,did,rel= line.split()
      if qid in topic_id_not_in_irj:
        continue
      if int(rel)>=1:
        if qid not in qid_dict:
          qid_dict[qid] = []
        qid_dict[qid].append(did)
  F = []
  F3 = []
  P = []
  R = []
  num_rel = []
  num_retrieved = []
  result_dict = {}
  if args.deduplicate:
    result_qrel_file = os.path.join(args.input_trec_folder, 'results_dedup.rel')
  else:
    result_qrel_file = os.path.join(args.input_trec_folder, 'results.rel')

  with open(result_qrel_file, 'w') as fw:
    for qid in tqdm(qid_dict):
      run_name = os.path.join(args.input_trec_folder, qid + '.trec')
      if not os.path.exists(run_name):
        continue
      result = eval(args.trec_eval,  args.input_qrels, run_name)

      #print(qid, result, len(set(qid_dict[qid])))
      if qid =="CD010438":
        continue
      num_rel.append(len(set(qid_dict[qid])))
      if 'set_P' in result:
        P.append(result['set_P'])
        F.append(result['set_F'])
        F3.append(result['set_F_3'])
        R.append(result['set_recall'])
        fw.write(f'{qid}\t{result["set_P"]}\t{result["set_F"]}\t{result["set_F_3"]}\t{result["set_recall"]}\n')
      else:
        P.append(0)
        F.append(0)
        F3.append(0)
        R.append(0)
        fw.write(f'{qid}\t0\t0\t0\t0\n')

    fw.write(f'All\t{sum(P)/len(P)}\t{sum(F)/len(F)}\t{sum(F3)/len(F3)}\t{sum(R)/len(R)}\n')
    print(len(P))
    print(sum(P)/len(P))
    print(sum(F)/len(F))
    print(sum(F3)/len(F3))
  print(sum(R)/len(R))

