# Evaluate NegEx's performance

# import
import sys
sys.path.insert(0,'open_negex/')
from negex import *
import csv

DATA_PATH = "csvs/"

total_num = 0
count_tp_tn_fp_fn = [0, 0, 0, 0]

# Evaluate one by one
with open("../open_negex/negex_triggers.txt") as rule_file:
    rules = sortRules(rule_file.readlines())
    with open(DATA_PATH + "upmc_neg2_negex.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for entry in reader:
            total_num += 1
            tagger = negTagger(sentence = entry[0], phrases = [entry[1]], rules = rules, negP=False)
            if entry[2] == "affirmed" and tagger.getNegationFlag() == "affirmed":
                count_tp_tn_fp_fn[0] += 1
            elif entry[2] == "negated" and tagger.getNegationFlag() == "negated":
                count_tp_tn_fp_fn[1] += 1
            elif entry[2] == "negated" and tagger.getNegationFlag() == "affirmed":
                count_tp_tn_fp_fn[2] += 1
            elif entry[2] == "affirmed" and tagger.getNegationFlag() == "negated":
                count_tp_tn_fp_fn[3] += 1

# results
precision = count_tp_tn_fp_fn[0] / (count_tp_tn_fp_fn[0] + count_tp_tn_fp_fn[2])
recall = count_tp_tn_fp_fn[0] / (count_tp_tn_fp_fn[0] + count_tp_tn_fp_fn[3])
f1 = 2 * (precision*recall) / (precision + recall)
print(precision, recall, f1)
