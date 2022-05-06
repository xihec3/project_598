"""
Transform the raw i2b2/VA for assertion classification dataset into five output files:

bidph.csv: A CSV file containing the subset of data from Beth Israel Deaconess Medical Center and from Partners Healthcare. It has two columns. The first column "text" is a sentence, with the target concept surrounded by and . The second column "label "is an int, 0-5 representing present, absent, hypothetical, possible, conditional, and associated_with_someone_else, respectively.

upmc.csv: A CSV file containing the subset of data from University of Pittsburgh Medical Center. It is of the same format as bidph.csv.

bidph_neg2.csv: A CSV file containing the data of present and absent type from subset of data from Beth Israel Deaconess Medical Center and from Partners Healthcare. It is of similar format as bidph.csv, except thatt he "label" column has only two possible values, 0 representing present and 1 representing absent.

upmc_neg2.csv: A CSV file containing the data of present and absent type from subset of data from University of Pittsburgh Medical Center. It is of the same format as bidph_neg2.csv.

upmc_neg2_negex.csv: A CSV file containing the data of present and absent type from subset of data from University of Pittsburgh Medical Center. It will be used as the input to NegEx. It has three column. The first column "text" is a sentence, the second column "concept" is the target concept, and the third column "label" is either "affirmed" or "negated".

All CSV file has the first row as header and has "\t" as the delimiter. All text are in lower case and with newline ('\n'), carriage returns ('\r'), punctuations removed.
"""

DATA_PATH = "ib2b_va_data/"
OUTPUT_PATH = "csvs/"

import os
import csv
import string
import re

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

label_mapping = {"present": 0, "absent": 1, "hypothetical":2, "possible": 3, "conditional": 4, "associated_with_someone_else": 5}
negex_mapping = {0: "affirmed", 1: "negated"}

f1 = open(OUTPUT_PATH + "bidph.txt", 'w+')
f2 = open(OUTPUT_PATH + "upmc.txt", 'w+')
f3 = open(OUTPUT_PATH + "bidph_neg2.txt", 'w+')
f4 = open(OUTPUT_PATH + "upmc_neg2.txt", 'w+')
f5 = open(OUTPUT_PATH + "upmc_neg2_negex.txt", 'w+')

c1 = csv.writer(f1, delimiter = '\t')
c1.writerow(["text", "label"])
c2 = csv.writer(f2, delimiter = '\t')
c2.writerow(["text", "label"])
c3 = csv.writer(f3, delimiter = '\t')
c3.writerow(["text", "label"])
c4 = csv.writer(f4, delimiter = '\t')
c4.writerow(["text", "label"])
c5 = csv.writer(f5, delimiter = '\t')
c5.writerow(["text", "concept", "label"])

bidph_label_count = [0, 0, 0, 0, 0, 0]
upmc_label_count = [0, 0, 0, 0, 0, 0]
bidph_neg2_count = 0
upmc_neg2_count = 0

def preprocess(s):
    s = s.lower()
    s = s.replace("-", " ")
    s = re.sub(r'[^\w\s]','',s)
    s = s.replace("\n", "")
    s = s.replace("\r", "")
    s = s.strip()
    s = ' '.join(s.split())
    return s

for filename in os.listdir(DATA_PATH + "BIDPH_assertion"):
    docname = filename.split(".")[0]
    with open(DATA_PATH + "BIDPH_txt/" + docname + ".txt", 'r') as txt:
        lines = txt.readlines()
    with open(DATA_PATH + "BIDPH_assertion/" + filename, 'r') as assertions:
        for a in assertions:
            a = a.strip()
            tokens = a.split("\"")
            temp_nums = tokens[2].replace(":", " ").replace("||", " ").split(" ", ) 
            # print(temp_nums) # Example: ['', '92', '14', '92', '14', 't=']
            linenum = int(temp_nums[1])
            sentence = lines[linenum - 1]
            sentence = preprocess(sentence)
            concept = tokens[1]
            concept = preprocess(concept)
            start_pos = int(temp_nums[2])
            end_pos = int(temp_nums[4])
            if sentence.find(concept) == -1:
                print(concept)
                print(sentence)
            else:
                modified_sentence = sentence.replace(concept, "<c> " + concept + " </c>", 1)
            
            label = label_mapping[tokens[5]]
            
            bidph_label_count[label] += 1
            
            c1.writerow([modified_sentence, label])
            if label == 0 or label == 1:
                bidph_neg2_count += 1
                c3.writerow([modified_sentence, label])
            
            
for filename in os.listdir(DATA_PATH + "UPMC_assertion"):
    docname = filename.split(".")[0]
    with open(DATA_PATH + "UPMC_txt/" + docname + ".txt", 'r') as txt:
        lines = txt.readlines()
    with open(DATA_PATH + "UPMC_assertion/" + filename, 'r') as assertions:
        # print(filename)
        for a in assertions:
            a = a.strip()
            tokens = a.split("\"")
            temp_nums = tokens[2].replace(":", " ").replace("||", " ").split(" ", ) 
            # print(temp_nums) # Example: ['', '92', '14', '92', '14', 't=']
            linenum = int(temp_nums[1])
            sentence = lines[linenum - 1]
            sentence = preprocess(sentence)
            concept = tokens[1]
            concept = preprocess(concept)
            start_pos = int(temp_nums[2])
            end_pos = int(temp_nums[4])
            if sentence.find(concept) == -1:
                print(concept)
                print(sentence)
            else:
                modified_sentence = sentence.replace(concept, "<c> " + concept + " </c>", 1)
            
            label = label_mapping[tokens[5]]
            
            upmc_label_count[label] += 1
            
            c2.writerow([modified_sentence, label])
            if label == 0 or label == 1:
                upmc_neg2_count += 1
                c4.writerow([modified_sentence, label])
                c5.writerow([sentence, concept, negex_mapping[label]])



f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

print(bidph_label_count)
print(upmc_label_count)
print(bidph_neg2_count)
print(upmc_neg2_count)
