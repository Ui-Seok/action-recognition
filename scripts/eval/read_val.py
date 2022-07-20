import os

datasetFolder="../../datasets"

val_file=os.path.join(datasetFolder,'settings','hmdb51',"val_rgb_split2.txt")

f_val = open(val_file, 'r')
val_list = f_val.readlines()
# print(val_list)

for i,line in enumerate(val_list):
    print(i, line)