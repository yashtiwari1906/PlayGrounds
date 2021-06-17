import os

path = r"Y:\\pytorch\\flower classification\\flower_data\\flower_data\\train"
file_names = []
labels = []
"""
for dirname, _, filenames in os.walk(path):
    print(dirname)
    #label = int(dirname.split("\\")[-1])
    for filename in filenames:
        print(os.path.join(dirname))
        break
        file_names.append(os.path.join(dirname, filename))
        labels.append(label)

assert len(labels) == len(file_names)
"""


labels = []
file_names = []
list_labels = os.listdir(path)
for label in list_labels:
    files = [label+"//"+ x for x in os.listdir(path+"//"+label)]
    file_names.extend(files)
    labels.extend([label]*len(files))

assert len(labels) == len(file_names)+1, "labels and file_names are not equa in length"
print("completed!")