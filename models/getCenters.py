#Collect focal points into one file after saving saliency maps for each image
import pickle
import glob
from ModelUtils import get_focal_points

num_points = 3
# num_points = 1
split = "train"
# split = "val"

a = glob.glob("/home/p2torabi/datasets/ImageNet/ILSVRC/Data/CLS-LOC/"+split+"/**/*.saliency.npy.gz", recursive=True)

results = {}

for i in range(len(a)):
    path = a[i]
    center = get_focal_points(path, num_points)
    save_path = path.split(".saliency.npy.gz")[0]+".JPEG"
    results[save_path] = center
#     #Periodically save every 1000 results
#     if i%1000 == 0:
#         pickle.dump(results, open("part{}.pkl".format(i), "wb"))
#         results = {}
#         print(i)

# pickle.dump(results, open("part{}.pkl".format(i), "wb"))

# #Aggregate and Check (or resume) that we have all centers
# parts = glob.glob("part**.pkl")
# results = {}
# for part in parts:
#     dic = pickle.load(open(part, "rb"))
#     new_dic = {i.split("CLS-LOC/")[1]:v for i,v in dic.items()} 
#     results = {**results, **new_dic}

# for path in a:
#     new_path = path.split("CLS-LOC/")[1].split(".saliency.npy.gz")[0]+".JPEG"
#     if new_path in results:
#         continue
#     center = get_focal_points(path, num_points)
#     results[new_path] = center


pickle.dump(results, open("{}FocalPoints_"+split+".pkl".format(num_points), "wb"))
