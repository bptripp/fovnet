import sys
from CartesianDataset import CartesianDataset
sys.path.append("..")
from BaseTrainer import main

centers_path = "../3FocalPoints"

main(CartesianDataset, centers_path, num_focal=3)
