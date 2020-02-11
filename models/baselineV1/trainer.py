import sys
from BaselineDataset import BaselineDataset
sys.path.append("..")
from BaseTrainer import main

centers_path = "../1FocalPoints"

main(BaselineDataset, centers_path, num_focal=1)
