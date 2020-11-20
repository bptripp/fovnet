import sys
from BaselineV2Dataset import BaselineV2Dataset
sys.path.append("..")
from BaseTrainer import main

centers_path = "../3FocalPoints"

main(BaselineV2Dataset, centers_path, num_focal=3)
