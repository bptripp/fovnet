import sys
from FovnetV2Dataset import FovnetV2Dataset
sys.path.append("..")
from BaseTrainer import main

centers_path = "../3FocalPoints"

main(FovnetV2Dataset, centers_path)
