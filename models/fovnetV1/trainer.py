import sys
from FovnetV1Dataset import FovnetV1Dataset
sys.path.append("..")
from BaseTrainer import main

centers_path = "../1FocalPoints"

main(FovnetV1Dataset, centers_path, num_focal=1)
