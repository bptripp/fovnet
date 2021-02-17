import sys
from BaselineDataset import BaselineDataset
sys.path.append("..")
from BaseTrainer import main

centers_path = "../1FocalPoints"

if __name__ == '__main__':
    main(BaselineDataset, centers_path, num_focal=1)
