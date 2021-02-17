import sys
from MultiResDataset import MultiResDataset
sys.path.append("..")
from BaseTrainer import main


centers_path = "../3FocalPoints"

main(MultiResDataset, centers_path)
