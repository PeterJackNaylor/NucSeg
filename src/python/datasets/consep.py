import os
import numpy as np

from glob import glob
from scipy.io import loadmat

from class_data import options, BaseData

class consep(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        files = glob(os.path.join(self.path, "Train", "Images", "*.png"))
        files += glob(os.path.join(self.path, "Test", "Images", "*.png"))

        for raw_path in files:
            gt__path = raw_path.replace('/Images/', '/Labels/').replace('.png', '.mat')
            organ = "colorectal"
            yield raw_path, gt__path, organ
    def gt_read(self, filename, raw_img):
        return loadmat(filename)['inst_map']

def main():
    opt = options()
    consep_dataset = consep(opt.path, opt.size, "consep")
    consep_dataset.create_dataset()


if __name__ == "__main__":
    main()
