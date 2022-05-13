import os
import numpy as np 

from glob import glob
from scipy.io import loadmat
from skimage.morphology import dilation, erosion, square
from class_data import options, BaseData

def generate_wsl(labelled_mat):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    Args:
        labelled_mat: 2-D labelled matrix.
    Returns:
        a 2-D labelled matrix where each integer component
        cooresponds to a seperation between two objects. 
        0 refers to the backrgound.
    """
    se_3 = square(3)
    ero = labelled_mat.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se_3)
    ero[labelled_mat == 0] = 0

    grad = dilation(labelled_mat, se_3) - ero
    grad[labelled_mat == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad

class consep(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        files = glob(os.path.join(self.path, "Train", "Images", "*.png"))
        files += glob(os.path.join(self.path, "Test", "Images", "*.png"))

        for raw_path in files:
            gt__path = raw_path.replace("/Images/", "/Labels/")
            gt__path = gt__path.replace(".png", ".mat")
            organ = "colorectal"
            yield raw_path, gt__path, organ

    def gt_read(self, filename, raw_img):
        inst_map = loadmat(filename)["inst_map"]
        lines = generate_wsl(inst_map)
        inst_map[lines > 0] = 0
        return inst_map


def main():
    opt = options()
    consep_dataset = consep(opt.path, opt.size, "consep", opt.target)
    consep_dataset.create_dataset()


if __name__ == "__main__":
    main()
