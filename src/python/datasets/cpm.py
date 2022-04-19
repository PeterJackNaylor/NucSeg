import os
import numpy as np

from glob import glob

import skimage.measure as meas
from monusac import square_padding
from class_data import options, BaseData

mapping = {
    "LUNG": "lung",
    "HNSC": "head&neck",
    "GBM": "brain",
    "LGG": "brain",
    "CPM17test": "unknown",
    "CPM18": "unknown",
}


class cpm(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        folders = glob(os.path.join(self.path, "Slide_*"))

        for f in folders:
            organ = os.path.basename(f).split("Slide_")[-1]
            organ = mapping[organ]
            f_gt = f.replace("Slide", "GT")
            for raw_path in glob(os.path.join(f, "*.png")):
                basename = os.path.basename(raw_path)
                basename = basename.replace("Slide_", "image")
                basename = basename.replace(".png", "_mask.txt")
                gt__path = os.path.join(f_gt, basename)
                yield raw_path, gt__path, organ

    def gt_read(self, filename, raw_img):
        f = open(filename, "r")
        content = f.readlines()
        size_x, size_y = content.pop(0)[:-1].split(" ")
        content = [int(c[:-1]) for c in content]
        if "CPM18" in filename:
            tmp = size_x
            size_x = size_y
            size_y = tmp
        gt = np.array(content).reshape((int(size_x), int(size_y)))
        return gt

    def post_process(self, raw, gt):
        gt_shape = gt.shape
        if gt_shape == (600, 600):
            raw = raw[50:-50, 50:-50]
            gt = gt[50:-50, 50:-50]
        elif gt_shape == (498, 461):
            raw, gt = square_padding(raw, gt, self.size)
        elif gt_shape != (500, 500):
            if gt_shape == (808, 1032):
                b = (29, -29, 16, -16)
            elif gt_shape == (669, 793):
                b = (84, -85, 22, -21)
            elif gt_shape == (696, 677):
                b = (98, -98, 88, -89)
            elif gt_shape == (589, 641):
                b = (44, -45, 70, -71)
            elif gt_shape == (596, 798):
                b = (48, -48, 24, -24)
            elif gt_shape == (526, 552):
                b = (13, -13, 26, -26)
            elif gt_shape == (888, 801):
                b = (69, -69, 25, -26)
            raw = raw[b[0]:b[1], b[2]:b[3]]
            gt = gt[b[0]:b[1], b[2]:b[3]]
        gt = meas.label(gt)
        return raw[:, :, :3], gt


def main():
    opt = options()
    cpm_dataset = cpm(opt.path, opt.size, "cpm", opt.target)
    cpm_dataset.create_dataset()


if __name__ == "__main__":
    main()
