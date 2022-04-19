import argparse
import numpy as np

from skimage.io import imread
import skimage.measure as meas
import staintools


def options():
    parser = argparse.ArgumentParser(description="setting up dataset")
    parser.add_argument("--path", type=str)
    parser.add_argument("--size", type=int)
    parser.add_argument("--target", type=str)
    args = parser.parse_args()
    args.parser = None if args.target == "None" else args.target
    return args


def split(image, label, size):
    s_steps = (size, size)
    f_step = label.shape

    for i in range(0, f_step[0], s_steps[0]):
        for j in range(0, f_step[1], s_steps[1]):
            box = (i, j, i + s_steps[0], j + s_steps[1])
            yield image[box[0]:box[2], box[1]:box[3]], label[
                box[0]: box[2], box[1]:box[3]
            ]


class BaseData:
    def __init__(self, path, size, name, target=None) -> None:
        self.path = path
        self.size = size
        self.name = name
        self.normalize_b = target is not None
        if self.normalize_b:
            self.setup_normalizer(target)

    def gt_read(self, filename, raw_img):
        return imread(filename)

    def generate_filename(self):
        pass

    def post_process(self, raw, gt):
        gt = meas.label(gt)
        return raw[:, :, :3], gt

    def setup_normalizer(self, path):
        target = staintools.read_image(path)
        target_lum = staintools.LuminosityStandardizer.standardize(target)
        self.target_he = staintools.StainNormalizer(method='vahadane')
        self.target_he.fit(target_lum) 

    def normalize(self, img):
        img = staintools.LuminosityStandardizer.standardize(img)
        return self.target_he.transform(img)

    def create_dataset(self):
        x, y, organs = [], [], []
        for raw_path, gt_path, organ in self.generate_filename():
            raw = imread(raw_path)
            if self.normalize_b:
                raw = self.normalize(raw)
            gt = self.gt_read(gt_path, raw)
            raw, gt = self.post_process(raw, gt)
            for c_raw, c_gt in split(raw, gt, self.size):
                x.append(c_raw)
                y.append(c_gt)
                organs.append(organ)

        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        organs = np.array(organs)
        np.savez(f"Xy_{self.name}.npz", x=x, y=y, organs=organs)
