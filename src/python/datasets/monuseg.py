import os
from glob import glob

from class_data import options, BaseData


def get_organ(path):
    organ = path.split("/")[-1].split("_")[-1]
    if organ not in ["stomach", "colorectal", "bladder"]:
        n = 4 if "test" in organ else 5
        organ = organ[n:]
    return organ


class monuseg(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        file_pattern = os.path.join(self.path, "Slide_*")
        for f in glob(file_pattern):
            organ = get_organ(f)
            for raw_f in glob(os.path.join(f, "Slide*.png")):
                gt_f = raw_f.replace("Slide_", "GT_")
                yield raw_f, gt_f, organ


def main():
    opt = options()
    monuseg_dataset = monuseg(opt.path, opt.size, "monuseg")
    monuseg_dataset.create_dataset()


if __name__ == "__main__":
    main()
