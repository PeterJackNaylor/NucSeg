import os
from glob import glob

import skimage.measure as meas

from class_data import options, BaseData


def get_folder_content(path):
    d = {}  # output dictionnary
    files = glob(os.path.join(path, "Slide_*"))
    for f in files:
        d[f.split("_")[-1]] = len(os.listdir(f))
    return d


class tnbc(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        file_pattern = os.path.join(self.path, "{}_{}/{}_{}.png")
        sample_length = get_folder_content(self.path)

        for i in range(1, 15):
            id_ = f"{i:02}"
            organ = "breast" if i <= 12 else "brain"
            for j in range(1, sample_length[id_] + 1):
                raw_path = file_pattern.format("Slide", id_, id_, j)
                gt__path = file_pattern.format("GT", id_, id_, j)
                yield raw_path, gt__path, organ
    def post_process(self, raw, gt):
        gt = meas.label(gt)
        return raw[6:-6, 6:-6, :3], gt[6:-6, 6:-6]

def main():
    opt = options()
    tnbc_dataset = tnbc(opt.path, opt.size, "tnbc")
    tnbc_dataset.create_dataset()


if __name__ == "__main__":
    main()
