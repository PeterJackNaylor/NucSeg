import os
import pandas as pd

import skimage.measure as meas

from class_data import options, BaseData

mapping = {
    "blca": "bladder",
    "brca": "breast",
    "cesc": "cervical",
    "coad": "ignore colon",
    "gbm": "brain",
    "luad": "lung",
    "lusc": "lung",
    "paad": "pancreas",
    "prad": "prostate",
    "read": "ignore colorectal",
    "skcm": "skin",
    "stad": "ignore stomach",
    "ucec": "pelvic",
    "uvm": "ignore eye",
}


class pancrops(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        f_pattern = os.path.join(self.path, "{}_crop.png")
        f_gt_pattern = f_pattern.replace("crop.", "labeled_mask_corrected.")
        table_file = os.path.join(self.path, "table.csv")
        table = pd.read_csv(table_file, index_col=0)
        invalid_index = [49, 59, 299, 310, 376, 450, 760, 981, 1312]
        for i in range(1, 1366):
            if i not in invalid_index:
                raw_path = f_pattern.format(i)
                gt__path = f_gt_pattern.format(i)
                organ = mapping[table.loc[i, "CancerType"]]
                if organ[:6] != "ignore":
                    yield raw_path, gt__path, organ

    def post_process(self, raw, gt):
        gt = meas.label(gt)
        return raw[3:-3, 3:-3, :3], gt[3:-3, 3:-3]


def main():
    opt = options()
    pancrops_dataset = pancrops(opt.path, opt.size, "pancrops")
    pancrops_dataset.create_dataset()


if __name__ == "__main__":
    main()
