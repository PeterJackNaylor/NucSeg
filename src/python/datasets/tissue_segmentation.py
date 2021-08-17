import os
from glob import glob
from numpy import logical_and
import skimage.measure as meas
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import threshold_otsu
from class_data import options, BaseData


def get_folder_content(path):
    d = {}  # output dictionnary
    files = glob(os.path.join(path, "Slide_*"))
    for f in files:
        d[f.split("_")[-1]] = len(os.listdir(f))
    return d


class tissue_semgentation(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        file_pattern = os.path.join(self.path, "{}/jpg/*.jpg")
        for f in ["train", "test", "val"]:
            files = glob(file_pattern.format(f))
            for raw_path in files:
                gt_path = raw_path.replace('/jpg/', '/lbl/')
                yield raw_path, gt_path, 'None'

    def post_process(self, raw, gt):   
        raw = resize(
            raw, 
            (self.size, self.size),
            preserve_range=True,
            anti_aliasing=True
        )
        gt = resize(
            gt,
            (self.size, self.size),
            preserve_range=True,
            anti_aliasing=True,
            order=0
        )
        gt[gt < 200] = 0
        gt[gt > 0] = 1
        raw = raw.astype('uint8')
        gt = gt.astype('uint8')
        
        gt = meas.label(gt)
        return raw[:, :, :3], gt

    def gt_read(self, filename, raw_img):
        gt =  imread(filename)
        # perform otsu to refine
        gt[gt < 200] = 0
        gt[gt > 0] = 255
        
        return gt

def main():
    opt = options()
    tissue_dataset = tissue_semgentation(opt.path, opt.size, "tissueSeg")
    tissue_dataset.create_dataset()


if __name__ == "__main__":
    main()
