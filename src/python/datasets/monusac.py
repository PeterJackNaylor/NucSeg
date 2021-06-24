import os
import numpy as np

from glob import glob

import skimage.measure as meas
from skimage.util import pad
import xml.etree.ElementTree as ET
from skimage import draw

from class_data import options, BaseData

mapping_dict = {
    "TCGA-55-1594": "lung",
    "TCGA-69-7760": "lung",
    "TCGA-69-A59K": "lung",
    "TCGA-73-4668": "lung",
    "TCGA-78-7220": "lung",
    "TCGA-86-7713": "lung",
    "TCGA-86-8672": "lung",
    "TCGA-L4-A4E5": "lung",
    "TCGA-MP-A4SY": "lung",
    "TCGA-MP-A4T7": "lung",
    "TCGA-5P-A9K0": "kidney",
    "TCGA-B9-A44B": "kidney",
    "TCGA-B9-A8YI": "kidney",
    "TCGA-DW-7841": "kidney",
    "TCGA-EV-5903": "kidney",
    "TCGA-F9-A97G": "kidney",
    "TCGA-G7-A8LD": "kidney",
    "TCGA-MH-A560": "kidney",
    "TCGA-P4-AAVK": "kidney",
    "TCGA-SX-A7SR": "kidney",
    "TCGA-UZ-A9PO": "kidney",
    "TCGA-UZ-A9PU": "kidney",
    "TCGA-A2-A0CV": "breast",
    "TCGA-A2-A0ES": "breast",
    "TCGA-B6-A0WZ": "breast",
    "TCGA-BH-A18T": "breast",
    "TCGA-D8-A1X5": "breast",
    "TCGA-E2-A154": "breast",
    "TCGA-E9-A22B": "breast",
    "TCGA-E9-A22G": "breast",
    "TCGA-EW-A6SD": "breast",
    "TCGA-S3-AA11": "breast",
    "TCGA-EJ-5495": "prostate",
    "TCGA-EJ-5505": "prostate",
    "TCGA-EJ-5517": "prostate",
    "TCGA-G9-6342": "prostate",
    "TCGA-G9-6499": "prostate",
    "TCGA-J4-A67Q": "prostate",
    "TCGA-J4-A67T": "prostate",
    "TCGA-KK-A59X": "prostate",
    "TCGA-KK-A6E0": "prostate",
    "TCGA-KK-A7AW": "prostate",
    "TCGA-V1-A8WL": "prostate",
    "TCGA-V1-A9O9": "prostate",
    "TCGA-X4-A8KQ": "prostate",
    "TCGA-YL-A9WY": "prostate",
}


def get_organ(path):
    sample = os.path.basename(path).split("-01Z")[0]
    return mapping_dict[sample]


def square_padding(raw, gt, size):
    x, y = gt.shape
    pad_width = []
    s_size = (size, size)
    for axe, val in enumerate([x, y]):
        n_tiles = val // s_size[axe]
        diff = s_size[axe] * (n_tiles + 1) - val
        m = diff / 2
        if m.is_integer():
            m = int(m)
            m = (m, m)
        else:
            m = int(m)
            m = (m, m + 1)
        pad_width.append(m)
    raw = pad(raw, pad_width + [(0, 0)], mode="reflect")
    gt = pad(gt, pad_width, mode="reflect")
    return raw, gt


def xml_parser(xml_file_name, raw_img):
    nx, ny, nz = raw_img.shape
    tree = ET.parse(xml_file_name)
    root = tree.getroot()
    binary_mask = np.zeros(shape=(nx, ny))
    cell_count = 1
    for k in range(len(root)):
        label = [x.attrib["Name"] for x in root[k][0]]
        label = label[0]

        for child in root[k]:
            for x in child:
                r = x.tag
                if r == "Attribute":
                    label = x.attrib["Name"]

                if r == "Region":
                    regions = []
                    vertices = x[1]
                    coords = np.zeros((len(vertices), 2))
                    for i, vertex in enumerate(vertices):
                        coords[i][0] = vertex.attrib["X"]
                        coords[i][1] = vertex.attrib["Y"]
                    regions.append(coords)

                    vertex_row_coords = regions[0][:, 0]
                    vertex_col_coords = regions[0][:, 1]
                    row_coords, col_coords = draw.polygon(
                        vertex_col_coords, vertex_row_coords, binary_mask.shape
                    )
                    binary_mask[row_coords, col_coords] = cell_count
                    cell_count += 1
    return binary_mask


class monusac(BaseData):
    def generate_filename(self):
        """
        Generator associating each raw/gt files
        """
        file_pattern = os.path.join(self.path, "TCGA-*")
        for f in glob(file_pattern):
            organ = get_organ(f)
            for raw_f in glob(os.path.join(f, "*.tif")):
                gt_f = raw_f.replace(".tif", ".xml")
                yield raw_f, gt_f, organ

    def gt_read(self, filename, raw_img):
        gt = xml_parser(filename, raw_img)
        return gt

    def post_process(self, raw, gt):
        """
        padding so that it becomes a multiple of 250
        """
        raw, gt = square_padding(raw, gt, self.size)
        gt = meas.label(gt)
        return raw[:, :, :3], gt


def main():
    opt = options()
    monusac_dataset = monusac(opt.path, opt.size, "monusac")
    monusac_dataset.create_dataset()


if __name__ == "__main__":
    main()
