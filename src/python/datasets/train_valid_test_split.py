import numpy as np
import sys
from glob import glob
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt

from multiprocessing import Pool

def distance_transform_array(bin_image):
    res = np.zeros_like(bin_image, dtype=np.float64)
    for j in range(1, int(bin_image.max()) + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_edt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    return res


def distance_transform_tensor(bin_image):
    pool = Pool()
    result = pool.map(distance_transform_array, bin_image)
    pool.close()
    pool.join()
    result = np.array(result)
    return result


def load_data():
    files = glob("*.npz")
    a_x, a_y, a_o, a_d = [], [], [], []
    for f in files:
        name = f.split("_")[-1].split(".")[0]
        data = np.load(f)
        x = data["x"]
        y = data["y"]
        organs = data["organs"]

        a_x.append(x)
        a_y.append(y)
        a_o.append(organs)
        data_set = np.zeros_like(organs)
        data_set[:] = name
        a_d.append(data_set)

    a_x = np.concatenate(a_x, axis=0)
    a_y = np.concatenate(a_y, axis=0)
    a_o = np.concatenate(a_o, axis=0)
    a_d = np.concatenate(a_d, axis=0)

    return a_x, a_y, a_o, a_d


def create_train_val_test(ptrain, pval, ptest, indice_max, array_to_stratefy):
    assert ptrain + pval + ptest == 1.0
    indices = np.arange(indice_max)
    train_ind, val_test_ind = train_test_split(
        indices, train_size=ptrain, stratify=array_to_stratefy
    )
    ratio_val_test = pval / (pval + ptest)
    val_ind, test_ind = train_test_split(
        indices[val_test_ind],
        train_size=ratio_val_test,
        stratify=array_to_stratefy[val_test_ind],
    )
    return train_ind, val_ind, test_ind


def main():
    x, y, o, d = load_data()
    # we stratefy with respect to the dataset and the organ type
    strat_array = np.char.add(o, d)
    n = len(o)
    train, val, test = create_train_val_test(0.6, 0.2, 0.2, n, strat_array)

    if sys.argv[1] == 'binary':
        y[y > 0] = 1
        y = y.astype('uint8')
    elif sys.argv[1] == 'distance':
        y = distance_transform_tensor(y)
    else:
        raise Exception('Unknown method')

    np.savez(f"Xy_train.npz", x=x[train], y=y[train], organs=o[train])
    np.savez(f"Xy_validation.npz", x=x[val], y=y[val], organs=o[val])
    np.savez(f"Xy_test.npz", x=x[test], y=y[test], organs=o[test])

if __name__ == "__main__":
    main()
