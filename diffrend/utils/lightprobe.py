import numpy as np


def load_lightprobe(filename, dim=None, endian='big', datatype='f4', flipud=True, clip_lb_thresh=0, normalize=True):
    """
    :param filename:
    :param endian: 'big' or 'little'
    :param datatype: i4: 4 byte integer
    :return:
    """
    endian_dtype = '<' if endian == 'big' else '>'
    endian_dtype += datatype

    data = np.fromfile(filename, dtype=endian_dtype)

    if dim is None:
        w = int(np.sqrt(data.size / 3))
        dim = (w, w, 3)

    data = data.reshape(dim)

    if flipud:
        data = data[::-1, ...]

    # process the data
    d = data[data > clip_lb_thresh]
    processed_data = (data - d.min()).astype(np.float64)
    if normalize:
        processed_data /= (d.max() - d.min())

    processed_data[processed_data < 0] = 0

    return data, processed_data


def display(filename):
    import matplotlib.pyplot as plt

    data, im = load_lightprobe(filename)

    plt.ion()
    plt.figure()
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    DATA_DIR = '../../data/envmap/'
    probe_files = ['grace_probe.float', 'rnl_probe.float', 'stpeters_probe.float']
    probe_idx = 2
    filename = DATA_DIR + probe_files[probe_idx]  # '../../data/grace_probe.float'
    display(filename)

