import numpy as np


def get_param_value(key, dict_var, default_val, required=False):
    if key in dict_var:
        return dict_var[key]
    elif required:
        raise ValueError('Missing required key {}'.format(key))

    return default_val


def contrast_stretch_percentile(im, nbins, hist_range, new_range=None,
                                low=0.05, high=0.95):
    hist, bins = np.histogram(im.ravel(), nbins, hist_range)
    cdf = hist.cumsum() / hist.sum()
    min_val = sum(cdf <= low)
    max_val = sum(cdf <= high)
    # print(UB, min_val, max_val)
    im = np.clip(im, min_val, max_val).astype(np.float32)
    if max_val > min_val:
        im = (im - min_val) / (max_val - min_val)

    if new_range is not None:
        im = im * (max(new_range) - min(new_range)) + min(new_range)

    return im


def save_xyz(filename, pos, normal):
    """
    Args:
        filename: full output filename
        pos: N-D Tensor with the last dimension being either 3 or 4
        normal: N-D Tensor wth last dimension being either 3 or 4
    """
    if normal is not None:
        data = np.concatenate([pos[..., :3].reshape(-1, 3), normal[..., :3].reshape(-1, 3)], axis=1)
    else:
        data = pos[..., :3].reshape(-1, 3)

    with open(filename, 'w') as fid:
        for sub_idx in range(data.shape[0]):
            fid.write('{}\n'.format(' '.join([str(x) for x in data[sub_idx]])))
