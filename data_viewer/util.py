import numpy as np 
import scipy as sp 
import scipy.io as sio
from scipy.ndimage.interpolation import rotate
import h5py
import datetime
import os
from PIL import Image
from PyQt4 import QtCore, QtGui


def cart2pol(x, y):
    """Summary
    
    Parameters
    ----------
    x : array_like
        x values in Cartesian coordinates
    y : array_like with the same shape of x
        y values in Cartesian coordinates
    
    Returns
    -------
    rho, theta: ndarray
        rho and theta values in polar coordinates
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))
    theta = np.arctan2(y, x)
    return rho, theta


def pol2cart(rho, theta):
    """Summary
    
    Parameters
    ----------
    rho : array_like
        rho values in polar coordinates
    theta : array_like
        theta values in polar coordinates
    
    Returns
    -------
    x, y: ndarray
        x and y values in Cardisian coordinates
    """
    rho = np.asarray(rho, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    assert rho.shape == theta.shape
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def save_as_Hawk_input_file(data, filename='hawk_input.h5', phased=False, scaled=False, shifted=False, center=None):
    """Summary
    
    Parameters
    ----------
    data : ndarray
        2D image.
    filename : str, optional
        Output filename.
    phased : bool, optional
        Phase information included. Not very sure.
    scaled : bool, optional
        True for amplitude, False for intensity.
    shifted : bool, optional
        True for 0-freq in the center, False for 0-freq at corners.
    center : ndarray, optional
        Image center, specified by 3 elements: cx, cy, cz=0.
    
    Returns
    -------
    TYPE
        None
    """
    f = h5py.File(filename, 'w')
    if center is None:
        center = [data.shape[0]//2, data.shape[1]//2, 0.]
    else:
        center = [center[0], center[1], 0.]
    center = np.asarray(center)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    if phased:
        f.create_dataset('/phased', data=1., shape=(1,), dtype='f4')
    else:
        f.create_dataset('/phased', data=0., shape=(1,), dtype='f4')
    if scaled:
        f.create_dataset('/scaled', data=1., shape=(1,), dtype='f4')
    else:
        f.create_dataset('/scaled', data=0., shape=(1,), dtype='f4')
    if shifted:
        f.create_dataset('/shifted', data=1., shape=(1,), dtype='f4')
    else:
        f.create_dataset('/shifted', data=0., shape=(1,), dtype='f4')
    f.create_dataset('/detector_distance', data=1., shape=(1,), dtype='f4')
    f.create_dataset('/image_center', data=center, shape=(3,), dtype='f4')
    f.create_dataset('/lambda', data=1., shape=(1,), dtype='f4')
    f.create_dataset('/mask', data=(data > 0), dtype='i4')
    f.create_dataset('/num_dimensions', data=2., shape=(1,), dtype='f4')
    f.create_dataset('/real', data=data, dtype='f4')
    f.create_dataset('/pixel_size', data=[0., 0., 0.], shape=(3,), dtype='f4')
    f.create_dataset('/version', data=2, shape=(1,), dtype='i4')
    f.close()


def get_small_dataset(filepath):
    f = h5py.File(filepath, 'r')
    datasets = []
    for key in f.keys():
        if key != 'frames' and key != 'paths':
            datasets.append(key)
    return datasets


def calc_radial_profile(image, center, binsize=1., mask=None, mode='sum'):
    """Summary
    
    Parameters
    ----------
    image : 2d array
        Input image to calculate radial profile
    center : array_like with 2 elements
        Center of input image
    binsize : float, optional
        By default, the binsize is 1 in pixel.
    mask : 2d array, optional
        Binary 2d array used in radial profile calculation. The shape must be same with image. 1 means valid while 0 not.
    mode : {'sum', 'mean'}, optional
        'sum'
        By default, mode is 'sum'. This returns the summation of each ring.
    
        'mean'
        Mode 'mean' returns the average value of each ring.
    
    Returns
    -------
    Radial profile: 1d array
        Output array, contains summation or mean value of each ring with binsize of 1 along rho axis.
    
    Raises
    ------
    ValueError
        Description
    """
    image = np.asarray(image, dtype=np.float64)
    assert len(image.shape) == 2
    center = np.asarray(center, dtype=np.float64)
    assert center.size == 2
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        assert mask.shape == image.shape
        assert mask.min() >= 0. and mask.max() <= 1.
        mask = (mask > 0.5).astype(np.float64)
    else:
        mask = np.ones_like(image)
    image *= mask  # apply mask on input image
    y, x = np.indices((image.shape))
    r = np.sqrt((x - center[0])**2. + (y - center[1])**2.)
    bin_r = r / binsize
    bin_r = np.round(bin_r).astype(int)
    radial_sum = np.bincount(bin_r.ravel(), image.ravel())  # summation of each ring

    radial_std = np.zeros_like(radial_sum)
    for x in xrange(bin_r.max()):
        radial_std[x] = image[bin_r == x].std()
    nr = np.bincount(bin_r.ravel(), mask.ravel())
    radial_mean = radial_sum / nr
    radial_mean[np.isinf(radial_mean)] = 0.
    radial_mean[np.isnan(radial_mean)] = 0.
    radial_rstd = radial_std / radial_mean
    radial_rstd[np.isinf(radial_rstd)] = 0.
    radial_rstd[np.isnan(radial_rstd)] = 0.

    if mode == 'sum':
        return radial_sum
    elif mode == 'mean':
        return radial_mean
    elif mode == 'std':
        return radial_std
    elif mode == 'rstd':
        return radial_rstd
    else:
        raise ValueError('Wrong mode: %s' %mode)


def calc_angular_profile(image, center, binsize=1., mask=None, mode='sum'):
    """Summary
    
    Parameters
    ----------
    image : 2d array
        Input image to calculate angular profile in range of 0 to 180 deg.
    center : array_like with 2 elements
        Center of input image
    binsize : float, optional
        By default, the binsize is 1 in degree.
    mask : 2d array, optional
        Binary 2d array used in angular profile calculation. The shape must be same with image. 1 means valid while 0 not.
    mode : {'sum', 'mean'}, optional
        'sum'
        By default, mode is 'sum'. This returns the summation of each ring.
    
        'mean'
        Mode 'mean' returns the average value of each ring.
    
    Returns
    -------
    Angular profile: 1d array
        Output array, contains summation or mean value of each ring with binsize of 1 along rho axis.
    """
    image = np.asarray(image, dtype=np.float64)
    assert len(image.shape) == 2
    center = np.asarray(center, dtype=np.float64)
    assert center.size == 2
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        assert mask.shape == image.shape
        assert mask.min() >= 0. and mask.max() <= 1.
        mask = (mask > 0.5).astype(np.float64)
    else:
        mask = np.ones_like(image)
    image *= mask 
    y, x = np.indices((image.shape))
    theta = np.rad2deg(np.arctan2(y-center[1], x-center[0]))
    bin_theta = theta.copy()
    bin_theta[bin_theta<0.] += 180.
    bin_theta = bin_theta / binsize
    bin_theta = np.round(bin_theta).astype(int)
    angular_sum = np.bincount(bin_theta.ravel(), image.ravel())  # summation of each ring

    if mode == 'sum':
        return angular_sum
    elif mode == 'mean':
        ntheta = np.bincount(bin_theta.ravel(), mask.ravel())
        angular_mean = angular_sum / ntheta
        angular_mean[np.isinf(angular_mean)] = 0.
        return angular_mean
    else:
        raise ValueError('Wrong mode: %s' %mode)


def calc_across_center_line_profile(image, center, angle=0., width=1, mask=None, mode='sum'):
    """Summary
    
    Parameters
    ----------
    image : 2d array
        Input image to calculate angular profile in range of 0 to 180 deg.
    center : array_like with 2 elements
        Center of input image
    angle : float, optional
        Line angle in degrees.
    width : int, optional
        Line width. The default is 1.
    mask : 2d array, optional
        Binary 2d array used in angular profile calculation. The shape must be same with image. 1 means valid while 0 not.
    mode : {'sum', 'mean'}, optional
        'sum'
        By default, mode is 'sum'. This returns the summation of each ring.
    
        'mean'
        Mode 'mean' returns the average value of each ring.
    
    Returns
    -------
    Across center line profile with given width at specified angle: 2d array
        Output array, contains summation or mean value alone the across center line and its indices with respect to the center.
    """
    image = np.asarray(image, dtype=np.float64)
    assert len(image.shape) == 2
    center = np.asarray(center, dtype=np.float64)
    assert center.size == 2
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        assert mask.shape == image.shape
        assert mask.min() >= 0. and mask.max() <= 1.
        mask = (mask > 0.5).astype(np.float64)
    else:
        mask = np.ones_like(image)
    image *= mask 
    # generate a larger image if the given center is not the center of the image.
    sy, sx = image.shape
    if sy % 2 == 0:
        # print('padding along first axis')
        image = np.pad(image, ((0,1), (0,0)), 'constant', constant_values=0)
    if sx % 2 == 0:
        # print('padding along second axis')
        image = np.pad(image, ((0,0), (0,1)), 'constant', constant_values=0)
    sy, sx = image.shape
    if center[0] < sx//2 and center[1] < sy//2:
        # print('case1')
        sx_p = int((sx - center[0]) * 2 - 1)
        sy_p = int((sy - center[1]) * 2 - 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[sy_p-sy:sy_p, sx_p-sx:sx_p] = image
    elif center[0] < sx//2 and center[1] > sy//2:
        # print('case2')
        sx_p = int((sx - center[0]) * 2 - 1)
        sy_p = int((center[1]) * 2 - 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[0:sy, sx_p-sx:sx_p] = image
    elif center[0] > sx//2 and center[1] < sy//2:
        sx_p = int((center[0]) * 2 - 1)
        sy_p = int((sy - center[1]) * 2 - 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[sy_p-sy:sy_p, 0:sx] = image
    else:
        # print('case4')
        sx_p = int((center[0]) * 2 + 1)
        sy_p = int((center[1]) * 2 + 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[0:sy, 0:sx] = image
    rot_img = rotate(ex_img, angle)
    rot_sy, rot_sx = rot_img.shape
    across_line = rot_img[rot_sy//2-width//2:rot_sy//2-width//2+width, :].copy()
    across_line_sum = np.sum(across_line, axis=0)
    line_indices = np.indices(across_line_sum.shape)[0] - rot_sx//2
    line_sum = np.bincount(np.abs(line_indices).ravel(), across_line_sum.ravel())
    if mode == 'sum':
        return line_sum
    elif mode == 'mean':
        line_mean = line_sum.astype(np.float) / width
        return line_mean
    else:
        raise ValueError('Wrong mode: %s' %mode)


def print_with_timestamp(s):
    now = datetime.datetime.now()
    print('%s: %s' %(now, str(s)))


def load_smalldata(filepath, dataset_name):
    filepath = str(filepath)
    dataset_name = str(dataset_name)
    if not os.path.isfile(filepath):
        raise os.error('File not exist: %s' %filepath)
    _, ext = os.path.splitext(filepath)
    assert ext == '.h5'
    f = h5py.File(filepath, 'r')
    smalldata = f[dataset_name].value
    paths = f['paths'].value
    frames = f['frames'].value
    f.close()
    return paths, frames, smalldata


def make_temp_file(filepath, frame, output):
    f = h5py.File(filepath, 'r')
    data = f['data'][frame]
    np.save(output, data)
    f.close()


def load_data(filepath, dataset_name):
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise os.error('File not exist: %s' %filepath)
    _, ext = os.path.splitext(filepath)
    if ext == '.npy':
        assert dataset_name == 'default'
        data = np.load(filepath)
    elif ext == '.npz':
        data = np.load(filepath)[dataset_name]
    elif ext == '.h5' or ext == '.cxi':
        data = h5py.File(filepath)[dataset_name]
    elif ext == '.mat':
        try:
            f = sio.loadmat(filepath)
            data = f[dataset_name]
        except NotImplementedError:  # v7.3 mat use h5py 
            f = h5py.File(filepath, 'r')
            data = f[dataset_name]
    elif ext == '.tif':
        assert dataset_name == 'default'
        data = np.asarray(Image.open(filepath), dtype=np.float64)
    elif ext == '.Iochamber':
        assert dataset_name == 'default'
        data = load_Iochamber(filepath)
    elif ext == '.Intensity':
        assert dataset_name == 'default'
        data = load_Intensity(filepath)
    data[np.isnan(data)] = 0.
    data[np.isinf(data)] = 0.
    return data


def get_data_info(filepath):
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise os.error('File not exist: %s' %filepath)
    _, ext = os.path.splitext(filepath)
    data_info = {}
    if ext == '.npy':
        data = np.load(filepath, 'r')
        data_info['default'] = {}
        data_info['default']['shape'] = data.shape
        if data.size == 1:
            data_info['default']['value'] = data 
        else:
            data_info['default']['value'] = None
    elif ext == '.npz':
        f = np.load(filepath, 'r')
        for key in f.keys():
            if len(f[key].shape) in [0,1,2,3]:
                data_info[key] = {}
                data_info[key]['shape'] = f[key].shape
                if f[key].size == 1:
                    data_info[key]['value'] = float(f[key])
                else:
                    data_info[key]['value'] = None
        f.close()
    elif ext == '.h5' or ext == '.cxi':
        f = h5py.File(filepath, 'r')
        keys = []
        def _get_all_dataset(key):
            if isinstance(f[key], h5py._hl.dataset.Dataset):
                keys.append(key)
        f.visit(_get_all_dataset)
        for key in keys:
            print key, f[key], f[key].shape, len(f[key].shape)
            if len(f[key].shape) in [0,1,2,3]:
                data_info[key] = {}
                data_info[key]['shape'] = f[key].shape
                if f[key].size == 1:
                    data_info[key]['value'] = f[key].value
                else:
                    data_info[key]['value'] = None
        f.close()
    elif ext == '.mat':
        try:
            f = sio.loadmat(filepath)
            for key in f.keys():
                if isinstance(f[key], np.ndarray):
                    if len(f[key].shape) in [0,1,2,3]:
                        data_info[key] = {}
                        data_info[key]['shape'] = f[key].shape
                        if f[key].size == 1:
                            data_info[key]['value'] = float(f[key].value)
                        else:
                            data_info[key]['value'] = None
        except NotImplementedError:  # v7.3 mat use h5py 
            f = h5py.File(filepath, 'r')
            for key in f.keys():
                if len(f[key].shape) in [0,1,2,3]:
                    data_info[key] = {}
                    data_info[key]['shape'] = f[key].shape
                    if f[key].size == 1:
                        data_info[key]['value'] = float(f[key].value)
                    else:
                        data_info[key]['value'] = None
            f.close()
    elif ext == '.tif':
        data = np.asarray(Image.open(filepath))
        data_info['default'] = {}
        data_info['default']['shape'] = data.shape
        data_info['default']['value'] = None
    elif ext == '.Iochamber':
        data = load_Iochamber(filepath)
        data_info['default'] = {}
        data_info['default']['shape'] = data.shape 
        data_info['default']['value'] = None
    elif ext == '.Intensity':
        data = load_Intensity(filepath)
        data_info['default'] = {}
        data_info['default']['shape'] = data.shape 
        data_info['default']['value'] = None
    return data_info


def load_Iochamber(filepath):
    f = open(filepath, 'r')
    intensity = []
    for i in range(3):  # skip first 3 lines
        f.readline()
    while True:
        line = f.readline()
        if line:
            intensity.append(float(line.split()[1]))
        else:
            break
    return np.asarray(intensity, dtype=np.float64)


def load_Intensity(filepath):
    f = open(filepath, 'r')
    intensity = []
    while True:
        line = f.readline()
        if line:
            try:
                I = float(line.split()[2])
                intensity.append(I)
                continue
            except:
                pass          
        else:
            break
    return np.asarray(intensity, dtype=np.float64)


def make_annulus(shape, inner_radii, outer_radii, fill_value=1., center=None):
    """Summary
    
    Parameters
    ----------
    imsize : array_like
        Image size. e.g. [40, 60]
    inner_radii : float
        Inner radii of annulus in pixel. Negative value will generate solid shape.
    outer_radii : float
        Outer radii of annulus in pixel. 
    fill_value : float, optional
        Values filled in annulus area.
    center : array_like, optional
        By default, the center is set to the center of image. You can also give a specified center: e.g. [0, 0]
    
    Returns
    -------
    ndarray
        Image with annulus inside
    """
    if center is not None:
        center = np.asarray(center)
        assert center.size == 2
    else:
        center = [shape[0]//2, shape[1]//2]
    img = np.ones((shape[0], shape[1])) * fill_value
    y, x = np.indices((img.shape))
    r = np.sqrt((x-center[0])**2. + (y-center[1])**2.)
    img[r<inner_radii] = 0.
    img[r>outer_radii] = 0.
    return img


def calc_Friedel_score(image, center, mask=None, mode='mean', ignore_negative=True):
    """Summary
    
    Parameters
    ----------
    image : 2d array
        Input image to calculate the Friedel score.
    center : array_like with 2 elements
        Center of input image
    mask : 2d array, optional
        Binary 2d array used in angular profile calculation. The shape must be same with image. 1 means valid while 0 not.
    mode : {'sum', 'mean'}, optional
        'mean'
        By default, mode is 'mean'. This returns sum(abs(diff(Friedel pair)))/num(Friedel pairs).
    
        'sum'
        Mode 'sum' returns sum(abs(Friedel pair)).
    
        'relative'
        Mode 'relative' returns sum(abs(diff(Friedel pair)/mean(Friedel pair)))/num(Friedel pairs)
    ignore_negative : bool, optional
        Ignore pixel <= 0.
    
    Returns
    -------
    Friedel score : float
        Centrosymmetric score.
    
    Raises
    ------
    Error
        Description
    """
    image = np.asarray(image, dtype=np.float64)
    assert len(image.shape) == 2
    center = np.asarray(center, dtype=np.int)
    assert center.size == 2
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        assert mask.shape == image.shape
        assert mask.min() >= 0. and mask.max() <= 1.
        mask = (mask > 0.5).astype(np.float64)
    else:
        mask = np.ones_like(image)
    cy, cx = center[0], center[1]
    sy, sx = image.shape
    lx = min(cx, sx-cx-1)
    ly = min(cy, sy-cy-1)
    image = image[cy-ly:cy+ly+1, cx-lx:cx+lx+1]
    mask = mask[cy-ly:cy+ly+1, cx-lx:cx+lx+1]
    if ignore_negative:
        mask *= (image > 0)
    mask = mask * np.rot90(np.rot90(mask))
    if mask.sum() == 0:
        return 0.  # no valid pixel
    if mode == 'sum':
        return (np.abs(image - np.rot90(np.rot90(image))) * mask).sum()
    elif mode == 'mean':
        return (np.abs(image - np.rot90(np.rot90(image))) * mask).sum() / mask.sum()
    elif mode == 'relative':
        mean_image = np.abs(0.5 * (image + np.rot90(np.rot90(image)))) + 1.E-10  # avoid divide 0 error
        return (np.abs((image - np.rot90(np.rot90(image))) / mean_image) * mask).sum() / mask.sum()
    else:
        raise Error('Unrecoganized mode: %s' %mode)

def getFilepathFromLocalFileID(localFileID):  # get real filepath from POSIX file in mac
    import CoreFoundation as CF
    import objc
    localFileQString = QtCore.QString(localFileID.toLocalFile())
    relCFStringRef = CF.CFStringCreateWithCString(
                     CF.kCFAllocatorDefault,
                     localFileQString.toUtf8(),
                     CF.kCFStringEncodingUTF8
                     )
    relCFURL = CF.CFURLCreateWithFileSystemPath(
               CF.kCFAllocatorDefault,
               relCFStringRef,
               CF.kCFURLPOSIXPathStyle,
               False   # is directory
               )
    absCFURL = CF.CFURLCreateFilePathURL(
               CF.kCFAllocatorDefault,
               relCFURL,
               objc.NULL
               )
    return QtCore.QUrl(str(absCFURL[0])).toLocalFile()