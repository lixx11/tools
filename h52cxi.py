#! /usr/bin/env python
#! coding=utf-8

"""
Convert multiple h5 files into a single cxi file.

Usage: 
    h52cxi.py <files.lst> [options]

Options:
    -h --help                   Show this screen.
    -o output.cxi               Filename of output cxi file [default: output.cxi].   
    -n num_hdf5                 Max number of hdf5 file to process [default: 0]. 
"""


from docopt import docopt
from tqdm import tqdm

import h5py
import numpy as np 


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    files_lst = argv['<files.lst>']
    output = argv['-o']
    nb_h5 = int(argv['-n'])

    with open(files_lst, 'r') as f:
        files = f.readlines()

    cxi = h5py.File(output, 'w')
    n_peaks = []
    peaks_x = []
    peaks_y = []
    peaks_intensity = []
    encoders = []
    wavelengths = []
    if nb_h5 == 0:
        N = len(files)
    else:
        N = min(len(files), nb_h5)
    for i in tqdm(range(N)):
        f = files[i][:-1]
        h5 = h5py.File(f, 'r')
        peak_x = np.zeros(1024)
        peak_y = np.zeros(1024)
        peak_intensity = np.zeros(1024)
        data = h5['processing/cheetah/peakinfo-raw'].value
        peak_mask = (data[:,0] > 0)* (data[:,1] > 0)
        n_peak = peak_mask.sum().astype(np.int)
        n_peaks.append(n_peak)
        peak_x[0:n_peak] = data[peak_mask,0]
        peak_y[0:n_peak] = data[peak_mask,1]
        peak_intensity [0:n_peak] = data[peak_mask,2]
        peaks_x.append(peak_x)
        peaks_y.append(peak_y)
        peaks_intensity.append(peak_intensity)
        # encoder value
        encoders.append(h5['/LCLS/detector0-EncoderValue'].value[0])
        # wavelength
        wavelengths.append(h5['/LCLS/photon_energy_eV'].value[0])

        pattern = h5['data/rawdata0'].value
        sy, sx = pattern.shape
        pattern = pattern.reshape((1, sy, sx))
        if i == 0:
            cxi.create_dataset('/entry_1/instrument_1/detector_1/detector_corrected/data',
                data=pattern, maxshape=(None, sy, sx))
        else:
            cxi['/entry_1/instrument_1/detector_1/detector_corrected/data'].resize(i+1, axis=0)
            cxi['/entry_1/instrument_1/detector_1/detector_corrected/data'][i] = pattern

    peaks_x = np.asarray(peaks_x)
    peaks_y = np.asarray(peaks_y)
    peaks_intensity = np.asarray(peaks_intensity)
    n_peaks = np.asarray(n_peaks)
    encoders = np.asarray(encoders)
    wavelengths = np.asarray(wavelengths)
    cxi.create_dataset('/entry_1/result_1/peakXPosRaw', data=peaks_x)
    cxi.create_dataset('/entry_1/result_1/peakYPosRaw', data=peaks_y)
    cxi.create_dataset('/LCLS/detector_1/EncoderValue', data=encoders)
    cxi.create_dataset('/LCLS/photon_energy_eV', data=wavelengths)
    cxi.create_dataset('/entry_1/result_1/nPeaks', data=n_peaks)
    cxi.create_dataset('/entry_1/result_1/peakTotalIntensity', data=peaks_intensity)
    cxi.close()