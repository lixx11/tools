#! /usr/bin/env python
#! coding=utf-8

"""
Convert multiple h5 files into a single cxi file.

Usage: 
    h52cxi.py <cxifile> [options]

Options:
    -h --help                   Show this screen.
    -o output_dir               Filename of output cxi file [default: output].   
    -n num_events               Max number of events to process [default: 0]. 
"""


from docopt import docopt
from tqdm import tqdm
import os

import h5py
import numpy as np 


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    cxi_file = argv['<cxifile>']
    output = argv['-o']
    nb_events = int(argv['-n'])

    cxi = h5py.File(cxi_file, 'r')
    if not os.path.isdir(output):
        os.mkdir(output)

    patterns = cxi['/entry_1/instrument_1/detector_1/detector_corrected/data']
    n_peaks = cxi['/entry_1/result_1/nPeaks']
    peaks_x = cxi['/entry_1/result_1/peakXPosRaw']
    peaks_y = cxi['/entry_1/result_1/peakYPosRaw']
    peaks_intensity = cxi['/entry_1/result_1/peakTotalIntensity']
    encoders = cxi['/LCLS/detector_1/EncoderValue']
    wavelengths = cxi['/LCLS/photon_energy_eV']

    if nb_events == 0:
        N = patterns.shape[0]
    else:
        N = min(patterns.shape[0], nb_events)
    for i in tqdm(range(N)):
        h5_file = '%s/event-%04d.h5' % (output, i)
        h5 = h5py.File(h5_file, 'w')

        peakinfo = np.zeros((n_peaks[i], 4))
        peakinfo[:,0] = peaks_x[i, 0:n_peaks[i]]
        peakinfo[:,1] = peaks_y[i, 0:n_peaks[i]]
        peakinfo[:,2] = peaks_intensity[i, 0:n_peaks[i]]

        h5.create_dataset('/LCLS/detector0-EncoderValue', data=encoders[i])
        h5.create_dataset('/LCLS/photon_energy_eV', data=wavelengths[i])
        h5.create_dataset('/processing/hitfinder/peakinfo', data=peakinfo)
        h5.create_dataset('/data/rawdata0', data=patterns[i])

        h5.close()
