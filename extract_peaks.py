#!/usr/bin/env python

"""
Usage:
    extract_peaks.py -f data.cxi -g geom.h5 [options]

Options:
    -h --help                                        Show this screen.
    -f data.cxi                                      Image file in cxi format.
    -g geom.h5                                       Geometry file in hdf5 format.
    -o output_dir                                    Output directory [default: output].
    --sort-by=<metric>                               Sort peaks by certain metric [default: SNR].
    --res=<resolution>                               Resolution threshold for high priority peaks in angstrom [default: 4.5].
"""

from docopt import docopt
import os
import h5py
from tqdm import tqdm  # show a progress bar for loops
import numpy as np
from math import sin, atan, sqrt


S_p = 110E-6  # pixel size in meters of CSPAD
d = 136.4028E-3  # detector distance in meters
lam = 1.306098E-10  # XFEL wavelength in meters


class Peak(object):
    """docstring for peak"""
    def __init__(self, x, y, res, SNR, total_intensity, encoder_value, photon_energy):
        """Summary
        
        Args:
            x (double): x coordinate in assembled detector in pixels.
            y (double): y coordinate in assembled detector in pixels.
            res (double): resolution in angstrom.
            SNR (double): peak signal to noise ratio.
            total_intensity (double): peak total intensity.
            encoder_value (double): camera offset.
            photon_energy (double): photom energy in eV.
        """
        super(Peak, self).__init__()
        self.x = x
        self.y = y
        self.res = res 
        self.SNR = SNR
        self.total_intensity = total_intensity
        self.encoder_value = encoder_value
        self.photon_energy = photon_energy


if __name__ == '__main__':
    argv = docopt(__doc__)
    cxi_file = argv['-f']
    geom_file = argv['-g']
    sort_by = argv['--sort-by']
    res_threshold = float(argv['--res']) * 1.E-10  # convert resolution in angstroms to meters.

    # load cxi file
    data = h5py.File(cxi_file,'r')
    basename, ext = os.path.splitext(os.path.basename(cxi_file))

    # load geometry
    geom = h5py.File(geom_file, 'r')
    geom_x, geom_y = geom['x'].value, geom['y'].value

    # mkdir for output
    output_dir = argv['-o']
    if not os.path.isdir(output_dir):
        os.makedirs('%s' % output_dir)

    nb_events = data['/cheetah/event_data/hit'][:].size
    print('Processing %s with geometry %s' % (cxi_file, geom_file))
    for event_id in tqdm(range(nb_events)):
        output = output_dir + '/' + basename + 'e' + str(event_id) + '.txt'
        nb_peaks = np.nonzero(data['entry_1/result_1/peakTotalIntensity'][event_id][:])[0].size
        peak_list = []
        for peak_id in range(nb_peaks):
            rawX = int(data['/entry_1/result_1/peakXPosRaw'][event_id][peak_id])
            rawY = int(data['/entry_1/result_1/peakYPosRaw'][event_id][peak_id])
            assX = geom_x[rawY][rawX] / S_p
            assY = geom_y[rawY][rawX] / S_p
        
            total_intensity = data['/entry_1/result_1/peakTotalIntensity'][event_id][peak_id]
            SNR = data['/entry_1/result_1/peakSNR'][event_id][peak_id]
            encoder_value = data['/LCLS/detector_1/EncoderValue'][event_id]
            photon_energy = data['LCLS/photon_energy_eV'][event_id]
            # calculate resolution
            r = sqrt(assX ** 2. + assY ** 2.)
            res = lam / (sin(0.5 * atan(r * S_p / d)) * 2)
            peak = Peak(assX, assY, res, SNR, total_intensity, encoder_value, photon_energy)
            peak_list.append(peak)

        peak_list.sort(key=lambda peak: peak.SNR, reverse=True)
        HP_ids = []  # high priority peak indices 
        LP_ids = []  # low priority peak indices
        for peak_id in range(len(peak_list)):
            peak = peak_list[peak_id]
            if peak.res > res_threshold:
                HP_ids.append(peak_id)
            else:
                LP_ids.append(peak_id)
        # rearange peaks order according to the priority
        new_peak_list = []
        for peak_id in HP_ids:
            new_peak_list.append(peak_list[peak_id])
        for peak_id in LP_ids:
            new_peak_list.append(peak_list[peak_id])
        # write to file
        f = open(output, 'w')
        for peak in (new_peak_list):
            f.write('%.5e %.5e %.5e %.5e %.5e %.5e\n' % 
                    (peak.x, peak.y, peak.total_intensity, peak.SNR, peak.photon_energy, peak.encoder_value))


