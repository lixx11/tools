#! /usr/bin/env python
#! coding=utf-8

"""
Process CrystFEL generated stream file, remove negative reflections.

Usage: 
    process_stream.py <stream_file> [options]

Options:
    -h --help                       Show this screen.
    --output-dir=output_dir         Output directory [default: output].
"""

import numpy as np 
import scipy as sp 
import os
import sys
from docopt import docopt
import re


class Reflection(object):
    def __init__(self, h, k, l, I, sigma_I, peak, background, fs, ss, panel):
        self.h = h
        self.k = k
        self.l = l
        self.I = I
        self.sigma_I = sigma_I
        self.peak = peak
        self.background = background
        self.fs = fs
        self.ss = ss
        self.panel = panel

    def __str__(self):
        return "Reflection(h:%d, k:%d, l:%d, I:%.2f, sigma_I:%.2f, peak:%.2f, background:%.2f, fs:%.2f, ss:%.2f, panel:%s)" \
                %(self.h, self.k, self.l, self.I, self.sigma_I, self.peak, self.background, self.fs, self.ss, self.panel)


if __name__ == '__main__':
    argv = docopt(__doc__)
    from_stream_file = argv['<stream_file>']
    basename = os.path.splitext(os.path.basename(from_stream_file))[0]
    output_dir = argv['--output-dir']
    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs('%s' %output_dir)
    to_stream_file = os.path.join(output_dir, '%s-processed.stream' % basename)
    print('Writing processed stream file and figures in %s' % output_dir)

    from_stream = open(from_stream_file, 'r')
    to_stream = open(to_stream_file, 'w')

    negative_reflections = []
    positive_reflections = []
    n_ref_Is = []
    p_ref_Is = []
    n_ref_ratio = []
    nb_peaks = []
    nb_reflections = 0
    while True:
        line = from_stream.readline()
        if line:
            if "indexed_by" in line:
                indexed_by = line.split("=")[1][1:-1]
                to_stream.write(line)
            elif "num_peaks" in line:
                nb_peak = int(line.split("=")[1])
                to_stream.write(line)
            elif "Reflections measured after indexing" in line:
                to_stream.write(line)
                line = from_stream.readline()
                to_stream.write(line)
                assert len(line.split()) == 10   # h, k, l, I, sigma, peak, bg, fs, ss, panel
                n_n = 0  # negative reflection count
                n_tol = 0  # total refletion count
                while True:
                    line = from_stream.readline()
                    if "End of reflections" in line:
                        to_stream.write(line)
                        break
                    h = int(line[:4])
                    k = int(line[4:9])
                    l = int(line[9:14])
                    I = float(line[14:25])
                    sigma_I = float(line[25:36])
                    peak = float(line[36:47])
                    background = float(line[47:58])
                    fs = float(line[58:65])
                    ss = float(line[65:72])
                    panel = line[72:-1]
                    reflection = Reflection(h, k, l, I, sigma_I, peak, background, fs, ss, panel)
                    if reflection.I > 0.:
                        to_stream.write(line)
                        positive_reflections.append(reflection)
                        p_ref_Is.append(I)
                    else:
                        negative_reflections.append(reflection)
                        n_ref_Is.append(I)
                        n_n += 1
                    n_tol += 1
                n_ref_ratio.append(float(n_n) / float(n_tol))
                nb_peaks.append(nb_peak)
                nb_reflections += n_tol
            else:
                to_stream.write(line)
        else:
            break
    n_ref_Is = np.asarray(n_ref_Is)
    p_ref_Is = np.asarray(p_ref_Is)
    n_ref_ratio = np.asarray(n_ref_ratio)
    nb_peaks = np.asarray(nb_peaks)

    # print summary, make figure
    print('=' * 60)
    print('Found %d reflections, %d negatives removed'  %(nb_reflections, n_ref_Is.size))
    print('=' * 60)

    fig = plt.figure()
    plt.hist(n_ref_Is, bins=np.arange(-1E4, 0, 100))
    plt.hist(p_ref_Is, bins=np.arange(0, 1E4, 100))
    plt.xlabel('Reflection Intensity')
    plt.ylabel('# of reflections')
    plt.title('Reflection Histogram')
    plt.savefig(os.path.join(output_dir, 'reflection_histogram.png'))
    plt.close()

    fig = plt.figure()
    plt.scatter(nb_peaks, n_ref_ratio)
    plt.xlim(xmin=0)
    plt.ylim((0.0, 1.0))
    plt.xlabel('# of peaks')
    plt.ylabel('negative reflection ratio')
    plt.title('# of peaks vs negative reflection ratio')
    plt.savefig(os.path.join(output_dir, 'peaks_vs_n_ref_ratio.png'))
    plt.close()