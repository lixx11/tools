#!/usr/bin/env python3

"""
Usage: 
  stream_parser.py <stream_file> [options]

Options:
  -h --help           Show this screen.
  --max-chunks NUM    Max chunks to process [default: inf].
"""

import numpy as np
from docopt import docopt
import re


class Chunk(object):
    def __init__(self, content):
        peak_start = 0
        for i in range(len(content)):
            if 'Peaks from peak search' in content[i]:
                peak_start = i
        if peak_start == 0:
            raise ValueError('No peak section found in this chunk!')
        for i in range(peak_start):
            if 'Image filename' in content[i]:
                self.image_filename = content[i].split(':')[1][1:-1]
            elif 'Event' in content[i]:
                self.event = int(content[i].split(':')[1][3:-1])
            elif 'Image serial number' in content[i]:
                self.image_serial_number = int(content[i].split(':')[1][1:-1])
            elif 'indexed_by' in content[i]:
                self.indexed_by = content[i].split('=')[1][1:-1]
            elif 'photon_energy_eV' in content[i]:
                self.photon_energy_eV = float(content[i].split('=')[1])
            elif 'beam_divergence' in content[i]:
                self.beam_divergence = float(content[i].split(' ')[2])
            elif 'beam_bandwidth' in content[i]:
                self.beam_bandwidth = float(content[i].split(' ')[2])
            elif 'average_camera_length' in content[i]:
                self.average_camera_length = float(content[i].split(' ')[2])
            elif 'num_peaks' in content[i]:
                self.num_peaks = int(content[i].split('=')[1])
            elif 'num_saturated_peaks' in content[i]:
                self.num_saturated_peaks = int(content[i].split('=')[1])
            else:
                pass
        self.peaks = []
        for i in range(peak_start+1, len(content)):
            if 'End of peak list' in content[i]:
                peak_end = i
                break
            else:
                peak = Peak(content[i])
                self.peaks.append(peak)
        self.crystals = []
        crystal_start = peak_end + 1
        for i in range(peak_end, len(content)):
            if 'End crystal' in content[i]:
                crystal_end = i
                crystal = Crystal(content[crystal_start:crystal_end])
                self.crystals.append(crystal)
                crystal_start = crystal_end + 1


class UnitCell(object):
    pass


def parse_abc_star(content):
    _, x, y, z, _ = content.split('=')[1].split(' ')
    x, y, z = float(x), float(y), float(z)
    x_star = np.asarray([x, y, z])
    return x_star


class Crystal(object):
    def __init__(self, content):
        self.unit_cell = UnitCell
        self.parse_cell_params(content[1])
        self.a_star = parse_abc_star(content[2])
        self.b_star = parse_abc_star(content[3])
        self.c_star = parse_abc_star(content[4])
        self.lattice_type = content[5].split('=')[1][1:-1]
        self.centering = content[6].split('=')[1][1:-1]
        self.unique_axis = content[7].split('=')[1][1:-1]
        self.profile_radius = float(content[8].split(' ')[2])
        self.det_shift_x = float(content[9].split(' ')[3])
        self.det_shift_y = float(content[9].split(' ')[6])
        self.diffraction_resolution_limit = float(content[10].split(' ')[5])
        self.num_reflections = int(content[11].split('=')[1])
        self.num_saturated_reflections = int(content[12].split('=')[1])
        self.num_implausible_reflections = int(content[13].split('=')[1])
        self.reflections = []
        for i in range(16, len(content)):
            if len(content[i].split()) == 10:
                reflection = Reflection(content[i])
                self.reflections.append(reflection)

    def parse_cell_params(self, cell_params_str):
        _, _, a, b, c, _, alpha, beta, gamma, _ = cell_params_str.split(' ')
        self.unit_cell.a = float(a)
        self.unit_cell.b = float(b)
        self.unit_cell.c = float(c)
        self.unit_cell.alpha = float(alpha)
        self.unit_cell.beta = float(beta)
        self.unit_cell.gamma = float(gamma)


class Peak(object):
    def __init__(self, content):
        fs, ss, one_over_d, intensity, panel = content.split()
        panel = panel[:-1]
        self.fs = fs
        self.ss = ss
        self.one_over_d = one_over_d
        self.intensity = intensity
        self.panel = panel


class Reflection(object):
    def __init__(self, content):
        h, k, l, I, sigma_I, peak, background, fs, ss, panel = content.split()
        panel = panel[:-1]
        self.h, self.k, self.l = int(h), int(k), int(l)
        self.I, self.sigma_I = float(I), float(sigma_I)
        self.peak, self.background = float(peak), float(background)
        self.fs, self.ss = float(fs), float(ss)
        self.panel = panel


class Stream(object):
    def __init__(self, filename, max_chunks=np.inf, debug=False):
        self.filename = filename
        self.max_chunks = max_chunks
        self.debug = debug
        self.index_stats = []
        self.chunks = []
        self.image_filename_list = []
        self.peaks = []
        self.crystals = []
        self.reflections = []
        self.indexers = []
        self.nb_indexed = 0
        self.nb_unindexed = 0
        self.parse_stream()

    def print_summary(self):
        print('total chunks: %d' % len(self.chunks))
        print('total indexed: %d' % self.nb_indexed)
        print('total unindexed: %d' % self.nb_unindexed)
        print('total crystals: %d' % len(self.crystals))
        print('total peaks: %d' % len(self.peaks))
        print('total reflections: %d' % len(self.reflections))

    def parse_stream(self):
        max_chunks = min(self.max_chunks, get_chunk_num(self.filename))
        stream = open(self.filename, 'r')
        count_chunk = 0
        while True:
            print('%.1f%% processed so far' %
                  (float(count_chunk) / max_chunks * 100.), end='\r')
            if count_chunk >= max_chunks:
                break
            line = stream.readline()
            if not line:
                break
            chunk_content = []
            if 'Begin chunk' in line:
                chunk_content.append(line)
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    elif 'End chunk' in line:
                        chunk_content.append(line)
                        chunk = Chunk(chunk_content)
                        self.chunks.append(chunk)
                        count_chunk += 1
                        break
                    chunk_content.append(line)
        print(flush=True)
        # collect summary
        indexers = []
        for i in range(len(self.chunks)):
            chunk = self.chunks[i]
            self.peaks += chunk.peaks
            indexer = chunk.indexed_by
            if 'none' in indexer:
                self.nb_unindexed += 1
            else:
                self.nb_indexed += 1
            indexers.append(indexer)
            self.crystals += chunk.crystals
            for j in range(len(chunk.crystals)):
                self.reflections += chunk.crystals[j].reflections
        self.indexers = list(set(indexers))


# some helper functions
def get_chunk_num(stream_file):
    nb_chunks = 0
    for line in open(stream_file, 'r'):
        if re.search("End chunk", line):
            nb_chunks += 1
    return nb_chunks


def main():
    argv = docopt(__doc__)
    stream_file = argv['<stream_file>']
    max_chunks = argv['--max-chunks']
    if max_chunks == 'inf':
        max_chunks = np.inf
    else:
        max_chunks = int(max_chunks)

    stream = Stream(stream_file, max_chunks=max_chunks)
    stream.print_summary()


if __name__ == '__main__':
    main()

