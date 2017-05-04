#! /opt/local/bin/python2.7
#! coding=utf-8

"""
Usage: 
    stream_parser.py <stream_file> [--max-chunks=<mc>]

Options:
    -h --help       Show this screen.
    --max-chunks=<mc>    Max chunks to process [default: inf].
"""

import numpy as np 
import scipy as sp 
import os
import sys
from docopt import docopt
import re


class Chunk(object):
    def __init__(self, image_filename, event):
        self.image_filename = image_filename
        self.event = event
        self.indexed_by = None
        self.photon_energy_eV = 0.
        self.beam_divergence = 0.
        self.beam_bandwidth = 0.
        self.EncoderValue = 0. # clen
        self.average_camera_length = 0.
        self.num_peaks = 0
        self.num_saturated_peaks = 0
        self.crystal = None
        self.peaks = []
        self.reflections = []
        self.reflection_peaks = []

    def __str__(self):
        return "%s-%d" %(self.image_filename, self.event)

    def get_info(self):
        print("============CHUNK INFO============")
        print("%-16s: %s" %("image_filename", self.image_filename))
        print("%-16s: %d" %("event", self.event))
        print("%-16s: %s" %("indexed_by", self.indexed_by))
        print("%-16s: %.2f" %("photon_energy_eV", self.photon_energy_eV))
        print("%-16s: %d" %("num_peaks", self.num_peaks))
        print("%-16s: %d" %("num_saturated_peaks", self.num_saturated_peaks))
        if self.crystal is not None:
            print("%-16s: %s" %("crystal", self.crystal.__str__()))
            print("%-16s: %s" %("lattice_type", self.crystal.lattice_type))
            print("%-16s: %s" %("centering", self.crystal.centering))
        print("===================================")


class Crystal(object):
    def __init__(self):
        self.cell_parameter = CellParameter(0, 0, 0, 0, 0, 0)
        self.astar = None
        self.bstar = None
        self.cstar = None
        self.lattice_type = None
        self.centering = None
        self.unique_axis = None
        self.profile_radius = 0.
        self.det_shift = 0.
        self.num_reflections = 0
        self.num_saturated_reflections = 0
        self.num_implausible_reflections = 0

    def __str__(self):
        return self.cell_parameter.__str__()


class Peak(object):
    def __init__(self, fs, ss, reverse_d, intensity, panel):
        self.fs = fs
        self.ss = ss 
        self.reverse_d = reverse_d
        self.intensity = intensity
        self.panel = panel 
    
    def __str__(self):
        return "Peak(fs:%.2f, ss:%.2f, 1/d:%.2f, I:%.2f, panel:%s)" \
               %(self.fs, self.ss, self.reverse_d, self.intensity, self.panel)


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


class CellParameter(object):
    def __init__(self, a, b, c, al, be, ga):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.al = float(al)
        self.be = float(be)
        self.ga = float(ga)

    def __str__(self):
        return "(%.2f, %.2f, %.2f A; %.2f, %.2f, %.2f deg)" \
               %(self.a, self.b, self.c, self.al, self.be, self.ga)

    def tolist(self):
        return [self.a, self.b, self.c, self.al, self.be, self.ga]
        

class IndexStat(object):
    """docstring for IndexStat"""
    def __init__(self, index_method):
        self.index_method = index_method
        self.indexed_num = 0
        self.chunks = []

    def __str__(self):
        return "stat of " + self.index_method

    def maybe_add_chunk(self, chunk):
        if chunk.indexed_by == self.index_method:
            self.chunks.append(chunk)
            self.indexed_num += 1
        else:
            pass

    def show_cell_parameter(self, bin_size=2., padding=10.):
        import matplotlib
        import matplotlib.pyplot as plt 

        if self.index_method == "none":
            print("Warning! None Indexing Method!")
        else:
            # collect cell parameters
            CPs = []
            for x in xrange(self.indexed_num):
                CP = self.chunks[x].crystal.cell_parameter.tolist()
                CPs.append(CP)
            CPs = np.asarray(CPs)
            param_names = ['a', 'b', 'c',
                           'alpha', 'beta', 'gamma']
            # plot cell parameter distributions
            fig = plt.figure(self.index_method)
            for i in xrange(6):
                cell_param = CPs[:,i]
                param_avg = cell_param.mean()
                param_std = cell_param.std()
                bins = np.arange(cell_param.mean() - padding, cell_param.mean() + padding, bin_size)

                ax = plt.subplot("23" + str(i+1))
                plt.hist(cell_param, bins)
                title = "%s -- %.2f, %.2f" %(param_names[i], param_avg, param_std)
                plt.title(title)
            plt.show()


def get_chunk_num(stream_file):
    nb_chunks = 0
    f = open(stream_file, 'r')
    for line in open(stream_file, 'r'):
        if re.search("Begin chunk", line):
            nb_chunks += 1
    return nb_chunks


def parse_abcstar(abcstar_str):
    _, xstarx, xstary, xstarz, _ = abcstar_str.split(" ")
    xstar = float(xstarx), float(xstary), float(xstarz)
    xstar = np.asarray([xstarx, xstary, xstarz], dtype=np.float) * 1E9  # in per m
    return xstar


def to_spind_file(index_stats, filepath):
    """Write abc star to file in spind format
    
    Args:
        filepath (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    # default value
    match_rate = 0.
    nb_pairs = 0
    pair_dist = 0.
    chunks = []
    for index_stat in index_stats:
        chunks += index_stat.chunks
    # sort chunks by event id
    chunks.sort(key=lambda x: x.event)
    f = open(filepath, 'w')
    for c in chunks:
        if c.indexed_by == 'none':
            f.write('%6d %.2f %4d %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n'
                  % (c.event, match_rate, nb_pairs, pair_dist, 
                     1, 0, 0, 0, 1, 0, 0, 0, 1))
        else:
            cr = c.crystal
            f.write('%6d %.2f %4d %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n'
                  % (c.event, match_rate, nb_pairs, pair_dist,
                    cr.astar[0], cr.astar[1], cr.astar[2],
                    cr.bstar[0], cr.bstar[1], cr.bstar[2],
                    cr.cstar[0], cr.cstar[1], cr.cstar[2]))


def parse_stream(filepath, max_chunks=np.inf):
    """Summary
    
    Args:
        filepath (string): stream filepath
        max_chunks (int, optional): max chunks to process
    
    Returns:
        List of index_stats: Description
    """
    total_chunks = get_chunk_num(filepath)
    stream_file = open(filepath)
    chunks = []
    indexed_bys = []
    image_filenames = []
    count_chunk = 0
    while count_chunk < max_chunks:
        line = stream_file.readline()
        if line:
            if "Begin chunk" in line:
                filenameL = stream_file.readline()
                eventL = stream_file.readline()
                filename = filenameL.split(":")[1][1:-1]
                event = int(eventL.split(":")[1][3:])
                chunk = Chunk(filename, event)
                count_chunk += 1
                print("\r%.1f%% chunks has been processed" %(float(count_chunk)/float(total_chunks)*100.)),
                image_filenames.append(os.path.basename(filename))

                while True:
                    newL = stream_file.readline()
                    if newL:
                        if "indexed_by" in newL:
                            chunk.indexed_by = newL.split("=")[1][1:-1]
                            indexed_bys.append(chunk.indexed_by)
                        elif "photon_energy_eV" in newL:
                            chunk.photon_energy_eV = float(newL.split("=")[1])
                        elif "num_peaks" in newL:
                            chunk.num_peaks = int(newL.split("=")[1])
                        elif "Peaks from peak search" in newL:
                            fs_ss = []  # all fs, ss in this chunk
                            newL = stream_file.readline()
                            assert int(len(newL.split())) == 5 # fs/px, ss/px, 1/d, Intensity, Panel
                            while True:
                                newL = stream_file.readline()
                                if newL:
                                    if "End of peak list" in newL:
                                        break
                                    fs = float(newL[:7])
                                    ss = float(newL[7:15])
                                    fs_ss.append([fs, ss])
                                    reverse_d = float(newL[15:27])
                                    intensity = float(newL[27:41])
                                    panel = newL[41:-1]
                                    peak = Peak(fs, ss, reverse_d, intensity, panel)
                                    chunk.peaks.append(peak)
                                else:
                                    break
                        elif "Begin crystal" in newL:
                            crystal = Crystal()
                            while True:
                                newL = stream_file.readline()
                                if newL:  
                                    if "Cell parameters" in newL:
                                        CP = newL.split(" ")
                                        if len(CP) == 6:
                                            a = float(CP[2]) * 10.
                                            b = float(CP[3]) * 10.
                                            c = float(CP[4]) * 10.
                                            al = float(CP[6])
                                            be = float(CP[7])
                                            ga = float(CP[8])
                                        else:  # abnormal cell parameters
                                            a, b, c = 0., 0., 0.
                                            al, be, ga = 0., 0., 0.
                                        cp = CellParameter(a, b, c, al, be, ga)
                                        crystal.cell_parameter = cp
                                    elif "astar" in newL:
                                        crystal.astar = parse_abcstar(newL.split("=")[1])
                                    elif "bstar" in newL:
                                        crystal.bstar = parse_abcstar(newL.split("=")[1])
                                    elif "cstar" in newL:
                                        crystal.cstar = parse_abcstar(newL.split("=")[1])
                                    elif "lattice_type" in newL:
                                        crystal.lattice_type = newL.split("=")[1][1:-1]
                                    elif "centering" in newL:
                                        crystal.centering = newL.split("=")[1][1:-1]
                                    elif "Reflections measured after indexing" in newL:
                                        newL = stream_file.readline()
                                        assert len(newL.split()) == 10   # h, k, l, I, sigma, peak, bg, fs, ss, panel
                                        fs_ss = np.asarray(fs_ss)
                                        while True:
                                            newL = stream_file.readline()
                                            if "End of reflections" in newL:
                                                break
                                            h = int(newL[:4])
                                            k = int(newL[4:9])
                                            l = int(newL[9:14])
                                            I = float(newL[14:25])
                                            sigma_I = float(newL[25:36])
                                            peak = float(newL[36:47])
                                            background = float(newL[47:58])
                                            fs = float(newL[58:65])
                                            ss = float(newL[65:72])
                                            panel = newL[72:-1]
                                            reflection = Reflection(h, k, l, I, sigma_I, peak, background, fs, ss, panel)
                                            chunk.reflections.append(reflection)
                                            min_dist = np.sqrt((fs_ss[:,0]-fs)**2. + (fs_ss[:,1]-ss)**2.).min()
                                            if  min_dist < 2.:
                                                chunk.reflection_peaks.append(reflection)
                                    elif "End crystal" in newL:
                                        chunk.crystal = crystal
                                        break
                                else:
                                    break
                        elif "End chunk" in newL:
                            chunks.append(chunk)
                            break
                        else:
                            pass
                            # print("unprocessed line: %s" %newL)
                    else:
                        break
        else:
            break
    index_stats = []
    indexed_bys = list(set(indexed_bys))
    image_filenames = list(set(image_filenames))
    chunks_num = len(chunks)
    chunks.sort(key=lambda x: x.indexed_by, reverse=True)
    for i in range(len(indexed_bys)):
        index_stat = IndexStat(indexed_bys[i])
        index_stats.append(index_stat)
        # print("%s index_stat created." %indexed_bys[i])
    for i in range(len(chunks)):
        chunk = chunks[i]
        for j in range(len(index_stats)):
            index_stats[j].maybe_add_chunk(chunk)
    index_stats.sort(key=lambda x: x.indexed_num, reverse=True)
    print("\n")
    print("===========STREAM SUMMARY=============")
    print("%16s: %d" %("Total Processed Chunks", chunks_num))
    for i in xrange(len(index_stats)):
        print("%-16s: %s" %("indexing method", index_stats[i].index_method))
        indexing_rate = float(index_stats[i].indexed_num) / chunks_num
        print("%-16s: %.3f" %("indexing rate", indexing_rate))
    print("--------PROCESSED FILES--------")
    image_filenames.sort()
    for i in xrange(len(image_filenames)):
        print(image_filenames[i])
    print("============END SUMMARY===============")
    return index_stats


if __name__ == '__main__':
    argv = docopt(__doc__)
    stream_file = argv['<stream_file>']
    max_chunks = argv['--max-chunks']
    if max_chunks == 'inf':
        max_chunks = np.inf
    else:
        max_chunks = int(max_chunks)

    index_stats = parse_stream(stream_file, max_chunks=max_chunks)
