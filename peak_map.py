#! coding=utf-8

import numpy as np 
import scipy as sp 
import sys
import matplotlib
import matplotlib.pyplot as plt 
from stream_parser import parse_stream

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify stream file to process")
        sys.exit()
    stream_file = sys.argv[1]
    if len(sys.argv) >= 3:
        map_type = sys.argv[2]
        if map_type in ["peak", "indexed_peak", "reflection"]:
            pass
        else:
            print("Unrecognized map type: %s" %map_type)
            print("Select map type from 'peak(default)', 'indexed_peak' and 'reflection'")
            sys.exit()
    else:
        map_type = "peak"
    index_stats = parse_stream(stream_file, max_chunk=np.inf)

    fs = []
    ss = []
    peak_intensity = []
    for x in xrange(len(index_stats)):
        index_stat = index_stats[x]
        if index_stat.index_method == 'mosflm-comb-latt-cell-retry-nomulti-refine':
            for chunk in index_stat.chunks:
                if map_type == "peak":
                    # print("generating peak map")
                    for peak in chunk.peaks:
                        fs.append(peak.fs)
                        ss.append(peak.ss)
                        peak_intensity.append(peak.intensity)
                if map_type == "indexed_peak":
                    # print("generating indexed peak map")
                    for reflection_peak in chunk.reflection_peaks:
                        fs.append(reflection_peak.fs)
                        ss.append(reflection_peak.ss)
                        peak_intensity.append(reflection_peak.intensity)
                if map_type == "reflection":
                    # print("generating reflection map")
                    for reflection in chunk.reflections:
                        fs.append(reflection.fs)
                        ss.append(reflection.ss)
                        peak_intensity.append(reflection.intensity)

    # make map
    fs, ss = np.round(np.asarray(fs)).astype(int), np.round(np.asarray(ss)).astype(int)
    sx, sy = 1480, 1552  # original detector matix size
    peak_map = np.zeros((sx, sy), dtype=np.int)
    peak_intensity_map = np.zeros((sx, sy), dtype=np.int)
    for x in xrange(fs.size):
        peak_map[ss[x], fs[x]] += 1
        peak_intensity_map[ss[x], fs[x]] += peak_intensity[x]
    peak_intensity_map /= peak_map

    # save map
    np.save("%s.npy" %map_type, peak_map)
    np.save("%s_intensity.npy" %map_type, peak_intensity_map)
    print("peak map saved in %s.npy" %map_type)
    print("peak intensity map saved in %s_intensity.npy" %map_type)

