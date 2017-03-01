#! /opt/local/bin/python2.7

"""
Usage:
    cxi_viewer.py -f cxi_file -g geometry_file [options]

Options:
    -h --help                                        Show this screen.
    -f data.cxi                                      Specify cxi file for visulization.
    -g geometry_file                                 Geometry file in cheetah, crystfel or psana format.
    --start=<start_from>                             First frame to render [default: 0].
    --spind-stream=<spind.stream>                 CrystFEL stream file using SPIND as auto-indexer.
    --crystfel-stream=<crystfel.stream>              Another crystFEL stream file.
    --cmin=<cmin>                                    Color level lower limit [default: 0].
    --cmax=<cmax>                                    Color level upper limit [default: 500].
"""

import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from docopt import docopt
import numpy as np
import h5py
from stream_parser import parse_stream
import psgeom
import os


pixel_size = 110E-6  # CSPAD pixel size


def get_reflection_from_stream(filename):
    chunks = []
    records = []
    index_stats = parse_stream(filename)
    for index_stat in index_stats:
        if index_stat.index_method != 'none':
            chunks += index_stat.chunks
    for c in chunks:
        reflections = []
        for r in c.reflections:
            reflection = [r.fs, r.ss]
            reflections.append(reflection)
        reflections = np.asarray(reflections)
        records.append({'event': c.event, 'predicted_spots': reflections})
    return records


def load_geom(filename):
    print "load %s" % filename
    ext = os.path.splitext(filename)[1]
    if ext == '.h5':
        f = h5py.File(filename, 'r')
        return f['x'].value, f['y'].value, f['z'].value
    elif ext == '.geom':
        from psgeom import camera
        cspad = camera.Cspad.from_crystfel_file(filename)
        cspad.to_cheetah_file('.geom.h5')
        f = h5py.File('.geom.h5', 'r')
        return f['x'].value, f['y'].value, f['z'].value
    elif ext == '.psana':
        cspad = camera.Cspad.from_psana_file(filename)
        cspad.to_cheetah_file('.geom.h5')
        f = h5py.File('.geom.h5', 'r')
        return f['x'].value, f['y'].value, f['z'].value
    else:
        print('Wrong geometry: %s. You must provide Cheetah, CrystFEL or psana geometry file.')
        return None


class CXIViewer(pg.ImageView):
    """docstring for CXIViewer"""
    def __init__(self, cxi_file, geom_file, start_from,
                 spind_stream, crystfel_stream,
                 cmin, cmax):
        super(CXIViewer, self).__init__()
        self.cxi_file = cxi_file
        self.geom_file = geom_file
        self.indexed_events = []

        if spind_stream is not None:
            self.spind_predicted = get_reflection_from_stream(spind_stream)
            self.show_spind_spots = True
            for i in range(len(self.spind_predicted)):
                self.indexed_events.append(self.spind_predicted[i]['event'])
        else:
            self.show_spind_spots = False
        if crystfel_stream is not None:
            self.crystfel_predicted = get_reflection_from_stream(crystfel_stream)
            self.show_crystfel_spots = True
            for i in range(len(self.crystfel_predicted)):
                self.indexed_events.append(self.crystfel_predicted[i]['event'])
        else:
            self.show_crystfel_spots = False
        self.indexed_events = np.asarray(list(set(self.indexed_events)))
        self.indexed_events.sort()
        self.level = [cmin, cmax]

        # load data from cxi file
        cxi = h5py.File(self.cxi_file, 'r')
        self.data = cxi['/entry_1/instrument_1/detector_1/detector_corrected/data']
        self.peaks_x = cxi['/entry_1/result_1/peakXPosRaw']
        self.peaks_y = cxi['/entry_1/result_1/peakYPosRaw']

        # load geometry
        # geom = h5py.File(geom_file, 'r')
        self.geom_x, self.geom_y, self.geom_z = load_geom(geom_file)
        # self.geom_x, self.geom_y = geom['x'].value, geom['y'].value
        x = np.int_(np.rint(self.geom_x / pixel_size))
        y = np.int_(np.rint(self.geom_y / pixel_size))
        self.offset_x = abs(x.min())
        self.offset_y = abs(y.min())
        x += self.offset_x
        y += self.offset_y
        self.nx, self.ny = x.max()+1, y.max()+1
        self.x, self.y = x, y

        # center item
        self.center_item = pg.ScatterPlotItem()
        self.center_item.setData([0+self.offset_x], [0+self.offset_y], symbol='+', size=20, brush=(255,255,255,0), pen='y')
        self.getView().addItem(self.center_item)

        self.spind_predicted_spot_items = []
        self.crystfel_predicted_spot_items = []

        self.frame = start_from
        self.peak_items = []
        self.update(self.frame)

    def update(self, frame):
        # clear peaks and predicted spots in last frame
        for p in self.peak_items:
            p.clear()
        for p in self.spind_predicted_spot_items:
            p.clear()
        for p in self.crystfel_predicted_spot_items:
            p.clear()
        img_remap = np.zeros((self.nx, self.ny))
        img_remap[self.x.ravel(), self.y.ravel()] = self.data[frame].ravel()
        img_remap[0,0] = 500
        self.setImage(img_remap, autoRange=False, autoLevels=False)
        self.setLevels(self.level[0], self.level[1])
        n_peaks = 0
        peaks_x = []
        peaks_y = []
        for i in range(len(self.peaks_x[frame])):
            peak_x = int(round(self.peaks_x[frame][i]))
            peak_y = int(round(self.peaks_y[frame][i]))
            if peak_x == 0.0 and peak_y == 0.0:
                break
            else:
                peak_remap_x = int(self.geom_x[peak_y][peak_x] / pixel_size) + self.offset_x
                peak_remap_y = int(self.geom_y[peak_y][peak_x] / pixel_size) + self.offset_y
                peaks_x.append(peak_remap_x)
                peaks_y.append(peak_remap_y)
                n_peaks += 1
        p = pg.ScatterPlotItem()
        self.getView().addItem(p)
        p.setData(peaks_x, peaks_y, symbol='x', size=6, brush=(255,255,255,0), pen='r')
        self.peak_items.append(p)
        print('%d peaks found in %d/%d frame' % (n_peaks, self.frame, self.data.shape[0]))
        # render spind predicted spots
        if self.show_spind_spots:
            event = frame
            for i in range(len(self.spind_predicted)):
                if self.spind_predicted[i]['event'] == event:
                    predicted_spots = self.spind_predicted[i]['predicted_spots']
                    print('%d spots predicted by spind' % predicted_spots.shape[0])
                    spots_x = []
                    spots_y = []
                    for j in range(predicted_spots.shape[0]):
                        ps_x, ps_y = predicted_spots[j,:]
                        ps_x, ps_y = int(round(ps_x)), int(round(ps_y))
                        crystfel_remap_x = int(self.geom_x[ps_y][ps_x] / pixel_size) + self.offset_x
                        crystfel_remap_y = int(self.geom_y[ps_y][ps_x] / pixel_size) + self.offset_y
                        spots_x.append(crystfel_remap_x)
                        spots_y.append(crystfel_remap_y)
                    p = pg.ScatterPlotItem()
                    self.getView().addItem(p)
                    self.crystfel_predicted_spot_items.append(p)
                    p.setData(spots_x, spots_y, symbol='o', size=10, brush=(255,255,255,0), pen='g')

        # render crystfel predicted spots
        if self.show_crystfel_spots:
            event = frame
            for i in range(len(self.crystfel_predicted)):
                if self.crystfel_predicted[i]['event'] == event:
                    predicted_spots = self.crystfel_predicted[i]['predicted_spots']
                    print('%d spots predicted by crystfel' % predicted_spots.shape[0])
                    spots_x = []
                    spots_y = []
                    for j in range(predicted_spots.shape[0]):
                        ps_x, ps_y = predicted_spots[j,:]
                        ps_x, ps_y = int(round(ps_x)), int(round(ps_y))
                        crystfel_remap_x = int(self.geom_x[ps_y][ps_x] / pixel_size) + self.offset_x
                        crystfel_remap_y = int(self.geom_y[ps_y][ps_x] / pixel_size) + self.offset_y
                        spots_x.append(crystfel_remap_x)
                        spots_y.append(crystfel_remap_y)
                    p = pg.ScatterPlotItem()
                    self.getView().addItem(p)
                    self.crystfel_predicted_spot_items.append(p)
                    p.setData(spots_x, spots_y, symbol='o', size=8, brush=(255,255,255,0), pen='y')

    def keyPressEvent(self, event):
        pg.ImageView.keyPressEvent(self, event)
        key = event.key()
        if key == QtCore.Qt.Key_Left:  # last frame
            self.frame -= 1
            self.update(self.frame)
        elif key == QtCore.Qt.Key_Right:  # next frame
            self.frame += 1
            self.update(self.frame)
        elif key == QtCore.Qt.Key_N:  # next indexed frame
            next_indexed_event = self.indexed_events[np.where((self.indexed_events - self.frame) > 0)[0][0]]
            self.frame = next_indexed_event
            self.update(self.frame)
        elif key == QtCore.Qt.Key_L:  # last indexed frame
            last_indexed_event = self.indexed_events[np.where((self.indexed_events - self.frame) < 0)[0][-1]]
            self.frame = last_indexed_event
            self.update(self.frame)
        elif key == QtCore.Qt.Key_Space:  # display information of mouse location
            pos = self.getView().mapToView(QtGui.QCursor().pos())
            x, y = pos.x(), pos.y()
            intensity = self.image[y, x]
            print('(%d, %d) -> (%d, %d), %.2f' % (x, y, x-self.offset_x, y-self.offset_y, intensity))
        elif key == QtCore.Qt.Key_S:  # show SPIND predicted spots or not
            if self.show_spind_spots == True:
                self.show_spind_spots = False
            elif self.show_spind_spots == False:
                self.show_spind_spots = True
            self.update(self.frame)
        elif key == QtCore.Qt.Key_C:  # show crystfel predicted spots or not
            if self.show_crystfel_spots == True:
                print 'disable crystfel spot display'
                self.show_crystfel_spots = False
            elif self.show_crystfel_spots == False:
                print 'enable crystfel spot display'
                self.show_crystfel_spots = True
            self.update(self.frame)


if __name__ == '__main__':
    # parse command line options
    argv = docopt(__doc__)
    cxi_file = argv['-f']
    geom_file = argv['-g']
    start_from = int(argv['--start'])
    spind_stream = argv['--spind-stream']
    crystfel_stream = argv['--crystfel-stream']
    cmin = float(argv['--cmin'])
    cmax = float(argv['--cmax'])
    print('=' * 60)
    print('Loading CXI File: %s' % cxi_file)
    print('Using Geometry File: %s' % geom_file)
    if spind_stream is not None:
        print('Rendering SPIND predicted spots from: %s' % spind_stream)
    if crystfel_stream is not None:
        print('Rendering CRYSTFEL predicted spots from: %s' % crystfel_stream)
    print('=' * 60)

    app = QtGui.QApplication(sys.argv)
    ## Create window with ImageView widget
    win = QtGui.QMainWindow()
    win.resize(800,800)
    cxi_viewer = CXIViewer(cxi_file, geom_file, start_from,
                           spind_stream, crystfel_stream,
                           cmin, cmax)
    win.setCentralWidget(cxi_viewer)
    win.show()
    win.setWindowTitle('CXI Viewer')    
    app.exec_()