#! /opt/local/bin/python2.7

"""
Usage:
    cxi_viewer.py -f cxi_file -g geom.h5

Options:
    -h --help       Show this screen.
    -f data.cxi     Specify cxi file for visulization.
    -g geom.h5      Geometry file in hdf5 format.
"""

import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from docopt import docopt
import numpy as np
import h5py


pixel_size = 110E-6  # CSPAD pixel size


class CXIViewer(pg.ImageView):
    """docstring for CXIViewer"""
    def __init__(self, cxi_file, geom_file):
        super(CXIViewer, self).__init__()
        self.cxi_file = cxi_file
        self.geom_file = geom_file
        self.level = [0, 500]

        # load data from cxi file
        cxi = h5py.File(self.cxi_file, 'r')
        self.data = cxi['/entry_1/instrument_1/detector_1/detector_corrected/data']
        self.peaks_x = cxi['/entry_1/result_1/peakXPosRaw']
        self.peaks_y = cxi['/entry_1/result_1/peakYPosRaw']

        # load geometry
        geom = h5py.File(geom_file, 'r')
        self.geom_x, self.geom_y = geom['x'].value, geom['y'].value
        x = np.int_(np.rint(self.geom_x / pixel_size))
        y = np.int_(np.rint(self.geom_y / pixel_size))
        self.offset_x = abs(x.min())
        self.offset_y = abs(y.min())
        x += self.offset_x
        y += self.offset_y
        self.nx, self.ny = x.max()+1, y.max()+1
        self.x, self.y = x, y

        self.frame = 0
        self.peak_items = []
        self.update(self.frame)

    def update(self, frame):
        # clear last frame
        for p in self.peak_items:
            p.clear()
        img_remap = np.zeros((self.nx, self.ny))
        img_remap[self.x.ravel(), self.y.ravel()] = self.data[frame].ravel()
        self.setImage(img_remap, autoRange=False, autoLevels=False)
        self.setLevels(self.level[0], self.level[1])
        n_peaks = 0
        for i in range(len(self.peaks_x[frame])):
            peak_x = self.peaks_x[frame][i]
            peak_y = self.peaks_y[frame][i]
            if peak_x == 0.0 and peak_y == 0.0:
                break
            else:
                peak_remap_x = int(self.geom_x[peak_y][peak_x] / pixel_size) + self.offset_x
                peak_remap_y = int(self.geom_y[peak_y][peak_x] / pixel_size) + self.offset_y
                p = pg.ScatterPlotItem()
                self.getView().addItem(p)
                p.setData([peak_remap_x], [peak_remap_y], symbol='o', size=10, brush=(255,255,255,0), pen='r')
                self.peak_items.append(p)
                n_peaks += 1
        print('Found %d peaks in %d/%d frame' % (n_peaks, self.frame, self.data.shape[0]))

    def keyPressEvent(self, event):
        pg.ImageView.keyPressEvent(self, event)
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self.frame -= 1
            self.update(self.frame)
        elif key == QtCore.Qt.Key_Right:
            self.frame += 1
            self.update(self.frame)


if __name__ == '__main__':
    # parse command line options
    argv = docopt(__doc__)
    cxi_file = argv['-f']
    geom_file = argv['-g']
    print('Loading CXI File: %s' % cxi_file)
    print('Using Geometry File: %s' % geom_file)

    app = QtGui.QApplication(sys.argv)
    ## Create window with ImageView widget
    win = QtGui.QMainWindow()
    win.resize(800,800)
    cxi_viewer = CXIViewer(cxi_file, geom_file)
    win.setCentralWidget(cxi_viewer)
    win.show()
    win.setWindowTitle('CXI Viewer')    
    app.exec_()