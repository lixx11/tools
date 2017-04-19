#!/usr/bin/env python

import sys
import os
from os.path import basename
from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4 import uic
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
import h5py
import numpy as np
from stream_parser import parse_stream


def load_data_from_stream(filename):
    """load reflection information from stream file"""
    chunks = []
    data = {}
    index_stats = parse_stream(filename)
    for index_stat in index_stats:
        if index_stat.index_method != 'none':
            chunks += index_stat.chunks
    for c in chunks:
        rawXYs = []
        HKLs = []
        Is = []
        for r in c.reflections:
            rawXYs.append([r.fs, r.ss])
            HKLs.append([r.h, r.k, r.l])
            Is.append(r.I)
        rawXYs = np.asarray(rawXYs)
        HKLs = np.asarray(HKLs)
        Is = np.asarray(Is)
        data[c.event] = {'rawXYs': rawXYs, 'HKLs': HKLs, 'Is': Is}
    return data


class Geometry(object):
    """docstring for Geometry"""
    def __init__(self, geom_file, pixel_size):
        self.geom_file = geom_file
        self.pixel_size = pixel_size
        self.geom_x, self.geom_y, self.geom_z = self.load_geom(geom_file)
        x = np.int_(np.rint(self.geom_x / self.pixel_size))
        y = np.int_(np.rint(self.geom_y / self.pixel_size))
        self.offset_x = abs(x.min())
        self.offset_y = abs(y.min())
        x += self.offset_x
        y += self.offset_y
        self.nx, self.ny = x.max()+1, y.max()+1
        self.x, self.y = x, y

    def rearrange(self, image):
        """Rearrange raw image to assembled pattern according to the geometry setup."""
        new_img = np.zeros((self.nx, self.ny))
        new_img[self.x.ravel(), self.y.ravel()] = image.ravel()
        return new_img

    def map(self, pos):
        """Map raw position to assembled position"""
        pos = np.int_(np.rint(pos))
        peak_remap_x_in_m = self.geom_x[pos[1], pos[0]]
        peak_remap_y_in_m = self.geom_y[pos[1], pos[0]]
        peak_remap_x_in_pixel = peak_remap_x_in_m / self.pixel_size + self.offset_x
        peak_remap_y_in_pixel = peak_remap_y_in_m / self.pixel_size + self.offset_y
        return peak_remap_x_in_pixel, peak_remap_y_in_pixel

    def load_geom(self, filename):
        """load geometry: x, y, z coordinates from cheetah, crystfel or psana geom file"""
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
            from psgeom import camera
            cspad = camera.Cspad.from_psana_file(filename)
            cspad.to_cheetah_file('.geom.h5')
            f = h5py.File('.geom.h5', 'r')
            return f['x'].value, f['y'].value, f['z'].value
        else:
            print('Wrong geometry: %s. You must provide Cheetah, CrystFEL or psana geometry file.')
            return None
        

class CXIWindow(QtGui.QMainWindow):
    """docstring for CXIWindow"""
    def __init__(self):
        super(CXIWindow, self).__init__()
        # load and adjust layout
        dir_ = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(dir_ + '/' + 'layout.ui', self)
        self.splitterH.setSizes([self.width()*0.7, self.width()*0.3])
        self.splitterV.setSizes([self.height()*0.7, self.height()*0.3])
        # setup menu slots
        self.actionLoadCXI.triggered.connect(self.loadCXISlot)
        self.actionLoadGeom.triggered.connect(self.loadGeomSlot)
        self.actionRefStream.triggered.connect(self.loadRefStreamSlot)
        self.actionTestStream.triggered.connect(self.loadTestStreamSlot)
        # initialize parameters
        self.frame = 0
        self.pixel_size = 110E-6  # 110um pixel of CSPAD
        self.cxi_file = None
        self.geom_file = None
        self.ref_stream_file = None
        self.test_stream_files = []
        self.peak_items = []
        self.reflection_items = []

        # stream plot
        self.scatterItem = pg.ScatterPlotItem()
        self.streamPlot.addItem(self.scatterItem)
        self.scatterItem.sigClicked.connect(self.scatterItemClicked)

        self.showReflection = {}
        self.showReflection['ref_stream'] = True
        self.showReflection['test_stream1'] = True
        self.showReflection['test_stream2'] = True
        self.showReflection['test_stream3'] = True
        self.dispOption = {}
        self.dispOption['autoRange'] = False
        self.dispOption['autoLevels'] = False
        self.dispOption['autoHistogramRange'] = False
                
        # setup parameter tree
        params_list = [
                        {'name': 'File Info', 'type': 'group', 'children': [
                            {'name': 'CXI File', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Pattern Number', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Geometry File', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Reference Stream', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Test Stream 1', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Test Stream 2', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Test Stream 3', 'type': 'str', 'value': 'not set', 'readonly': True},
                        ]},
                        {'name': 'Basic Operation', 'type': 'group', 'children': [
                            {'name': 'Frame', 'type': 'int', 'value': self.frame},
                            {'name': 'Show Reflections from', 'type': 'group', 'children': [
                                {'name': 'Reference Stream', 'type': 'bool', 'value': self.showReflection['ref_stream']},
                                {'name': 'Test Stream 1', 'type': 'bool', 'value': self.showReflection['test_stream1']},
                                {'name': 'Test Stream 2', 'type': 'bool', 'value': self.showReflection['test_stream2']},
                                {'name': 'Test Stream 3', 'type': 'bool', 'value': self.showReflection['test_stream3']},
                            ]}
                        ]},
                        {'name': 'Display Option', 'type': 'group', 'children': [
                            {'name': 'AutoRange', 'type': 'bool', 'value': self.dispOption['autoRange']},
                            {'name': 'AutoLevel', 'type': 'bool', 'value': self.dispOption['autoLevels']},
                            {'name': 'AutoHistogram', 'type': 'bool', 'value': self.dispOption['autoHistogramRange']},
                        ]},
                      ]
        self.params = Parameter.create(name='params', type='group', children=params_list)
        self.parameterTree.setParameters(self.params, showTop=False)
        # parameter connection
        self.params.param('Basic Operation', 'Frame').sigValueChanged.connect(self.frameChangedSlot)

    def scatterItemClicked(self, _, points):
        pos = points[0].pos()
        index = int(points[0].pos()[0])
        print('clicked %d' % index)
        self.frame = index
        self.updateDisp()

    def frameChangedSlot(self, _, frame):
        self.frame = frame
        self.updateDisp()

    def loadTestStreamSlot(self):
        """Load test stream files (<=3)"""
        fpaths = QtGui.QFileDialog.getOpenFileNames(self, 'Open file', '', 'Stream File (*.stream)')
        self.test_stream_files = []
        self.test_stream_data = []
        for fpath in fpaths:
            self.test_stream_files.append(str(fpath))
            self.test_stream_data.append(load_data_from_stream(str(fpath)))
        self.updateStreamPlot()

    def loadRefStreamSlot(self):
        fpath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '', 'Stream File (*.stream)')
        self.ref_stream_file = str(fpath)
        self.ref_stream_data = load_data_from_stream(self.ref_stream_file)
        self.params.param('File Info', 'Reference Stream').setValue(basename(self.ref_stream_file))
        self.updateStreamPlot()

    def loadCXISlot(self):
        fpath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '', 'CXI File (*.cxi)')
        self.cxi_file = str(fpath)
        cxi = h5py.File(self.cxi_file, 'r')
        self.data = cxi['/entry_1/instrument_1/detector_1/detector_corrected/data']
        self.nb_pattern = self.data.shape[0]
        self.peaks_x = cxi['/entry_1/result_1/peakXPosRaw']
        self.peaks_y = cxi['/entry_1/result_1/peakYPosRaw']
        self.params.param('File Info', 'CXI File').setValue(basename(self.cxi_file))
        self.params.param('File Info', 'Pattern Number').setValue(self.nb_pattern)
        self.updateDisp()
        self.updateStreamPlot()

    def loadGeomSlot(self):
        fpath = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', '', 'Geom File (*.geom *.h5 *.data)'))
        if fpath is '':
            return
        self.geom_file = fpath
        self.geom = Geometry(self.geom_file, self.pixel_size)
        self.params.param('File Info', 'Geometry File').setValue(basename(self.geom_file))
        self.updateDisp()

    def testSlot(self):
        print('hello world')

    def render_reflections(self, stream_data, frame, anchor=(0, 0), pen='g', tag=''):
        if not stream_data.has_key(frame):
            return None  # this event not exist
        event = stream_data[frame]
        rawXYs = event['rawXYs']
        Is = event['Is']
        print('%d reflections including %d negative tagged by %s' 
               % (Is.size, (Is < 0.).sum(), tag))
        spots_x, spots_y = list(), list()
        for j in range(Is.size):
            spot_x, spot_y = self.geom.map((rawXYs[j,:]))
            spots_x.append(spot_x)
            spots_y.append(spot_y)
        p = pg.ScatterPlotItem()
        self.imageView.getView().addItem(p)
        self.reflection_items.append(p)
        p.setData(spots_x, spots_y, symbol='o', size=10, brush=(255,255,255,0), pen=pen)

    def updateStreamPlot(self):
        y = 0
        self.scatterItem.clear()
        if self.cxi_file is not None:
            self.scatterItem.addPoints(np.arange(self.nb_pattern), np.ones(self.nb_pattern)*y)
            y += 1
        if self.ref_stream_file is not None:
            event_ids = self.ref_stream_data.keys()
            p = pg.ScatterPlotItem()
            self.streamPlot.addItem(p)
            self.scatterItem.addPoints(event_ids, np.ones_like(event_ids)*y)
            y += 1
        # todo: render test streams

    def updateDisp(self):
        # clean peaks and reflections in last frame
        for item in self.peak_items:
            self.imageView.getView().removeItem(item)
        for item in self.reflection_items:
            self.imageView.getView().removeItem(item)
        if self.cxi_file is None:
            return
        image = self.data[self.frame]
        if self.geom_file is not None:
            image = self.geom.rearrange(image)
            image[0,0] = 500.
        self.imageView.setImage(image, autoRange=self.dispOption['autoRange'],
                             autoLevels=self.dispOption['autoLevels'],
                             autoHistogramRange=self.dispOption['autoHistogramRange'])
        # render peaks extracted from cxi file
        n_peaks = 0
        peaks_x, peaks_y = list(), list()
        for i in range(len(self.peaks_x[self.frame])):
            peak_x = int(round(self.peaks_x[self.frame][i]))
            peak_y = int(round(self.peaks_y[self.frame][i]))
            if peak_x == 0.0 and peak_y == 0.0:
                break
            else:
                if self.geom_file is not None:
                    peak_x, peak_y = self.geom.map((peak_x, peak_y))
                else:
                    peak_x, peak_y = peak_y, peak_x
                peaks_x.append(peak_x)
                peaks_y.append(peak_y)
                n_peaks += 1
        p = pg.ScatterPlotItem()
        self.imageView.getView().addItem(p)
        p.setData(peaks_x, peaks_y, symbol='x', size=6, brush=(255,255,255,0), pen='r')
        self.peak_items.append(p)
        print('%d peaks found in %d/%d frame' % (n_peaks, self.frame, self.data.shape[0]))
        # render reference stream reflections
        if self.showReflection['ref_stream']:
            if self.ref_stream_file is not None:
                self.render_reflections(self.ref_stream_data, self.frame, tag='ref', anchor=(1.1,1.5), pen='g')
            else:
                pass
        # todo: render test stream reflections


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = CXIWindow()
    win.setWindowTitle("CXI Viewer")
    win.show()
    app.exec_()