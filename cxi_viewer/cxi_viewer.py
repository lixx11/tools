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


# pens for plot items, visit 
# http://www.pyqtgraph.org/documentation/functions.html
# for more details
pens = {'peak': '#ff0000',
        'ref': '#5dff35',
        'test1': '#35fffb',
        'test2': '#3496ff',
        'test3': '#ddff00'}

# sizes for reflection plot
sizes = {'peak': 3,
        'ref': 5,
        'test1': 7,
        'test2': 9,
        'test3': 11}

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
            print('Wrong geometry: %s. You must provide Cheetah, \
                CrystFEL or psana geometry file.')
            return None
        

class CXIWindow(QtGui.QMainWindow):
    """docstring for CXIWindow"""
    def __init__(self, state_file):
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
        self.actionLoadState.triggered.connect(self.loadStateSlot)
        self.actionSaveState.triggered.connect(self.saveStateSlot)
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
        self.streamPlot.getPlotItem().setTitle('Stream Plot')
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
                        ]},
                        {'name': 'Display Option', 'type': 'group', 'children': [
                            {'name': 'Show Reflections from', 'type': 'group', 'children': [
                                {'name': 'Reference Stream', 'type': 'bool', 'value': self.showReflection['ref_stream']},
                                {'name': 'Test Stream 1', 'type': 'bool', 'value': self.showReflection['test_stream1']},
                                {'name': 'Test Stream 2', 'type': 'bool', 'value': self.showReflection['test_stream2']},
                                {'name': 'Test Stream 3', 'type': 'bool', 'value': self.showReflection['test_stream3']},
                            ]},
                            {'name': 'AutoRange', 'type': 'bool', 'value': self.dispOption['autoRange']},
                            {'name': 'AutoLevel', 'type': 'bool', 'value': self.dispOption['autoLevels']},
                            {'name': 'AutoHistogram', 'type': 'bool', 'value': self.dispOption['autoHistogramRange']},
                        ]},
                      ]
        self.params = Parameter.create(name='params', type='group', children=params_list)
        self.parameterTree.setParameters(self.params, showTop=False)
        # parameter connection
        self.params.param('Basic Operation', 
            'Frame').sigValueChanged.connect(self.frameChangedSlot)
        self.params.param('Display Option', 
            'Show Reflections from', 
            'Reference Stream').sigValueChanged.connect(self.showRefSlot)
        self.params.param('Display Option', 
            'Show Reflections from', 
            'Test Stream 1').sigValueChanged.connect(self.showTest1Slot)
        self.params.param('Display Option', 
            'Show Reflections from', 
            'Test Stream 2').sigValueChanged.connect(self.showTest2Slot)
        self.params.param('Display Option', 
            'Show Reflections from', 
            'Test Stream 3').sigValueChanged.connect(self.showTest3Slot)
        # mouse move connection
        self.imageView.scene.sigMouseMoved.connect(self.mouseMoved)
        # load state if given
        self.maybeLoadState(state_file)

    def mouseMoved(self, pos):
        if self.cxi_file is None:
            return
        mouse_pos = self.imageView.view.mapToView(pos)
        image = self.imageView.image
        x, y = int(mouse_pos.x()), int(mouse_pos.y())
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            self.statusbar.showMessage("x:%d y:%d I:%.2E" % 
                (x, y, image[x, y]), 5000)

    def showRefSlot(self, _, show):
        self.showReflection['ref_stream'] = show
        self.updateDisp()

    def showTest1Slot(self, _, show):
        self.showReflection['test_stream1'] = show
        self.updateDisp()

    def showTest2Slot(self, _, show):
        self.showReflection['test_stream2'] = show
        self.updateDisp()

    def showTest3Slot(self, _, show):
        self.showReflection['test_stream3'] = show
        self.updateDisp()

    def scatterItemClicked(self, _, points):
        pos = points[0].pos()
        index = int(points[0].pos()[0])
        print('clicked %d' % index)
        self.params.param('Basic Operation', 
            'Frame').setValue(index)

    def frameChangedSlot(self, _, frame):
        self.frame = frame
        self.updateDisp()

    def loadStateSlot(self):
        """Load configuration from state file"""
        fpath = str(QtGui.QFileDialog.getOpenFileName(self, 
            'Open File', '', 'State File (*.yml)'))
        self.maybeLoadState(fpath)

    def maybeLoadState(self, fpath):
        import yaml
        yml_file = file(fpath, 'r')
        state = yaml.safe_load(yml_file)
        yml_file.close()
        self.maybeLoadCXI(state['cxi_file'])
        self.maybeLoadGeom(state['geom_file'])
        self.maybeLoadRefStream(state['ref_stream_file']) 
        self.maybeLoadTestStream(state['test_stream_files'])

    def saveStateSlot(self):
        """Save configuration to state file"""
        import yaml
        state = dict()
        state['cxi_file'] = self.cxi_file
        state['geom_file'] = self.geom_file
        state['ref_stream_file'] = self.ref_stream_file
        state['test_stream_files'] = self.test_stream_files
        fpath = QtGui.QFileDialog.getSaveFileName(self, 
            'Save File', '', 'State File (*.yml)')
        yml_file = file(fpath, 'w')
        yaml.dump(state, yml_file)
        yml_file.close()

    def loadTestStreamSlot(self):
        """Load test stream files (<=3)"""
        fpaths = QtGui.QFileDialog.getOpenFileNames(self, 
            'Open File', '', 'Stream File (*.stream)')
        self.maybeLoadTestStream(fpaths)

    def maybeLoadTestStream(self, fpaths):
        if len(fpaths) == 0:
            return
        self.test_stream_files = []
        self.test_stream_data = []
        for i in range(len(fpaths)):
            fpath = str(fpaths[i])
            self.test_stream_files.append(str(fpath))
            self.test_stream_data.append(load_data_from_stream(str(fpath)))
            self.params.param('File Info', 
                'Test Stream %d' % (i+1)).setValue(basename(str(fpath)))
        self.updateStreamPlot()

    def loadRefStreamSlot(self):
        fpath = str(QtGui.QFileDialog.getOpenFileName(self, 
            'Open File', '', 'Stream File (*.stream)'))
        self.maybeLoadRefStream(fpath)

    def maybeLoadRefStream(self, fpath):
        if fpath is '':
            return
        self.ref_stream_file = str(fpath)
        self.ref_stream_data = load_data_from_stream(self.ref_stream_file)
        self.params.param('File Info', 
            'Reference Stream').setValue(basename(self.ref_stream_file))
        self.updateStreamPlot()

    def loadCXISlot(self):
        fpath = str(QtGui.QFileDialog.getOpenFileName(self, 
            'Open File', '', 'CXI File (*.cxi)'))
        self.maybeLoadCXI(fpath)

    def maybeLoadCXI(self, fpath):
        if fpath is '':
            return
        self.cxi_file = fpath
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
        fpath = str(QtGui.QFileDialog.getOpenFileName(self, 
            'Open file', '', 'Geom File (*.geom *.h5 *.data)'))
        self.maybeLoadGeom(fpath)

    def maybeLoadGeom(self, fpath):
        if fpath is '':
            return
        self.geom_file = fpath
        self.geom = Geometry(self.geom_file, self.pixel_size)
        self.params.param('File Info', 
            'Geometry File').setValue(basename(self.geom_file))
        self.updateDisp()

    def render_reflections(self, stream_data, frame, 
        anchor=(0, 0), pen='r', size=5, tag=''):
        """Render reflections from stream file"""
        if not stream_data.has_key(frame):
            return None  # this event not exist
        event = stream_data[frame]
        rawXYs = event['rawXYs']
        Is = event['Is']
        print('%d reflections including %d negative tagged by %s' 
               % (Is.size, (Is < 0.).sum(), tag))
        spots_x, spots_y = [], []
        for j in range(Is.size):
            spot_x, spot_y = self.geom.map((rawXYs[j,:]))
            spots_x.append(spot_x)
            spots_y.append(spot_y)
        p = pg.ScatterPlotItem()
        self.imageView.getView().addItem(p)
        self.reflection_items.append(p)
        p.setData(spots_x, spots_y, symbol='o', 
            size=size, brush=(255,255,255,0), pen=pen)

    def updateStreamPlot(self):
        y = 0
        self.scatterItem.clear()
        ticks = []
        ax = self.streamPlot.getAxis('left')
        # scatter for all patterns
        if self.cxi_file is not None:
            self.scatterItem.addPoints(
                np.arange(self.nb_pattern), 
                np.ones(self.nb_pattern)*y,
                pen=pens['peak'])
            ticks.append((0, 'all'))
            y += 1
        # scatter for reference stream
        if self.ref_stream_file is not None:
            event_ids = self.ref_stream_data.keys()
            p = pg.ScatterPlotItem()
            self.streamPlot.addItem(p)
            self.scatterItem.addPoints(
                event_ids, np.ones_like(event_ids)*y,
                pen=pens['ref'])
            ticks.append((1, 'ref'))
            y += 1
        # scatter for test streams
        if len(self.test_stream_files) > 0:
            for i in range(len(self.test_stream_data)):
                event_ids = self.test_stream_data[i].keys()
                p = pg.ScatterPlotItem()
                self.streamPlot.addItem(p)
                self.scatterItem.addPoints(
                    event_ids, np.ones_like(event_ids)*y,
                    pen=pens['test%d' % (i+1)])
                ticks.append((i+2, 'test%d' % (i+1)))
                y += 1
        ax.setTicks([ticks])

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
            image[0,0] = 500.  # set a reference point
        self.imageView.setImage(image, 
            autoRange=self.dispOption['autoRange'],
            autoLevels=self.dispOption['autoLevels'],
            autoHistogramRange=self.dispOption['autoHistogramRange'])
        # render peaks extracted from cxi file
        n_peaks = 0
        peaks_x, peaks_y = [], []
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
        p.setData(peaks_x, peaks_y, symbol='x', 
            size=6, brush=(255,255,255,0), pen=pens['peak'])
        self.peak_items.append(p)
        print('%d peaks found in %d/%d frame' % 
            (n_peaks, self.frame, self.data.shape[0]))
        # render reference stream reflections
        if self.showReflection['ref_stream']:
            if self.ref_stream_file is not None:
                self.render_reflections(self.ref_stream_data, 
                    self.frame, tag='ref', size=sizes['ref'], 
                    anchor=(1.1,1.5), pen=pens['ref'])
        # render test stream reflectins
        if self.showReflection['test_stream1']:
            if len(self.test_stream_files) >= 1:
                self.render_reflections(self.test_stream_data[0], 
                    self.frame, tag='test1', size=sizes['test1'],
                    anchor=(1.1,1.5), pen=pens['test1'])
        if self.showReflection['test_stream2']:
            if len(self.test_stream_files) >= 2:
                self.render_reflections(self.test_stream_data[1], 
                    self.frame, tag='test1', size=sizes['test2'],
                    anchor=(1.1,1.5), pen=pens['test2'])
        if self.showReflection['test_stream3']:
            if len(self.test_stream_files) >= 3:
                self.render_reflections(self.test_stream_data[2], 
                    self.frame, tag='test1', size=sizes['test3'],
                    anchor=(1.1,1.5), pen=pens['test3'])


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    if len(sys.argv) == 2:
        state_file = sys.argv[1]
    else:
        state_file = ''
    win = CXIWindow(state_file)
    win.setWindowTitle("CXI Viewer")
    win.show()
    app.exec_()