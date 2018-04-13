#!/usr/bin/env python3

import sys
import os

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
from pyqtgraph.parametertree import Parameter

import h5py
import numpy as np
from numpy.linalg import norm
import yaml
from functools import partial
from settings import Settings
from stream_parser import Stream
from geometry import Geometry, det2fourier, get_hkl


def load_data_from_stream(filename):
    """load information from stream file"""
    chunks = []
    data = {}
    stream = Stream(filename)
    index_stats = stream.index_stats
    # collect indexed chunks
    for index_stat in index_stats:
        if index_stat.index_method != 'none':
            chunks += index_stat.chunks
    for c in chunks:
        # a, b, c star
        astar = c.crystal.astar
        bstar = c.crystal.bstar
        cstar = c.crystal.cstar
        # peaks
        peakXYs = []
        for p in c.peaks:
            peakXYs.append([p.fs, p.ss])
        peakXYs = np.asarray(peakXYs)
        # collect reflections
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
        data[c.event] = {
            'rawXYs': rawXYs,
            'HKLs': HKLs,
            'Is': Is,
            'peakXYs': peakXYs,
            'astar': astar,
            'bstar': bstar,
            'cstar': cstar
        }
    return data


def makeTabelItem(text):
    """Make table item with text centered"""
    item = QtGui.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignCenter)
    # item.setFlags(QtCore.Qt.ItemIsEditable)
    return item


class StreamTable(QtGui.QDialog):
    def __init__(self, parent=None):
        super(StreamTable, self).__init__(parent)
        dir_ = os.path.dirname(os.path.abspath(__file__))
        loadUi(dir_ + '/' + 'table.ui', self)
        # add slot for buttons
        self.plotHistButton.clicked.connect(
            self.onClickedPlotHistSlot)
        self.exportDataButton.clicked.connect(
            self.onClickedExportDataSlot)
        self.centeringAnaButton.clicked.connect(
            self.onClickCenteringAnaSlot)
        # add slot for table
        self.table.cellClicked.connect(
            self.onCellClickedSlot)
        self.streams = {}

    def onClickCenteringAnaSlot(self):
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        nb_row = self.table.rowCount()
        nb_paired_peaks = 0
        nb_I_peaks = 0
        nb_C_peaks = 0
        nb_A_peaks = 0
        nb_B_peaks = 0
        for i in range(nb_row):
            dataset = str(self.table.item(i, 1).text())
            if dataset == 'ref':  # only collect from reference
                nb_paired_peaks += int(self.table.item(i, 10).text())
                nb_I_peaks += int(self.table.item(i, 11).text())
                nb_C_peaks += int(self.table.item(i, 12).text())
                nb_A_peaks += int(self.table.item(i, 13).text())
                nb_B_peaks += int(self.table.item(i, 14).text())

        I_peak_ratio = float(nb_I_peaks) / float(nb_paired_peaks)
        C_peak_ratio = float(nb_C_peaks) / float(nb_paired_peaks)
        A_peak_ratio = float(nb_A_peaks) / float(nb_paired_peaks)
        B_peak_ratio = float(nb_B_peaks) / float(nb_paired_peaks)

        # make plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs = np.arange(4)
        ys = [I_peak_ratio, C_peak_ratio, A_peak_ratio, B_peak_ratio]
        ticks = ['I type', 'C type', 'A type', 'B type']
        plt.bar(xs, ys)
        plt.xticks(xs + 0.5, ticks)
        plt.ylim((0., 1))
        for i in range(len(ax.patches)):
            patch = ax.patches[i]
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2,
                    height + 0.04, '%.2f' % (ys[i]),
                    ha='center', va='bottom')
        plt.show(block=False)

    def onCellClickedSlot(self, row, column):
        item = self.table.item(row, 0)
        frame = int(str(item.text()))
        self.parent().params.param('Basic Operation',
                                   'Frame').setValue(frame)

    def onClickedPlotHistSlot(self):
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        error = self.collect_error()
        for dataset in error.keys():
            if len(error[dataset] > 0):
                print('plot for %s' % dataset)
                data = error[dataset]
                valid_idx = np.max(data[:, 1:], axis=1) < 0.5

                plt.figure(dataset)
                plt.subplot(131)
                astar_err = data[valid_idx, 1]
                plt.hist(astar_err)
                plt.title('Number: %d  Mean: %.2E' %
                          (astar_err.size, astar_err.mean()))
                plt.xlabel('a star error')

                plt.subplot(132)
                bstar_err = data[valid_idx, 2]
                plt.hist(bstar_err)
                plt.title('Number: %d  Mean: %.2E' %
                          (bstar_err.size, bstar_err.mean()))
                plt.xlabel('b star error')

                plt.subplot(133)
                cstar_err = data[valid_idx, 3]
                plt.hist(cstar_err)
                plt.title('Number: %d  Mean: %.2E' %
                          (cstar_err.size, cstar_err.mean()))
                plt.xlabel('c star error')
                plt.tight_layout()
                plt.show(block=False)

    def onClickedExportDataSlot(self):
        error = self.collect_error()
        for dataset in error:
            if len(error[dataset]) > 0:
                fname = '%s-error.txt' % dataset
                np.savetxt(fname,
                           error[dataset],
                           fmt='%5d %.5f %.5f %.5f')
                print('%s error saved to %s' %
                      (dataset, fname))

    def collect_error(self):
        nb_row = self.table.rowCount()
        error = {}
        for i in [1, 2, 3]:
            error['test%d' % i] = []
        for i in range(nb_row):
            dataset = str(self.table.item(i, 1).text())
            astar_text = str(self.table.item(i, 5).text())
            bstar_text = str(self.table.item(i, 6).text())
            cstar_text = str(self.table.item(i, 7).text())
            if '/' in astar_text:
                astar_err = float(
                    astar_text.split('/')[-1][:-1]) / 100.  # relative error
                bstar_err = float(bstar_text.split('/')[-1][:-1]) / 100.
                cstar_err = float(cstar_text.split('/')[-1][:-1]) / 100.
                error[dataset].append([i, astar_err, bstar_err, cstar_err])
        for i in [1, 2, 3]:
            error['test%d' % i] = np.asarray(
                error['test%d' % i])
        return error

    def merge_streams(self):
        """
        Merge multiple streams and organize as a list
        """
        event_ids = []
        for name, stream in self.streams.items():
            event_ids += stream['data'].keys()
        event_ids = list(set(event_ids))  # remove duplicates
        event_ids.sort()

        self.merged_stream = []
        for i in range(len(event_ids)):
            event_id = event_ids[i]
            event_data = {}
            if self.streams.has_key('ref'):
                if self.streams['ref']['data'].has_key(event_id):
                    event_data['ref'] = self.streams['ref']['data'][event_id]
            for test_id in [1, 2, 3]:
                if self.streams.has_key('test%d' % test_id):
                    if self.streams['test%d' % test_id]['data'].has_key(
                            event_id):
                        event_data['test%d' % test_id] = \
                            self.streams['test%d' % test_id]['data'][event_id]
                        if event_data.has_key('ref'):
                            for xstar in ['astar', 'bstar', 'cstar']:
                                vec = event_data['test%d' % test_id][xstar]
                                vec_ref = event_data['ref'][xstar]
                                abs_err = norm(vec - vec_ref)  # absolute error
                                rel_err = abs_err / norm(
                                    vec_ref)  # relative error
                                event_data['test%d' % test_id][
                                    '%s_err' % xstar] = \
                                    (abs_err, rel_err)
            self.merged_stream.append((event_id, event_data))

    def fillTableRow(self, row, event_id, label, data):
        """fill stream table with given event data and organize
        * event [event id]
        * stream label [ref or testx]
        * a, b, c star
        * a, b, c star error
        * # of reflections
        * # of peaks
        * # of paired peaks [hkl error < 0.25]
        * # of I-type peaks [h+k+l=2n]
        * # of C-type peaks [h+k=2n]
        * # of A-type peaks [k+l=2n]
        * # of B-type peaks [h+l=2n]

        Parameters
        ----------
        row : TYPE
            Description
        event_id : TYPE
            Description
        label : TYPE
            Description
        data : TYPE
            Description
        """
        self.table.insertRow(row)
        astar = data[label]['astar'] / 1E6  # in um
        bstar = data[label]['bstar'] / 1E6
        cstar = data[label]['cstar'] / 1E6
        # event
        self.table.setItem(row, 0,
                           makeTabelItem('%d' % (event_id)))
        # stream label
        self.table.setItem(row, 1,
                           makeTabelItem('%s' % label))
        # astar
        self.table.setItem(row, 2,
                           makeTabelItem('%.2f, %.2f, %.2f' %
                                         (astar[0], astar[1], astar[2])))
        # bstar
        self.table.setItem(row, 3,
                           makeTabelItem('%.2f, %.2f, %.2f' %
                                         (bstar[0], bstar[1], bstar[2])))
        # cstar
        self.table.setItem(row, 4,
                           makeTabelItem('%.2f, %.2f, %.2f' %
                                         (cstar[0], cstar[1], cstar[2])))
        # astar_err
        if data[label].has_key('astar_err'):
            astar_err_abs = data[label]['astar_err'][0] / 1E6
            astar_err_rel = data[label]['astar_err'][1]
            self.table.setItem(row, 5,
                               makeTabelItem('%.2f / %.2f%%' %
                                             (astar_err_abs,
                                              astar_err_rel * 100)))
        else:
            self.table.setItem(row, 5,
                               makeTabelItem('--'))
        # bstar_err
        if data[label].has_key('bstar_err'):
            astar_err_abs = data[label]['bstar_err'][0] / 1E6
            astar_err_rel = data[label]['bstar_err'][1]
            self.table.setItem(row, 6,
                               makeTabelItem('%.2f / %.2f%%' %
                                             (astar_err_abs,
                                              astar_err_rel * 100)))
        else:
            self.table.setItem(row, 6,
                               makeTabelItem('--'))
        # cstar_err
        if data[label].has_key('astar_err'):
            astar_err_abs = data[label]['cstar_err'][0] / 1E6
            astar_err_rel = data[label]['cstar_err'][1]
            self.table.setItem(row, 7,
                               makeTabelItem('%.2f / %.2f%%' %
                                             (astar_err_abs,
                                              astar_err_rel * 100)))
        else:
            self.table.setItem(row, 7,
                               makeTabelItem('--'))
        # num of reflections
        self.table.setItem(row, 8,
                           makeTabelItem('%d' % len(data[label]['rawXYs'])))
        # num of peaks
        peakXYs = data[label]['peakXYs']
        self.table.setItem(row, 9,
                           makeTabelItem('%d' % len(peakXYs)))
        # num of paired peaks
        geom = self.parent().geom
        assXYs = geom.batch_map_from_raw_in_m(peakXYs)
        det_dist = self.parent().det_dist  # detector distance in meters
        wavelength = self.parent().wavelength[
            event_id]  # XFEL wavelength in meters
        qs = det2fourier(assXYs, wavelength, det_dist)
        A = np.ones((3, 3))
        A[:, 0] = astar * 1E-9
        A[:, 1] = bstar * 1E-9
        A[:, 2] = cstar * 1E-9
        HKL = get_hkl(qs, A=A)  # decimal hkls
        rHKL = np.int_(np.round(HKL))  # integer hkls
        print(rHKL)
        eHKL = np.abs(HKL - rHKL)  # hkl error
        pair_idx = np.max(eHKL, axis=1) < 0.25
        nb_paired_peaks = pair_idx.sum()
        self.table.setItem(row, 10,
                           makeTabelItem('%d' % nb_paired_peaks))
        # num of X-type peaks
        pair_h = rHKL[pair_idx, 0]
        pair_k = rHKL[pair_idx, 1]
        pair_l = rHKL[pair_idx, 2]
        # I-type peaks
        nb_I_peaks = np.sum((pair_h + pair_k + pair_l) % 2 == 0)
        self.table.setItem(row, 11,
                           makeTabelItem('%d' % nb_I_peaks))
        # C-type peaks
        nb_C_peaks = np.sum((pair_h + pair_k) % 2 == 0)
        self.table.setItem(row, 12,
                           makeTabelItem('%d' % nb_C_peaks))
        # A-type peaks
        nb_A_peaks = np.sum((pair_k + pair_l) % 2 == 0)
        self.table.setItem(row, 13,
                           makeTabelItem('%d' % nb_A_peaks))
        # B-type peaks
        nb_B_peaks = np.sum((pair_h + pair_l) % 2 == 0)
        self.table.setItem(row, 14,
                           makeTabelItem('%d' % nb_B_peaks))

    def updateTable(self, current_event_id):
        self.table.setRowCount(0)  # clear all rows
        self.merge_streams()
        row_counter = 0
        print('scroll to %d' % current_event_id)
        for i in range(len(self.merged_stream)):
            event_id, event_data = self.merged_stream[i]
            if event_data.has_key('ref'):
                self.fillTableRow(row_counter,
                                  event_id, 'ref', event_data)
                row_counter += 1
            if event_data.has_key('test1'):
                self.fillTableRow(row_counter,
                                  event_id, 'test1', event_data)
                row_counter += 1
            if event_data.has_key('test2'):
                self.fillTableRow(row_counter,
                                  event_id, 'test2', event_data)
                row_counter += 1
            if event_data.has_key('test3'):
                self.fillTableRow(row_counter,
                                  event_id, 'test3', event_data)
                row_counter += 1
            if len(event_data) > 1:
                self.table.setSpan(row_counter - len(event_data), 0,
                                   len(event_data), 1)
        for i in range(row_counter):
            item = self.table.item(i, 0)
            if int(item.text()) == current_event_id:
                self.table.scrollToItem(item)
        nb_column = self.table.columnCount()
        table_width = 0
        for i in range(nb_column):
            self.table.resizeColumnToContents(i)
            table_width += self.table.columnWidth(i)
        self.table.setFixedWidth(table_width)


class GUI(QtGui.QMainWindow):
    def __init__(self, settings):
        super(GUI, self).__init__()
        # load settings
        self.workdir = settings.workdir
        self.data_location = settings.data_location
        self.peak_info_location = settings.peak_info_location
        self.pixel_size = settings.pixel_size
        self.det_dist = settings.detector_distance
        self.cxi_file = settings.cxi_file
        self.ref_stream_file = settings.ref_stream_file
        self.test_stream_file = settings.test_stream_file

        # setup ui
        dir_ = os.path.dirname(os.path.abspath(__file__))
        loadUi('%s/layout.ui' % dir_, self)
        self.splitterH.setSizes([self.width() * 0.7, self.width() * 0.3])
        self.splitterV.setSizes([self.height() * 0.7, self.height() * 0.3])
        # add stream table window
        # self.streamTable = StreamTable(parent=self)

        self.nb_frame = 0
        self.frame = 0
        self.data = None
        self.geom_file = None
        self.geom = None
        self.ref_stream = None
        self.ref_events = []
        self.ref_reflections = {}
        self.test_stream = None
        self.test_events = []
        self.test_reflections = {}
        self.peaks = None
        self.nb_peaks = None
        self.show_hkl = False
        self.show_ref_stream = True
        self.show_test_stream = True

        # plot items
        self.peak_item = pg.ScatterPlotItem(
            symbol='x', size=10, pen='r', brush=(255, 255, 255, 0)
        )
        self.ref_reflection_item = pg.ScatterPlotItem(
            symbol='o', size=12, pen='g', brush=(255, 255, 255, 0)
        )
        self.test_reflection_item = pg.ScatterPlotItem(
            symbol='o', size=14, pen='y', brush=(255, 255, 255, 0)
        )
        self.peak_stream_item = pg.ScatterPlotItem(
            symbol='d', size=5, pen='r', brush=(255, 255, 255, 0)
        )
        self.ref_stream_item = pg.ScatterPlotItem(
            symbol='t', size=5, pen='g', brush=(255, 255, 255, 0)
        )
        self.test_stream_item = pg.ScatterPlotItem(
            symbol='t1', size=5, pen='y', brush=(255, 255, 255, 0)
        )

        self.image_view.getView().addItem(self.peak_item)
        self.image_view.getView().addItem(self.ref_reflection_item)
        self.image_view.getView().addItem(self.test_reflection_item)
        self.stream_plot.addItem(self.ref_stream_item)
        self.stream_plot.addItem(self.test_stream_item)
        self.stream_plot.addItem(self.peak_stream_item)

        # setup parameter tree
        params = [
            {
                'name': 'cxi file', 'type': 'str', 'value': self.cxi_file,
                'readonly': True
            },
            {
                'name': 'total frame', 'type': 'str',
                'value': None, 'readonly': True
            },
            {
                'name': 'geom file', 'type': 'str',
                'value': 'not set',
                'readonly': True
            },
            {
                'name': 'ref stream', 'type': 'str',
                'value': 'not set', 'readonly': True
            },
            {
                'name': 'test stream', 'type': 'str',
                'value': 'not set',
                'readonly': True
            },
            {
                'name': 'det dist', 'type': 'float', 'siPrefix': True,
                'suffix': 'mm', 'value': self.det_dist,
            },
            {
                'name': 'pixel size', 'type': 'float',
                'siPrefix': True, 'suffix': 'μm', 'value': self.pixel_size
            },
            {
                'name': 'current frame', 'type': 'int',
                'value': self.frame
            },
            {
                'name': 'show ref stream', 'type': 'bool',
                'value': self.show_ref_stream
            },
            {
                'name': 'show test stream', 'type': 'bool',
                'value': self.show_test_stream
            }
        ]
        self.params = Parameter.create(name='params', type='group',
                                       children=params)
        self.parameterTree.setParameters(self.params, showTop=False)

        if self.cxi_file is not None:
            self.load_cxi_file(self.cxi_file)
        if self.ref_stream_file is not None:
            self.load_stream_file(self.ref_stream_file, 'ref')
        if self.test_stream_file is not None:
            self.load_stream_file(self.test_stream_file, 'test')

        # menu bar actions
        self.action_load_cxi.triggered.connect(self.load_cxi)
        self.action_load_geom.triggered.connect(self.load_geom)
        self.action_load_ref_stream.triggered.connect(
            partial(self.load_stream, flag='ref'))
        self.action_load_test_stream.triggered.connect(
            partial(self.load_stream, flag='test')
        )

        # parameter tree slots
        self.params.param(
            'current frame').sigValueChanged.connect(self.change_frame)
        self.params.param(
            'show ref stream').sigValueChanged.connect(
            partial(self.change_show_stream, flag='ref')
        )
        self.params.param(
            'show test stream').sigValueChanged.connect(
            partial(self.change_show_stream, flag='test')
        )

        # image view / stream plot slots
        self.peak_stream_item.sigClicked.connect(self.stream_item_clicked)
        self.ref_stream_item.sigClicked.connect(self.stream_item_clicked)
        self.test_stream_item.sigClicked.connect(self.stream_item_clicked)

    @pyqtSlot()
    def load_cxi(self):
        filepath, _ = QtGui.QFileDialog.getOpenFileName(
            self, 'Open File', self.workdir, 'CXI File (*.cxi)')
        if filepath == '':
            return
        self.load_cxi_file(filepath)

    def load_cxi_file(self, cxi_file):
        try:
            h5_obj = h5py.File(cxi_file, 'r')
            data = h5_obj[self.data_location]
        except IOError:
            print('Failed to load %s' % cxi_file)
            return
        self.cxi_file = cxi_file
        self.data = data
        self.nb_frame = self.data.shape[0]
        # collect peaks from cxi, XPosRaw/YPosRaw is fs, ss, respectively
        peaks_x = h5_obj['%s/peakXPosRaw' % self.peak_info_location].value
        peaks_y = h5_obj['%s/peakYPosRaw' % self.peak_info_location].value
        nb_peaks = h5_obj['%s/nPeaks' % self.peak_info_location].value
        self.peaks = []
        self.nb_peaks = nb_peaks
        for i in range(len(nb_peaks)):
            self.peaks.append(
                np.concatenate(
                    [
                        peaks_y[i, :nb_peaks[i]].reshape(-1, 1),
                        peaks_x[i, :nb_peaks[i]].reshape(-1, 1)
                    ],
                    axis=1
                )
            )
        self.params.param('cxi file').setValue(self.cxi_file)
        self.params.param('total frame').setValue(self.nb_frame)
        self.update_display()
        self.update_stream_plot()

    def load_geom(self):
        filepath, _ = QtGui.QFileDialog.getOpenFileName(
            self, 'Open file', self.workdir, 'Geom File (*.geom *.h5 *.data)')
        if filepath == '':
            return
        geom_file = filepath
        geom = Geometry(geom_file, self.pixel_size)
        self.geom_file = geom_file
        self.geom = geom
        self.params.param('geometry file').setValue(self.geom_file)
        self.update_display()

    @pyqtSlot(object, object)
    def change_frame(self, _, frame):
        if frame < 0:
            frame = 0
        elif frame > self.nb_frame:
            frame = self.nb_frame - 1
        self.frame = frame
        self.update_display()

    @pyqtSlot(str)
    def load_stream(self, flag):
        filepath, _ = QtGui.QFileDialog.getOpenFileName(
            self, 'Open File', self.workdir, 'Stream File (*.stream)')
        if filepath == '':
            return
        self.load_stream_file(filepath, flag)

    def load_stream_file(self, stream_file, flag):
        stream = Stream(stream_file)
        # collect reflections
        all_reflections = {}
        events = []
        for i in range(len(stream.chunks)):
            chunk = stream.chunks[i]
            event = chunk.event
            reflections = []
            if len(chunk.crystals) > 0:
                events.append(event)
            for j in range(len(chunk.crystals)):
                crystal = chunk.crystals[j]
                for k in range(len(crystal.reflections)):
                    reflection = crystal.reflections[k]
                    reflections.append(
                        [reflection.ss, reflection.fs]
                    )
            all_reflections[event] = np.array(reflections)
        if flag == 'ref':
            self.ref_stream_file = stream_file
            self.ref_stream = stream
            self.ref_reflections = all_reflections
            self.ref_events = events
            self.params.param('ref stream').setValue(self.ref_stream_file)
        elif flag == 'test':
            self.test_stream_file = stream_file
            self.test_stream = stream
            self.test_reflections = all_reflections
            self.test_events = events
            self.params.param('test stream').setValue(self.test_stream_file)
        else:
            print('Undefined flag: %s' % flag)
        self.update_stream_plot()

    @pyqtSlot(object, object)
    def stream_item_clicked(self, _, pos):
        event = int(pos[0].pos()[0])
        self.params.param('current frame').setValue(event)

    @pyqtSlot(object, object, str)
    def change_show_stream(self, _, show, flag):
        if flag == 'ref':
            self.show_ref_stream = not self.show_ref_stream
        elif flag == 'test':
            self.show_test_stream = not self.show_test_stream
        else:
            raise ValueError('Undefined flag: %s' % flag)
        self.update_display()

    def mouseMoved(self, pos):
        if self.cxi_file is None:
            return
        mouse_pos = self.imageView.view.mapToView(pos)
        image = self.imageView.image
        x, y = int(mouse_pos.x()), int(mouse_pos.y())

        message = ''
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            message += "x:%d y:%d I:%.2E" % \
                       (x, y, image[x, y])
        if self.geom_file is not None:
            geom = self.geom
            centered_x = x - geom.offset_x
            centered_y = y - geom.offset_y
            message += ' centered x: %d, centered y: %d' % (
            centered_x, centered_y)
        self.statusbar.showMessage(message, 5000)

    def showHKLSlot(self, _, showHKL):
        self.showHKL = showHKL
        self.updateDisp()

    def detDistChangedSlot(self, _, dist):
        self.det_dist = float(dist) * 1.E-3  # convert to meter

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
                                                      'Open File', '',
                                                      'State File (*.yml)'))
        self.maybeLoadState(fpath)

    def maybeLoadState(self, fpath):
        if fpath == '':
            return
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
                                                  'Save File', '',
                                                  'State File (*.yml)')
        yml_file = file(fpath, 'w')
        yaml.dump(state, yml_file)
        yml_file.close()

    def loadTestStreamSlot(self):
        """Load test stream files (<=3)"""
        fpaths = QtGui.QFileDialog.getOpenFileNames(self,
                                                    'Open File', '',
                                                    'Stream File (*.stream)')
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
            self.streams['test%d' % (i + 1)] = {
                'file': self.test_stream_files[-1],
                'data': self.test_stream_data[-1]
            }
            self.params.param('File Info',
                              'Test Stream %d' % (i + 1)).setValue(
                basename(str(fpath)))
        self.updateStreamPlot()

    def loadRefStreamSlot(self):
        fpath = str(QtGui.QFileDialog.getOpenFileName(self,
                                                      'Open File', '',
                                                      'Stream File (*.stream)'))
        self.maybeLoadRefStream(fpath)

    def maybeLoadRefStream(self, fpath):
        if fpath is '':
            return
        self.ref_stream_file = str(fpath)
        self.ref_stream_data = load_data_from_stream(self.ref_stream_file)
        self.streams['ref'] = {
            'file': self.ref_stream_file,
            'data': self.ref_stream_data
        }
        self.params.param('File Info',
                          'Reference Stream').setValue(
            basename(self.ref_stream_file))
        self.updateStreamPlot()

    def maybeLoadGeom(self, fpath):
        if fpath is '':
            return
        self.geom_file = fpath
        self.geom = Geometry(self.geom_file, self.pixel_size)
        self.params.param('File Info',
                          'Geometry File').setValue(basename(self.geom_file))
        self.updateDisp()

    def showStreamTabelSlot(self):
        self.streamTable.streams = self.streams
        self.streamTable.updateTable(self.frame)
        self.streamTable.exec_()

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
            spot_x, spot_y = self.geom.map((rawXYs[j, :]))
            spots_x.append(spot_x)
            spots_y.append(spot_y)
        p = pg.ScatterPlotItem()
        self.imageView.getView().addItem(p)
        self.reflection_items.append(p)
        p.setData(spots_x, spots_y, symbol='o',
                  size=size, brush=(255, 255, 255, 0), pen=pen)

    def update_stream_plot(self):
        if self.cxi_file is not None:
            pos = np.concatenate(
                [
                    np.arange(self.nb_frame).reshape(-1, 1),
                    np.zeros(self.nb_frame).reshape(-1, 1)
                ],
                axis=1
            )
            self.peak_stream_item.setData(pos=pos)
        if len(self.ref_events) > 0:
            pos = np.concatenate(
                [
                    np.array(self.ref_events).reshape(-1, 1),
                    np.ones(len(self.ref_events)).reshape(-1, 1)
                ],
                axis=1,
            )
            self.ref_stream_item.setData(pos=pos)
        if len(self.test_events) > 0:
            pos = np.concatenate(
                [
                    np.array(self.test_events).reshape(-1, 1),
                    np.ones(len(self.test_events)).reshape(-1, 1) * 2
                ],
                axis=1
            )
            self.test_stream_item.setData(pos=pos)

    def update_display(self):
        if self.data is None:
            return
        image = self.data[self.frame]
        if self.geom is not None:
            image = self.geom.rearrange(image)
        self.image_view.setImage(
            image, autoRange=False, autoLevels=False,
            autoHistogramRange=False
        )

        # plot peaks
        self.peak_item.clear()
        self.peak_item.setData(pos=self.peaks[self.frame] + 0.5)

        # plot reflections
        self.ref_reflection_item.clear()
        if self.show_ref_stream and self.frame in self.ref_events:
            reflections = self.ref_reflections[self.frame]
            if len(reflections) > 0:
                self.ref_reflection_item.setData(pos=reflections + 0.5)
        self.test_reflection_item.clear()
        if self.show_test_stream and self.frame in self.test_events:
            reflections = self.test_reflections[self.frame]
            if len(reflections) > 0:
                self.test_reflection_item.setData(pos=reflections + 0.5)


def main():
    if len(sys.argv) > 1:
        print('using setting from %s' % sys.argv[1])
        with open(sys.argv[1], 'r') as f:
            settings_dict = yaml.load(f)
    else:
        settings_dict = {}
        print('using default settings')
    settings = Settings(settings_dict)
    app = QtGui.QApplication(sys.argv)
    win = GUI(settings)
    win.setWindowTitle("CXI Viewer")
    win.show()
    app.exec_()


if __name__ == '__main__':
    main()
