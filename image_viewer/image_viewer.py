#! /opt/local/bin/python2.7
#! coding=utf-8

"""
Usage: 
    image_viewer.py

Options:
    -h --help       Show this screen.
"""

import sys
import os
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QMainWindow, QApplication
from PyQt4.QtCore import pyqtSignal, pyqtSlot
import pyqtgraph as pg 
from pyqtgraph.parametertree import ParameterTree, Parameter
from pyqtgraph import PlotDataItem
import numpy as np
import scipy as sp
from scipy.signal import savgol_filter
from docopt import docopt
import datetime
from util import *

def getFilepathFromLocalFileID(localFileID):
    import CoreFoundation as CF
    import objc
    localFileQString = QtCore.QString(localFileID.toLocalFile())
    relCFStringRef = CF.CFStringCreateWithCString(
                     CF.kCFAllocatorDefault,
                     localFileQString.toUtf8(),
                     CF.kCFStringEncodingUTF8
                     )
    relCFURL = CF.CFURLCreateWithFileSystemPath(
               CF.kCFAllocatorDefault,
               relCFStringRef,
               CF.kCFURLPOSIXPathStyle,
               False   # is directory
               )
    absCFURL = CF.CFURLCreateFilePathURL(
               CF.kCFAllocatorDefault,
               relCFURL,
               objc.NULL
               )
    return QtCore.QUrl(str(absCFURL[0])).toLocalFile()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        uic.loadUi('layout-dev.ui', self)
        self.plotWidget.hide()
        self.splitter_2.setSizes([self.width()*0.7, self.width()*0.3])
        self.setAcceptDrops(True)

        self.filepath = None  # current filepath
        self.imageData = None  # original image data, 2d or 3d
        self.imageShape = None
        self.dispData = None  # 2d data for plot
        self.dispShape = None
        self.mask = None
        self.acceptedFiletypes = [u'npy', u'png', u'mat']

        self.dispItem = self.imageView.getImageItem()
        self.ringItem = pg.ScatterPlotItem()
        self.imageView.getView().addItem(self.ringItem)

        self.axis = 'x'
        self.frameIndex = 0
        self.maskFlag = False
        self.binaryFlag = False 
        self.dispThreshold = 0
        self.center = [0, 0]
        self.showRings = False 
        self.ringRadiis = []

        self.profileItem = PlotDataItem(pen=pg.mkPen('y', width=1, style=QtCore.Qt.SolidLine))
        self.smoothItem = PlotDataItem(pen=pg.mkPen('g', width=2, style=QtCore.Qt.DotLine))
        self.plotWidget.addItem(self.profileItem)
        self.plotWidget.addItem(self.smoothItem)
        self.showProfile = False
        self.profileType = 'radial'
        self.angularRmin = 0.
        self.angularRmax = np.inf
        self.profileMode = 'sum'
        self.smoothFlag = False 
        self.smoothWinSize = 15
        self.polyOrder = 3

        self.profileLog = False
        self.imageAutoRange = True 
        self.imageAutoLevels = True 
        self.imageAutoHistogramRange = True
        self.plotAutoRange = True 

        params_list = [
                      {'name': 'Image State Information', 'type': 'group', 'children': [
                          {'name': 'Filename', 'type': 'str', 'value': 'filename', 'readonly': True},
                          {'name': 'Image Shape', 'type': 'str', 'value': 'imageshape', 'readonly': True},
                      ]},
                      {'name': 'Basic Operation', 'type': 'group', 'children': [
                          {'name': 'Axis', 'type': 'list', 'values': ['x','y','z'], 'value': 'x'},
                          {'name': 'Frame Index', 'type': 'int', 'value': self.frameIndex},
                          {'name': 'Apply Mask', 'type': 'bool', 'value': self.maskFlag},
                          {'name': 'binaryzation', 'type': 'bool', 'value': self.binaryFlag},
                          {'name': 'Threshold', 'type': 'float', 'value': self.dispThreshold},
                          {'name': 'Center x', 'type': 'int', 'value': self.center[1]},
                          {'name': 'Center y', 'type': 'int', 'value': self.center[0]},
                          {'name': 'Show Rings', 'type': 'bool'},
                          {'name': 'Ring Radiis', 'type': 'str', 'value': ''},
                      ]},
                      {'name': 'Radial or Angular Profile', 'type': 'group', 'children': [
                          {'name': 'Show Profile', 'type': 'bool'},
                          {'name': 'Radial or Angular', 'type': 'list', 'values': ['radial','angular'], 'value': 'radial'},
                          {'name': 'Profile Mode', 'type': 'list', 'values': ['sum','mean'], 'value': 'sum'},
                          {'name': 'Angular Option', 'type': 'group', 'children': [
                              {'name': 'R min', 'type': 'float', 'value': self.angularRmin},
                              {'name': 'R max', 'type': 'float', 'value': self.angularRmax},
                          ]},
                          {'name': 'Smoothing', 'type': 'group', 'children': [
                              {'name': 'Enable Smoothing', 'type': 'bool', 'value': self.smoothFlag},
                              {'name': 'Window Size', 'type': 'int', 'value': self.smoothWinSize},
                              {'name': 'Poly-Order', 'type': 'int', 'value': self.polyOrder},
                          ]}
                      ]},
                      {'name': 'Display Option', 'type': 'group', 'children': [
                          {'name': 'Image Option', 'type': 'group', 'children': [
                              {'name': 'autoRange', 'type': 'bool', 'value': self.imageAutoRange},
                              {'name': 'autoLevels', 'type': 'bool', 'value': self.imageAutoLevels},
                              {'name': 'autoHistogramRange', 'type': 'bool', 'value': self.imageAutoHistogramRange},
                          ]},
                          {'name': 'Plot Option', 'type': 'group',  'children': [
                              {'name': 'autoRange', 'type': 'bool', 'value': self.plotAutoRange},
                              {'name': 'Log', 'type': 'bool', 'value': self.profileLog},
                          ]},
                      ]}
                  ]
        self.params = Parameter.create(name='', type='group', children=params_list)
        self.parameterTree.setParameters(self.params, showTop=False)

        self.fileList.itemDoubleClicked.connect(self.changeImageSlot)
        self.fileList.customContextMenuRequested.connect(self.showFileMenuSlot)
        self.params.param('Basic Operation', 'Axis').sigValueChanged.connect(self.axisChangedSlot)
        self.params.param('Basic Operation', 'Frame Index').sigValueChanged.connect(self.frameIndexChangedSlot)
        self.params.param('Basic Operation', 'Apply Mask').sigValueChanged.connect(self.applyMaskSlot)
        self.params.param('Basic Operation', 'binaryzation').sigValueChanged.connect(self.binaryImageSlot)
        self.params.param('Basic Operation', 'Threshold').sigValueChanged.connect(self.setDispThresholdSlot)
        self.params.param('Basic Operation', 'Center x').sigValueChanged.connect(self.centerXChangedSlot)
        self.params.param('Basic Operation', 'Center y').sigValueChanged.connect(self.centerYChangedSlot)
        self.params.param('Basic Operation', 'Show Rings').sigValueChanged.connect(self.showRingsSlot)
        self.params.param('Basic Operation', 'Ring Radiis').sigValueChanged.connect(self.ringRadiiSlot)
        self.params.param('Radial or Angular Profile', 'Show Profile').sigValueChanged.connect(self.showProfileSlot)
        self.params.param('Radial or Angular Profile', 'Radial or Angular').sigValueChanged.connect(self.setProfileTypeSlot)
        self.params.param('Radial or Angular Profile', 'Profile Mode').sigValueChanged.connect(self.setProfileModeSlot)
        self.params.param('Radial or Angular Profile', 'Angular Option', 'R min').sigValueChanged.connect(self.setAngularRmin)
        self.params.param('Radial or Angular Profile', 'Angular Option', 'R max').sigValueChanged.connect(self.setAngularRmax)
        self.params.param('Radial or Angular Profile', 'Smoothing', 'Enable Smoothing').sigValueChanged.connect(self.setSmooth)
        self.params.param('Radial or Angular Profile', 'Smoothing', 'Window Size').sigValueChanged.connect(self.setWinSize)
        self.params.param('Radial or Angular Profile', 'Smoothing', 'Poly-Order').sigValueChanged.connect(self.setPolyOrder)
        self.params.param('Display Option', 'Image Option', 'autoRange').sigValueChanged.connect(self.imageAutoRangeSlot)
        self.params.param('Display Option', 'Image Option', 'autoLevels').sigValueChanged.connect(self.imageAutoLevelsSlot)
        self.params.param('Display Option', 'Image Option', 'autoHistogramRange').sigValueChanged.connect(self.imageAutoHistogramRangeSlot)
        self.params.param('Display Option', 'Plot Option', 'autoRange').sigValueChanged.connect(self.plotAutoRangeSlot)
        self.params.param('Display Option', 'Plot Option', 'Log').sigValueChanged.connect(self.setLogModeSlot)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) > 1:
            print_with_timestamp("WARNING! Please drag in 1 file only!")
            event.ignore()
            return None
        url = urls[0]
        if QtCore.QString(url.toLocalFile()).startsWith('/.file/id='):
            # print_with_timestamp("POSIX file in mac.")
            drop_file = getFilepathFromLocalFileID(url)
        file_info = QtCore.QFileInfo(drop_file)
        ext = file_info.suffix()
        if ext not in self.acceptedFiletypes:
            print_with_timestamp("unaccepted ext: %s" %ext)
            event.ignore()
            return None
        else:
            print_with_timestamp("accepted ext: %s" %ext)
            event.accept()
            return None

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        url = urls[0]
        localFile = QtCore.QString(url.toLocalFile())
        if localFile.startsWith('/.file/id='):
            # print_with_timestamp("POSIX file in mac.")
            localFile = getFilepathFromLocalFileID(url)
        self.filepath = localFile
        self.loadData(self.filepath)
        self.changeDisp()
        # add file to fileList
        basename = os.path.basename(str(self.filepath))
        item = FilenameListItem(basename)
        item.filepath = self.filepath
        item.setToolTip(item.filepath)
        self.fileList.addItem(item)

    def loadData(self, filepath):
        filepath = str(filepath)
        basename = os.path.basename(filepath)
        self.imageData = load_data(filepath)
        self.imageShape = self.imageData.shape
        _shape_str = ''
        if len(self.imageShape) == 2:
            _x, _y = self.imageShape
            _shape_str = 'x: %d, y: %d' %(_x, _y)
        elif len(self.imageShape) == 3:
            _x, _y, _z = self.imageShape
            _shape_str = 'x: %d, y: %d, z: %d' %(_x, _y, _z)
        self.params.param('Image State Information', 'Image Shape').setValue(_shape_str)
        self.params.param('Image State Information', 'Filename').setValue(basename)

    def maybePlotProfile(self):
        if self.showProfile:
            print_with_timestamp("OK, I'll plot %s profile" %self.profileType)
            if self.plotWidget.isVisible() == False:
                self.plotWidget.show()
                self.splitter.setSizes([self.height()*0.7, self.height()*0.3])
            viewBox = self.plotWidget.getViewBox()
            if self.plotAutoRange:
                viewBox.enableAutoRange()
            else:
                viewBox.disableAutoRange()
            if self.dispData is not None:
                if self.maskFlag == True:
                    assert self.mask.shape == self.dispData.shape 
                else:
                    self.mask = np.ones_like(self.dispData)
                if self.profileType == 'radial':
                    profile = calc_radial_profile(self.dispData, self.center, mask=self.mask, mode=self.profileMode)
                else:
                    annulus_mask = make_annulus(self.dispShape, self.angularRmin, self.angularRmax)
                    profile = calc_angular_profile(self.dispData, self.center, mask=self.mask*annulus_mask, mode=self.profileMode)
                self.profileItem.setData(profile)
                if self.profileType == 'radial':
                    self.plotWidget.setTitle('Radial Profile')
                    self.plotWidget.setLabels(bottom='r/pixel')
                else:
                    self.plotWidget.setTitle('Angular Profile')
                    self.plotWidget.setLabels(bottom='theta/deg')
                if self.smoothFlag == True:
                    smoothed_profile = savgol_filter(profile, self.smoothWinSize, self.polyOrder)
                    # print(smoothed_profile)
                    self.smoothItem.setData(smoothed_profile)
                else:
                    self.smoothItem.clear()
                if self.profileLog == True:
                    radialProfile[radialProfile < 1.] = 1.
                    self.profileItem.setData(radialProfile)
                    self.plotWidget.setLogMode(y=True)
                else:
                    self.plotWidget.setLogMode(y=False)
        else:
            self.plotWidget.hide()

    def calcDispData(self):
        if self.imageData is None:
            return None
        elif len(self.imageShape) == 3:
            _x, _y, _z = self.imageShape
            if self.axis == 'x':
                if 0 <= self.frameIndex < _x:
                    dispData = self.imageData[self.frameIndex, :, :]
                else:
                    print_with_timestamp("ERROR! Index out of range. %s axis frame %d" %(self.axis, self.frameIndex))
            elif self.axis == 'y':
                if 0 <= self.frameIndex < _y:
                    dispData = self.imageData[:, self.frameIndex, :]
                else:
                    print_with_timestamp("ERROR! Index out of range. %s axis frame %d" %(self.axis, self.frameIndex))
            else:
                if 0 <= self.frameIndex < _z:
                    dispData = self.imageData[:, :, self.frameIndex]
                else:
                    print_with_timestamp("ERROR! Index out of range. %s axis frame %d" %(self.axis, self.frameIndex))
        elif len(self.imageShape) == 2:
            dispData = self.imageData
        return dispData

    @pyqtSlot(QtGui.QListWidgetItem)
    def changeImageSlot(self, item):
        self.filepath = item.filepath 
        self.loadData(self.filepath)
        self.changeDisp()
        self.maybePlotProfile()

    def changeDisp(self):
        if self.maskFlag:
            if self.mask is not None:
                assert self.mask.shape == self.dispShape
                dispData = self.mask * self.calcDispData()
        else:
            dispData = self.calcDispData()
        if self.binaryFlag:
            dispData[dispData < self.dispThreshold] = 0.
            dispData[dispData >= self.dispThreshold] = 1.
        self.dispData = dispData
        self.dispShape = self.dispData.shape
        self.center = [self.dispShape[0]//2, self.dispShape[1]//2]
        self.params.param('Basic Operation', 'Center x').setValue(self.center[0])
        self.params.param('Basic Operation', 'Center y').setValue(self.center[1])
        # set dispData to distItem. Note: transpose the dispData to show image with same manner of matplotlib
        self.imageView.setImage(self.dispData.T, autoRange=self.imageAutoRange, autoLevels=self.imageAutoLevels, autoHistogramRange=self.imageAutoHistogramRange) 
        if self.showRings:
            if self.ringRadiis is not None:
                _cx = np.ones_like(self.ringRadiis) * self.center[0]
                _cy = np.ones_like(self.ringRadiis) * self.center[1]
            self.ringItem.setData(_cx, _cy, size=self.ringRadiis*2., symbol='o', brush=(255,255,255,0), pen='r', pxMode=False)            

    @pyqtSlot(object)
    def axisChangedSlot(self, axis):
        print_with_timestamp('axis changed.')
        self.axis = axis.value()
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def frameIndexChangedSlot(self, frameIndex):
        print_with_timestamp('frame index changed')
        self.frameIndex = frameIndex.value()
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def showProfileSlot(self, showProfile):
        print_with_timestamp('show or hide radial profile changed')
        self.showProfile = showProfile.value()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setLogModeSlot(self, log):
        print_with_timestamp('log mode changed')
        self.radialProfileLog = log.value()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def centerXChangedSlot(self, centerX):
        print_with_timestamp('center X changed')
        self.center[0] = centerX.value()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def centerYChangedSlot(self, centerY):
        print_with_timestamp('center Y changed')
        self.center[0] = centerY.value()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def showFileMenuSlot(self, position):
        print_with_timestamp(position)
        fileMenu = QtGui.QMenu()
        markAsMask = fileMenu.addAction("Mark As Mask")
        action = fileMenu.exec_(self.fileList.mapToGlobal(position))
        if action == markAsMask:
            currentFile = self.fileList.currentItem().filepath
            data = load_data(currentFile)
            if len(data.shape) != 2:
                raise ValueError('%s can not be used as mask. Mask data must be 2d.' %currentFile)
            assert len(data.shape) == 2
            self.mask = data 

    @pyqtSlot(object)
    def applyMaskSlot(self, mask):
        print_with_timestamp('turn on/off mask')
        self.maskFlag = mask.value()
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def imageAutoRangeSlot(self, imageAutoRange):
        print_with_timestamp('set image autorange: %s' %str(imageAutoRange.value()))
        self.imageAutoRange = imageAutoRange.value()
        self.changeDisp()

    @pyqtSlot(object)
    def imageAutoLevelsSlot(self, imageAutoLevels):
        print_with_timestamp('set image autolevels: %s' %str(imageAutoLevels.value()))
        self.imageAutoLevels = imageAutoLevels.value()
        self.changeDisp()

    @pyqtSlot(object)
    def imageAutoHistogramRangeSlot(self, imageAutoHistogramRange):
        print_with_timestamp('set image autohistogram: %s' %str(imageAutoHistogramRange.value()))
        self.imageAutoHistogramRange = imageAutoHistogramRange.value()
        self.changeDisp()

    @pyqtSlot(object)
    def plotAutoRangeSlot(self, plotAutoRange):
        print_with_timestamp('set plot autorange: %s' %str(plotAutoRange.value()))
        self.plotAutoRange = plotAutoRange.value()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def showRingsSlot(self, showRings):
        print_with_timestamp('show rings: %s' %showRings.value())
        self.showRings = showRings.value()
        self.changeDisp()

    @pyqtSlot(object)
    def ringRadiiSlot(self, ringRadiis):
        print_with_timestamp('set ring radiis: %s' %str(ringRadiis.value()))
        ringRadiisStrList = ringRadiis.value().split()
        ringRadiis = []
        for ringRadiisStr in ringRadiisStrList:
            ringRadiis.append(float(ringRadiisStr))
        self.ringRadiis = np.asarray(ringRadiis)
        self.changeDisp()

    @pyqtSlot(object)
    def setProfileTypeSlot(self, profileType):
        print_with_timestamp('set profile type: %s' %str(profileType.value()))
        self.profileType = profileType.value()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setProfileModeSlot(self, profileMode):
        print_with_timestamp('set profile mode: %s' %str(profileMode.value()))
        self.profileMode = profileMode.value()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def binaryImageSlot(self, binaryImage):
        print_with_timestamp('set binary flat: %s' %str(binaryImage.value()))
        self.binaryFlag = binaryImage.value()
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setDispThresholdSlot(self, threshold):
        print_with_timestamp('set disp threshold: %.1f' %threshold.value())
        self.dispThreshold = threshold.value()
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setSmooth(self, smooth):
        print_with_timestamp('set smooth: %s' %str(smooth.value()))
        self.smoothFlag = smooth.value()
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setWinSize(self, winSize):
        winSize = winSize.value()
        if winSize % 2 == 0:
            winSize += 1  # winSize must be odd
        print_with_timestamp('set smooth winsize: %d' %winSize)
        self.smoothWinSize = winSize
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setPolyOrder(self, polyOrder):
        print_with_timestamp('set poly order: %d' %polyOrder.value())
        self.polyOrder = polyOrder.value()
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setAngularRmin(self, Rmin):
        Rmin = float(Rmin.value())
        print_with_timestamp('set angular Rmin to %.1f' %Rmin)
        self.angularRmin = Rmin 
        self.changeDisp()
        self.maybePlotProfile()

    @pyqtSlot(object)
    def setAngularRmax(self, Rmax):
        Rmax = float(Rmax.value())
        print_with_timestamp('set angular Rmax to %.1f' %Rmax)
        self.angularRmax = Rmax
        self.changeDisp()
        self.maybePlotProfile()

class MyImageView(pg.ImageView):
    """docstring for MyImageView"""
    def __init__(self, parent=None, *args):
        super(MyImageView, self).__init__(parent, view=pg.PlotItem(), *args)


class MyPlotWidget(pg.PlotWidget):
    """docstring for MyPlotWidget"""
    def __init__(self, parent=None):
        super(MyPlotWidget, self).__init__(parent=parent)

        
class MyParameterTree(ParameterTree):
    """docstring for MyParameterTree"""
    def __init__(self, parent=None):
        super(MyParameterTree, self).__init__(parent=parent)


def mouseMoved(event):
    imageView = win.imageView
    dispItem = win.dispItem
    data = dispItem.image
    if win.filepath == None:
        return None
    mouse_point = imageView.view.mapToView(event[0])
    x, y = int(mouse_point.x()), int(mouse_point.y())
    filename = os.path.basename(str(win.filepath))
    if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
        win.statusbar.showMessage("%s x:%d y:%d I:%.2f" %(filename, x, y, data[x, y]), 5000)
    else:
        # win.statusbar.showMessage("out of bounds")
        pass


class FilenameListWidget(QtGui.QListWidget):
    """docstring for FilenameListWidget"""
    def __init__(self, parent=None, *args):
        super(FilenameListWidget, self).__init__(parent, *args)


class FilenameListItem(QtGui.QListWidgetItem):
    """docstring for FilenameListItem"""
    def __init__(self, parent=None, *args):
        super(FilenameListItem, self).__init__(parent, *args)
        self.filepath = None


if __name__ == '__main__':
    # add signal to enable CTRL-C 
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    argv = docopt(__doc__)
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 600)
    win.setWindowTitle("Image Viewer")
    
    proxy = pg.SignalProxy(win.imageView.scene.sigMouseMoved, rateLimit=10, slot=mouseMoved)
    win.show()
    app.exec_()