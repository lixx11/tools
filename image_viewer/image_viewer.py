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
import numpy as np
import scipy as sp
from docopt import docopt


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


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
    def __init__(self, parent=None, ui=None):
        super(MainWindow, self).__init__(parent)
        uic.loadUi(ui, self)
        self.imageView.addImageSignal.connect(self.listWidget.addImageSlot)
        self.listWidget.itemDoubleClicked.connect(self.imageView.showImage)


class MyImageView(pg.ImageView):
    """docstring for MyImageView"""
    addImageSignal = pyqtSignal(str)

    def __init__(self, parent=None, *args):
        super(MyImageView, self).__init__(parent, *args)
        self.setAcceptDrops(True)
        self.filepath = None
        self.accepted_filetypes = [u'npy', u'png']
        
    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) > 1:
            print("WARNING! Please drag in 1 file only!")
            event.ignore()
            return None
        url = urls[0]
        if QtCore.QString(url.toLocalFile()).startsWith('/.file/id='):
            # print("POSIX file in mac.")
            drop_file = getFilepathFromLocalFileID(url)
        file_info = QtCore.QFileInfo(drop_file)
        ext = file_info.suffix()
        if ext not in self.accepted_filetypes:
            print("unaccepted ext: %s" %ext)
            event.ignore()
            return None
        else:
            print("accepted ext: %s" %ext)
            event.accept()
            self.filepath = drop_file
            return None

    def dropEvent(self, event):
        if QtCore.QFileInfo(self.filepath).suffix() == u'npy':
            print("display npy file")
            self.setImage(np.load(str(self.filepath)))
        elif QtCore.QFileInfo(self.filepath).suffix() == u'png':
            print("png file display not implemented.")
        self.addImageSignal.emit(str(self.filepath))

    @pyqtSlot(QtGui.QListWidgetItem)
    def showImage(self, item):
        self.filepath = item.filepath 
        self.setImage(np.load(str(self.filepath)))


def mouseMoved(event):
    imageView = win.imageView
    data = imageView.image
    if imageView.filepath == None:
        return None
    mouse_point = imageView.view.mapSceneToView(event[0])
    x, y = int(mouse_point.x()), int(mouse_point.y())
    filename = os.path.basename(str(win.imageView.filepath))
    if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
        win.statusbar.showMessage("%s x:%d y:%d I:%.2f" %(filename, x, y, data[x, y]), 5000)
    else:
        win.statusbar.showMessage("out of bounds")


class FilenameListWidget(QtGui.QListWidget):
    """docstring for FilenameListWidget"""
    def __init__(self, parent=None, *args):
        super(FilenameListWidget, self).__init__(parent, *args)
        self.current_filepath = None
        self.filepath_list = []

    @pyqtSlot(str)
    def addImageSlot(self, filepath):
        self.current_filepath = filepath
        self.filepath_list.append(filepath)
        basename = os.path.basename(str(filepath))
        item = FilenameListItem(basename)
        item.filepath = filepath
        item.setToolTip(filepath)
        self.addItem(item)


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
    win = MainWindow(ui='layout.ui')
    win.setWindowTitle("Image Viewer")
    
    proxy = pg.SignalProxy(win.imageView.scene.sigMouseMoved, rateLimit=10, slot=mouseMoved)
    win.show()
    app.exec_()