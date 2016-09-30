# -*- coding: utf-8 -*-
"""
Display 2D image using pyqtgraph. The input image can be .npy, balabala...
"""

import numpy as np 
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg 


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


class MyImageView(pg.ImageView):
    """docstring for MyImageView"""
    def __init__(self):
        super(MyImageView, self).__init__()
        self.setAcceptDrops(True)
        self.current_image = None
        self.image_list = []
        self.image_data = None

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) > 1:
            print("WARNING! Please drag in 1 file only!")
            event.ignore()
            return None
        url = urls[0]
        if QtCore.QString(url.toLocalFile()).startsWith('/.file/id='):
            # print("POSIX file in mac.")
            global drop_file
            drop_file = getFilepathFromLocalFileID(url)
        file_info = QtCore.QFileInfo(drop_file)
        ext = file_info.suffix()
        if ext not in [u'png', u'npy']:
            print("unaccepted ext: %s" %ext)
            event.ignore()
            return None
        else:
            print("accepted ext: %s" %ext)
            event.accept()
            self.current_image = drop_file
            if drop_file not in self.image_list:
                self.image_list.append(drop_file)
            return None


    def dropEvent(self, event):
        print(self.current_image)
        print(self.image_list)
        if QtCore.QFileInfo(self.current_image).suffix() == u'npy':
            print("display npy file")
            self.image_data = np.load(str(self.current_image))  
            self.setImage(self.image_data)          
        elif QtCore.QFileInfo(self.current_image).suffix() == u'png':
            print("png file display not implemented.")
        # self.setImage()


def mouse_moved(event):
    mouse_point = imv.view.mapSceneToView(event[0])
    x, y = int(mouse_point.x()), int(mouse_point.y())
    if x >= 0 and x < data.shape[0] and y >= 0 and y < data.shape[1]:
        I = data[x,y]
    else:
        I = 0
    print("x:%d, y:%d, I:%.2f" %(x, y, I))


app = QtGui.QApplication([])
win = QtGui.QMainWindow()
win.resize(800, 800)
imv = MyImageView()
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('Image Viewer')

# Set image data
data = np.load('powder.npz')['powder_p']
# data = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
imv.setImage(data)
# imv.setImage(np.log(np.abs(data)+1.))

# proxy = pg.SignalProxy(imv.scene.sigMouseMoved, rateLimit=1, slot=mouse_moved)

# Start Qt event loop
QtGui.QApplication.instance().exec_()