#! /opt/local/bin/python2.7
#! coding=utf-8

"""
Usage:
    CSPAD_image_converter.py <raw_data> -g <geom_file> [-o <reassemble.npy>]

Options:
    -h --help       Show this screen.
    -g geom_file    Geometry file. Please provide CrystFEL, psana or Cheetah geom file.
    -o reassemble.npy   Output pattern after reassemble [default: reassemble.npy].
"""

import numpy as np
from scipy.interpolate import griddata
import h5py
from psgeom import camera
from docopt import docopt
import os
import sys



class CSPADImageConverter(object):
    """docstring for CSPADImageConverter"""
    def __init__(self, raw_data, geom_file, method='nearest', padding=10.E-3, pixel_size=0.11E-3, show=False):
        """Summary
        
        Args:
            raw_data (numpy.ndarray): raw data, the size must be 1480*1552.
            geom_file (string): CrystFEL, psana or Cheetah geometry file.
            method (str, optional): Interpolation method. Default: nearest.
            padding (float, optional): add padding around converted pattern, in m.
            pixel_size (float, optional): detector pixel size in m.
            show (bool, optional): Wether show converted data or not.
        """
        if type(raw_data) is not np.ndarray:
            print("ERROR! Raw data type must be numpy.ndarray!")
            sys.exit()
        if raw_data.shape != (1480, 1552): # fixed shape of CSPAD detector effective area
            print('ERROR! Raw data shape must be 1480*1552!')
            print('The given raw data has shape:'),
            print(raw_data.shape)
            sys.exit()
        self.raw_data = raw_data
        self.geom_file = geom_file
        self.method = method
        self.padding = padding  # add padding around converted pattern, in m.
        self.pixel_size = pixel_size  # in m.
        self.show = show
        self.x = None
        self.y = None
        self.z = None
        self.center = None  # center coordinate in pixel.
        self.reassembled_data = self._reassemble()


    def _reassemble(self):
        temp_h5 = '.temp.h5' # Converted Cheetah geom file
        filename, file_ext = os.path.splitext(self.geom_file)
        if file_ext == '.geom':
            print('using CrystFEL geom file')
            cspad = camera.Cspad.from_crystfel_file(self.geom_file)
            cspad.to_cheetah_file(temp_h5)
            geom_data = h5py.File(temp_h5, 'r')
        elif file_ext == '.data':
            print('using psana geom file')
            cspad = camera.Cspad.from_psana_file(self.geom_file)
            cspad.to_cheetah_file(temp_h5)
            geom_data = h5py.File(temp_h5, 'r')
        elif file_ext == '.h5':
            print('using Cheetah geom file')
            geom_data = h5py.File(self.geom_file, 'r')
        else:
            print('ERROR! Unrecognized geom type: %s' %file_ext)
            print('Plese provide CrysFEL, psana or Cheetah geom file.')
        self.x = geom_data['x'].value
        self.y = geom_data['y'].value 
        self.z = geom_data['z'].value 
        geom_data.close()

        x = self.x.reshape(-1)
        y = self.y.reshape(-1)
        z = self.z.reshape(-1)
        xyz = np.zeros((x.shape[0], 3))
        xyz[:,0] = x
        xyz[:,1] = y 
        xyz[:,2] = z 
        raw_data_1d = self.raw_data.reshape(-1)
        # calculate new pattern coorodinates
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        xx_range = x_range + 2 * self.padding
        yy_range = y_range + 2 * self.padding      
        xx_size = xx_range // self.pixel_size
        yy_size = yy_range // self.pixel_size      
        _xx = np.arange(xx_size) * self.pixel_size  - xx_range / 2.
        _yy = np.arange(yy_size) * self.pixel_size  - yy_range / 2.        
        xx, yy = np.meshgrid(_xx, _yy)
        center_x = np.where(np.abs(xx) == np.abs(xx).min())[1][0]
        center_y = np.where(np.abs(yy) == np.abs(yy).min())[0][0]
        center = np.asarray((center_x, center_y))
        self.center = center
        interp_data = griddata(xyz[:,0:2], raw_data_1d, (xx, yy), method=self.method, fill_value=0)

        if self.show:
            import matplotlib.pyplot as plt 
            plt.imshow(interp_data, interpolation='nearest')
            plt.show(block=True)
        return interp_data

    def map2new_coor(self, raw_coor):
        """Summary
        
        Args:
            raw_coor (array_like, shape N*2): Raw coordinate (ss, fs) to be converted to new coordinate in reassemble pattern. 
                                   Note: ss range: 0-1479, fs range: 0-1551
        
        Returns:
            new_coor (ndarray, shape N*2): New coordinate in reassemble pattern.
        """
        raw_coor = np.asarray(raw_coor)
        if raw_coor.size == 2:
            raw_coor = raw_coor.reshape((1,2))
        if len(raw_coor.shape) != 2 or raw_coor.shape[1] != 2:
            print('ERROR! Not valid raw coordinate. The shape should be N*2.')
            sys.exit()
        if  raw_coor[:,0].min() < 0 or raw_coor[:,0].max() > 1479 \
            or raw_coor[:,1].min() < 0 or raw_coor[:,1].max() > 1551:
            print("ERROR! Not valid raw coordinate. Raw coor: ss(0-1479), fs(0-1551)")
            sys.exit()
        num_coor = raw_coor.shape[0]
        raw_coor = np.round(raw_coor).astype(int)
        new_coor_x = self.center[0] + self.x[raw_coor[:,0], raw_coor[:,1]] / self.pixel_size
        new_coor_y = self.center[1] + self.y[raw_coor[:,0], raw_coor[:,1]] / self.pixel_size
        new_coor = np.zeros((num_coor, 2))
        new_coor[:,0] = new_coor_x
        new_coor[:,1] = new_coor_y
        return new_coor


if __name__ == '__main__':
    argv = docopt(__doc__)
    raw_data = argv['<raw_data>']
    geom_file = argv['-g']
    output = argv['-o']

    raw_data = np.load(raw_data)
    image_converter = CSPADImageConverter(raw_data, geom_file, show=True)

    np.save(output, image_converter.reassembled_data)
    print('Converted pattern saved to %s!' %output)