"""
Convert single particle patterns into hdf5 format file. The structure of the output h5 file is as follows:
    'data': 
        3D data with shape (Np, Nx, Ny) where Np is the number of patters, and Nx, Ny are the size of first and second axis.
    'labels': 
        classification results of the data with shape (Np,). 0 for no particle, 1 for single particle, 2 for multiple particle
    'wavelength':
        Optional. The XFEL wavelength in angstrom(1E-10m).
    'detector distance':
        Optional. The detector distance to intersection point in mm.
"""

import h5py
import numpy as np 


class SPIData(object):
    """docstring for SPIData"""
    def __init__(self, output='output.h5', wavelength=None, detectorDistance=None):
        """Summary
        
        Args:
            output (str, optional): Output filename. Default is 'output.h5'.
            wavelength (float, optional): XFEL wavelength in angstrom.
            detectorDistance (float, optional): Detector distance to intersection point in mm.
        """
        self.output = str(output)
        self.wavelength = float(wavelength)
        self.detectorDistance = float(detectorDistance)
        self.h5File = self.initH5()
        self.patternShape = None
        self.labels = []

    def initH5(self):
        h5File = h5py.File(self.output, 'w')
        print('===============================================')
        print('Creating h5 file: %s' %self.output)
        if self.wavelength is not None:
            h5File.create_dataset('wavelength', data=self.wavelength) 
            print('%-20s: %.2fA added to %s' %('wavelength', self.wavelength, self.output))
        if self.detectorDistance is not None:
            h5File.create_dataset('detector distance', data=self.detectorDistance)
            print('%-20s: %.2fmm added to %s' %('detector distance', self.detectorDistance, self.output))
        print('h5 file creation completed.')
        print('===============================================')
        return h5File

    def addPatternWithLabel(self, pattern, label):
        """Summary
        
        Args:
            pattern (array like): 2D pattern.
            label (int): 0 for no particle, 1 for single particle, 2 for multiple particle
        
        Returns:
            None: Description
        """
        pattern = np.asarray(pattern)
        assert len(pattern.shape) == 2  # must be 2D pattern
        label = int(label)
        if label == 0:
            print('adding no particle pattern')
        elif label == 1:
            print('adding single particle pattern')
        elif label == 2:
            print('adding multiple particle pattern')
        else:
            raise ValueError('label must be 0, 1 or 2 for no, single and multiple particle pattern')

        if self.patternShape is None:
            self.patternShape = pattern.shape
            print('pattern shape is set to %s' %str(pattern.shape))
            pattern = pattern.reshape((1, pattern.shape[0], pattern.shape[1]))
            self.h5File.create_dataset('data', data=pattern, maxshape=(None, self.patternShape[0], self.patternShape[1]))
            self.labels.append(label)
        else:
            if pattern.shape != self.patternShape:
                print('Warning!!! Pattern has different shape. Discard this pattern')  # the pattern shape must be same in one dataset
                return None
            Np = self.h5File['data'].shape[0]
            self.h5File['data'].resize(Np+1, axis=0)
            self.h5File['data'][Np] = pattern
            self.labels.append(label)


    def close(self):
        labels = np.asarray(self.labels, dtype=np.int8)
        self.h5File.create_dataset('labels', data=labels)
        self.h5File.close()


if __name__ == '__main__':
    spiData = SPIData('test.h5', wavelength=2.06, detectorDistance=135)  # create spiData
    spiData.addPatternWithLabel(np.random.rand(200,100), 2)  # add pattern with label
    spiData.addPatternWithLabel(np.random.rand(200,200), 1)  # add another pattern with label
    spiData.addPatternWithLabel(np.random.rand(200,100), 0)  # add another pattern with label
    spiData.close()  # close h5file
    