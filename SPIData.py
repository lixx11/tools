"""
Convert single particle patterns into hdf5 format file. The structure of the output h5 file is as follows:
    'data': 
        3D data with shape (Np, Nx, Ny) where Np is the number of patters, and Nx, Ny are the size of first and second axis.
    'labels': 
        classification results of the data with shape (Np,). 0 for no particle, 1 for single particle, 2 for multiple particle
    'Nn':
        Number of paterns with no particle.
    'Ns':
        Number of patterns with single particle.
    'Nm':
        Number of patterns with multiple particles.
    'Np':
        Number of totoal patterns.
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
        self.wavelength = wavelength
        self.detectorDistance = detectorDistance
        self.h5File = self.initH5()
        self.patternShape = None
        self.labels = []
        self.Nn = 0  # Num of no particle pattern
        self.Ns = 0  # Num of single particle pattern
        self.Nm = 0  # Num of multiple particle pattern
        self.Np = 0  # Num of total patterns

    def initH5(self):
        h5File = h5py.File(self.output, 'w')
        print('===============================================')
        print('Creating h5 file: %s' %self.output)
        if self.wavelength is not None:
            h5File.create_dataset('wavelength', data=float(self.wavelength)) 
            print('%-20s: %.2fA added to %s' %('wavelength', self.wavelength, self.output))
        if self.detectorDistance is not None:
            h5File.create_dataset('detector distance', data=float(self.detectorDistance))
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
        if self.patternShape is not None and pattern.shape != self.patternShape:
            print('Warning!!! Pattern has different shape. Discard this pattern.')  # the pattern shape must be same in one dataset
            return None
        if label not in [0,1,2]:
            raise ValueError('Label must be 0, 1 or 2 for no, single and multiple particle pattern.')

        if label == 0:
            print('Adding no particle pattern.')
            self.Nn += 1
        elif label == 1:
            print('Adding single particle pattern.')
            self.Ns += 1
        elif label == 2:
            print('Adding multiple particle pattern.')
            self.Nm += 1
        self.labels.append(label)

        newShape = (1, pattern.shape[0], pattern.shape[1])
        pattern = pattern.reshape(newShape)
        if self.Np == 0:
            self.patternShape = pattern.shape[1:]
            print('First adding. Pattern shape is set to %s' %str(pattern.shape))
            self.h5File.create_dataset('data', data=pattern, maxshape=(None, self.patternShape[0], self.patternShape[1]), chunks=newShape)
        else:
            self.h5File['data'].resize(self.Np+1, axis=0)
            self.h5File['data'][self.Np] = pattern
        self.Np += 1

    def close(self):
        labels = np.asarray(self.labels, dtype=np.int8)
        self.h5File.create_dataset('labels', data=labels)
        self.h5File.create_dataset('Nn', data=self.Nn)
        self.h5File.create_dataset('Ns', data=self.Ns)
        self.h5File.create_dataset('Nm', data=self.Nm)
        self.h5File.create_dataset('Np', data=self.Np)
        self.h5File.close()
        print('====================SUMMARY====================')
        print('Np %-35s: %d' %('(num of total patterns)', self.Np))
        print('Nn %-35s: %d' %('(num of no particel patterns)', self.Nn))
        print('Ns %-35s: %d' %('(num of single particle patterns)', self.Ns))
        print('Nm %-35s: %d' %('(num of multiple patterns)', self.Nm))


if __name__ == '__main__':
    spiData = SPIData('test.h5', wavelength=2.06, detectorDistance=135)  # create spiData
    spiData.addPatternWithLabel(np.random.rand(200,100), 2)  # add multiple particle pattern
    spiData.addPatternWithLabel(np.random.rand(200,200), 1)  # add single particle pattern
    spiData.addPatternWithLabel(np.random.rand(200,200), 0)  # add no particle pattern
    spiData.close()  # close h5file
    