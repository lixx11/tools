"""
Store scalar attribute for SPIData patterns such as score in classification. The `paths` and `frames` are required fields. Any other customized fields can be added, but should be a list.
    ## REQUIRED FIELDS ##
    'paths'(list of str): 
        Path for each record. Must be SPIData format.
    'frames'(list of int): 
        Frame of each record. 
    ## CUSTOMIZED FIELDS ##
    # example given below #
    'scores'(list of float):
        Score for each record, which may be the classification score.
"""

import h5py
import numpy as np 


class SmallData(object):
    """docstring for SmallData"""
    def __init__(self, output='small-data.h5', smallDataNames=None):
        """Summary
        
        Args:
            output (str, optional): Output filename. Default is 'output.h5'.
            smallDataNames (None, optional): List of srings of customized small data names. Must provide 1 datanames at least like ['scores', ]
        
        Raises:
            Exception: smallDataNames must be a list of strings. error raised if not.
        """
        super(SmallData, self).__init__()
        self.output = output
        if smallDataNames is None:
            raise Exception('Must provide one smallDataNames at least, like ["scores",].')
            sys.exit()
        if not isinstance(smallDataNames, list):
            raise Exception('smallDataNames must be list of strings, like ["scores",].')
            sys.exit()
        self.smallDataNames = smallDataNames
        self.h5File = self.initH5()
        self.paths = []
        self.frames = []
        self.smallDataDict = {}
        for dataname in self.smallDataNames:
            self.smallDataDict[dataname] = []

    def initH5(self):
        h5File = h5py.File(self.output, 'w')
        print('%s created' %self.output)
        return h5File

    def addRecord(self, path, frame, **kwargs):
        """Summary
        
        Args:
            path (str): Corresponding filepath of this record.
            frame (int): Correspoding frame of this record in the file.
            **kwargs (TYPE): key-value pairs. The keys must match the small data names. Use singular form.
        
        Returns:
            TYPE: None
        
        Raises:
            Exception: Keys in kwargs must match the small data. Error raised if not.
        """
        keys = kwargs.keys()
        plural_keys = []
        for key in keys:
            plural_keys.append(key+'s')
        if not set(plural_keys) == set(self.smallDataNames):
            raise Exception('Record(%s) not match small data names: %s' %(plural_keys, self.smallDataNames))
        for key, item in kwargs.iteritems():
            plural_key = key + 's'
            self.smallDataDict[plural_key].append(item)
        self.paths.append(str(path))
        self.frames.append(int(frame))

    def addRecords(self, paths, frames, **kwargs):
        """Summary
        
        Args:
            paths (list of str): Corresponding filepath of this record.
            frames (list of int): Correspoding frame of this record in the file.
            **kwargs (TYPE): key-value pairs. The keys must match the small data names. Use plural form.
        
        Returns:
            TYPE: None
        
        Raises:
            Exception: Keys in kwargs must match the small data. Error raised if not.
        """
        keys = kwargs.keys()
        if not set(keys) == set(self.smallDataNames):
            raise Exception('Record(%s) not match small data names: %s' %(keys, self.smallDataNames))
        for key, item in kwargs.iteritems():
            self.smallDataDict[key].extend(item)
        self.paths.extend(paths)
        self.frames.extend(frames)

    def close(self):
        self.h5File.create_dataset('paths', data=self.paths, dtype=h5py.special_dtype(vlen=unicode))
        self.h5File.create_dataset('frames', data=self.frames)
        for dataName in self.smallDataNames:
            self.h5File.create_dataset(dataName, data=self.smallDataDict[dataName])
        self.h5File.close()
        print('%s closed' %self.output)  
        print('====================SUMMARY====================')
        print('Num of records : %d' %len(self.paths))


if __name__ == '__main__':
    smallData = SmallData(output='test.h5', smallDataNames=['scores', 'aas', 'bbs'])   # init smalldata with 3 customized fields: scores, aas, bbs
    smallData.addRecord('path1', 1, score=1, aa=12, bb=1)  # add one record using singular form keyword
    smallData.addRecords(['path1', 'path2'], [1, 2], scores=[3,4], aas=[1,2], bbs=[3,4])  # add multiple records using plural form keyword
    smallData.close()  # write to h5 file and close