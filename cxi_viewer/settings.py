"""
cxi viewer settings.
"""
import os


class Settings(object):
    def __init__(self, settings_dict):
        # main window
        self.workdir = settings_dict.get('work dir', os.path.dirname(__file__))
        self.data_location = settings_dict.get('data location', None)
        self.peak_info_location = settings_dict.get('peak info location', None)
        self.pixel_size = settings_dict.get('pixel size', 100)
        self.detector_distance = settings_dict.get('detector distance', 100)

    def __str__(self):
        attrs = dir(self)
        s = ''
        for attr in attrs:
            if attr[:2] != '__':
                s += '%s: %s\n' % (attr, getattr(self, attr))
        return s
