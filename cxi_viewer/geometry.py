import numpy as np 
import h5py
import os


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
  