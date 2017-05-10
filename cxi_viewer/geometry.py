import numpy as np 
import h5py
import os


def det2fourier(det_xy, wave_length, det_dist):
  """Detector 2d coordinates to fourier 3d coordinates
  
  Args:
      det_xy (TYPE): Description
      wave_length (TYPE): Description
      det_dist (TYPE): Description
  
  Returns:
      TYPE: 3d fourier coordinates in angstrom^-1
  
  """
  nb_xy = len(det_xy)
  det_dist = np.ones(nb_xy) * det_dist
  det_dist = np.reshape(det_dist, (-1, 1))
  q1 = np.hstack((det_xy, det_dist))
  q1_norm = np.sqrt(np.diag(q1.dot(q1.T)))
  q1_norm = q1_norm.reshape((-1, 1)).repeat(3, axis=1)
  q1 = q1 / q1_norm
  q0 = np.asarray([0., 0., 1.])
  q0 = q0.reshape((1,-1)).repeat(nb_xy, axis=0)
  q = 1. / wave_length * (q1 - q0)
  return q


def get_hkl(q, A=None, A_inv=None):
  """calculate hkl from q vectors
  
  Args:
      q (ndarray, [N, 3]): fourier vectors
      A (ndarray, [3, 3], optional): transformational matrix
      A_inv (ndarray, [3, 3], optional): inverse transformational matrix
  
  Returns:
      ndarray, [N, 3]: hkl
  """
  if A_inv is not None:
    hkl = A_inv.dot(q.T)
  else:
    assert A is not None  # must provide A or A_inv
    A_inv = np.linalg.inv(A)
    hkl = A_inv.dot(q.T)
  return hkl.T


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
    """
    Rearrange raw image to assembled pattern according to 
    the geometry setup.
    """
    new_img = np.zeros((self.nx, self.ny))
    new_img[self.x.ravel(), self.y.ravel()] = image.ravel()
    return new_img

  def map(self, pos):
    """
    Map raw position to assembled position
    """
    pos = np.int_(np.rint(pos))
    # map raw coorinates to assembled coordinates in meters
    peak_remap_x_in_m = self.geom_x[pos[1], pos[0]]
    peak_remap_y_in_m = self.geom_y[pos[1], pos[0]]
    # map assembled coordinates in meters to in pixels
    peak_remap_x_in_pixel = peak_remap_x_in_m / self.pixel_size
    peak_remap_x_in_pixel += self.offset_x
    peak_remap_y_in_pixel = peak_remap_y_in_m / self.pixel_size 
    peak_remap_y_in_pixel += self.offset_y
    return peak_remap_x_in_pixel, peak_remap_y_in_pixel

  def batch_map_in_m(self, raw_XYs):
    raw_XYs = np.int_(np.rint(raw_XYs))
    peak_remap_x_in_m = self.geom_x[raw_XYs[:,1], raw_XYs[:,0]]
    peak_remap_y_in_m = self.geom_y[raw_XYs[:,1], raw_XYs[:,0]]
    peak_remap_xy_in_m = np.vstack((
      peak_remap_x_in_m,
      peak_remap_y_in_m
      )).T
    return peak_remap_xy_in_m

  def load_geom(self, filename):
    """
    load geometry: x, y, z coordinates from cheetah, 
    crystfel or psana geom file
    """
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
  