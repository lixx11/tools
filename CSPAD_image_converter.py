#!/opt/local/bin/python2.7

from pylab import *
import h5py
from scipy.interpolate import griddata
from psgeom import camera

# set file path
# geom_psana = '0-end.data'
# geom_crystfel = 'cspad-jun16-SSC-nz1.geom' # geom from Nadia
geom_crystfel = 'Dsd-best.geom' # geom from Tom Grant
# geom_h5 = 'cheetah_geom_move_quadv1-2.h5'
geom_h5 = '.temp.h5'
# powder_pattern = 'virtual_powder_max.npy'
powder_pattern = '/Users/lixuanxuan/playground/index/peak_intensity.npy'
# powder_pattern = 'r0022-detector0-class1-sum.h5'
# set linear or log display
display_scale = 'linear' # 'linear' or 'log'
# set virtual detector parameter
padding = 10.E-3 # add 10mm padding 
ds = 0.11E-3 # pixel size: 0.11 mm
# output
output = 'powder'

vmax = 1.
vmin = 0.

# convert psana machine geometry to cheetah geometry
# cspad = camera.Cspad.from_psana_file(geom_psana)
cspad = camera.Cspad.from_crystfel_file(geom_crystfel)
cspad.to_cheetah_file(geom_h5)

# load data
geom_f = h5py.File(geom_h5, 'r')
# powder_f = h5py.File(powder_pattern, 'r')
x, y, z = geom_f['x'].value, geom_f['y'].value, geom_f['z'].value
x = x.reshape(-1)
y = y.reshape(-1)
z = z.reshape(-1)
xyz = np.zeros((x.shape[0], 3))
xyz[:,0] = x
xyz[:,1] = y 
xyz[:,2] = z 
# nframes = powder_f['data']['nframes'].value
# sum_p = powder_f['data']['correcteddata'].value / nframes
sum_p = np.load(powder_pattern)
sum_p_1d = sum_p.reshape(-1)

# calculate new pattern coorodinates
x_range = x.max() - x.min()
y_range = y.max() - y.min()
xx_range = x_range + 2 * padding
yy_range = y_range + 2 * padding

xx_size = xx_range // ds
yy_size = yy_range // ds 

_xx = np.arange(xx_size) * ds  - xx_range / 2.
_yy = np.arange(yy_size) * ds  - yy_range / 2.

xx, yy = np.meshgrid(_xx, _yy)
center_x = np.where(np.abs(xx) == np.abs(xx).min())[1][0]
center_y = np.where(np.abs(yy) == np.abs(yy).min())[0][0]
center = np.asarray((center_x, center_y))
interp_data = griddata(xyz[:,0:2], sum_p_1d, (xx, yy), method='linear', fill_value=0)

# plot powder pattern before re-assemble
fig1 = plt.figure()
if display_scale == 'linear':
    plt.imshow(sum_p)
elif display_scale == 'log':
    plt.imshow(np.log(np.abs(sum_p) + 1.)) # plus 1 to avoid divided by 0
else:
    print("ERROR! Undefined Display Style: %s" %display_scale)
plt.title('before re-assemble')
plt.clim(vmin, vmax)
plt.show()

# plot powder pattern after re-assemble
fig2 = plt.figure()
if display_scale == 'linear':
    plt.imshow(interp_data)
elif display_scale == 'log':
    plt.imshow(np.log(np.abs(interp_data) + 1.))
else:
    print("ERROR! Undefined Display Style: %s" %display_scale)
plt.title('after re-assemble')
# plt.clim(vmin, vmax)
plt.show()

geom_f.close()
np.savez(output, 
         powder_p = interp_data,
         center = center)