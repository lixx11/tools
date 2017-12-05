"""
Calculate 7x7 rmsd matrix.

Usage:
    pymol -cq calc_rmsd_matrix.py -- config.yml
"""


import yaml
import sys
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def get_TM_sele(TM_list, ref_name, mob_name):
    assert len(TM_list) == 7  # 7 TM helix
    ref_TM_sele = {}
    mob_TM_sele = {}
    for i in range(7):
        TM = map(int, TM_list[i].split('-'))
        ref_TM_sele['TM%d' % (i+1)] = '/%s///%d-%d/c+ca+n' % (ref_name, TM[0], TM[1])
        mob_TM_sele['TM%d' % (i+1)] = '/%s///%d-%d/c+ca+n' % (mob_name, TM[0], TM[1])
    return ref_TM_sele, mob_TM_sele


# define object name
ref_name = 'ref'
mob_name = 'mob'

# load and parse configuration
config_file = sys.argv[1]
config = yaml.load(open(config_file, 'r'))

# load structure and trajectory
ref_mol = cmd.load(config['reference']['structure'], ref_name)
ref_resi = config['reference']['resi']
cmd.alter('/%s///%s' % (ref_name, ref_resi), 'chain=""')
cmd.alter('/%s///%s' % (ref_name, ref_resi), 'segi=""')
mob_mol = cmd.load(config['mobile']['structure'], mob_name)
mob_traj = config['mobile']['trajectory']
mob_interval = config['mobile']['interval']

print('loading trajectories...')
if mob_traj is not None:
    trajs = config['mobile']['trajectory']
    for i in range(len(trajs)):
        cmd.load_traj(trajs[i], mob_name, interval=mob_interval)
mob_resi = config['mobile']['resi']
cmd.alter('/%s///%s' % (mob_name, mob_resi), 'chain=""')
cmd.alter('/%s///%s' % (mob_name, mob_resi), 'segi=""')

# build TM selections
TM_list = config['TM']
ref_TM_sele, mob_TM_sele = get_TM_sele(TM_list, ref_name, mob_name)

# calculate rmsd matrix
n_frames = cmd.count_frames()
print('calculating RMSD matrix...')
RMSD_MATRIX = np.zeros((n_frames,7,7))
for fid in tqdm(range(n_frames)):
    for i in range(7):
        cmd.fit(mob_TM_sele['TM%d' % (i+1)], ref_TM_sele['TM%d' % (i+1)], mobile_state=(fid+1))
        for j in range(7):
            ref_xyz = cmd.get_coords(ref_TM_sele['TM%d' % (j+1)], 1)
            mob_xyz = cmd.get_coords(mob_TM_sele['TM%d' % (j+1)], fid+1)
            rms = np.sqrt(((ref_xyz - mob_xyz) ** 2).sum(axis=1).mean())
            RMSD_MATRIX[fid,i,j] = rms

# write npy array and create movie
np.save(os.path.join(config['dirname'], '%s.npy' % config['prefix']), RMSD_MATRIX)

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='RMSD MATRIX', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=5, metadata=metadata)
fig = plt.figure()
l = plt.imshow(RMSD_MATRIX[0,:,:], interpolation='nearest')
plt.title('1/%d' % n_frames)
plt.clim(0, 10)
plt.colorbar()
plt.tight_layout()

# plot ticks
yticks = ['TM%d' % (i+1) for i in range(7)]
xticks = ['TM%d-TM%d' % (i+1, i+1) for i in range(7)]
plt.yticks(np.arange(7), yticks)
plt.xticks(np.arange(7), xticks, fontsize=8)

offset = -0.3, 0.15

print('making movie...')
with writer.saving(fig, "%s.mp4" % config['prefix'], 100):
    for i in tqdm(range(n_frames)):
        l.set_data(RMSD_MATRIX[i,:,:])
        plt.title('%d/%d' % (i+1, n_frames))

        texts = []
        for row in range(7):
            for col in range(7):
                texts.append(plt.text(col+offset[0], row+offset[1], '%3.1f' % RMSD_MATRIX[i,row,col]))
        writer.grab_frame()
        for text in texts:
            text.remove()
