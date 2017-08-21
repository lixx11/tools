"""
Calculate 7x7 rmsd matrix.

Usage:
    pymol -cq calc_rmsd_matrix.py -- config.yml
"""


import yaml
import sys
from tqdm import tqdm

import numpy as np


def get_TM_sele(TM_list, ref_name, mob_name):
    assert len(TM_list) == 7  # 7 TM helix
    ref_TM_sele = {}
    mob_TM_sele = {}
    for i in range(7):
        TM = map(int, TM_list[i].split('-'))
        ref_TM_sele['TM%d' % (i+1)] = '/%s//A/%d-%d/c+ca+n' % (ref_name, TM[0], TM[1])
        mob_TM_sele['TM%d' % (i+1)] = '/%s//A/%d-%d/c+ca+n' % (mob_name, TM[0], TM[1])
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
cmd.alter('/%s///%s' % (ref_name, ref_resi), 'chain="A"')
mob_mol = cmd.load(config['mobile']['structure'], mob_name)
mob_traj = config['mobile']['trajectory']
mob_interval = config['mobile']['interval']
if mob_traj is not None:
    cmd.load_traj(mob_traj, mob_name, interval=mob_interval)
mob_resi = config['mobile']['resi']
cmd.alter('/%s///%s' % (mob_name, mob_resi), 'chain="A"')

# build TM selections
TM_list = config['TM']
ref_TM_sele, mob_TM_sele = get_TM_sele(TM_list, ref_name, mob_name)

# calculate rmsd matrix
n_frames = cmd.count_frames()
print('%d frames to process...' % n_frames)
RMSD_MATRIX = np.zeros((n_frames,7,7))
for fid in tqdm(range(n_frames)):
    # cmd.frame(fid+1)
    for i in range(7):
        cmd.fit(mob_TM_sele['TM%d' % (i+1)], ref_TM_sele['TM%d' % (i+1)], mobile_state=(fid+1))
        for j in range(7):
            ref_xyz = cmd.get_coords(ref_TM_sele['TM%d' % (j+1)], 1)
            mob_xyz = cmd.get_coords(mob_TM_sele['TM%d' % (j+1)], fid+1)
            rms = np.sqrt(((ref_xyz - mob_xyz) ** 2).sum(axis=1)).mean()
            RMSD_MATRIX[fid,i,j] = rms
np.save('rmsd.npy', RMSD_MATRIX)