"""
Usage:
    RAW_dat_parser.py <dat_files>... --ref=<ref_dat> [options]

Options:
    -h --help           Show this screen.
    --ref=<ref_dat>     Reference RAW dat file.
    --qmin=<qmin>       qmin to calculate scaling ratio [default: 0].
    --qmax=<qmax>       qmax to calculate scaling ratio [default: 1E10].
    --diff-mode=<diff_mode>     Plot difference in absolute scale or relative scale [default: relative].
"""

import numpy as np 
from docopt import docopt
import matplotlib.pyplot as plt
from scipy.stats import linregress


def load_RAW_dat(filepath):
    f = open(filepath, 'r')
    qs = []  # q
    Is = []  # intensity
    Es = []  # error
    while True:
        line = f.readline()
        if line:
            if line[:10] == '### HEADER':
                # print('processing HEADER section')
                pass
            elif line[:8] == '### DATA':
                # print('processing DATA section')
                for i in range(3):
                    f.readline()
                while True:
                    line = f.readline()
                    data = line.split()
                    if len(data) == 3:
                        qs.append(float(data[0]))
                        Is.append(float(data[1]))
                        Es.append(float(data[2]))
                    else:
                        break
        else:
            break
    qs = np.asarray(qs)
    Is = np.asarray(Is)
    Es = np.asarray(Es)
    return qs, Is, Es


if __name__ == '__main__':
    # add signal to enable CTRL-C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    argv = docopt(__doc__)
    ref_dat = argv['--ref']
    raw_dats = argv['<dat_files>']
    qmin = float(argv['--qmin'])
    qmax = float(argv['--qmax'])
    diff_mode = argv['--diff-mode']
    ref_qs, ref_Is, ref_Es = load_RAW_dat(ref_dat)
    qmin_id = np.argmin(np.abs(ref_qs - qmin))
    qmax_id = np.argmin(np.abs(ref_qs - qmax))
    data_list = []
    for raw_dat in raw_dats:
        filename = raw_dat.split('/')[-1]
        qs, Is, Es = load_RAW_dat(raw_dat)
        data_dict = {}
        data_dict['filename'] = filename
        data_dict['qs'] = qs 
        data_dict['Is'] = Is 
        data_dict['Es'] = Es 
        data_dict['abs_diff'] = Is - ref_Is
        data_dict['rel_diff'] = (Is - ref_Is) / ref_Is
        scaling_factor =  ref_Is[qmin_id:qmax_id].sum() / Is[qmin_id:qmax_id].sum()
        data_dict['scaling_Is'] = scaling_factor * Is
        data_dict['abs_scaling_diff'] = scaling_factor * Is - ref_Is
        data_dict['rel_scaling_diff'] = (scaling_factor * Is - ref_Is) / ref_Is
        data_list.append(data_dict)
        print('scaling factor for %s: %.3f' % (raw_dat, scaling_factor))

    # making plots
    fig1 = plt.figure()
    plt.subplot(221)
    for data in data_list:
        plt.semilogy(data['qs'], data['Is'], label=data['filename'])
    plt.legend()
    plt.title('SAXS profiles before scaling')
    plt.subplot(222)
    for data in data_list:
        plt.semilogy(data['qs'], data['scaling_Is'], label=data['filename'])
    plt.legend()
    plt.title('SAXS profiles after scaling')
    plt.subplot(223)
    for data in data_list:
        if diff_mode == 'absolute':
            plt.plot(data['qs'], data['abs_diff'], label=data['filename'])
        elif diff_mode == 'relative':
            plt.plot(data['qs'], data['rel_diff'], label=data['filename'])
    plt.legend()
    plt.title('%s difference before scaling' % diff_mode)
    plt.subplot(224)
    for data in data_list:
        if diff_mode == 'absolute':
            plt.plot(data['qs'], data['abs_scaling_diff'], label=data['filename'])
        elif diff_mode == 'relative':
            plt.plot(data['qs'], data['rel_scaling_diff'], label=data['filename'])
    plt.legend()
    plt.title('%s difference after scaling' % diff_mode)
    plt.show()
