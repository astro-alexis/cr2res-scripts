import numpy as np
import glob
from astropy.io import fits
import os
A,B = [], []
fi = np.sort(glob.glob('CR*fits'))
for i in range(len(fi)):
    f = fits.open(fi[i])
    head = f[0].header
    isnod = head['HIERARCH ESO DPR TECH'].find('NODDING')
    if isnod>=0:
        nodpos = head["HIERARCH ESO SEQ NODPOS"].strip()
        if nodpos == "A": A.append(fi[i])
        elif nodpos == "B": B.append(fi[i])
        else: print(f[i]+" is not a nodding obs")

if len(A)!=len(B): print("A and B have different lengths", len(A), len(B))

listfile = open("filelist-raw-to-reduced.ascii", "w")
listfile.write('{:10s}\t{:30s}\t\t{:30s}\n'.format('pair','raw file','NODPOS'))
# Taking care of pairs
for i in range(len(A)):
    count=str(i).zfill(2)
    dirname = "pair"+count
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass
    soffile = dirname + "/" + dirname + ".sof"
    cal = open("calibs.sof", "r")
    with open(soffile, 'w') as sof:
       sof.write('../'+A[i]+'\t OBS_NODDING_OTHER\n')
       sof.write('../'+B[i]+'\t OBS_NODDING_OTHER\n')
       sof.writelines(cal.readlines())
    cal.close()
    listfile.write('{:10s}\t{:30s}\t{:1s}\n'.format(dirname,A[i],'A'))
    listfile.write('{:10s}\t{:30s}\t{:1s}\n'.format(dirname,B[i],'B'))

listfile.close()
# master spectrum

dirname="master"
try:
    os.mkdir(dirname)
except FileExistsError:
    pass
soffile = dirname + "/" + dirname + ".sof"
cal = open("calibs.sof", "r")
with open(soffile, 'w') as sof:
    for i in range(len(A)):
        sof.write('../'+A[i]+'\t OBS_NODDING_OTHER\n')
        sof.write('../'+B[i]+'\t OBS_NODDING_OTHER\n')
    sof.writelines(cal.readlines())
cal.close()

