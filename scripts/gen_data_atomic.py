import os
import numpy as np
import subprocess
import time
import distutils
from distutils.spawn import find_executable
import sys
import shutil
import itertools
import argparse
import collections

#nbos_array = [1, 5, 16]
nbos_array = [1,2,3,4,5,6,8,12,16]
#nbos_array = [8,12,16]
nbos_array = [3]
#nbos_array = [1,2,3,4,5,8,16]
#nbos_array = [6,5,4,3,2,1][::-1]
#nferm_array = [16,32,48,64]
nferm_array = [4,8,16,24,32,48,64,96,128,160]
#nferm_array=[16,24,32,48,64]
#nferm_array=[4,8,12]#16,24,32,48,64]
#nferm_array = [48]
#kpts_array = [4, 8, 16, 32]#, 32]
#kpts_array = [4,8,16,24,32,48,64]
kpts_array =[16]

U=20.0
beta = 1.0

dry_run = False # True

inp_section="atomic"
data_dir="data_atomic"
resume=False
plaintext=0
df_sc_iter = 150
df_sc_cutoff = 1e-8

def main():
    counter = 0
    origdir = os.path.abspath(os.getcwd())
    for nbos in nbos_array:
            for nferm in nferm_array:
                for kpts in kpts_array:
                    exec_dir=origdir + os.path.sep+data_dir+os.path.sep + "b" + str(nbos) + os.path.sep + "f" + str(nferm) + os.path.sep + "k" + str(kpts)
                    os.makedirs(exec_dir) if not os.path.exists(exec_dir) else None
                    os.chdir(exec_dir)
                    
                    prepare_string = ["generate_atomic", 
                        "--nfermionic", str(nferm),
                        "--nbosonic", str(nbos),
                        "--U", str(U),
                        "--beta", str(beta),
                        "--output", "atomic.h5",
                        "--plaintext", str(int(plaintext))
                    ] 
                    df_string = ["hub_df_cubic2d", 
                        "--input", "atomic.h5", 
                        "--output", "output.h5", 
                        "--inp_section", inp_section,
                        "--df_sc_iter", str(df_sc_iter), 
                        "--df_sc_cutoff", str(df_sc_cutoff),
                        "--nbosonic", str(nbos),
                        "--resume", str(int(resume)),
                        "--kpts", str(kpts),
                        "--plaintext", str(int(plaintext))
                    ]
                    if not dry_run:
                        print " ".join(prepare_string)
                        print subprocess.call(" ".join(prepare_string), shell=True)
                        print " ".join(df_string)
                        print subprocess.call(" ".join(df_string), shell=True)
                        
                    counter = counter + 1

                    os.chdir(origdir)

    print counter, "calcs ran"

if __name__ == "__main__":
    main()
