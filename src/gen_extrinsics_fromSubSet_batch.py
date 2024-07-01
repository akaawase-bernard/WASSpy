#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:37:15 2024

@author: bernard
"""


import os
import shutil
import subprocess
import time
import numpy as np
from multiprocessing import pool
import multiprocessing
import glob
from os import path
import wasspy as wp

VERSION = "1.0.0"

num_processors = 2  # Number of processors to use
first_frame = 1
last_frame = 9 #21551 #15
interval = 2# 50 #1
clear_output = False
auto_avg_ext = False #True always wass to cal. the avg ext for you. False if you want to see how each frame ext aries
n0 = 1  # The first frame where we start analyzing data

# Initialize constants
EXE_DIR = '/Users/bernard/software/wass/dist/bin/'



# Set the paths here
DATA_ROOT = '../tests/'
OUT_ROOT = DATA_ROOT
DATA = DATA_ROOT

# # Set the paths here ... the main paths.. uncomment for main data
# DATA_ROOT = '/Users/bernard/airsea/data/wass/'
# OUT_ROOT = '/Users/bernard/airsea/data/bernard/wass/'
# EXP = 'BS_2013'
# DATE = '2013-09-30_10-20-01_12Hz'
# DATA = os.path.join(DATA_ROOT, EXP, DATE)
#MY_OUT = os.path.join(OUT_ROOT, EXP, DATE, FOUT)

CONFIG_DIR = os.path.join(DATA, 'config')
INPUT_C0_DIR = os.path.join(DATA, 'input', 'cam1')
INPUT_C1_DIR = os.path.join(DATA, 'input', 'cam2')
FOUT = 'extrinsic_calibration_test'


if os.name == 'nt':  # Windows
    EXE_SUFFIX = '.exe'
    ENV_SET = ''
else:  # Unix/Linux/MacOS
    EXE_SUFFIX = ''
    ENV_SET = 'LD_LIBRARY_PATH="" && '

PREPARE_EXE = os.path.join(EXE_DIR, f'wass_prepare{EXE_SUFFIX}')
MATCH_EXE = os.path.join(EXE_DIR, f'wass_match{EXE_SUFFIX}')
AUTOCAL_EXE = os.path.join(EXE_DIR, f'wass_autocalibrate{EXE_SUFFIX}')
STEREO_EXE = os.path.join(EXE_DIR, f'wass_stereo{EXE_SUFFIX}')

# Set & make output directory
MY_OUT = os.path.join(OUT_ROOT, FOUT)
os.makedirs(MY_OUT, exist_ok=True)
OUT_DIR = MY_OUT

# Sanity checks
def check_path_exists(path, is_file=False):
    if is_file:
        assert os.path.isfile(path), f"{path} does not exist."
    else:
        assert os.path.isdir(path), f"{path} does not exist."

check_path_exists(EXE_DIR)
check_path_exists(PREPARE_EXE, is_file=True)
check_path_exists(MATCH_EXE, is_file=True)
check_path_exists(AUTOCAL_EXE, is_file=True)
check_path_exists(STEREO_EXE, is_file=True)
check_path_exists(DATA)
check_path_exists(CONFIG_DIR)
check_path_exists(INPUT_C0_DIR)
check_path_exists(INPUT_C1_DIR)

# List frames
input_frames = []

cam0_frames = sorted(f for f in os.listdir(INPUT_C0_DIR) if os.path.isfile(os.path.join(INPUT_C0_DIR, f)))
for i, frame in enumerate(cam0_frames):
    if os.path.getsize(os.path.join(INPUT_C0_DIR, frame)) > 0:
        input_frames.append({
            'Cam0': os.path.join(INPUT_C0_DIR, frame),
            'wd': os.path.join(OUT_DIR, f"{i:06d}_wd/")
        })

cam1_frames = sorted(f for f in os.listdir(INPUT_C1_DIR) if os.path.isfile(os.path.join(INPUT_C1_DIR, f)))
for i, frame in enumerate(cam1_frames):
    if os.path.getsize(os.path.join(INPUT_C1_DIR, frame)) > 0:
        input_frames[i]['Cam1'] = os.path.join(INPUT_C1_DIR, frame)

print(f"{len(input_frames)} stereo frames found.")

# Prepare output directory
if clear_output and os.path.isdir(OUT_DIR):
    print(f"{OUT_DIR} already exists, removing it.")
    shutil.rmtree(OUT_DIR)
print(f"Creating {OUT_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)

def run_command(cmd):
    print(f"Running command: {cmd}")
    result = subprocess.call(cmd, shell=True)
    assert result == 0, 'Component exited with non-zero return code'




def run_extSubset(ind):

    # Run WASS prepare
    print("RUNNING wass_prepare")
    start_time = time.time()
    for ii in ind:
        wd = input_frames[ii - 1]['wd']
        if os.path.isdir(wd):
            shutil.rmtree(wd)
        cmd = f"{ENV_SET}{PREPARE_EXE} --workdir {wd} --calibdir {CONFIG_DIR} --c0 {input_frames[ii - 1]['Cam0']} --c1 {input_frames[ii - 1]['Cam1']}"
        run_command(cmd)
    print(f"Done in {time.time() - start_time} secs.")
    
    # Run WASS match
    print("RUNNING wass_match")
    start_time = time.time()
    for ii in ind:
        cmd = f"{ENV_SET}{MATCH_EXE} {os.path.join(CONFIG_DIR, 'matcher_config.txt')} {input_frames[ii - 1]['wd']}"
        run_command(cmd)
    print(f"Done in {time.time() - start_time} secs.")
    
    # Run WASS autocalibrate
    print("RUNNING wass_autocalibrate")
    start_time = time.time()
    workspaces_file = os.path.join(OUT_DIR, 'workspaces.txt')
    # mode = 'w' if ind[0] == n0 else 'a+'
    # with open(workspaces_file, mode) as fid:
    #     for ii in ind:
    #         fid.write(f"{input_frames[ii - 1]['wd']}\n")
    # cmd = f"{ENV_SET}{AUTOCAL_EXE} {workspaces_file}"
    # run_command(cmd)
    # print(f"Done in {time.time() - start_time} secs.")
    
    if auto_avg_ext:
        # Avoid auto-averaging the extrinsic calibration
        with open(os.path.join(OUT_DIR, 'workspaces.txt'), 'w') as fid:
            fid.write(f"{input_frames[ind]['wd']}\n")
    else:
        # Create workspaces file for auto-averaging .. 
        mode = 'w' if ind[0] == n0 else 'a+'
        with open(os.path.join(OUT_DIR, 'workspaces.txt'), mode) as fid:
            for ii in ind:
                fid.write(f"{input_frames[ii]['wd']}\n")

#--------------------------------------- multi processing -----------------------------------------------------


def main():
    
    print("WASS processing v.", VERSION)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nCopyright (C) UCONN AirSea-Lab 2024 \n")
    
    iterable = np.arange(first_frame, last_frame, interval)
    iterable = [[int(i)] for i in iterable]  # Wrap each index in a list

    pool = multiprocessing.Pool(num_processors)
    pool.map(run_extSubset, iterable)

    #cal. the median extrinsics & plot
    parent_dir = OUT_DIR
    filename1 = 'ext_T.xml'
    filename2 = 'ext_R.xml'

    wp.cal_median_extT_and_plot(parent_dir, filename1) #T
    wp.median_matrix_and_plot_histograms_extR(parent_dir, filename2) #R
    print('All Done!')
    
    pool.close()
    pool.join()
    

    return iterable

if __name__ == "__main__":
    main()


