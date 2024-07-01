#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:12:27 2024

@author: bernard
"""

import os
import shutil
import subprocess
import time


first_frame = 0
last_frame = 2 #15
interval = 1
clear_output = False
use_avg_ext = False

ind = range(first_frame, last_frame, interval) #[1]
n0 = 1



# Initialize constants
EXE_DIR = '/Users/bernard/software/wass/dist/bin/'

# Set the paths here
DATA_ROOT = '/Users/bernard/airsea/data/wass/'
OUT_ROOT = '/Users/bernard/airsea/data/bernard/wass/'
EXP = 'BS_2013'
DATE = '2013-09-30_10-20-01_12Hz'
DATA = os.path.join(DATA_ROOT, EXP, DATE)

CONFIG_DIR = os.path.join(DATA, 'config')
INPUT_C0_DIR = os.path.join(DATA, 'input', 'cam1')
INPUT_C1_DIR = os.path.join(DATA, 'input', 'cam2')
FOUT = 'output'
FCONFIG = 'extrinsic_calibration'
MY_EXT = os.path.join(OUT_ROOT, EXP, DATE, FCONFIG)


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
MY_OUT = os.path.join(OUT_ROOT, EXP, DATE, FOUT)
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

# Sync the calculated avg extrinsics
if use_avg_ext:
    if ind[0] > n0:
        wdr = os.listdir(MY_EXT)
        os.system(f'rsync -avu {os.path.join(MY_EXT, wdr[2], "ext_*.xml")} {CONFIG_DIR}')
        print('Successfully copied the extrinsics to config folder')

# Run WASS prepare
print("---------------------------------------------------")
print("|                RUNNING wass_prepare             |")
print("---------------------------------------------------")

start_time = time.time()
for ii in ind:
    wd = input_frames[ii - 1]['wd']
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    cmd = f"{ENV_SET}{PREPARE_EXE} --workdir {wd} --calibdir {CONFIG_DIR} --c0 {input_frames[ii - 1]['Cam0']} --c1 {input_frames[ii - 1]['Cam1']}"
    subprocess.check_call(cmd, shell=True)
print("***************************************************")
print(f"Done in {time.time() - start_time} secs.")

# Run WASS stereo
print("---------------------------------------------------")
print("|                RUNNING wass_stereo              |")
print("---------------------------------------------------")

start_time = time.time()
for ii in ind:
    cmd = f"{ENV_SET}{STEREO_EXE} {os.path.join(CONFIG_DIR, 'stereo_config.txt')} {input_frames[ii - 1]['wd']}"
    subprocess.check_call(cmd, shell=True)
print("***************************************************")
print(f"Done in {time.time() - start_time} secs.")
