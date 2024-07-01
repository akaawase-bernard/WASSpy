#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:21:04 2024
@author: bernard akaawase
"""

import os
import shutil
import subprocess
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing
import glob
from os import path
import cv2 as cv
import wasspy as wp


num_processors = 2  # Number of processors to use
first_frame = 1 #starts from 1 not 0
last_frame = 8 #15
interval = 1
clear_output = False
use_avg_ext = False
cal_median_plane = True
make_plots = True
n0 = 1
VERSION = "1.0.0"
baseline = 1.87
resolution = 0.2 # 0.2

# Initialize constants
EXE_DIR = '/Users/bernard/software/wass/dist/bin/'

# # Set the paths here
# DATA_ROOT = '/Users/bernard/airsea/data/wass/'
# OUT_ROOT = '/Users/bernard/airsea/data/bernard/wass/'
# EXP = 'BS_2013'
# DATE = '2013-09-30_10-20-01_12Hz'
# DATA = os.path.join(DATA_ROOT, EXP, DATE)


# Set the paths here
DATA_ROOT = '../tests/'
OUT_ROOT = DATA_ROOT
DATA = DATA_ROOT

CONFIG_DIR = os.path.join(DATA, 'config')
INPUT_C0_DIR = os.path.join(DATA, 'input', 'cam1')
INPUT_C1_DIR = os.path.join(DATA, 'input', 'cam2')
FOUT = 'output_full'
FCONFIG = 'extrinsic_calibration'

#MY_EXT = os.path.join(OUT_ROOT, EXP, DATE, FCONFIG) #main data
MY_EXT = os.path.join(OUT_ROOT, FCONFIG)

def run_wass(ii):
    if os.name == 'nt':  # Windows
        EXE_SUFFIX = '.exe'
        ENV_SET = ''
    else:  # Unix/Linux/MacOS
        EXE_SUFFIX = ''
        ENV_SET = 'LD_LIBRARY_PATH="" && '

    PREPARE_EXE = os.path.join(EXE_DIR, f'wass_prepare{EXE_SUFFIX}')
    STEREO_EXE = os.path.join(EXE_DIR, f'wass_stereo{EXE_SUFFIX}')

    # Set & make output directory
    #MY_OUT = os.path.join(OUT_ROOT, EXP, DATE, FOUT)
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
        if ii > n0:
            wdr = os.listdir(MY_EXT)
            os.system(f'rsync -avu {os.path.join(MY_EXT, wdr[2], "ext_*.xml")} {CONFIG_DIR}')
            print('Successfully copied the extrinsics to config folder')

    # Run WASS prepare
    print("---------------------------------------------------")
    print("|                RUNNING wass_prepare             |")
    print("---------------------------------------------------")

    start_time = time.time()
    wd = input_frames[ii]['wd']
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    cmd = f"{ENV_SET}{PREPARE_EXE} --workdir {wd} --calibdir {CONFIG_DIR} --c0 {input_frames[ii]['Cam0']} --c1 {input_frames[ii]['Cam1']}"
    subprocess.check_call(cmd, shell=True)
    print("***************************************************")
    print(f"Done in {time.time() - start_time} secs.")

    # Run WASS stereo
    print("---------------------------------------------------")
    print("|                RUNNING wass_stereo              |")
    print("---------------------------------------------------")

    start_time = time.time()
    cmd = f"{ENV_SET}{STEREO_EXE} {os.path.join(CONFIG_DIR, 'stereo_config.txt')} {input_frames[ii]['wd']}"
    subprocess.check_call(cmd, shell=True)
    print("-----------------------------------------")
    print(f"Done in {time.time() - start_time} secs.")


        
        


#--------------------------------------- multi processing -----------------------------------------------------

def plotter(num, workdirs, baseline, resolution, MY_OUT):
    outdir = os.path.join(MY_OUT, 'figs')
    os.makedirs(outdir, exist_ok=True)

    # WASS xyz loading
    iw, jw, x, y, zp = wp.load_camXYZ(MY_OUT, n=num, baseline=baseline, checks=0)

    # the entire data
    xmax = np.nanmax(x)
    xmin = np.nanmin(x)
    ymax = np.nanmax(y)
    ymin = np.nanmin(y)

    dx = resolution
    dy = resolution
    xvector = np.arange(xmin, xmax, dx)
    yvector = np.arange(ymin, ymax, dy)

    # run the ffgridding function
    X, Y, Z = wp.ffgrid(x, y, zp, xvector, yvector, dx, dy)
    x_gridded = np.squeeze(X.flatten())
    y_gridded = np.squeeze(Y.flatten())
    z_gridded = np.squeeze(Z.flatten())
    ind_nonans = np.where(~np.isnan(z_gridded))  # indices that are free from nans

    # run our interp fc
    interpolator_z = wp.interpolate(x_gridded[ind_nonans], y_gridded[ind_nonans], z_gridded[ind_nonans])
    ind_nan = np.where(np.isnan(z_gridded))
    Zi = z_gridded
    Zi[ind_nan] = interpolator_z(x_gridded[ind_nan], y_gridded[ind_nan])
    Zi = Zi.reshape(np.shape(X))

    # Load image
    Img = cv.imread(path.join(workdirs[num], "undistorted", "00000001.png"), cv.IMREAD_ANYCOLOR)

    # Plot the data
    wp.plots_wass_data(X, Y, Zi, num, outdir, Img, iw, jw, zp)

def calculate_median_plane():
    if cal_median_plane:
        print('Now calculating the median plane from the estimated planes')
        #MY_OUT = os.path.join(OUT_ROOT, EXP, DATE, FOUT)
        MY_OUT = os.path.join(OUT_ROOT, FOUT)

        fname = os.path.join(MY_OUT, 'median_plane.txt')
        fname_all = os.path.join(MY_OUT, 'all_planes.txt')
        
        workdirs = sorted(glob.glob(MY_OUT + '/*_wd'))
        all_planes = []
        
        for item in range(len(workdirs)):
            planefile = os.path.join(workdirs[item], "plane.txt")
            if os.path.exists(planefile):
                plane = np.loadtxt(planefile)
                assert len(plane) == 4, "Plane must be a 4-element vector"
                all_planes.append(plane)
        
        all_planes = np.asarray(all_planes)
        median_plane = np.nanmedian(all_planes, axis=0)   # average by column all_planes
        
        np.savetxt(fname, median_plane, delimiter=", ", fmt='%s')
        np.savetxt(fname_all, all_planes, delimiter=", ", fmt='%s')
        print('Saving the median plane')
        
        
        greater_count = np.sum(all_planes > median_plane, axis=0)
        
        rows_greater_than_median = np.sum(np.all(all_planes > median_plane, axis=1))
        print("Medians:", median_plane)
        print("Elements greater than the median:", greater_count)
        print("All elements are greater than the corresponding column median:", rows_greater_than_median)

def main():
    print("WASS processing v.", VERSION)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nCopyright (C) UCONN AirSea-Lab 2024 \n")
    
    iterable = list(np.arange(first_frame, last_frame, interval))
    
    pool = multiprocessing.Pool(num_processors)
    with Pool(num_processors) as pool:
        pool.map(run_wass, iterable)
    
    calculate_median_plane()
    
    if make_plots:
        #MY_OUT = os.path.join(OUT_ROOT, EXP, DATE, FOUT)
        MY_OUT = os.path.join(OUT_ROOT, FOUT)

        workdirs = sorted(glob.glob(MY_OUT + '/*_wd'))
        
        nums = range(len(workdirs))
        if nums:
            with Pool(num_processors) as pool:
                pool.starmap(plotter, [(num, workdirs, baseline, resolution, MY_OUT) for num in nums])
    #wp.confirm_wass_processing(OUT_DIR, first_frame, last_frame)

    return iterable

if __name__ == "__main__":
    main()
    
    
