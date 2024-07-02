        import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib widget
import sys
import glob
import cv2 as cv
from os import path
import numpy as np
import colorama #pip install colorama
colorama.init()
from os import path
import struct

import scipy
from scipy.interpolate import LinearNDInterpolator
from scipy import ndimage
from scipy.stats import binned_statistic
import pickle
from scipy.stats import binned_statistic_2d


def interpolate(X,Y,Z):
    '''
    creates an interpolator
    '''
    from scipy import interpolate
    Xl = list(X.flatten())
    Yl = list(Y.flatten())
    Zl = list(Z.flatten())
    interpolator=interpolate.LinearNDInterpolator(np.array([Xl,Yl]).T,Zl)
    return interpolator

def parse_matrix_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = root.find(".//data").text.split()
    rows = int(root.find(".//rows").text)
    cols = int(root.find(".//cols").text)
    data = np.array(data, dtype=float).reshape((rows, cols))
    return data

def write_matrix_to_xml(matrix, xml_path):
    rows, cols = matrix.shape
    lines = [
        '<?xml version="1.0"?>',
        '<opencv_storage>',
        '<ext_R type_id="opencv-matrix">',
        f'  <rows>{rows}</rows>',
        f'  <cols>{cols}</cols>',
        '  <dt>d</dt>',
        '  <data>',
    ]

    # Flatten the matrix and split the data across multiple lines as required
    flat_matrix = matrix.flatten()
    for i in range(0, len(flat_matrix), 2):
        line_data = ' '.join(f'{value:.16e}' for value in flat_matrix[i:i+2])
        lines.append('    ' + line_data)

    lines.append('  </data>')
    lines.append('</ext_R>')
    lines.append('</opencv_storage>')

    with open(xml_path, 'w') as f:
        f.write('\n'.join(lines))

def plot_histograms(matrices, save_dir):
    matrices = np.array(matrices)
    rows, cols = matrices.shape[1], matrices.shape[2]
    
    for i in range(rows):
        for j in range(cols):
            data = matrices[:, i, j]
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            gradient = np.gradient(data)

            plt.figure(figsize=(15, 7))

            # Histogram subplot
            plt.subplot(1, 3, 1)
            plt.hist(data, bins=100, alpha=0.7, color='blue')
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=5, label=f'Mean: {mean_val:.6f}')
            plt.axvline(median_val, color='green', linestyle='dashed', linewidth=3, label=f'Median: {median_val:.6f}')
            plt.title(f'Histogram of element [{i},{j}]')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.text(0.95, 0.45, f'STD: {std_val:.6f}', verticalalignment='top', 
                     horizontalalignment='right', transform=plt.gca().transAxes, color='black', fontsize=15, bbox=dict(facecolor='white', alpha=0.7))

            # Time series subplot
            plt.subplot(1, 3, 2)
            plt.plot(data, marker='+', linestyle='', color='k')
            plt.title(f'Time series of element [{i},{j}]')
            plt.xlabel('Frame index')
            plt.ylabel('Value')
            plt.ylim(min(data), max(data))

            # Gradient subplot
            plt.subplot(1, 3, 3)
            plt.plot(gradient, marker='*', linestyle='', color='r')
            plt.title(f'Gradient of element [{i},{j}]')
            plt.xlabel('Frame index')
            plt.ylabel('Gradient')
            plt.ylim(min(gradient), max(gradient))

            # Save the combined plot
            save_path = os.path.join(save_dir, f'plot_element_{i}_{j}.png')
            plt.tight_layout()
            plt.savefig(save_path)
            #plt.close()
            plt.show()
def median_matrix_and_plot_histograms_extR(parent_dir, filename):
    matrices = []
    
    # Iterate over all folders in the parent directory
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        if os.path.isdir(folder_path):
            xml_path = os.path.join(folder_path, filename)
            if os.path.exists(xml_path):
                matrix = parse_matrix_from_xml(xml_path)
                matrices.append(matrix)
    
    # Calculate mean matrix and save it
    if matrices:
        #mean_matrix = np.mean(matrices, axis=0)
        median_matrix = np.median(matrices, axis=0)

        #output_file = os.path.join(parent_dir, f'mean_{filename}')
        output_file = os.path.join(parent_dir, f'median_{filename}')

        #write_matrix_to_xml(mean_matrix, output_file)
        write_matrix_to_xml(median_matrix, output_file)

        print(f'Mean matrix for {filename} saved to {output_file}')
        
        # Plot histograms and time series for each element
        hist_dir = os.path.join(parent_dir, 'plots_ext_R')
        os.makedirs(hist_dir, exist_ok=True)
        plot_histograms(matrices, hist_dir)
        print(f'Plots saved to {hist_dir}')
    else:
        print(f'No {filename} files found in any folder.')



def cal_median_extT_and_plot(parent_dir, filename):
    

    matrices = []
    # Iterate over all folders in the parent directory
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        if os.path.isdir(folder_path):
            xml_path = os.path.join(folder_path, filename)
            if os.path.exists(xml_path):
                # Parse the matrix from the XML file
                tree = ET.parse(xml_path)
                root = tree.getroot()
                data = list(map(float, root.find(".//data").text.split()))
                rows = int(root.find(".//rows").text)
                cols = int(root.find(".//cols").text)
                matrix = np.array(data).reshape((rows, cols))
                matrices.append(matrix)
    
    # Calculate mean matrix and save it if there are matrices
    if matrices:
        matrices = np.array(matrices)
        #mean_matrix = np.mean(matrices, axis=0)
        median_matrix = np.median(matrices, axis=0)
    
        # Create XML content for the mean matrix
        rows, cols = median_matrix.shape

        lines = [
            '<?xml version="1.0"?>',
            '<opencv_storage>',
            '<ext_T type_id="opencv-matrix">',
            f'  <rows>{rows}</rows>',
            f'  <cols>{cols}</cols>',
            '  <dt>d</dt>',
            '  <data>',
        ]
    
        # Flatten the matrix and split the data across multiple lines as required
        #flat_matrix = mean_matrix.flatten()
        flat_matrix = median_matrix.flatten()
    
        for i in range(0, len(flat_matrix), 3):
            line_data = ' '.join(f'{value:.16e}' for value in flat_matrix[i:i+3])
            lines.append('    ' + line_data)
    
        lines.append('  </data>')
        lines.append('</ext_T>')
        lines.append('</opencv_storage>')
    
        #output_file = os.path.join(parent_dir, f'mean_{filename}')
        output_file = os.path.join(parent_dir, f'median_{filename}')
    
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
    
        print(f'Mean matrix for {filename} saved to {output_file}')
    
        # Plot histograms and time series for each element
        hist_dir = os.path.join(parent_dir, 'plots_ext_T')
        os.makedirs(hist_dir, exist_ok=True)
    
        for i in range(rows):
            for j in range(cols):
                data = matrices[:, i, j]
                mean_val = np.mean(data)
                std_val = np.std(data)
                median_val = np.median(data)
                gradient = np.gradient(data)
    
                plt.figure(figsize=(15, 7))
    
                # Histogram subplot
                plt.subplot(1, 3, 1)
                plt.hist(data, bins=100, alpha=0.7, color='blue')
                plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=5, label=f'Mean: {mean_val:.4f}')
                plt.axvline(median_val, color='green', linestyle='dashed', linewidth=3, label=f'Median: {median_val:.4f}')
                plt.title(f'Histogram of element [{i},{j}]')
                plt.xlabel('Value')
                plt.ylabel('Count')
                plt.legend()
                plt.text(0.95, 0.45, f'STD: {std_val:.4f}', verticalalignment='center', 
                         horizontalalignment='right', transform=plt.gca().transAxes, color='black', fontsize=25, bbox=dict(facecolor='white', alpha=0.7))
    
                # Time series subplot
                plt.subplot(1, 3, 2)
                plt.plot(data, marker='o', linestyle='-', color='lime')
                plt.title(f'Time series of element [{i},{j}]')
                plt.xlabel('Frame index')
                plt.ylabel('Value')
                plt.ylim(min(data), max(data))
    
                # Gradient subplot
                plt.subplot(1, 3, 3)
                plt.plot(gradient, marker='o', linestyle='-', color='r')
                plt.title(f'Gradient of element [{i},{j}]')
                plt.xlabel('Frame index')
                plt.ylabel('Gradient')
                plt.ylim(min(gradient), max(gradient))
    
                # Save the combined plot
                save_path = os.path.join(hist_dir, f'plot_element_{i}_{j}.png')
                plt.tight_layout()
                plt.savefig(save_path)
                #plt.close()
                plt.show()
    
        print(f'Plots saved to {hist_dir}')
    else:
        print(f'No {filename} files found in any folder.')



def confirm_wass_processing(base_dir, first_frame, last_frame):
    
    missing_files_info = []
    
    # List of required files
    required_files = [
        "00000000_features.png", "00000000_s.png", "00000001_features.png", "00000001_s.png",
        "Cam0_poseR.txt", "Cam0_poseT.txt", "Cam1_poseR.txt", "Cam1_poseT.txt",
        "disparity_coverage.jpg", "disparity_final_scaled.png", "disparity_stereo_ouput.png",
        "ext_R.xml", "ext_T.xml", "graph_components.jpg", "intrinsics_00000000.xml",
        "intrinsics_00000001.xml", "K0_small.txt", "K1_small.txt", "matcher_stats.csv",
        "matches_epifilter.png", "matches.png", "matches.txt", "mesh_cam.xyzC",
        "P0cam.txt", "P1cam.txt", "plane_refinement_inliers.xyz", "plane.txt",
        "scale.txt", "stereo_input.jpg", "stereo.jpg", "undistorted", "wass_stereo_log.txt"
    ]
    
    # Iterate through all working directories from 000000_wd to 21577_wd
    for i in range(first_frame, last_frame):
        # Format the directory name with leading zeros
        dir_name = f"{i:06d}_wd"
        dir_path = os.path.join(base_dir, dir_name)
        
        # Check if the directory exists
        if os.path.isdir(dir_path):
            missing_files = []
            # Check for each required file
            for file_name in required_files:
                file_path = os.path.join(dir_path, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)
            
            # If there are missing files, record the directory and the missing files
            if missing_files:
                missing_files_info.append({
                    'directory': dir_name,
                    'missing_files': missing_files
                })
        else:
            missing_files_info.append({
                'directory': dir_name,
                'missing_files': required_files
            })
    
    # Output the result
    if missing_files_info:
        print(f"Number of incomplete directories: {len(missing_files_info)}")
    else:
        print("The processing ran successfully without any missing files.")


def run_interpolation(X,Y,Z):
    '''
    creates an interpolator
    '''
    from scipy import interpolate
    Xflat = list(X.flatten())
    Yflat = list(Y.flatten())
    Zflat = list(Z.flatten())
    interpolator=interpolate.LinearNDInterpolator(np.array([Xflat,Yflat]).T,Zflat)
    return interpolator
    

def load_camera_mesh( meshfile ):
    with open(meshfile, "rb") as mf:
        npts = struct.unpack( "I", mf.read( 4 ) )[0]
        limits = np.array( struct.unpack( "dddddd", mf.read( 6*8 ) ) )
        Rinv = np.reshape( np.array(struct.unpack("ddddddddd", mf.read(9*8) )), (3,3) )
        Tinv = np.reshape( np.array(struct.unpack("ddd", mf.read(3*8) )), (3,1) ) 
                
        data = np.reshape( np.array( bytearray(mf.read( npts*3*2 )), dtype=np.uint8 ).view(dtype=np.uint16), (3,npts), order="F" )
        
        mesh_cam = data.astype( np.float32 )
        mesh_cam = mesh_cam / np.expand_dims( limits[0:3], axis=1) + np.expand_dims( limits[3:6], axis=1 );
        mesh_cam = Rinv@mesh_cam + Tinv;
    
        return mesh_cam
    
    
def compute_sea_plane_RT( plane  ):
    assert len(plane)==4, "Plane must be a 4-element vector"
    a=plane[0]
    b=plane[1]
    c=plane[2]
    d=plane[3];
    q = (1-c)/(a*a + b*b)
    R=np.array([[1-a*a*q, -a*b*q, -a], [-a*b*q, 1-b*b*q, -b], [a, b, c] ] )
    T=np.expand_dims( np.array([0,0,d]), axis=1)
    
    return R, T
    


def align_on_sea_plane_RT( mesh, R, T ):
    # Rotate, translate
    mesh_aligned = R@mesh + T;
    # Invert z axis
    mesh_aligned[2,:]*=-1.0;
    return mesh_aligned


def align_on_sea_plane( mesh, plane ):
    assert mesh.shape[0]==3, "Mesh must be a 3xN numpy array"
    R,T = compute_sea_plane_RT( plane )
    return align_on_sea_plane_RT(mesh, R,T)


def filter_mesh_outliers( mesh_aligned, ransac_inlier_threshold=0.2, debug=True ):
    def filter_plane_ransac( pts, inlier_threshold = 0.2 ):
        """ Return the plane outliers
        """
        assert ptsR.shape[0] == 3, "Points must be a 3xN array"
        Npts = ptsR.shape[1]
        #print("Running RANSAC on %d points"%Npts)
        best_inliers = np.zeros(Npts)
        best_n_inliers = 0
        best_N = None
        best_P = None

        if Npts>3:
            for iit in range(100):
                p0 = ptsR[:, np.random.randint(0,Npts-1) ]
                p1 = ptsR[:, np.random.randint(0,Npts-1) ]
                p2 = ptsR[:, np.random.randint(0,Npts-1) ]
                N = np.cross( p1-p0, p2-p0 )
                Nnorm = np.linalg.norm(N)
                if Nnorm < 1E-5:
                    continue

                N = N / Nnorm
                P = np.expand_dims( np.mean( np.vstack( [p0,p1,p2]), axis=0 ), axis=1 )
                distances = ptsR - P
                distances = np.abs( distances[0,:]*N[0] + distances[1,:]*N[1] + distances[2,:]*N[2] )
                inliers = distances < inlier_threshold
                curr_n_inliers = np.sum(inliers)
                if curr_n_inliers > best_n_inliers:
                    best_n_inliers = curr_n_inliers
                    best_inliers = inliers
                    best_N = N
                    best_P = P

        return (1-best_inliers).astype(np.bool)

    from scipy.spatial import KDTree
    tree = KDTree( mesh_aligned[0:2,:].T )
    np.random.seed( None )
    votes = np.zeros( mesh_aligned.shape[1] )
    processed = np.zeros( mesh_aligned.shape[1] )
    limits_x = [ np.amin( mesh_aligned[0,:] ), np.amax( mesh_aligned[0,:] ) ]
    limits_y = [ np.amin( mesh_aligned[1,:] ), np.amax( mesh_aligned[1,:] ) ]
    xx,yy = np.meshgrid( np.linspace(limits_x[0],limits_x[1],15), np.linspace(limits_y[0],limits_y[1],15) )
    scan_pts = np.vstack( [xx.flatten(), yy.flatten() ] )

    for ii in range( scan_pts.shape[1]):
        randpt = scan_pts[:,ii]
        pts_idx = np.array( tree.query_ball_point( (randpt[0],randpt[1]),0.5 ) )
        
        if len(pts_idx)>0:
            processed[ pts_idx ] += 1
            ptsR = mesh_aligned[:,pts_idx]
            outliers = filter_plane_ransac(ptsR, inlier_threshold=ransac_inlier_threshold )
            outliers_idx = pts_idx[outliers]
            votes[outliers_idx] += 1
    mesh_aligned_filtered = mesh_aligned[:, votes<1]

    if debug:
        import matplotlib.pyplot as plt
        plt.figure( figsize=(10,10) )
        plt.scatter( mesh_aligned[0,:], mesh_aligned[1,:], c=votes)
        plt.colorbar()
        plt.figure( figsize=(10,10) )
        plt.scatter( mesh_aligned[0,:], mesh_aligned[1,:], c=processed)
        plt.colorbar()
        
    return mesh_aligned_filtered

    
def ffgrid(x,y,z, xvec, yvec, dx,dy):

    from scipy.stats import binned_statistic
    """
    The input:
        The function recieves ungridded x, y, z as 1-D arrays, 
        generates a grid of resolution specified by the user as 
        xvec & yvec. Note, xvec & yvec should be center bins.

    The ouput: 
        The gridded Z on the X,Y space 
    """

    X,Y = np.meshgrid(xvec, yvec)

    #establish edge bins
    x_edge = xvec - (dx/2.)
    x_edge = np.append(x_edge, max(x_edge)+ dx)

    y_edge = yvec - (dy/2.)
    y_edge = np.append(y_edge, max(y_edge)+ dy)

    #call binning function
    ret = binned_statistic_2d(x, y, z, 'mean', bins=[x_edge, y_edge], 
        expand_binnumbers=False)

    Z = ret.statistic
    Z = Z.T #need to transpose to match the meshgrid orientation with is (J,I)
    return X, Y, Z 



def teri():
    """
    A custom cmap that goes from white to red.
    """
    import matplotlib.colors as mcolors

    upper = plt.cm.jet(np.arange(256))
    lower = np.ones((int(256/4), 4))
    for i in range(3):
        lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])
    cmap = np.vstack((lower, upper))
    
    # Convert to matplotlib colormap
    cmap = mcolors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
    return cmap


def load_camXYZ(workdir, n, baseline =2.545, checks=1):
    '''
    input: the working directory, as generated by during 3d reconstruction of sea surface eg with 
            WASS. This typically lives in the out directory. The should be of the 
            form 0000023_wd
    output: brings out the x,y,z and the projection in pixel. if checks is set as 1, plots 
            will be made.    
    '''


    #print("Looking for WASS reconstructed stereo frames in ",workdir )
    wass_frames = glob.glob( path.join(workdir, "*_wd" ))
    wass_frames.sort()
    #print("%d frames found."%len(wass_frames))
  
    planefile = path.join( workdir,  "median_plane.txt" ) #use median plane
    #planefile = path.join( wass_frames[0],  "plane.txt" ) #use the first frame

    medianplane = None
    if path.exists( planefile ):
        #print("Found! Loading...")
        medianplane = np.loadtxt( planefile )
        #meanplane = np.mean( all_planes, axis=0)
        #meanplane = np.median( all_planes, axis=0)
    
    meshname = path.join(wass_frames[n],"mesh_cam.xyzC")
    #print("Loading ", meshname )
    
    R = np.loadtxt(path.join(wass_frames[n],'Cam0_poseR.txt'))
    T = np.loadtxt(path.join(wass_frames[n],'Cam0_poseT.txt'))
    P0Cam =  np.vstack( (np.loadtxt( path.join(wass_frames[n],"P0cam.txt"))  ,[0, 0, 0, 1] ) )
    P1Cam =  np.vstack( (np.loadtxt( path.join(wass_frames[n],"P1cam.txt"))  ,[0, 0, 0, 1] ) )

    I = cv.imread(  path.join(wass_frames[n],"undistorted","00000000.png"), cv.IMREAD_ANYCOLOR )
    Iw,Ih = I.shape[1],I.shape[0]
    if Iw is None or Ih is None:
        #print("Unable to determine the camera image size. Please set it manually with -Iw,-Ih program arguments")
        sys.exit(-3)

    Rpl, Tpl = compute_sea_plane_RT( medianplane )
    mesh = load_camera_mesh(meshname)
    mesh_aligned = align_on_sea_plane( mesh, medianplane) * baseline
    
    Ri = Rpl.T
    Ti = -Rpl.T@Tpl
    RTplane = np.vstack( (np.hstack( (Ri,Ti) ),[0,0,0,1]) )
    toNorm = np.array( [[ 2.0/Iw, 0     , -1, 0],
                        [ 0     , 2.0/Ih, -1, 0],
                        [ 0,      0,       1, 0],
                        [ 0,      0,       0, 1]], dtype=np.float64 )

    SCALEi = 1.0/baseline
    P0plane = toNorm @ P0Cam @ RTplane @ np.diag((SCALEi,SCALEi,-SCALEi, 1))
    Npts = len(mesh[2,:]);
    my_ones = np.ones(len(mesh[2,:]))
    #my_ones = np.ones(len(mesh_aligned[2,:]))
    mesh_reshaped = np.vstack([mesh,my_ones])
    #mesh_reshaped = np.vstack([mesh_aligned,my_ones])
    
    P1 = np.resize(P1Cam,(3,4))
    pt2d = np.matmul(P1,mesh_reshaped);
    pt2d = pt2d / np.tile( pt2d[2,:],(3,1));
    
    plane_reshaped = np.tile(medianplane,(1,Npts) )
    my_plane = np.resize(plane_reshaped, (4,Npts))
    #elevations = mesh_aligned[2,::50]
    elevations = mesh_aligned[2,:]

    if checks:
        plt.figure(figsize=(11,8))
        plt.imshow(I, cmap = 'gray', origin = 'upper')
        c= plt.scatter(pt2d[0,:],pt2d[1,:],
                       c=mesh_aligned[2,:], s =0.02, 
                       cmap= 'jet',
                        vmin=-1.5, vmax = 1.5, 
                       marker = '.')
        
        plt.title('Frame ' + str(n))
        c=plt.colorbar(c)
        c.set_label('elevation (m)')
        plt.show()
        
    i = pt2d[0,:]
    j = pt2d[1,:]
    x = mesh_aligned[0,:]
    y = mesh_aligned[1,:]

    
    
    return i, j, x, y, elevations


def load_xyz_and_map2IJ(wass_frame, fidstr, plane, baseline =2.5, distance = 60, xcentre = 0, ycentre = -30, N = 1024, resolution=0.2, checks=0):
    from scipy import interpolate
    '''
    This is the latest version. The code loads and maps xyz to image plane at your region of choosing only.
    
    input: the working directory, as generated by during 3d reconstruction of sea surface eg with 
            WASS. This typically lives in the out directory. The should be of the 
            form 0000023_wd
            
            This function uses a mean plane file provided as input to the function.
            loads xyx,removes plane, put xyzp to regular grid and then maps to IJ
            
    output: brings out the x,y,z and the projection in pixel. if checks is set as 1, plots 
            will be made. 
    '''

    #ycentre = -30 #the y distance is -ve from camera
    meshname = path.join(wass_frame,"mesh_cam.xyzC")
    print("Loading ", meshname )

    P1Cam =  np.vstack( (np.loadtxt( path.join(wass_frame,"P1cam.txt"))  ,[0, 0, 0, 1] ) )
    I = cv.imread(  path.join(wass_frame,"undistorted","00000001.png"), cv.IMREAD_ANYCOLOR )


    xyz = load_camera_mesh(meshname)   
    assert len(plane)==4, "Plane must be a 4-element vector"
    a= plane[0]
    b= plane[1]
    c= plane[2]
    d= plane[3];
    q = (1-c)/(a*a + b*b)
    Rpl=np.array([[1-a*a*q, -a*b*q, -a], [-a*b*q, 1-b*b*q, -b], [a, b, c] ] )
    Tpl=np.expand_dims( np.array([0,0,d]), axis=1)
    
    #remove the plane
    assert xyz.shape[0]==3, "Mesh must be a 3xN numpy array"    
    xyzp = (Rpl@xyz + Tpl)*baseline
    
    #extract the wass xyz
    x = xyzp[0,:]
    y = xyzp[1,:]
    zp = xyzp[2,:]

    #consider wass nc_extent
    # xmax = xcentre + distance/2
    # xmin = xcentre - distance/2
    # ymax = ycentre + distance/2
    # ymin = ycentre - distance/2

    #the entire data
    xmax = np.nanmax(x)
    xmin = np.nanmin(x)
    ymax = np.nanmax(y)
    ymin = np.nanmin(y)

    #xvec , yvec are centre bins
    xvector = np.linspace(xmin,xmax,N)
    yvector = np.linspace(ymin,ymax,N)

    # dx = abs(np.mean(np.diff(xvector)))
    # dy = abs(np.mean(np.diff(yvector)))
    
    dx = resolution
    dy = resolution
    xvector = np.arange(xmin,xmax,dx)
    yvector = np.arange(ymin,ymax,dy)


    #run the ffgridding function
    xx, yy, zz = ffgrid(x, y, zp, xvector, yvector, dx, dy)
    x_gridded = np.squeeze(xx.flatten())
    y_gridded = np.squeeze(yy.flatten())
    z_gridded = np.squeeze(zz.flatten())
    ind_nonans = np.where(~np.isnan(z_gridded)) #indices that are free from nans


    #run our interp fc
    interpolator_z = interpolate(x_gridded[ind_nonans], y_gridded[ind_nonans], z_gridded[ind_nonans])
    ind_nan = np.where(np.isnan(z_gridded))
    Zi = z_gridded
    Zi[ind_nan] = interpolator_z(x_gridded[ind_nan], y_gridded[ind_nan])
    Zi = Zi.reshape(np.shape(xx))

    if checks:
        fig = plt.figure(figsize=(8,5) )
        plt.pcolor( xx,yy,Zi, cmap='RdBu', vmin =-1.2, vmax =1.2)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title('Frame ' + fidstr)
        #figfile = path.join(outdir,"interpolated_data.png" )
        #fig.savefig(figfile,bbox_inches='tight')
        
        

    XYZi = np.vstack([x_gridded, y_gridded, Zi.flatten()])
    elevations = XYZi[2,:]*-1
    
    XYZinv =  np.linalg.inv(Rpl)@((XYZi/baseline) - Tpl)
    my_ones = np.ones(len(XYZinv[2,:]))
    XYZinv_reshaped = np.vstack([XYZinv,my_ones])

    #PROJECTS TO IJ (image plane)
    P1 = np.resize(P1Cam,(3,4))
    pt2di = np.matmul(P1,XYZinv_reshaped);
    pt2di = pt2di / np.tile( pt2di[2,:],(3,1));

    if checks:
        plt.figure(figsize=(11,5))
        plt.imshow(I, cmap = 'gray')
        c= plt.scatter(pt2di[0,:],pt2di[1,:],
                       c=elevations, s =0.03, 
                       cmap= 'jet',
                        vmin=-1., vmax = 1., 
                       marker = '.')

        plt.title('Frame ' + fidstr)
        c=plt.colorbar(c)
        c.set_label('elevation (m)')
    
    ii = pt2di[0,:] #x pixels values
    jj = pt2di[1,:] #y pixels values
    xx = XYZi[0,:]
    yy = XYZi[1,:]
    zp = elevations
    return ii, jj, xx, yy, zp





def plots_wass_data(X, Y, Zi, num, outdir, I, iw, jw, zp):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot the interpolated frame on the first subplot
    ax = axs[0]
    c1 = ax.pcolor(X, Y, Zi, cmap='RdBu_r', vmin=-1., vmax=1.)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-50, -10)
    ax.set_xlabel('X (m)')  
    ax.set_ylabel('Y (m)')  
    ax.invert_yaxis()
    fig.colorbar(c1, ax=ax)
    ax.set_title('Interpolated: Frame ' + str(num).zfill(6))
    
    # Plot the raw frame on the second subplot
    ax = axs[1]
    ax.imshow(I, cmap='gray', origin='upper')
    c2 = ax.scatter(iw, jw, c=zp, s=0.03, cmap='RdBu_r', vmin=-1., vmax=1., marker='.')
    ax.set_xlim(0, I.shape[1])
    ax.set_ylim(I.shape[0], 0)  
    ax.set_title('Raw: Frame ' + str(num).zfill(6))
    ax.set_xlabel('I (px)')  
    ax.set_ylabel('J (px)') 
    cbar = fig.colorbar(c2, ax=ax)
    cbar.set_label('elevation (m)')
    
    # Save the figure
    figfile = path.join(outdir, "wass_data_" + str(num).zfill(6) + ".png")
    fig.savefig(figfile, bbox_inches='tight')
    plt.close()





def grid_interpolate_wassXYZ(wass_frame, num, all_planes, 
                           baseline =2.5, distance = 70, xcentre = 0, 
                           ycentre = -30, N = 1024, checks=0):
    '''
    The function loads the raw xyz, removes the mean plane, bin grid and interpolate to 20cm spatial resolation.
    it returns the gridded XYZ only. 
    
    NB: y distance is -ve from camera
    '''

    meshname = path.join(wass_frame,"mesh_cam.xyzC")
    print("Loading ", meshname )
    P1Cam =  np.vstack( (np.loadtxt( path.join(wass_frame,"P1cam.txt"))  ,[0, 0, 0, 1] ) )
 
    xyz = load_camera_mesh(meshname)   
    assert len(all_planes)==4, "Plane must be a 4-element vector"
    a=all_planes[0]
    b=all_planes[1]
    c=all_planes[2]
    d=all_planes[3];
    q = (1-c)/(a*a + b*b)
    Rpl=np.array([[1-a*a*q, -a*b*q, -a], [-a*b*q, 1-b*b*q, -b], [a, b, c] ] )
    Tpl=np.expand_dims( np.array([0,0,d]), axis=1)
    
    #remove the plane
    assert xyz.shape[0]==3, "Mesh must be a 3xN numpy array"    
    xyzp = (Rpl@xyz + Tpl)*baseline
    mesh_aligned = xyzp
    
    #extract the wass xyz
    x = xyzp[0,:]
    y = xyzp[1,:]
    zp = xyzp[2,:]


    xmax = xcentre + distance/2
    xmin = xcentre - distance/2
    ymax = ycentre + distance/2
    ymin = ycentre - distance/2

    #xvec , yvec are centre bins
    xvector = np.linspace(xmin,xmax,N)
    yvector = np.linspace(ymin,ymax,N)
    dx = abs(np.mean(np.diff(xvector)))
    dy = abs(np.mean(np.diff(yvector)))

    #print(f'The dx is : {dx}')
    #print(f'The dy is : {dy}')

    #run the ffgridding function
    xx, yy, zz = ffgrid(x, y, zp, xvector, yvector, dx, dy)
    x_gridded = np.squeeze(xx.flatten())
    y_gridded = np.squeeze(yy.flatten())
    z_gridded = np.squeeze(zz.flatten())
    ind_nonans = np.where(~np.isnan(z_gridded)) #indices that are free from nans
    #tqdm.write("Interpolating... ", end="")

    #run our interp fc
    interpolator_z = run_interpolation(x_gridded[ind_nonans], y_gridded[ind_nonans], z_gridded[ind_nonans])
    ind_nan = np.where(np.isnan(z_gridded))
    Zi = z_gridded
    Zi[ind_nan] = interpolator_z(x_gridded[ind_nan], y_gridded[ind_nan])
    ZZ = Zi.reshape(np.shape(xx))
    

    #ZZ = scipy.ndimage.median_filter(Zi, size=7)
    if 0:
        fig = plt.figure( figsize=(10,10))
        plt.imshow( ZZ, vmin=-1, vmax=1)
        #figfile = path.join(outdir,"points.png" )
        plt.title('point')
        #fig.savefig(figfile,bbox_inches='tight')
        #plt.close()


    if checks:
        fig = plt.figure( figsize=(10,6))
        plt.pcolormesh(xx,yy, ZZ, cmap = 'RdBu_r', vmin = -1.2, vmax = 1.2, shading='gouraud')
        #figfile = path.join(outdir,"interpolated_data.png" )
        plt.gca().invert_yaxis()
        plt.title('Frame ' + str(num), fontsize = 20)
        plt.xlabel('X (m)', fontsize = 20)
        plt.ylabel('Y (m)', fontsize = 20)
        c =plt.colorbar()
        plt.gca().set_aspect('equal')
        c.set_label("Elevations (m)", fontsize=20)
        plt.gca().tick_params(axis='x', labelsize=16)
        plt.gca().tick_params(axis='y', labelsize=16)
        c.ax.tick_params(labelsize=13)
    #print('Done')  
    if 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        
        # Plot the original data
        im1 = axes[0].scatter(x, y, c=zp,s = 0.1, cmap='RdBu_r', vmin=-1.2, vmax=1.2)
        axes[0].invert_yaxis()
        axes[0].set_title('Raw', fontsize=20)
        axes[0].set_xlabel('X (m)', fontsize=20)
        axes[0].set_ylabel('Y (m)', fontsize=20)
        c1 = fig.colorbar(im1, ax=axes[0])
        c1.set_label("Elevations (m)", fontsize=20)
        axes[0].set_aspect('equal')
        axes[0].tick_params(axis='x', labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        #c1.ax.tick_params(labelsize=13)
        
        # Plot the data after scipy median filter
        im2 = axes[1].pcolormesh(xx, yy, ZZ, cmap='RdBu_r', vmin=-1.2, vmax=1.2, shading='gouraud')
        axes[1].invert_yaxis()
        axes[1].set_title('Interpolated', fontsize=20)
        axes[1].set_xlabel('X (m)', fontsize=20)
        axes[1].set_ylabel('Y (m)', fontsize=20)
        c2 = fig.colorbar(im2, ax=axes[1])
        c2.set_label("Elevations (m)", fontsize=20)
        axes[1].set_aspect('equal')
        axes[1].tick_params(axis='x', labelsize=16)
        axes[1].tick_params(axis='y', labelsize=16)
        c2.ax.tick_params(labelsize=13)

        plt.tight_layout()
        plt.show()

    return xx, yy, ZZ


def status(name):
    return f"The wasspy installation went well, {name}!"
    
