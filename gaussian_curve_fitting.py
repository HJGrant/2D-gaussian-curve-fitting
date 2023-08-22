import numpy as np
import rawpy
import cv2
import platform
import os
from pathlib import Path
import matplotlib.pyplot as plt
from camera_imaging_pipeline.src.image_processing import imageProcessing
import json
from skimage.measure import block_reduce
from skimage.feature import peak_local_max
from skimage import img_as_float, data
from camera_imaging_pipeline.utils.helpers import scale_data
from scipy.signal import correlate
from scipy.spatial import distance
from scipy.ndimage import sobel
from scipy import ndimage as ndi
from scipy.stats import norm, fit
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import dictances
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from random import choice
import warnings
import csv

file_name = 'P5050051.ORF'
#data_path = Path("data/test_set")
data_path = Path("data/FSL-01D_Fred")
params_path = Path("./camera_imaging_pipeline/params1.json")
processor = imageProcessing()

labels_path = Path("label_data/FSL-01D_Fred_07_23.csv")
label_data = []

# opening the CSV file
with open(labels_path, mode ='r')as file:
    # reading the CSV file
    csvFile = csv.DictReader(file)
    #getting coordinates
    for lines in csvFile:
        label_data.append(lines)
        
 

#a function for displaying cuts of an image taken from a certain coloumn and row 
def image_cuts(img, display=False):

        x_cut = img[:, int(img.shape[1]/2)]
        y_cut = img[int(img.shape[0]/2), :]
        x_vals = np.linspace(0, img.shape[0], img.shape[0])
        y_vals = np.linspace(0, img.shape[1], img.shape[1])

        if display == True:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(y_vals, x_cut)
            axs[1].plot(x_vals, y_cut)

            plt.figure("Image")
            plt.imshow(img)

            plt.show()
            return x_cut, y_cut
        
        else: 
            return x_cut, y_cut

#displaying a surface plot of the intensity values of an image
def display_3D_plot(arr):
     
    # Load and format data
    z = arr
    nrows, ncols = arr.shape
    x = np.linspace(0, ncols, ncols)
    y = np.linspace(0, nrows, nrows)
    x, y = np.meshgrid(x, y)

    # region = np.s_[5:50, 5:50]
    # x, y, z = x[region], y[region], z[region]

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)

    plt.show()

#a function for displaying a tile
def display_tile(tile1, tile2=None, title=None):
     
    # if tile2 == None:
    #     std_div = np.sqrt(np.var(tile1))
    #     mean_val = np.mean(tile1)

    #     fig = plt.figure("Tile")
    #     plt.imshow(tile1, cmap='gray')
    #     plt.title(f'Standard Deviation is: {std_div}, and the mean is: {mean_val}')
    #     plt.show()
        
    # elif tile2.all() != None:

    # std_div_1 = np.sqrt(np.var(tile1))
    # mean_val_1 = np.mean(tile1)

    # std_div_2 = np.sqrt(np.var(tile2))
    # mean_val_2 = np.mean(tile2)

    #x = np.linspace(-400, 400, 800)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    axs[0].imshow(tile1, cmap='gray')
    axs[1].imshow(tile2, cmap='gray')
    axs[1].plot(25, 25, 'r.')
    #plt.imshow(tile1, cmap='gray')
    plt.show()

def bhatta_dist(X1, X2, method='continuous'):
    #Calculate the Bhattacharyya distance between X1 and X2. X1 and X2 should be 1D numpy arrays representing the same
    # feature in two separate classes. 

    def get_density(x, cov_factor=0.1):
        #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
        density = gaussian_kde(x)
        density.covariance_factor = lambda:cov_factor
        density._compute_covariance()
        return density

    #Combine X1 and X2, we'll use it later:
    cX = np.concatenate((X1,X2))

    if method == 'noiseless':
        ###This method works well when the feature is qualitative (rather than quantitative). Each unique value is
        ### treated as an individual bin.
        uX = np.unique(cX)
        A1 = len(X1) * (max(cX)-min(cX)) / len(uX)
        A2 = len(X2) * (max(cX)-min(cX)) / len(uX)
        bht = 0
        for x in uX:
            p1 = (X1==x).sum() / A1
            p2 = (X2==x).sum() / A2
            bht += np.sqrt(p1*p2) * (max(cX)-min(cX))/len(uX)

    elif method == 'hist':
        ###Bin the values into a hardcoded number of bins (This is sensitive to N_BINS)
        N_BINS = 10
        #Bin the values:
        h1 = np.histogram(X1,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        h2 = np.histogram(X2,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        #Calc coeff from bin densities:
        bht = 0
        for i in range(N_BINS):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1*p2) * (max(cX)-min(cX))/N_BINS

    elif method == 'autohist':
        ###Bin the values into bins automatically set by np.histogram:
        #Create bins from the combined sets:
        # bins = np.histogram(cX, bins='fd')[1]
        bins = np.histogram(cX, bins='doane')[1] #Seems to work best
        # bins = np.histogram(cX, bins='auto')[1]

        h1 = np.histogram(X1,bins=bins, density=True)[0]
        h2 = np.histogram(X2,bins=bins, density=True)[0]

        #Calc coeff from bin densities:
        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1*p2) * (max(cX)-min(cX))/len(h1)

    elif method == 'continuous':
        ###Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):
        N_STEPS = 200
        #Get density functions:
        d1 = get_density(X1)
        d2 = get_density(X2)
        #Calc coeff:
        xs = np.linspace(min(cX),max(cX),N_STEPS)
        bht = 0
        for x in xs:
            p1 = d1(x)
            p2 = d2(x)
            bht += np.sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS

    else:
        raise ValueError("The value of the 'method' parameter does not match any known method")

    ###Lastly, convert the coefficient into distance:
    if bht==0:
        return float('Inf')
    else:
        return -np.log(bht)

#a function for returning a set of tiles based on list of coordinates
def get_tiles(img, coordinates):
    #takes in gray-scale image
    #get 20 by 20 tiles that are centred around the points in coordinates
    N = 50
    M = 50
    buf = []
    coords_buf = []

    print(coordinates.shape)

    img_tiles = [img[int(x-M/2):int(x+M/2),int(y-N/2):int(y+N/2)] for x,y in coordinates]
    
    for i, tile in enumerate(img_tiles):
        if tile.shape == (N, M):
            buf.append(tile)
            coords_buf.append(coordinates[i])

    coordinates = np.asarray(coords_buf)
    img_tiles = np.asarray(buf)
    print(img_tiles.shape)
    
    return img_tiles, coordinates

#a function for finding the tile that contains the laser
def get_best_tile(laser_tile, img_tiles):
    potential_tile_list = []
    tile_index = []
    min_dist = 0.1 
    # laser_mean = 102.0475
    # std_div_min = 77.0
    # std_div_max = 80.0
    laser_mean = 54.7477
    std_div_min = 54
    std_div_max = 56.5
    i = 0

    for tile in tqdm(img_tiles):
        img_tile = np.asarray(tile)

        if (35, 35) != img_tile.shape:
                i+=1
                continue
        
        #dist = distance.directed_hausdorff(laser_tile, img_tile)[0]
        std_div = np.sqrt(np.var(img_tile))
        mean_val = np.mean(img_tile)

        if  std_div_min < std_div and std_div < std_div_max and 51 < mean_val < 55:
            std_div_opt = std_div
            potential_tile_list.append(img_tile)
            tile_index.append(i)

        i += 1

    return potential_tile_list, tile_index

#take in coordinates, and search the proximity of those coordinates for other high intensity values
#adjust the coordinates to be at the center of that cluster of high intensity values
def center_coordinates(image, coordinates):
    new_coordinates = []
    area = []
    r = 20              #radius of the area to check for max vals
    d = r * 2           #diameter of the area to check for max vals
    y_offset = 0
    x_offset = 0
     
    for coordinate in coordinates:
        area_buf = []
        x = coordinate[0]
        y = coordinate[1]
        for j in range(d):
            for i in range(d):
                area.append(image[x + j - r, y + i - r])
                if image[x+ j - r, y + i - r] > 70:        #115
                     area_buf.append([x + j - r, y + i - r])

        if len(area_buf) == 0 or len(area_buf) == 1:
            for j in range(d):
                for i in range(d):
                    area.append(image[x + j - r, y + i - r])
                    if image[x+ j - r, y + i - r] > 30:        #115
                        area_buf.append([x + j - r, y + i - r])

        if len(area_buf) == 0:
            centroid_x = x
            centroid_y = y 
            new_coordinates.append([centroid_x, centroid_y])
            area_buf = [new_coordinates[0]]
            return new_coordinates, area_buf

        area_buf = np.asarray(area_buf)
        centroid_x = area_buf[:, 0].sum() / len(area_buf[:, 0]) + y_offset
        centroid_y = area_buf[:, 1].sum() / len(area_buf[:, 1]) + x_offset
        new_coordinates.append([centroid_x, centroid_y])

    return new_coordinates, area_buf

#function for getting coordiantes from labels or from computing the local maxima
def get_coordinates(img=False, file=False, label_data=False):
    #read in a rgb image or a .csv file for getting the coordinates of the laser
    #if a image is read in, then we compute all the local maxima of the image

    if img.any():
        print('computing local maxima')
        coordinates = peak_local_max(img[:, :, 0], threshold_abs=170, min_distance=10)

    elif label_data:
        print('reading in laser coordinates from data')
        for label in label_data:
            label_path = Path(label['name'])
            img_path = Path(file)
            if img_path.stem == label_path.stem:
                coordinates = [[int(label['laser.y']), int(label['laser.x'])]]
                break

    return coordinates

#take in an image, and return the tiles at certain coordinates in the image
def get_laser_tiles(data_path, fit):
    #lists to store each tile and it's x and y cut values
    x_cuts = []
    y_cuts = []
    new_tiles_list = []
    coordinates = []

    for file in tqdm(os.listdir(data_path.as_posix())):
       
        #read in and process the image
        filepath = data_path.joinpath(file)
        params = json.load(open(params_path))
        img, _ = processor.applyToImage(filepath, params)

        #display_3D_plot(img[:, :, 0])

        #make a gloabl copy of the image for use in other function
        global img_copy 
        img_copy = img.copy()

        coordinates = get_coordinates(img)


        # plt.imshow(img[:, :, 0], cmap='gray')
        # plt.plot(coordinates[:, 1], coordinates[:, 0], 'g.')
        # plt.show()

        #image_max = ndi.maximum_filter(img, size=20, mode='constant')

        #apply a filter to find all of the local maxima in an image and draw a red point on them 
        #then calculate the centroid of the laser dot, and adjust the coordinates so that they are positioned in the center of the laser
        #coordinates = peak_local_max(img[:, :, 0], threshold_abs=170, min_distance=10)
        image_tiles, _  = get_tiles(img[:,:,0], coordinates)

        new_coordinates, area_coords= center_coordinates(img[:, :, 0], coordinates[975:985])
        new_coordinates = np.asarray(new_coordinates)


        if len(area_coords) == 1:
            print('Skipping this tile')
            continue

        plt.imshow(img[:, :, 0], cmap='gray')
        plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        plt.plot(new_coordinates[:, 1], new_coordinates[:, 0], 'g.')
        plt.show()
        

        #get the image tiles based on the local maxima and then compute the tiles that match the laser tile the most
        #image_tiles  = get_tiles(img[:,:,0], [[232, 3920], [324, 458]])
        new_tiles, coordinates = get_tiles(img[:, :, 0], new_coordinates)

        #best_tiles, index = get_best_tile(laser_tile, image_tiles)
        ideal_fit_parameters = np.load('gaussian_parameters.npy')
        print(ideal_fit_parameters)
        print("\n")
        # [  2.73491993  -2.25390338  33.33195819  21.63807409 255.        ]

        for i, tile in tqdm(enumerate(new_tiles)):
            #dist = distance.directed_hausdorff(fit, tile)
            #print(tile.shape)
            try:
                tile_fit, tile_params = curve_fitting_2D(tile)
            except:
                #print("can't create fit for tile")
                continue
            
            # if np.abs(np.abs(tile_params[0]) - np.abs(ideal_fit_parameters[0])) > 0.5 or np.abs(np.abs(tile_params[1]) - np.abs(ideal_fit_parameters[1])) > 0.5:
            #     continue
            
            # if np.abs(np.abs(tile_params[2]) - np.abs(ideal_fit_parameters[2])) > 10 or np.abs(np.abs(tile_params[3]) - np.abs(ideal_fit_parameters[3])) > 10:
            #     continue

            # if tile_params[4] < 200 or tile_params[4] > 265:
            #     continue

            dist = bhatta_dist(fit.flatten(), tile_fit.flatten(), method='continuous') 
            # print(dist)
            #tile[tile<50] = 0
            print(tile_fit.shape)
            print(fit.shape)
            display_tile(fit, tile_fit, str(f'Bhatta dist is: {dist}'))
            print(tile_params)

            point = coordinates[i]

            plt.imshow(img[:, :, 0], cmap='gray')
            plt.plot(point[1], point[0], 'r.')
            plt.show()
        
        #laser_tiles_list.append(new_tiles[0]) 

    return laser_tiles_list

#2D gaussian function
def gaussian(x, y, x0, y0, xalpha, yalpha, A):
    return A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

#callable gaussian that understands raveled data
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += gaussian(x, y, *args[i*5:i*5+5])

    return arr

#function for fitting a 2D gaussian to a matrix
def curve_fitting_2D(tile):
    N = 50
    M = 50

    x, y = np.linspace(-25, 25, N), np.linspace(-25, 25, M)
    X, Y = np.meshgrid(x, y)

    guess_prms = [(0, 0, 1, 1, 255)]
    # Flatten the initial guess parameter list.
    p0 = [p for prms in guess_prms for p in prms]
    #print(p0)

    xdata = np.vstack((X.ravel(), Y.ravel()))

    popt, pcov = curve_fit(_gaussian, xdata, tile.ravel(), p0)
    fit = np.zeros(tile.shape)
    for i in range(len(popt)//5):
        fit += gaussian(X, Y, *popt[i*5:i*5+5])
    #print('Fitted parameters:')
    #print(popt)

    #display_3D_plot(tile)
    #display_3D_plot(fit)

    return fit, popt


#take in a list of tiles with lasers, and compute the fitted curve for each. Then compute the average of all the computed curves. 
#return the gaussian parameters for plotting an ideal gaussian function. 
def get_ideal_gaussian_fit(tiles_list):
    xdata = np.linspace(-25, 25, 50)
    x0_list = []
    x_sigma_list = []
    y0_list = []
    y_sigma_list = []

    for tile in tiles_list:
        try:
            x0, y0, x_sigma, y_sigma, A = curve_fitting_2D(tile)
            x0_list.append(x0)
            y0_list.append(y0)
            x_sigma_list.append(x_sigma)
            y_sigma_list.append(y_sigma)
        except:
            print('optimal parameters not found')
            continue

        #transform the arrays to numpy array for further processing
        x0_list = np.asarray(x0_list)
        x_std_dev_list = np.asarray(x_sigma_list)
        y0_list = np.asarray(y0_list)
        y_std_dev_list = np.asarray(y_sigma_list)

        for i, val in enumerate(y_std_dev_list):
            if val > 100: 
                y_std_dev_list[i] = 1
            if val < -100:
                y_std_dev_list[i] = 1

        for i, val in enumerate(y0_list):
            if val < -50:
                y0_list[i] = 0
            if val > 50:
                y0_list[i] = 0

        for i, val in enumerate(x0_list):
            if val < -50:
                x0_list[i] = 0
            if val > 50:
                x0_list[i] = 0

        #calculate the sample mean and std dev 
        x_mean = x0_list.sum() / len(x0_list)
        x_std_dev = x_std_dev_list.sum() / (len(x_std_dev_list))

        y_mean = y0_list.sum() / len(y0_list)
        y_std_dev = y_std_dev_list.sum() / (len(y_std_dev_list))

        #calculate the ideal fit form the average means and std devs
        popt = [x_mean, y_mean, x_std_dev, y_std_dev, 255]
        fit = np.zeros(tile.shape)
        for i in range(len(popt)//5):
            fit += gaussian(X, Y, *popt[i*5:i*5+5])

    return fit, popt


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    x, y = np.linspace(-25, 25, 50), np.linspace(-25, 25, 50)
    X, Y = np.meshgrid(x, y)

    fit = np.load('ideal_gaussian_fit.npy')
    display_3D_plot(fit) 

    #take in the data and 
    laser_tiles_list = get_laser_tiles(data_path, fit)

    #calculate the fitted gaussian function for the y and x cut curves
    #this function is used when the position of the laser is known.
    #fit, popt = get_ideal_gaussian_fit(tiles_list)

    # print('The final estimated parameters are: ')
    # print(popt)
    # np.save('ideal_gaussian_fit', fit)
    fit = np.load('ideal_gaussian_fit.npy')
    display_3D_plot(fit)

    #random_tile = np.random.randn(tile.shape[0], tile.shape[1])
    random_tile = get_tiles(img_copy[:,:,0], [[234, 455]])
    random_tile = np.asarray(random_tile)
    print(random_tile[0].shape)
    dist = bhatta_dist(fit.flatten(), random_tile[0].flatten(), method='continuous') 
    #display_3D_plot(random_tile[0])

    print(f'Rondom tile distance is: {dist}')

    for tile in laser_tiles_list:
        #dist = distance.directed_hausdorff(fit, tile)
        dist = bhatta_dist(fit.flatten(), tile.flatten(), method='continuous') 
        print(dist)
        tile[tile<50] = 0
        display_tile(fit, tile, str(f'Bhatta dist is: {dist}'))