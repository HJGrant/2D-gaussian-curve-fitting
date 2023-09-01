# FishSense - gaussian curve fitting for a laser dot.
This software takes in image data with laser dots, where each of the laser dots is labeled and the coordinates of the lasers are stored in a .csv file. It then fits a 2D gaussian curve to each of the laser dots, and then computes the average of all of those fitted curves. 
It then saves the parameters for the gaussian and the numpy array representing that gaussin to .npy files. 

You can then use the averaged fitted gaussian to calculate similarity to other tiles in the image, in the hopes that the tile with the highest similarity index is the tile that contains the laser dot. 

## Steps to use.
1. Pass a list of tiles that contain the laser dot, and that have been centered around that dot, to the get_ideal_gaussian_fit() function. This will return the averaged gaussian function fit of all of those lasers and the gaussian parameters, such as the x and y axis mean and the x and y axis standard deviation, as well as the max value. You can then save the averaged gaussian fit and the parameters to .npy files. 
You can get this list of laser dot tiles by passing the image data and the .csv with the laser dot coordinates to the get_coordinates() function, there is also the possibility to look for all the local maxima in the image. You then pass those coordinates to the center_coordinates() function and then to the get_tiles() function. The get_tiles() function will return a list of N x M tiles with the lasers, and it will also retirn the coordinates where the tiles were taken from. 

2. Most of the work is done in the get_laser_tiles() function. By passing the this function the path to the image data and the ideal gaussian fit you have calculated before. The function will take care of looking for all the local maxima within the image, then compute a gaussian curve fit for each of those points. It will then compute the eucleadean distance between your ideal gaussian fit computed previously, and each of the gaussian fits for the tiles in the tiles list. You can then check the tiles for the one with the least distance to your ideal fit. 

## Additional information.
There are some more functions in the gaussian_curve_fitting.py that are there to help you work with the tiles and coordinates. 
- image_cuts() plots or returns the x and y axis center cuts of the tiles. 
- display_3D_plot() allows you to quickly plot any tile or other matrix in 3D.
- bhatta_dist() allows you to compute the bhattacharyya distance instead of the euclidean distance between to arrays. 
  