#################### PACKAGES ########################################
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
######################################################################


######################################################################
########## FUNCTIONS ##################################################
######################################################################


########## Convert image to array Greytone ##########
def convert_image2array(image_name):
    """image_name : str
        Name of the image file
    ----------------------------------
    ----------------------------------
    f : array (2D)
        2D array of the image
    ----------------------------------"""
    f = Image.open(image_name).convert('L')     # 'L' = greytone
    f = np.array(f)                             # convert to 2D-array
    return f



########## Convert image to array RGB ##########
def convert_image2array_color(image_name):
    """image_name : str
        Name of the image file
    ----------------------------------
    ----------------------------------
    f : array (2D)
        2D array of the image
    ----------------------------------"""
    f = Image.open(image_name).convert('RGB')     # 'RGB' = RGB colors
    f = np.array(f)                               # convert to 2D-array
    return f



########## Find saturated pixels ##########
def find_saturated(f):
    """f : array (2D)
        Can be graytone or RGB
    ----------------------------------
    ----------------------------------
    saturated : array
        The pixels that have at least 1 color channel == 255
    ----------------------------------"""
    saturated = np.any(f == 255, axis=-1)
    return saturated



########## Find all white pixels ##########
def find_white(f):
    """f : array (2D)
        Can be graytone or RGB
    ----------------------------------
    ----------------------------------
    white : array
        The pixels that have at all color channels == 255
    ----------------------------------"""
    white = np.all(f == 255, axis=-1)
    return white




########## Find both saturated and white ##########
def find_saturated_white(f):
    """f : array (2D)
        Can be graytone or RGB
    ----------------------------------
    ----------------------------------
    saturated : array
        The pixels that have at least 1 color channel == 255
    ----------------------------------
    white : array
        The pixels that have at all color channels == 255
    ----------------------------------"""
    saturated = np.any(f == 255, axis=-1)
    white = np.all(f == 255, axis=-1)
    return saturated, white



########## Plot image min,max = [0,255] ##########
def plot_image(f, title=None):
    """f : array (2D)
    ----------------------------------
    ----------------------------------
    Plot grey scale of image matrix f
    ----------------------------------"""
    plt.imshow(f, cmap='gray', vmin=0, vmax=255, aspect='auto') # 0=black, 255=white
    if title is not None: plt.title(title)
    plt.show()


########## Plot image without min,max ##########
def plot_image_simple(f, title=None):
    """f : array (2D)
    ----------------------------------
    ----------------------------------
    Plot grey scale of image matrix f
    ----------------------------------"""
    #plt.imshow(f, cmap='gray', vmin=0, vmax=1, aspect='auto') 
    plt.imshow(f, cmap='gray', aspect='auto') # Binary format
    if title is not None: plt.title(title)
    plt.show()





########## crop the image ##########
def crop_image(f, startx, endx, starty, endy): # x, y are within bounds of dimensions f
    """f : array (2D)
    ----------------------------------
    startx, endx : int, int
        start and end along x-axis
    ----------------------------------
    starty, endy : int, int
        start and end along y-axis
    ----------------------------------
    ----------------------------------
    g : array (2D)
        Cropped image of f
    ----------------------------------"""
    rows = endx - startx + 1; cols = endy - starty + 1
    g = np.zeros((rows, cols))
    for x in range(startx, endx+1):
        for y in range(starty, endy+1):
            g[x - startx, y - starty] = f[x, y]
    return g




########## plot side by side comparison ##########
def plot_image_2(f, g):
    """f : array (2D)
    ----------------------------------
    g : array (2D)
    ----------------------------------
    ----------------------------------
    Plot f and g side by side
    ----------------------------------"""
    ##### Configure frame
    fig, axes = plt.subplots(1,2,figsize=(16,8))
    ##### f
    im_0 = axes[0].imshow(f, cmap="grey")
    ##### g
    im_1 = axes[1].imshow(g, cmap="grey")
    plt.tight_layout(); plt.show()






########## pixel histogram of image f ##########
def make_histogram(f, title=None):
    f_flat = f.flatten()
    plt.hist(f_flat, bins = range(0,257))
    if title is not None: plt.title(title)
    plt.show()



########## middle-value mu, std sigma ##########
def mu_sigma(f):
    """f : array (2D)
    ----------------------------------
    ----------------------------------
    mu : float
        Total avg of the pixels in image f
    ----------------------------------
    s_v : float
        Standard deviation
    ----------------------------------"""
    f_flat = f.flatten()                # convert 2D to 1D
    sum_pixel = np.sum(f_flat)
    mu = sum_pixel//len(f_flat)
    
    sum_var = np.zeros(len(f_flat))
    sum_var = (f_flat - mu)**2
    sum_var = np.sum(sum_var)
    var = sum_var//len(f_flat)          # variance
    s_v = int(np.sqrt(var))             # std
    return mu, s_v



########## Find contrast coefficients a and b ##########
def find_ab(mu_T, std_T, mu, std):
    """mu_T : float
        Arbitrary total avg
    ----------------------------------
    std_T : float
        Arbitrary std
    ----------------------------------
    mu : float
        Total avg
    ----------------------------------
    std : float
        Std
    ----------------------------------
    ----------------------------------
    a : float
    ----------------------------------
    b : float
    ----------------------------------"""
    a = std_T/std 
    b = mu_T - a * mu
    return a, b




########## Enhance contrast in image f ##########
def make_contrast(f, a, b):
    rows = len(f); cols = len(f[0])
    g = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            val = int(a * f[r, c] + b) 
            """ Making sure value is within scale [0,255] """
            if val < 0: val = 0 
            if val > 255: val = 255
            g[r, c] = val
    return g



########## Maximize contrast in image f ##########
def make_max_contrast(f, mu):
    """f : array (2D)
    ----------------------------------
    mu : float
    ----------------------------------
    ----------------------------------
    g : array (2D)
        The contrast image
    ----------------------------------"""
    rows = len(f); cols = len(f[0])
    g = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            val = f[r, c]
            """ Making sure value is within scale [0,255] """
            if val < 0 or val <= mu: val = 0 
            else: val = 255
            g[r, c] = val
    return g







######################################################################
#################### EXPERIMENTS ########################################
######################################################################
if __name__ == "__main__":
    print("----- TEK4710 exercise 2 -----\n")
    ############################################################
    print("\n----- Exercise 2a) -----\n")
    ########## Initial parameters ##########
    image_name = "siemens_star_mini.jpg"

    ########## Convert image to matrix ##########
    #f = convert_image2array(image_name)        # (224, 224)

    ########## Plot image ##########
    #plot_image(f)



    ############################################################
    print("\n----- Exercise 2c) -----\n")
    #image_c = "c_iso25_1_909s.jpg"
    #image_c = "c_iso50_1_1000s.jpg"
    #image_c = "c_iso50_1_500s.jpg"
    image_c = "c_iso50_1_50s.jpg"

    #f_c = convert_image2array_color(image_c)
    
    #saturated = find_saturated(f_c)
    #saturated = crop_image(saturated, 200, 1200, 200, 1200)
    #plot_image_simple(saturated)

    #white = find_white(f_c)
    #plot_image_simple(white)
    



    ############################################################
    print("\n----- Exercise 2b) -----\n")
    image_b = "b_crop_brightness_000.jpg"; print(image_b)
    f = convert_image2array(image_b)
    #plot_image_simple(f)
    #mu_T, std_T = 127, 64
    mu, std = mu_sigma(f)
    #print(mu)
    #a, b = find_ab(mu_T, std_T, mu, std)
    #g = make_contrast(f, a, b)

    #g = make_max_contrast(f, mu)
    #plot_image(g)



    ############################################################
    print("\n----- Exercise 2f) -----\n")
    image_f = "f_image.jpg"; print(image_f)
    f = convert_image2array(image_f)
    mu, std = mu_sigma(f)
    print("Max: ", np.max(f)); print("Min: ", np.min(f))
    print("Mu: ", mu); print("Std: ", std)
    g = make_max_contrast(f, mu+5)
    plot_image(g)


    




