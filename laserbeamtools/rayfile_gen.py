# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Routines for generating ZEMAX rayfiles from images.

Full documentation is available at TODO>
"""

import numpy as np
import random
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import laserbeamtools as lbs

__all__ = ('Rayfile_gen',
           'Weighted_far_field',
           'open_spectrum'
           )


class Rayfile_gen:
    """
    
    Attributes:
        nf: near field image
        ff: far field image
        wff: weighted far field
    """
    def load_nf(self,
                nf_img, 
                pixel_size_um=4.4, 
                magnification=0.59):    
        """
        Loads in the near image

        Args:
            nf_img: near field image
            pixel_size_um: pixel size in micron
            magnification: magnification of the image
        """

        # Near field array
        self.nf = np.copy(nf_img)    

        # Camera shape
        self.vv, self.hh = self.nf.shape
        self.s = pixel_size_um / magnification
        self.h_s = self.hh * self.s
        self.v_s = self.vv * self.s
        self.x1 = self.h_s / 2
        self.y1 = self.v_s / 2

        return None
    
    def load_ff(self,
                ff_img, 
                pixel_size_um=4.4, 
                flen_mm=80,
                floor=0):
        """
        Loads the far field data. All far field data is embedded in the wff object.

        Args:
            ff_img:
            pixel_size:
            flen_mm:
            floor:
        """
        self.wff = Weighted_far_field(ff_img=ff_img, pixel_size_um=pixel_size_um, 
                                      flen_mm=flen_mm, floor=floor)
        return None

    def preview_data(self) -> None:
        """
        Preview data, useful before generating rayfile to ensure proper generation.

        Args:

        """
        # Near field data
        plt.imshow(self.nf, cmap='jet', extent=[-self.x1, self.x1, -self.y1, self.y1])
        plt.xlabel('(um)')
        plt.ylabel('(um)')

        return None


    def open_spectrum_file(self, file_name, delimiter='\t', skip_start = 1):
        """
        Method to load spectrum data from a text file.

        Args:
            file_name: full file path to file.
            delimiter: delimiters in file.
            skip_start: skip start rows
        
        Returns:
            [wavelengths, intensities]: wavelengths are their corresponding intensities in the signal.
        """
        # Create empty lists for data to be appended to
        x = []
        y = []

        # Open file
        with open(file_name,"r") as file:
            data=file.read()
            dataList = data.split('\n')

            # Loop through rows in file
            for i in range(0,len(dataList)-1):
                
                # Look for rows to skip
                if i >= skip_start:
                    # Split data and append it to lists
                    temp = dataList[i].split(delimiter)
                    x.append(float(temp[0]))
                    y.append(float(temp[1]))

        self.load_spectrum(x,y)
        
        return None


    def load_spectrum(self, wavelengths, intensities):
        """
        Method to load spectrum data from a text file.

        Args:
            file_name: full file path to file.
            delimiter: delimiters in file.
            skip_start: skip start rows
        
        Returns:
            [wavelengths, intensities]: wavelengths are their corresponding intensities in the signal.
        """
        # wavelength data
        self.wavelengths = np.asarray(wavelengths)

        # intensity data
        intensities = np.asarray(intensities)
        intensities = intensities - np.min(intensities)
        self.intensities = intensities/np.sum(intensities)

        return None
        
    def generate(self, output_filename='custom_rayfile.DAT') -> None:
        """
        Method to actually generate the rayfile.

        Args:
            output_filename:
        """
        print("Generating rayfile, this may take some time...")

        # generate wavelength array
        wavelength_array = np.random.choice(self.wavelengths, self.vv * self.hh, p=self.intensities)

        x_axis = np.linspace(-self.x1, self.x1, self.hh)
        y_axis = np.linspace(-self.y1, self.y1, self.vv)

        index = 0
        line = []
        for j in range(0,self.hh):
            for i in range(0,self.vv):

                #x y z l m n i w
                if(self.nf[i,j] > 0):
                    # Get a unit vector string from weighted far field object
                    uvs = self.wff.get_vector()

                    # Get wavelength value
                    wl = wavelength_array[index]

                    # Append line
                    line.append("{:.5f} {:.5f} 0 {} {:.5f} {:.4f}\n".format(x_axis[j]/1000, y_axis[i]/1000, uvs, self.nf[i,j]*100000, wl/1000))
                    
                    # Increment index
                    index += 1
                
        # Randomize ray order
        print("Radomizing ray order...")
        random.shuffle(line)

        # Create output file
        with open(output_filename, 'w') as writer:
            writer.write("{} 4\n".format(index))
            for l in line:
                writer.write(l)

        print("Rayfile succesfully generated, saved under:\n\t%s" % output_filename)
        
        return None

class Weighted_far_field:
    """
    A class representing an employee.
 
    Attributes:
        ff (numpy array): pixel array from the far field camera.
        v (list): list off all unit vectors built from the far field image.
    """

    def __init__(self, ff_img, pixel_size_um=4.4, flen_mm=80, floor=0):
        """
        Initialize the Weighted_far_field object.

        Args:
            ff_img: far field array
            pixel_size: camera pixel size, must be in um
            flen: far field lens effective focal length, must also be in um 
            floor: floor value that a pixel must exceed to be considered for a vector
        """
        # Convert flen to um
        flen = flen_mm * 1000 #um
        
        # Copy input array
        self.ff = np.copy(ff_img)
        self.ff = self.ff/np.sum(self.ff)
        
        # Camera shape
        vv, hh = self.ff.shape
        s = pixel_size_um / flen
        h_s = hh * s
        v_s = vv * s
        x1 = h_s / 2
        y1 = v_s / 2
  
        # Empty lists
        self.__cff = [] # Cumulative far field
        self.v = [] # Vector string list
        
        # Temp variable
        tmp = 0.0

        # Scan over entire image
        for j in range(0,hh):
            for i in range(0,vv):

                # Check if pixel value is greater than 0
                if(self.ff[i,j] > floor):
                    
                    # Temp keeps track of cumulative sum
                    tmp = tmp + self.ff[i,j] 
                    
                    # Ray pointing
                    tx_ = (j * s) - x1          # Pointing in x
                    ty_ = (i * s) - y1          # Pointing in y
                    
                    # Convert to unit vector
                    zvc = (1 + tx_**2 + ty_**2)**(-1/2) # Z component of vector
                    xvc = tx_ * zvc                     # X component of vector
                    yvc = ty_ * zvc                     # Y component of vector
                    
                    # Format unit vector string
                    vstr = "{} {} {}".format(xvc, yvc, zvc)
                    
                    # Append values to list
                    self.v.append(vstr)
                    self.__cff.append(tmp)
        
        # Verify sum
        print(self.__cff[-1])
        
        return

    def get_vector(self):
        """
        Input a value between 0 and 1, returns a ray string.

        Args:
            num: number between 0 and 1, picks the ray
        Returns:
            vstr: ray unit vector string
        """
        num = random.random()
        for i in range(0,len(self.__cff)):
            if(num < self.__cff[i]):
                return self.v[i]

        return "0 0 1"


def open_spectrum(file_name, delimiter='\t', skip_start = 1, normalize=True):
    """
    Method to load spectrum data from a text file.

    Args:
        file_name: full file path to file.
        delimiter: delimiters in file.
        skip_start: skip start rows
    
    Returns:
        [wavelengths, intensities]: wavelengths are their corresponding intensities in the signal.
    """
    # Create empty lists for data to be appended to
    x = []
    y = []

    # Open file
    with open(file_name,"r") as file:
        data=file.read()
        dataList = data.split('\n')

        # Loop through rows in file
        for i in range(0,len(dataList)-1):
            
            # Look for rows to skip
            if i >= skip_start:
                # Split data and append it to lists
                temp = dataList[i].split(delimiter)
                x.append(float(temp[0]))
                y.append(float(temp[1]))

    # Convert data to numpy arrays
    wavelengths = np.asarray(x)
    intensities = np.asarray(y)

    # Normalize data
    if normalize:
        intensities = intensities/np.sum(intensities)

    return [wavelengths, intensities]