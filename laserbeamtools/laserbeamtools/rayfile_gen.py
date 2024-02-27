# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Routines for generating ZEMAX rayfiles from images.

Full documentation is available at TODO>
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import laserbeamtools as lbs

__all__ = ('Rayfile_gen',
           'Weighted_far_field',
           'open_spectrum'
           )

class Weighted_far_field:
    """
    A class representing an employee.
 
    Attributes:
        ff (numpy array): pixel array from the far field camera.
        v (list): list off all unit vectors built from the far field image.
    """

    def __init__(self, 
                 ff_img: np, 
                 pixel_size_um: float, 
                 flen_mm: float, 
                 floor: float=0):
        """
        Initialize the Weighted_far_field object.

        Args:
            ff_img: far field array
            pixel_size: camera pixel size, must be in um
            flen: far field lens effective focal length, must also be in um 
            floor: (optional) floor value that a pixel must exceed to be considered for a vector
        """
        # TODO:
        # Floor option
        
        # Convert flen to um
        flen = flen_mm * 1000 #um
        
        # Copy input array and normalize
        self.ff = np.copy(ff_img)
        self.ff[self.ff < abs(floor)] = 0.0
        self.ff = self.ff/np.sum(self.ff)

        # Flatten array
        self.ff_flat = self.ff.flatten()

        # Generate index list
        self.i = np.linspace(0, len(self.ff_flat), len(self.ff_flat), endpoint=False, dtype=int)
        
        # Camera shape and scaled values
        vv, hh = self.ff.shape
        self.s = pixel_size_um / flen
        h_s = hh * self.s
        v_s = vv * self.s

        # Create index arrays (vi : vertical index array) (hi : horizontal index array)
        vi, hi = (np.indices(self.ff.shape))

        # Index arrays (vi_s : vertical index scaled) (hi_s : horizontal index scaled)
        vi_s = (vi * self.s) - (v_s / 2)
        hi_s = (hi * self.s) - (h_s / 2)

        # Projection of vectors
        z_proj = (1 + hi_s**2 + vi_s**2)**(-1/2)
        x_proj = hi_s * z_proj
        y_proj = vi_s * z_proj

        # Flattened arrays
        self.z_proj_flat = z_proj.flatten()
        self.x_proj_flat = x_proj.flatten()
        self.y_proj_flat = y_proj.flatten()

        return

    def get_vectors(self, num: int):
        """
        Returns a list of vectors chosen by the weighted distribution

        Args:
            num: number of vectors to choose

        Returns:
            x_list: list of x components of unit vectors
            y_list: list of y components of unit vectors
            z_list: list of z components of unit vectors
        """
        indexes = np.random.choice(self.i, num, p=self.ff_flat)
        x_list = self.x_proj_flat[indexes]
        y_list = self.y_proj_flat[indexes]
        z_list = self.z_proj_flat[indexes]

        return x_list, y_list, z_list

    def preview_ff(self) -> None:
        """
        Preview the data from the far field.
        """
        lbs.plot_knife_edge_analysis(self.ff, 
                                     pixel_size=self.s * 1000, 
                                     units='mrad', 
                                     title='Far Field Knife-Edge Analysis')
        return
    
    def preview_dist(self) -> None:
        ff_flat = self.ff.flatten()

        plt.plot(ff_flat)

        return


class Rayfile_gen:
    """
    
    Attributes:
        nf: near field image
        ff: far field image
        wff: weighted far field
    """
    def load_nf(self,
                nf_img: np, 
                pixel_size_um: float, 
                magnification: float,
                quash_noise: bool=True,
                crop: bool=True
                ) -> None:    
        """
        Loads in the near image

        Args:
            nf_img: near field image
            pixel_size_um: pixel size in micron
            magnification: magnification of the image
            quash_noise: (optional) zero all noise below 3*nT
            crop: (optional) crop rectangle to 3*D4sigma fullwidth
        """

        # Near field array
        self.nf = np.copy(nf_img)      

        if quash_noise:
            self.nf = lbs.subtract_corner_background(self.nf, iso_noise=False)

        if crop:
            x, y, dx, dy, phi = lbs.beam_size(self.nf)
            self.nf, _, _ = lbs.crop_image_to_integration_rect(self.nf, x, y, dx, dy, phi)

        # Near field parameters
        self.mag = magnification
        self.pixel_size_um = pixel_size_um   

        # Camera shape
        self.vv, self.hh = self.nf.shape
        self.s = pixel_size_um / magnification
        self.h_s = self.hh * self.s
        self.v_s = self.vv * self.s
        self.x1 = self.h_s / 2
        self.y1 = self.v_s / 2

        return None
    

    def preview_nf(self) -> None:
        """
        Preview data, useful before generating rayfile to ensure proper generation.

        Args:

        """
        lbs.plot_knife_edge_analysis(self.nf, 
                                     pixel_size=self.pixel_size_um/self.mag/1000, 
                                     units='mm',
                                     title='Near Field Knife-Edge Analsis')
        
        return None


    def load_ff(self,
                ff_img: np, 
                pixel_size_um: float, 
                flen_mm: float,
                floor: float=0,
                quash_noise: bool=True,
                crop: bool=True,
                precrop: bool=True,
                precrop_frac: float=0.7
                ) -> None:
        """
        Loads the far field data. All far field data is embedded in the wff object.

        Args:
            ff_img: far field image
            pixel_size: pixel size on the far field camera
            flen_mm: focal length of the lens used to image the beam on the far field camera
            floor: (optional) floor value, only consider pixels above this value
            crop: (optional) crop the beam to 3 time the D4simga fullwidth
            precrop: (optional) crop the beam before finding the D4sigma values used to crop the beam
        """
        ff = np.copy(ff_img)

        if precrop:
            h_frac = precrop_frac/2.0
            vv,hh = ff.shape
            vd = int(vv * h_frac)
            hd = int(hh * h_frac)
            ff = ff[vd:(vv-vd), hd:(hh-hd)]

        if quash_noise:
            ff = lbs.subtract_corner_background(ff, iso_noise=False)

        if crop:
            x, y, dx, dy, phi = lbs.beam_size(ff)
            ff, _, _ = lbs.crop_image_to_integration_rect(ff, x, y, dx, dy, phi)

        self.wff = Weighted_far_field(ff_img=ff, 
                                      pixel_size_um=pixel_size_um, 
                                      flen_mm=flen_mm, 
                                      floor=floor)
        
        return None
        
    def preview_ff(self):
        """
        Preview the far field data
        """
        self.wff.preview_ff()

        return

    def open_spectrum_file(self, 
                           file_name: str, 
                           delimiter: str='\t', 
                           skip_start: int=1
                           )->None:
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


    def load_spectrum(self, 
                      wavelengths_um:np, 
                      intensities:np
                      )->None:
        """
        Method to load spectrum data from a text file.

        Args:
            wavelenghts: wavelength array.
            intensities: intensity array.
        
        Returns:
            [wavelengths, intensities]: wavelengths are their corresponding intensities in the signal.
        """

        # wavelength data
        self.wavelengths = np.asarray(wavelengths_um)

        if np.average(self.wavelengths) > 100:
            print("Wavelengths look to be in nm, convert to um...")
            self.wavelengths = self.wavelengths/1000

        # intensity data
        intensities = np.asarray(intensities)

        if np.min(intensities) < 0:
            intensities = intensities - np.min(intensities)

        self.intensities = intensities/np.sum(intensities)        

        return None
        
    def generate(self,
                 output_filename: str='custom_rayfile.DAT',
                 rays: int=10000, 
                 )->None:
        """
        Method to actually generate the rayfile.

        Args:
            output_filename:
        """
        print("Generating rayfile with %s rays, this may take some time..." % rays)

        floor = 0

        # generate wavelength array
        # rays = np.sum(self.nf > floor)

        # Weighted random wavelength list
        wl_vals = np.random.choice(self.wavelengths, rays, p=self.intensities) # wavelength in micron

        # Weighted random ray unit vector lists
        vx, vy, vz = self.wff.get_vectors(rays)

        # flattened near field
        self.nf_flat = self.nf.flatten()
        self.nf_flat = self.nf_flat / np.sum(self.nf_flat)

        # Create an index list
        self.i = np.linspace(0, len(self.nf_flat), len(self.nf_flat), endpoint=False, dtype=int)

        # Only pick indexes that are above floor value
        i_removed_floor = self.i[self.nf_flat > floor]

        # Weighted random indexes
        indexes = np.random.choice(i_removed_floor, rays)

        # Camera shape and scaled values
        vv, hh = self.nf.shape
        h_s = hh * self.s
        v_s = vv * self.s
        vi, hi = (np.indices(self.nf.shape))
        vi_s = (vi * self.s) - (v_s / 2)
        hi_s = (hi * self.s) - (h_s / 2)
        vi_s_flat = vi_s.flatten()
        hi_s_flat = hi_s.flatten()

        # Randomized lists
        x_coords = hi_s_flat[indexes]/1000 # x coords in mm
        y_coords = vi_s_flat[indexes]/1000 # y coords in mm
        power_vals = self.nf_flat[indexes]

        power_vals = power_vals / np.max(power_vals) # Make max value 1.0

        # Create output file
        with open(output_filename, 'w') as writer:
            
            # Line 1
            writer.write("{} 4\n".format(rays))
            
            # Remaining lines
            for i in range(0,rays):
                line_str = "{:.5f} {:.5f} 0 {:.10f} {:.10f} {:10f} {:.4f} {:.4f}\n".format(x_coords[i], y_coords[i],
                                                                                            vx[i], vy[i], vz[i],
                                                                                            power_vals[i],
                                                                                            wl_vals[i])
                writer.write(line_str)

        print("Rayfile succesfully generated, saved under:\n\t%s\n\t\t%s" % (os.getcwd(), output_filename))
        
        return None
    
def open_spectrum(file_name: str, 
                  delimiter: str='\t', 
                  skip_start: int=1, 
                  normalize: bool=True
                  )->None:
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