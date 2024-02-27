laserbeamtools
==============

by Ryan Robinson 

Simple and fast calculation of beam sizes from a single monochrome image based
on the ISO 11146 method of variances.  Some effort has been made to make the 
algorithm less sensitive to background offset and noise.

This module also supports M² calculations based on a series of images
collected at various distances from the focused beam. 

Installation
------------

Package is not on PyPi, pip must be pointed to the downloaded directory to install.

Use ``pip``::
    
    pip install -e ./laserbeamtools

ZEMAX Rayfile generation
-------------------------

ZEMAX non-sequatial rayfiles may be generated given a near-field and far-field image.

Example::

    # Create rayfile generation object
    rg = lbt.Rayfile_gen()

    # Load near-field array
    rg.load_nf(nf_c, pixel_size_um=4.4, magnification=0.59)

    # Load far-field array
    rg.load_ff(ff_c, pixel_size_um=4.4, flen_mm=80)

    # Load Spectrum data
    rg.load_spectrum(spec_x_t, spec_y_t)

    # Generate rayfile
    rg.generate('BLE_09_100k.DAT', rays = 100000)

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/rayfile_example2.png
   :alt: Rayfile

Determining the beam size in an image
-------------------------------------

Near and Far Field Report Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A report that outlines the near a far field characteristics of a beam can be generated
given the near and far field beam images.

Example::

    # Import laserbeamtools
    import laserbeamtools as lbt

    # Load images into arrays
    folder = r'N:/PRODUCTION/BL/BeamExpander/COL-006/BLE-023/Raw'
    ff_img = lbt.load_img(folder+'/Module_All_FF.bmp')
    nf_img = lbt.load_img(folder+'/Module_All_NF.bmp')

    # Run analysis method
    lbt.near_and_far_profiles(nf_img,
                            ff_img, 
                            title='BLE-023 All', 
                            ff_lens=300,
                            ff_units='mrad',
                            ffprecrop=0,
                            ff_pixel_size=2.2,
                            nf_pixel_size=2.2,
                            nfprecrop=0.4, 
                            nf_mag=0.1307, 
                            nf_scale_down=1000, 
                            nf_units='mm'
                            )

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/BLE-023_All.png
   :alt: Report 1

Knife-Edge
^^^^^^^^^^^

To find the knife-edge widths of a beam::

    import laserbeamtools as lbt

    # Load image file
    img = lbt.load_img(nf1_file)

    # Call plotting method
    lbt.plot_knife_edge_analysis(img, pixel_size=4.4 / 0.59 / 1000, units='mm')

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/ke01.png
   :alt: Knife-Edge 1

or::
    
    import laserbeamtools as lbt

    # Load image file
    img = lbt.load_img(ff2_file)

    # Call plotting method
    lbt.plot_knife_edge_analysis(img, pixel_size=4.4 / 80, units='mrad')

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/ke02.png
   :alt: Knife-Edge 2

D4_sigma
^^^^^^^^^

Finding the center and dimensions of a good beam image::

    import imageio.v3 as iio
    import laserbeamsize as lbs
    
    file = "https://github.com/ryantro/laserbeamtools/blob/master/docs/t-hene.pgm"
    image = iio.imread(file)
    
    x, y, dx, dy, phi = lbs.beam_size(image)
    print("The center of the beam ellipse is at (%.0f, %.0f)" % (x, y))
    print("The ellipse diameter (closest to horizontal) is %.0f pixels" % dx)
    print("The ellipse diameter (closest to   vertical) is %.0f pixels" % dy)
    print("The ellipse is rotated %.0f° ccw from the horizontal" % (phi * 180/3.1416))

to produce::

    The center of the beam ellipse is at (651, 492)
    The ellipse diameter (closest to horizontal) is 369 pixels
    The ellipse diameter (closest to   vertical) is 347 pixels
    The ellipse is rotated -12° ccw from the horizontal

A visual report can be done with one function call::

    lbs.plot_image_analysis(beam)
    plt.show()
    
produces something like

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/hene-report.png
   :alt: HeNe report

or::

    lbs.plot_image_analysis(beam, r"Original Image $\lambda$=4µm beam", pixel_size = 12, units='µm')
    plt.show()

produces something like

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/astigmatic-report.png
   :alt: astigmatic report

Non-gaussian beams work too::

    # 12-bit pixel image stored as high-order bits in 16-bit values
    tem02 = imageio.imread("TEM02_100mm.pgm") >> 4
    lbs.plot_image_analysis(tem02, title = r"TEM$_{02}$ at z=100mm", pixel_size=3.75)
    plt.show()

produces

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/tem02.png
   :alt: TEM02

Determining the beam divergence of a far field image
-----------------------------------------------------

TODO

Determining M² 
--------------

Determining M² for a laser beam is also straightforward.  Just collect beam diameters from
five beam locations within one Rayleigh distance of the focus and from five locations more
than two Rayleigh distances::

    lambda1=308e-9 # meters
    z1_all=np.array([-200,-180,-160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,99,120,140,160,180,200])*1e-3
    d1_all=2*np.array([416,384,366,311,279,245,216,176,151,120,101,93,102,120,147,177,217,256,291,316,348])*1e-6
    lbs.M2_radius_plot(z1_all, d1_all, lambda1, strict=True)
    plt.show()

produces

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/m2fit.png
   :alt: fit for M2

Here is an analysis of a set of images that do not meet the ISO 11146
requirements for determining M² (because the image locations are not taken
in right locations relative to the focus).  These beam images are from a HeNe
laser with slightly misaligned mirrors to primarily lase in a TEM₀₁ transverse mode.
The laser resonator had a fixed rotation of 38.7° from the plane of
the optical table.::

    lambda0 = 632.8e-9 # meters
    z10 = np.array([247,251,259,266,281,292])*1e-3 # meters
    filenames = ["sb_%.0fmm_10.pgm" % (number*1e3) for number in z10]

    # the 12-bit pixel images are stored in high-order bits in 16-bit values
    tem10 = [imageio.imread(name)>>4 for name in filenames]

    # remove top to eliminate artifact 
    for i in range(len(z10)):
        tem10[i] = tem10[i][200:,:]

    # find beam rotated by 38.7° in all images
    fixed_rotation = np.radians(38.7)
    options = {'pixel_size': 3.75, 'units': "µm", 'crop': [1400,1400], 'z':z10, 'phi':fixed_rotation}
    dy, dx= lbs.beam_size_montage(tem10, **options)  # dy and dx in microns
    plt.show()

produces

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/sbmontage.png
   :alt: montage of laser images

Here is one way to plot the fit using the above diameters::

    lbs.M2_diameter_plot(z10, dx*1e-6, lambda0, dy=dy*1e-6)
    plt.show()

In the graph on the below right, the dashed line shows the expected divergence
of a pure gaussian beam.  Since real beams should diverge faster than this (not slower)
there is some problem with the measurements (too few!).  On the other hand, the M² value 
the semi-major axis 2.6±0.7 is consistent with the expected value of 3 for the TEM₁₀ mode.

.. image:: https://github.com/ryantro/laserbeamtools/blob/master/docs/sbfit.png
   :alt: fit


Determining M² using near and far field beam profiles
------------------------------------------------------

TODO

- Knife edge measurements, e.g., 10%-90% and 5%-95% in minor and major axis.
- Knife edge plots.
- Measuring and lotting in angle space for far field images.
- M² determination from near filed and far field camera images.
- Rayfile generation for ZEMAX non-sequatial mode.
- Live beam updates with for Baslar Cameras imaging near and far fields.
   - Maybe this should be a seperate library?

License
-------

``laserbeamtools`` is licensed under the terms of the MIT license.
