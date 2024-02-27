# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=too-many-statements

"""
A module for generating a graphical analysis of beam size fitting.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

A graphic showing the image and extracted beam parameters is achieved by::

    >>> import imageio.v3 as iio
    >>> import matplotlib.pyplot as plt
    >>> import laserbeamsize as lbs
    >>>
    >>> repo = "https://github.com/scottprahl/laserbeamsize/raw/master/docs/"
    >>> image = iio.imread(repo + 't-hene.pgm')
    >>>
    >>> lbs.plot_image_analysis(image)
    >>> plt.show()

A mosaic of images might be created by::

    >>> import imageio.v3 as iio
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import laserbeamsize as lbs
    >>>
    >>> repo = "https://github.com/scottprahl/laserbeamsize/raw/master/docs/"
    >>> z1 = np.array([168,210,280,348,414,480], dtype=float)
    >>> fn1 = [repo + "t-%dmm.pgm" % number for number in z1]
    >>> images = [iio.imread(fn) for fn in fn1]
    >>>
    >>> options = {'z':z1/1000, 'pixel_size':0.00375, 'units':'mm', 'crop':True}
    >>> lbs.plot_image_montage(images, **options, iso_noise=False)
    >>> plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import laserbeamtools as lbs

__all__ = ('beam_ellipticity',
           'plot_beam_diagram',
           'plot_image_analysis',
           'plot_image_and_fit',
           'plot_image_montage',
           'plot_knife_edge_analysis',
           'plot_knife_edge_analysis_slow'
           )


def beam_ellipticity(dx, dy):
    """
    Calculate the ellipticity of the beam.

    The ISO 11146 standard defines ellipticity as the "ratio between the
    minimum and maximum beam widths".  These widths (diameters) returned
    by `beam_size()` can be used to make this calculation.

    When `ellipticity > 0.87`, then the beam profile may be considered to have
    circular symmetry. The equivalent beam diameter is the root mean square
    of the beam diameters.

    Args:
        dx: x diameter of the beam spot
        dy: y diameter of the beam spot
    Returns:
        ellipticity: varies from 0 (line) to 1 (round)
        d_circular: equivalent diameter of a circular beam
    """
    if dy < dx:
        ellipticity = dy / dx
    elif dx < dy:
        ellipticity = dx / dy
    else:
        ellipticity = 1

    d_circular = np.sqrt((dx**2 + dy**2) / 2)

    return ellipticity, d_circular


def plot_beam_diagram():
    """Draw a simple astigmatic beam ellipse with labels."""
    theta = np.radians(30)
    xc, yc, dx, dy = 0, 0, 50, 25

    plt.subplots(1, 1, figsize=(6, 6))

    # If the aspect ratio is not `equal` then the major and minor radii
    # do not appear to be orthogonal to each other!
    plt.axes().set_aspect('equal')

    xp, yp = lbs.ellipse_arrays(xc, yc, dx, dy, theta)
    plt.plot(xp, yp, 'k', lw=2)

    xp, yp = lbs.rotated_rect_arrays(xc, yc, dx, dy, theta)
    plt.plot(xp, yp, ':b', lw=2)

    sint = np.sin(theta) / 2
    cost = np.cos(theta) / 2
    plt.plot([xc - dx * cost, xc + dx * cost], [yc + dx * sint, yc - dx * sint], ':b')
    plt.plot([xc + dy * sint, xc - dy * sint], [yc + dy * cost, yc - dy * cost], ':r')

    # draw axes
    plt.annotate("x'", xy=(-25, 0), xytext=(25, 0),
                 arrowprops={'arrowstyle': '<-'}, va='center', fontsize=16)

    plt.annotate("y'", xy=(0, 25), xytext=(0, -25),
                 arrowprops={'arrowstyle': '<-'}, ha='center', fontsize=16)

    plt.annotate(r'$\phi$', xy=(13, -2.5), fontsize=16)
    plt.annotate('', xy=(15.5, 0), xytext=(14, -8.0),
                 arrowprops={'arrowstyle': '<-', 'connectionstyle': 'arc3, rad=-0.2'})

    plt.annotate(r'$d_x$', xy=(-17, 7), color='blue', fontsize=16)
    plt.annotate(r'$d_y$', xy=(-4, -8), color='red', fontsize=16)

    plt.xlim(-30, 30)
    plt.ylim(30, -30)  # inverted to match image coordinates!
    plt.axis('off')


def plot_visible_dotted_line(xpts, ypts):
    """Draw a dotted line that is is visible against images."""
    plt.plot(xpts, ypts, '-', color='#FFD700')
    plt.plot(xpts, ypts, ':', color='#0057B8')


def plot_image_and_fit(o_image,
                       pixel_size=None,
                       vmin=None,
                       vmax=None,
                       units='µm',
                       crop=False,
                       colorbar=False,
                       cmap='gist_ncar',
                       corner_fraction=0.035,
                       nT=3,
                       iso_noise=True,
                       **kwargs):
    """
    Plot the image, fitted ellipse, integration area, and semi-major/minor axes.

    If pixel_size is defined, then the returned measurements are in units of
    pixel_size.

    This function helpful when creating a mosaics of all images captured for an
    experiment.

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    All cropping is done after analysis and therefosre only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        o_image: 2D array of image with beam spot
        pixel_size: (optional) size of pixels
        vmin: (optional) minimum value for colorbar
        vmax: (optional) maximum value for colorbar
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        colorbar (optional) show the color bar,
        cmap: (optional) colormap to use

    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    # only pass along arguments that apply to beam_size()
    beamsize_keys = ['mask_diameters', 'max_iter', 'phi']
    bs_args = dict((k, kwargs[k]) for k in beamsize_keys if k in kwargs)
    bs_args['iso_noise'] = iso_noise
    bs_args['nT'] = nT
    bs_args['corner_fraction'] = corner_fraction

    # find center and diameters
    xc, yc, dx, dy, phi = lbs.beam_size(o_image, **bs_args)

    # establish scale and correct label
    if pixel_size is None:
        scale = 1
        label = 'Pixels'
    else:
        scale = pixel_size
        label = 'Position (%s)' % units

    # crop image if necessary
    if isinstance(crop, list):
        ymin = yc - crop[0] / 2 / scale  # in pixels
        ymax = yc + crop[0] / 2 / scale
        xmin = xc - crop[1] / 2 / scale
        xmax = xc + crop[1] / 2 / scale
        image, xc, yc = lbs.crop_image_to_rect(o_image, xc, yc, xmin, xmax, ymin, ymax)
    elif crop:
        image, xc, yc = lbs.crop_image_to_integration_rect(o_image, xc, yc, dx, dy, phi)
    else:
        image = o_image

    # establish maximum colorbar value
    if vmax is None:
        vmax = image.max()
    if vmin is None:
        vmin = image.min()

    # extents may be changed by scale
    v, h = image.shape
    extent = np.array([-xc, h - xc, v - yc, -yc]) * scale

    # display image and axes labels
    im = plt.imshow(image, extent=extent, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.xlabel(label)
    plt.ylabel(label)

    # draw semi-major and semi-minor axes
    xp, yp = lbs.axes_arrays(xc, yc, dx, dy, phi)
    plot_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # show ellipse around beam
    xp, yp = lbs.ellipse_arrays(xc, yc, dx, dy, phi)
    plot_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # show integration area around beam
    xp, yp = lbs.rotated_rect_arrays(xc, yc, dx, dy, phi)
    plot_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # set limits on axes
    plt.xlim(-xc * scale, (h - xc) * scale)
    plt.ylim((v - yc) * scale, -yc * scale)

    # show colorbar
    if colorbar:
        v, h = image.shape
        plt.colorbar(im, fraction=0.046 * v / h, pad=0.04)

    return xc * scale, yc * scale, dx * scale, dy * scale, phi


def plot_image_analysis(o_image,
                        title='Original',
                        pixel_size=None,
                        units='µm',
                        crop=False,
                        cmap='gist_ncar',
                        corner_fraction=0.035,
                        nT=3,
                        iso_noise=True,
                        **kwargs):
    """
    Create a visual report for image fitting.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    All cropping is done after analysis and therefosre only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        o_image: 2D image of laser beam
        title: (optional) title for upper left plot
        pixel_size: (optional) size of pixels
        far_field_lens: (optional) far field lens focal length
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        cmap: (optional) colormap to use
    Returns:
        nothing

    """
    # only pass along arguments that apply to beam_size()
    bs_args = dict((k, kwargs[k]) for k in ['mask_diameters', 'max_iter', 'phi'] if k in kwargs)
    bs_args['iso_noise'] = iso_noise
    bs_args['nT'] = nT
    bs_args['corner_fraction'] = corner_fraction

    # find center and diameters
    xc, yc, dx, dy, phi = lbs.beam_size(o_image, **bs_args)

    # determine scaling and labels
    if pixel_size is None:
        scale = 1
        unit_str = ''
        units = 'pixels'
        label = 'Pixels from Center'
    else:
        scale = pixel_size
        unit_str = '[%s]' % units
        label = 'Distance from Center %s' % unit_str
    

    # crop image as appropriate
    if isinstance(crop, list):
        ymin = yc - crop[0] / 2 / scale  # in pixels
        ymax = yc + crop[0] / 2 / scale
        xmin = xc - crop[1] / 2 / scale
        xmax = xc + crop[1] / 2 / scale
        image, xc, yc = lbs.crop_image_to_rect(o_image, xc, yc, xmin, xmax, ymin, ymax)
    elif crop:
        image, xc, yc = lbs.crop_image_to_integration_rect(o_image, xc, yc, dx, dy, phi)
    else:
        image = o_image

    # subtract background
    working_image = lbs.subtract_iso_background(image, corner_fraction=corner_fraction,
                                                nT=nT, iso_noise=iso_noise)
    bkgnd, _ = lbs.iso_background(image, corner_fraction=corner_fraction, nT=nT)

    min_ = image.min()
    max_ = image.max()
    vv, hh = image.shape

    # determine the sizes of the semi-major and semi-minor axes
    r_major = max(dx, dy) / 2.0
    r_minor = min(dx, dy) / 2.0

    # scale all the dimensions to convert pixels to um, mrad, etc...
    v_s = vv * scale # y image size
    h_s = hh * scale # x image size
    xc_s = xc * scale # beam center location in x
    yc_s = yc * scale # beam center location in y
    r_mag_s = r_major * scale # semi-major radius
    d_mag_s = r_mag_s * 2 # semi-major diameter
    r_min_s = r_minor * scale # semi-minor radius
    d_min_s = r_min_s * 2 # semi-minor diameter

    plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(right=1.0)

    # original image
    plt.subplot(2, 2, 1)
    im = plt.imshow(image, cmap=cmap)
    plt.colorbar(im, fraction=0.046 * v_s / h_s, pad=0.04)
    plt.clim(min_, max_)
    plt.xlabel('Position (pixels)')
    plt.ylabel('Position (pixels)')
    plt.title(title)

    # working image
    plt.subplot(2, 2, 2)
    extent = np.array([-xc_s, h_s - xc_s, v_s - yc_s, -yc_s])
    # extent = np.array([-h_s/2, h_s/2, v_s/2, -v_s/2]) # would requre changing all shape coords as well
    im = plt.imshow(working_image, extent=extent, cmap=cmap)
    xp, yp = lbs.ellipse_arrays(xc, yc, dx, dy, phi) * scale
    plot_visible_dotted_line(xp - xc_s, yp - yc_s)

    xp, yp = lbs.axes_arrays(xc, yc, dx, dy, phi) * scale
    plot_visible_dotted_line(xp - xc_s, yp - yc_s)

    xp, yp = lbs.rotated_rect_arrays(xc, yc, dx, dy, phi) * scale # TODO: Use these for KE calcs
    plot_visible_dotted_line(xp - xc_s, yp - yc_s)

    plt.colorbar(im, fraction=0.046 * v_s / h_s, pad=0.04)
#    plt.clim(min_, max_)
    plt.xlim(-xc_s, h_s - xc_s)
    plt.ylim(v_s - yc_s, -yc_s)
    plt.xlabel(label)
    plt.ylabel(label)
    if pixel_size is None:
        plt.title('Image w/o background, center: (%.0f, %.0f) %s' % (xc_s - (h_s/2), (v_s/2) - yc_s, units))
    else:
        plt.title('Image w/o background, center: (%.3f, %.3f) %s' % (xc_s - (h_s/2), (v_s/2) - yc_s, units))
    # xc_s is relative to left edge in either pixel coords or um coords
    # yc_s is relative to bottom ...


    # plot of values along semi-major axis
    _, _, z, s = lbs.major_axis_arrays(image, xc, yc, dx, dy, phi)
    a = np.sqrt(2 / np.pi) / r_major * abs(np.sum(z - bkgnd) * (s[1] - s[0]))
    baseline = a * np.exp(-2) + bkgnd

    plt.subplot(2, 2, 3)
    plt.plot(s * scale, z, 'sb', markersize=2)
    plt.plot(s * scale, z, '-b', lw=0.5)
    z_values = bkgnd + a * np.exp(-2 * (s / r_major)**2)
    plt.plot(s * scale, z_values, 'k')
    plt.annotate('', (-r_mag_s, baseline), (r_mag_s, baseline),
                 arrowprops={'arrowstyle': '<->'})
    plt.text(0, 1.1 * baseline, 'dx=%.0f %s' % (d_mag_s, units), va='bottom', ha='center')
    plt.text(0, bkgnd + a, '  Gaussian Fit')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Major Axis')
    plt.title('Semi-Major Axis')
    # plt.gca().set_ylim(bottom=0)

    # plot of values along semi-minor axis
    _, _, z, s = lbs.minor_axis_arrays(image, xc, yc, dx, dy, phi)
    a = np.sqrt(2 / np.pi) / r_minor * abs(np.sum(z - bkgnd) * (s[1] - s[0]))
    baseline = a * np.exp(-2) + bkgnd

    plt.subplot(2, 2, 4)
    plt.plot(s * scale, z, 'sb', markersize=2)
    plt.plot(s * scale, z, '-b', lw=0.5)
    z_values = bkgnd + a * np.exp(-2 * (s / r_minor)**2)
    plt.plot(s * scale, z_values, 'k')
    plt.annotate('', (-r_min_s, baseline), (r_min_s, baseline),
                 arrowprops={'arrowstyle': '<->'})
    plt.text(0, 1.1 * baseline, 'dy=%.0f %s' % (d_min_s, units), va='bottom', ha='center')
    plt.text(0, bkgnd + a, '  Gaussian Fit')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Minor Axis')
    plt.title('Semi-Minor Axis')
    # plt.gca().set_ylim(bottom=0)

    # add more horizontal space between plots
    plt.subplots_adjust(wspace=0.3)


def plot_image_montage(images,
                       z=None,
                       cols=3,
                       pixel_size=None,
                       vmax=None,
                       vmin=None,
                       units='µm',
                       crop=False,
                       cmap='gist_ncar',
                       corner_fraction=0.035,
                       nT=3,
                       iso_noise=True,
                       **kwargs):
    """
    Create a beam size montage for a set of images.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    All cropping is done after analysis and therefosre only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        images: array of 2D images of the laser beam
        z: (optional) array of axial positions of images (always in meters!)
        cols: (optional) number of columns in the montage
        pixel_size: (optional) size of pixels
        vmax: (optional) maximum gray level to use
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        cmap: (optional) colormap to use
    Returns:
        dx: semi-major diameter
        dy: semi-minor diameter
    """
    # arrays to save diameters
    dx = np.zeros(len(images))
    dy = np.zeros(len(images))

    # calculate the number of rows needed in the montage
    rows = (len(images) - 1) // cols + 1

    # when pixel_size is not specified, units default to pixels
    if pixel_size is None:
        units = 'pixels'

    # gather all the options that are fixed for every image in the montage
    options = {'pixel_size': pixel_size,
               'vmax': vmax,
               'vmin': vmin,
               'units': units,
               'crop': crop,
               'cmap': cmap,
               'corner_fraction': corner_fraction,
               'nT': nT,
               'iso_noise': iso_noise,
               **kwargs}

    # now set up the grid of subplots
    plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    for i, im in enumerate(images):
        plt.subplot(rows, cols, i + 1)

        # should we add color bar?
        cb = not (vmax is None) and (i + 1 == cols)

        # plot the image and gather the beam diameters
        _, _, dx[i], dy[i], _ = plot_image_and_fit(im, **options, colorbar=cb)

        # add a title
        if units == 'mm':
            s = "dx=%.2f%s, dy=%.2f%s" % (dx[i], units, dy[i], units)
        else:
            s = "dx=%.0f%s, dy=%.0f%s" % (dx[i], units, dy[i], units)
        if z is None:
            plt.title(s)
        else:
            plt.title("z=%.0fmm, %s" % (z[i] * 1e3, s))

        # omit y-labels on all but first column
        if i % cols:
            plt.ylabel("")
            if isinstance(crop, list):
                plt.yticks([])

        # omit x-labels on all but last row
        if i < (rows - 1) * cols:
            plt.xlabel("")
            if isinstance(crop, list):
                plt.xticks([])

    for i in range(len(images), rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.axis("off")

    return dx, dy


def plot_knife_edge_analysis(img,
                             title='Knife-Edge Analysis', 
                             pixel_size=None, 
                             units='µm', 
                             kep=[0.05, 0.95],
                             rotate=True,
                             cmap='gist_ncar',
                             corner_fraction=0.035,
                             nT=3,
                             iso_noise=True,
                             **kwargs):
    """
    Fast knife-edge method, works by rotating the image and then summing the up all the pixels along the X and Y axis.

    Args:
        img: noise subtracted image
        title: plot title
        pixel_size: (optional) size of pixels
        units: image units, can be used for converting far field
        kep: (optional) knife-edge points to measure beam from, default is 5%-95%
        rotate: Allow knife-edge to rotate with major/minor axis
    Returns:
        nothing
    """
    # determine scaling and labels
    if pixel_size is None:
        scale = 1
        unit_str = ''
        units = 'pixels'
        # label = 'Pixels from Center'
    else:
        scale = pixel_size
        unit_str = units
        # label = 'Distance from Center %s' % unit_str
    
    # only pass along arguments that apply to beam_size()
    bs_args = dict((k, kwargs[k]) for k in ['mask_diameters', 'max_iter', 'phi'] if k in kwargs)
    bs_args['iso_noise'] = iso_noise
    bs_args['nT'] = nT
    bs_args['corner_fraction'] = corner_fraction

    # find center and diameters
    x, y, dx, dy, phi = lbs.beam_size(img, **bs_args)

    # scale all the dimensions to units of interest
    vv,hh = img.shape
    v_s = vv * scale
    h_s = hh * scale
    x_s = x * scale
    y_s = y * scale
    dx_s = dx * scale
    dy_s = dy * scale
    major_s = np.max([dx_s, dy_s])
    minor_s = np.min([dx_s, dy_s])

    # Subtract background
    wimg = lbs.subtract_iso_background(img)

    # Mask beam
    mask = lbs.rotated_rect_mask(wimg, x, y, dx, dy, phi)
    
    # Working image
    mwimg = np.copy(wimg)
    
    # Apply mask
    mwimg[mask < 0] = 0
    
    # Rotated image
    # --------------------------------------------------------------------------------------------------- #
    if rotate:
        img_r = lbs.rotate_image(mwimg, x, y, -phi)
        x_r_, y_r_, dx_r_, dy_r_, _ = lbs.beam_size(img_r, **bs_args)
        img_r, _, _  = lbs.crop_image_to_integration_rect(img_r, x_r_, y_r_, dx_r_, dy_r_, 0)
        x_r, y_r, dx_r, dy_r, phi_r = lbs.beam_size(img_r, **bs_args)
    else:
        img_r = np.copy(mwimg)
        x_r_, y_r_, dx_r_, dy_r_, _ = lbs.beam_size(img_r, **bs_args)
        img_r, _, _  = lbs.crop_image_to_integration_rect(img_r, x_r_, y_r_, dx_r_, dy_r_, phi)
        x_r, y_r, dx_r, dy_r, phi_r = lbs.beam_size(img_r, **bs_args)

    vv_r,hh_r = img_r.shape
    v_rs = vv_r * scale
    h_rs = hh_r * scale
    x_rs = x_r * scale
    y_rs = y_r * scale
    dx_rs = dx_r * scale
    dy_rs = dy_r * scale
    major_rs = np.max([dx_rs, dy_rs])
    minor_rs = np.min([dx_rs, dy_rs])

    # Knife edge math
    x_scan_y_axis, x_scan_x_axis, bwsx = lbs.knife_edge(img_r, 0, kep)
    y_scan_y_axis, y_scan_x_axis, bwsy = lbs.knife_edge(img_r, 1, kep)

    # plotting
    # --------------------------------------------------------------------------------------------------- #
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=30)

    # [0,0] - Basic Plot
    # --------------------------------------------------------------------------------------------------- #
    extent = np.array([-h_s/2, h_s/2, v_s/2, -v_s/2])
    im = axs[0,0].imshow(mwimg, extent=extent, cmap=cmap) #, cmap='gist_ncar')
    plt.colorbar(im, ax=axs[0,0], fraction=0.046 * v_s / h_s, pad=0.04)

    # Rectangular array
    xp,yp = lbs.rotated_rect_arrays(x, y, dx, dy, phi) * scale
    axs[0,0].plot(xp-h_s/2,yp-v_s/2,':y')

    # Ellipse array
    xp,yp = lbs.ellipse_arrays(x, y, dx, dy, phi) * scale
    axs[0,0].plot(xp-h_s/2,yp-v_s/2,':y')

    # Axes array
    xp, yp = lbs.axes_arrays(x, y, dx, dy, phi) * scale
    axs[0,0].plot(xp-h_s/2,yp-v_s/2,':y')

    # Crosshair array
    xp, yp = lbs.axes_arrays(hh/2, vv/2, hh/3, vv/3, 0) * scale
    axs[0,0].plot(xp-h_s/2,yp-v_s/2,':r')#,linewidth=1)

    # Plot formatting
    axs[0,0].set_title('ISO Noise Subtracted Image', fontsize=22)
    axs[0,0].set_xlim(-h_s/2, h_s/2)
    axs[0,0].set_ylim(v_s/2, -v_s/2)
    axs[0,0].set_xlabel("X Coordinate (%s)" % unit_str, fontsize=16)
    axs[0,0].set_ylabel("Y Coordinate (%s)" % unit_str, fontsize=16)

    # Beam stats
    m1 = 'Centroid: (%.2f, %.2f) %s' % (x*scale - h_s/2, y*scale - v_s/2, unit_str)
    m2 = 'D4\u03C3:        (%.2f, %.2f) %s' % (dx_s, dy_s, unit_str)
    m3 = 'D4\u03C3 Circ: %.2f' % (minor_s/major_s)
    m4 = 'Angle:      %.2f\N{DEGREE SIGN}' % (phi*180/np.pi)
    axs[0,0].text(-h_s/2 * 0.95, v_s/2*0.95, '\n'.join([m1,m2,m3,m4]), va='bottom', ha='left',c='white',fontsize=10)
    
    # [0,1] Rotated Plot
    # --------------------------------------------------------------------------------------------------- #
    extent_r = np.array([-h_rs/2, h_rs/2, v_rs/2, -v_rs/2])
    im = axs[0,1].imshow(img_r, extent=extent_r, cmap=cmap)
    plt.colorbar(im, ax=axs[0,1], fraction=0.046 * v_s / h_s, pad=0.04)

    # Rectangular array
    xp,yp = lbs.rotated_rect_arrays(x_r, y_r, dx_r, dy_r, phi_r) * scale
    axs[0,1].plot(xp-h_rs/2,yp-v_rs/2,':y')

    # Ellipse array
    xp,yp = lbs.ellipse_arrays(x_r, y_r, dx_r, dy_r, phi_r) * scale
    axs[0,1].plot(xp-h_rs/2,yp-v_rs/2,':y')

    # Axes array
    xp, yp = lbs.axes_arrays(x_r, y_r, dx_r, dy_r, phi_r) * scale
    axs[0,1].plot(xp-h_rs/2,yp-v_rs/2,':y')

    # Crosshair array
    xp, yp = lbs.axes_arrays(hh_r/2, vv_r/2, hh_r/3, vv_r/3, 0) * scale
    axs[0,1].plot(xp-h_rs/2,yp-v_rs/2,':r')

    # Plot formatting
    axs[0,1].set_title('Image Oriented To Knife-Edge Axis', fontsize=22)
    axs[0,1].set_xlim(-h_rs/2, h_rs/2)
    axs[0,1].set_ylim(v_rs/2, -v_rs/2)
    axs[0,1].set_xlabel("X Coordinate (%s)" % unit_str, fontsize=16)
    axs[0,1].set_ylabel("Y Coordinate (%s)" % unit_str, fontsize=16)
    
    # Beam stats
    m1 = 'Centroid: (%.2f, %.2f) %s' % (x_r*scale - h_rs/2, y_r*scale - v_rs/2, unit_str)
    m2 = 'D4\u03C3:        (%.2f, %.2f) %s' % (dx_rs, dy_rs, unit_str)
    m3 = 'D4\u03C3 Circ: %.2f' % (minor_rs/major_rs)
    m4 = 'Angle:      %.2f\N{DEGREE SIGN}' % (phi_r*180/np.pi)
    axs[0,1].text(-h_rs/2 * 0.95, v_rs/2*0.95, '\n'.join([m1,m2,m3,m4]), va='bottom', ha='left',c='white',fontsize=10)

    # [1,X] - Knife edge plot for x scan
    # --------------------------------------------------------------------------------------------------- #
    axs[1,0].set_title("Knife-Edge Plot, Minor Axis", fontsize=22)
    axs[1,1].set_title("Knife-Edge Plot, Major Axis", fontsize=22)
    
    # Find major and minor axis
    if dx < dy:
        # x is minor axis
        ax1 = (1,0)
        ax2 = (1,1)
    else:
        # x is major axis
        ax1 = (1,1)
        ax2 = (1,0)

    axs[ax1].set_xlabel("Knife Position (%s)" % unit_str, fontsize=16)
    axs[ax1].set_ylabel("Fractional Power", fontsize = 16)
    axs[ax1].plot(x_scan_x_axis * scale, x_scan_y_axis)
    axs[ax1].grid('on')
    
    # Plot lines
    axs[ax1].axvline(x=bwsx[0] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[0]*100), ls='--')
    axs[ax1].axvline(x=bwsx[1] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[1]*100), ls='--')

    # Show distance on plot
    axs[ax1].annotate('', (bwsx[0] * scale, kep[0]), (bwsx[1] * scale, kep[0]), arrowprops={'arrowstyle': '<->'})
    axs[ax1].text(np.average(bwsx) * scale, 1.1 * kep[0], 'dx=%.2f %s' % ((bwsx[1]-bwsx[0]) * scale, unit_str), va='bottom', ha='center')
    
    # [1,!X] - Knife edge plot for y scan
    # --------------------------------------------------------------------------------------------------- #
    axs[ax2].set_xlabel("Knife Position (%s)" % unit_str, fontsize=16)
    axs[ax2].set_ylabel("Fractional Power", fontsize = 16)
    axs[ax2].plot(y_scan_x_axis * scale, y_scan_y_axis)
    axs[ax2].grid('on')

    # Plot lines
    axs[ax2].axvline(x=bwsy[0] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[0]*100), ls='--')
    axs[ax2].axvline(x=bwsy[1] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[1]*100), ls='--')

    # Show distance on plot
    axs[ax2].annotate('', (bwsy[0] * scale, kep[0]), (bwsy[1] * scale, kep[0]), arrowprops={'arrowstyle': '<->'})
    axs[ax2].text(np.average(bwsy) * scale, 1.1 * kep[0], 'dy=%.2f %s' % ((bwsy[1]-bwsy[0]) * scale, unit_str), va='bottom', ha='center')

    fig.tight_layout(pad=2)
    
    plt.show()

    return

def plot_knife_edge_analysis_slow(img, 
                                  pixel_size=None, 
                                  units='µm', 
                                  points = 50, 
                                  kep=[0.05, 0.95], 
                                  title='Knife-Edge Analysis'):
    """
    Slow knife-edge method. Works by creating a rotated mask that expands to cover varying fractions of the beam. The 'points' 
    input specifies how many points to include between the start and end knife points. Knife region is defines as the rectangle
    that is 3 times the d4sigma width.

    Args:
        img: noise subtracted image
        pixel_size: (optional) size of pixels
        points: number of knife-edge points
        kep: knife-edge points to measure beam from
        title: plot title
    """
    # determine scaling and labels
    if pixel_size is None:
        scale = 1
        unit_str = ''
        units = 'pixels'
        # label = 'Pixels from Center'
    else:
        scale = pixel_size
        unit_str = units
        # label = 'Distance from Center %s' % unit_str
    
    # Parse beam
    x, y, dx, dy, phi = lbs.beam_size(img)

    # scale all the dimensions to units of interest
    vv,hh = img.shape
    v_s = vv * scale
    h_s = hh * scale
    x_s = x * scale
    y_s = y * scale
    dx_s = dx * scale
    dy_s = dy * scale
    major_s = np.max([dx_s, dy_s])
    minor_s = np.min([dx_s, dy_s])

    # Subtract background
    wimg = lbs.subtract_iso_background(img)

    # Mask beam
    mask = lbs.rotated_rect_mask(wimg, x, y, dx, dy, phi)
    
    # Working image
    mwimg = np.copy(wimg)
    
    # Apply mask
    mwimg[mask < 0] = 0
    
    # Knife edge math
    # --------------------------------------------------------------------------------------------------- #

    # Create arrays for x axis
    x_scan_x_axis = np.linspace(0,1,points)
    x_scan_y_axis = np.full_like(x_scan_x_axis, 0.0)
    
    # Create arrays for y axis
    y_scan_x_axis = np.linspace(0,1,points)
    y_scan_y_axis = np.full_like(x_scan_x_axis, 0.0)

    # Collect knife-edge data for x scan
    for i, x_val in enumerate(x_scan_x_axis):
        mask = lbs.knife_edge_mask(mwimg, x, y, dx, dy, phi, x_val, dir = 'x')
        x_scan_y_axis[i] = np.sum(wimg * mask)

    # Normalize knife edge
    x_scan_y_axis = x_scan_y_axis/np.max(x_scan_y_axis)
    
    # Collect knife-edge data for y scan
    for i, x_val in enumerate(y_scan_x_axis):
        mask = lbs.knife_edge_mask(mwimg, x, y, dx, dy, phi, x_val, dir = 'y')
        y_scan_y_axis[i] = np.sum(wimg * mask)

    # Normalize knife edge
    y_scan_y_axis = y_scan_y_axis/np.max(y_scan_y_axis)

    # plotting
    # --------------------------------------------------------------------------------------------------- #
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    fig.suptitle(title, fontsize=30)

    # Plot 1
    # --------------------------------------------------------------------------------------------------- #
    im = axs[0,0].imshow(img) #, cmap='gist_ncar')
    plt.colorbar(im, ax=axs[0,0])
    axs[0,0].set_title('Raw Image', fontsize=22)
    axs[0,0].set_xlabel("X Coordinate (pixel)", fontsize=16)
    axs[0,0].set_ylabel("Y Coordinate (pixel)", fontsize=16)

    # Plot 2
    # --------------------------------------------------------------------------------------------------- #
    extent = np.array([-h_s/2, h_s/2, v_s/2, -v_s/2])
    axs[0,1].imshow(mwimg, extent=extent) #, cmap='gist_ncar')
    plt.colorbar(im, ax=axs[0,1])
    axs[0,1].set_title('ISO Noise Subtracted Image', fontsize=22)

    # Plot masks
    xp,yp = lbs.rotated_rect_arrays(x, y, dx, dy, phi) * scale
    axs[0,1].plot(xp-h_s/2,yp-v_s/2,':y')

    xp,yp = lbs.ellipse_arrays(x, y, dx, dy, phi) * scale
    axs[0,1].plot(xp-h_s/2,yp-v_s/2,':y')


    xp, yp = lbs.axes_arrays(x, y, dx, dy, phi) * scale
    axs[0,1].plot(xp-h_s/2,yp-v_s/2,':y')

    # Center Crosshairs
    xp, yp = lbs.axes_arrays(hh/2, vv/2, hh/3, vv/3, 0) * scale
    axs[0,1].plot(xp-h_s/2,yp-v_s/2,':r')#,linewidth=1)

    # Limit plot size
    #vv, hh = wimg.shape
    axs[0,1].set_xlim(-h_s/2, h_s/2)
    axs[0,1].set_ylim(v_s/2, -v_s/2)
    axs[0,1].set_xlabel("X Coordinate (%s)" % unit_str, fontsize=16)
    axs[0,1].set_ylabel("Y Coordinate (%s)" % unit_str, fontsize=16)

    m1 = 'Centroid: (%.2f, %.2f) %s' % (x*scale - h_s/2, y*scale - v_s/2, unit_str)
    m2 = 'D4\u03C3:        (%.2f, %.2f) %s' % (dx_s, dy_s, unit_str)
    m3 = 'D4\u03C3 Circ: %.2f' % (minor_s/major_s)
    m4 = 'Angle:      %.2f\N{DEGREE SIGN}' % (phi*180/np.pi)
    axs[0,1].text(h_s/2*0.05, v_s/2*0.95, '\n'.join([m1,m2,m3,m4]), va='bottom', ha='left',c='white',fontsize=10)
    
    # Knife edge plots
    # --------------------------------------------------------------------------------------------------- #
    axs[1,0].set_title("Knife-Edge Plot, Minor Axis", fontsize=22)
    axs[1,1].set_title("Knife-Edge Plot, Major Axis", fontsize=22)
    # Find major and minor axis
    if dx < dy:
        # x is minor axis
        ax1 = (1,0)
        ax2 = (1,1)
    else:
        # x is major axis
        ax1 = (1,1)
        ax2 = (1,0)

    # Convert x axis to pixels
    x_scan_x_axis_p = x_scan_x_axis * dx * 3

    # Interpolate values
    bws = np.interp([kep[0], kep[1]], x_scan_y_axis, x_scan_x_axis_p)

    axs[ax1].set_xlabel("Knife Position (%s)" % unit_str, fontsize=16)
    axs[ax1].set_ylabel("Fractional Power", fontsize = 16)
    axs[ax1].plot(x_scan_x_axis_p * scale, x_scan_y_axis)
    axs[ax1].grid('on')
    
    # Plot lines
    axs[ax1].axvline(x=bws[0] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[0]*100), ls='--')
    axs[ax1].axvline(x=bws[1] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[1]*100), ls='--')

    # Show distance on plot
    axs[ax1].annotate('', (bws[0] * scale, kep[0]), (bws[1] * scale, kep[0]), arrowprops={'arrowstyle': '<->'})
    axs[ax1].text(np.average(bws) * scale, 1.1 * kep[0], 'dx=%.2f %s' % ((bws[1]-bws[0]) * scale, unit_str), va='bottom', ha='center')
    
    # Knife edge plots
    # --------------------------------------------------------------------------------------------------- #
    # Convert x axis to pixels
    y_scan_x_axis_p = y_scan_x_axis * dy * 3

    # Interpolate values
    bws = np.interp([kep[0], kep[1]], y_scan_y_axis, y_scan_x_axis_p)

    axs[ax2].set_xlabel("Knife Position (%s)" % unit_str, fontsize=16)
    axs[ax2].set_ylabel("Fractional Power", fontsize = 16)
    axs[ax2].plot(y_scan_x_axis_p * scale, y_scan_y_axis)
    axs[ax2].grid('on')

    # Plot lines
    axs[ax2].axvline(x=bws[0] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[0]*100), ls='--')
    axs[ax2].axvline(x=bws[1] * scale, ymin=0.05, ymax=0.95, color='b', label='{:.0f}% Point'.format(kep[1]*100), ls='--')

    # Show distance on plot
    axs[ax2].annotate('', (bws[0] * scale, kep[0]), (bws[1] * scale, kep[0]), arrowprops={'arrowstyle': '<->'})
    axs[ax2].text(np.average(bws) * scale, 1.1 * kep[0], 'dy=%.2f %s' % ((bws[1]-bws[0]) * scale, unit_str), va='bottom', ha='center')

    fig.tight_layout(pad=2)

    plt.show()

    return


