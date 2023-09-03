import numpy as np
import pytest
import laserbeamsize

########## rotate_points 

def test_rotate_points_0_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, 0)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)

def test_rotate_points_90_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, np.pi/2)
    assert np.isclose(x, 0, atol=1e-8)
    assert np.isclose(y, -1, atol=1e-8)

def test_rotate_points_180_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, np.pi)
    assert np.isclose(x, -1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)

def test_rotate_points_360_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, 2*np.pi)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)

########## values_along_line 

def test_values_along_line():
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 1, 1, 2)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678,  0.70710678]))

def test_values_along_line_vertical():
    image = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 0, 3, 4)
    assert np.all(x == np.array([0, 0, 0, 0]))
    assert np.all(y == np.array([0, 1, 2, 3]))
    assert np.all(z == np.array([0, 2, 4, 6]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))

def test_values_along_line_horizontal():
    image = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 3, 0, 4)
    assert np.all(x == np.array([0, 1, 2, 3]))
    assert np.all(y == np.array([0, 0, 0, 0]))
    assert np.all(z == np.array([0, 1, 2, 3]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))

def test_values_along_line_diagonal_small():
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 1, 1, 2)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678,  0.70710678]))

############## major_axis_arrays
def test_major_axis_arrays_horizontal_major():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 4, 3, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_major_axis_arrays_vertical_major():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 3, 4, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_major_axis_arrays_large_diameter():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 10, 2, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_major_axis_arrays_rotated():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 4, 2, np.pi/4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 3)
    assert np.isclose(y[0], 3)
    assert np.isclose(y[-1], 0)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

############## minor_axis_arrays
def test_minor_axis_arrays_horizontal():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 4, 3, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_minor_axis_arrays_vertical():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 3, 4, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_minor_axis_arrays_large_diameter():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 10, 2, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_minor_axis_arrays_rotated():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 4, 2, np.pi/4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 3)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 3)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

########### subtract_background_image

def test_basic_subtraction():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=float)
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)
    
    result = laserbeamsize.subtract_background_image(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtraction_iso_false():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = np.array([[10, 15, 20], [15, 20, 25]], dtype=float)
    expected = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
    
    result = laserbeamsize.subtract_background_image(original, background, iso_noise=False)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtraction_iso_true():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = np.array([[10, 15, 20], [15, 20, 25]], dtype=float)
    expected = np.array([[-5, -5, -5], [-5, -5, -5]], dtype=float)
    
    result = laserbeamsize.subtract_background_image(original, background, iso_noise=True)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_subtraction_type_float():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=np.uint8)
    
    result = laserbeamsize.subtract_background_image(original, background, iso_noise=False)
    assert result.dtype == float

########### subtract_constant

def test_basic_subtract_constant():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = 5
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)
    
    result = laserbeamsize.subtract_constant(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtract_constant_iso_false():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[0, 0, 5], [0, 5, 10]], dtype=float)
    
    result = laserbeamsize.subtract_constant(original, background, iso_noise=False)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtract_constant_iso_true():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[-5, 0, 5], [0, 5, 10]], dtype=float)
    
    result = laserbeamsize.subtract_constant(original, background, iso_noise=True)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_subtract_constant_type_float():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = 5
    
    result = laserbeamsize.subtract_constant(original, background)
    assert result.dtype == np.float64

########### subtract_constant

def test_no_rotation():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = laserbeamsize.rotate_image(original, 1, 1, 0)
    assert np.array_equal(original, result)

def test_full_rotation():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = laserbeamsize.rotate_image(original, 1, 1, 2 * np.pi)
    assert np.array_equal(original, result)

def test_half_rotation():
    original = np.array([[200, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    expected = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 200]], dtype=np.uint8)
    result = laserbeamsize.rotate_image(original, 1, 1, np.pi)
    assert np.array_equal(expected, result)

def test_quarter_rotation():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    expected = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
    result = laserbeamsize.rotate_image(original, 1, 1, np.pi/2)
    assert np.array_equal(expected, result)

def test_rotate_and_crop():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = laserbeamsize.rotate_image(original, 1, 1, np.pi / 4)
    assert original.shape == result.shape
