import timeit
import time
import csv
import matplotlib.pyplot as plt
from skimage.transform import warp, warp_coords
from skimage.filters import gaussian
from skimage import data
from skimage.viewer import ImageViewer
from scipy.ndimage import map_coordinates
import numpy as np
from skimage.transform import SimilarityTransform

class RGCMap():
    """
    Remaps images with foveated resolution modelled on macaque retinal ganglion cells. The map is
    organized in an eccentricity-angle grid. Foveated cortex-like representations can be produced
    by convolution of this map. As in the macaque, a map includes only one half of the scene, and
    there are maps with different resolutions for parvocellular and magnocellular systems. The
    caller specifies the left or right field, and parvo or magnocellular.
    """

    def __init__(self, image_width, right=True, parvo=True, pixels_per_radius=0.5, radii_per_step=2, show_fit=False):
        assert radii_per_step > 0
        assert pixels_per_radius > 0

        self.image_width = image_width
        self.right_field = right
        self.parvo = parvo
        self.pixels_per_foveal_radius = pixels_per_radius
        self.radii_per_step = radii_per_step

        ce, cr = get_RCG_radii(parvo=self.parvo, centre=True)
        self.centre_radii = np.poly1d(np.polyfit(ce, cr, 2))

        se, sr = get_RCG_radii(parvo=self.parvo, centre=False)
        self.surround_radii = np.poly1d(np.polyfit(se, sr, 2))

        if show_fit:
            xx = np.linspace(0, 90, 20)
            plt.plot(ce, cr, '.', xx, self.centre_radii(xx), '-')
            plt.plot(se, sr, 'x', xx, self.surround_radii(xx), '--')
            plt.show()


    def get_radial_positions(self):
        pixels_per_degree = self.pixels_per_foveal_radius / self.centre_radii(0)

        def get_euler_step(eccentricity):
            return self.centre_radii(eccentricity) * self.radii_per_step

        def get_trapezoidal_step(eccentricity):
            euler_step = get_euler_step(eccentricity)
            radius_before_step = self.centre_radii(eccentricity)
            radius_after_step = self.centre_radii(eccentricity + euler_step)
            return (radius_before_step + radius_after_step) / 2 * self.radii_per_step

        degrees = [get_trapezoidal_step(0)]
        while pixels_per_degree * degrees[-1] <= self.image_width / 2:
            # print(degrees[-1])
            step = get_trapezoidal_step(degrees[-1])
            degrees.append(degrees[-1] + step)

        degrees = np.array(degrees[:-2]) #last one is overshoot
        return degrees, pixels_per_degree * degrees

    def remap(self, image):
        pass



def mean_centre_radius_over_eccentricity(parvo=True):
    eccentricities, radii = get_RCG_radii(parvo, centre=True)
    return np.mean(radii) / np.mean(eccentricities)


def get_RCG_radii(parvo=True, centre=True):
    figure = 'a' if centre else 'b'
    series = 'P' if parvo else 'M'
    filename = './data/caplan_1995_4{}_{}'.format(figure, series)

    eccentricities = []
    radii = []
    with open(filename) as file:
        r = csv.reader(file)
        for row in r:
            eccentricities.append(float(row[0]))
            radii.append(float(row[1]))

    return np.array(eccentricities), np.array(radii)


def make_target_image(shape=(400,400,3)):
    image = np.zeros(shape, dtype='uint8')
    print(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            radius = np.sqrt((i - shape[0]/2)**2 + (j - shape[1]/2)**2)
            if int(radius / 20) % 2 == 0:
                image[i,j,0] = 255
            else:
                image[i,j,2] = 255

            if radius < 20:
                image[i,j,2] = 255

    return image

image = data.chelsea()
# image = make_target_image(image.shape)

image = gaussian(image, 1.0)

class AngleEccentricityMap:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __call__(self, xy):
        print(self.input_shape)
        angle = xy[:, 0] / self.output_shape[0] * 2 * np.pi # in radians
        fraction_eccentricity = (np.exp(2 * xy[:, 1] / self.output_shape[1]) - 1) / np.exp(2)
        eccentricity = fraction_eccentricity * (np.min(self.input_shape[:2]) / 2) # in pixels
        xy[:, 0] = self.input_shape[1] / 2 + eccentricity * np.sin(angle)
        xy[:, 1] = self.input_shape[0] / 2 - eccentricity * np.cos(angle)
        # print(xy)
        print('angle range: {} to {}'.format(min(angle), max(angle)))
        print('eccentricity range: {} to {}'.format(min(eccentricity), max(eccentricity)))
        # print(np.min(xy))
        # print(np.max(xy))
        return xy


rgcm = RGCMap(256, show_fit=False)
degrees, pixels = rgcm.get_radial_positions()
print(degrees)
print(pixels.shape)

# get_RCG_radii(parvo=False, centre=False)
# print(mean_centre_radius_over_eccentricity())

# map = AngleEccentricityMap(image.shape, (200, 200))
# # coords = warp_coords(map, map.input_shape)
# coords = warp_coords(map, (200,200,3))
# # coords = warp_coords(map, image.shape)
# # print(coords.shape)
# print(image.shape)
#
# # before = time.time()
# warped = map_coordinates(image, coords)
# # print(time.time() - before)
# print(warped.shape)
#
# viewer = ImageViewer(image)
# # viewer = ImageViewer(warped)
# viewer.show()
