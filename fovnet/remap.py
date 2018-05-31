import matplotlib.pyplot as plt
from skimage.transform import warp, warp_coords
from skimage.filters import gaussian
from skimage import data
from skimage.viewer import ImageViewer
from scipy.ndimage import map_coordinates
import numpy as np
from scipy.interpolate import interp1d
from skimage.transform import SimilarityTransform
from fovnet.data.retina import get_RCG_radii
from fovnet.data.retina import get_density_fit, get_centre_radius_fit, get_surround_radius_fit


class RGCMap():
    def __init__(self, source_pixels, source_degrees, scale, angle_steps, right=True, parvo=True, centre=True):
        self.source_pixels = source_pixels
        self.source_degrees = source_degrees
        self.pixels_per_degree = source_pixels / source_degrees
        self.scale = scale
        self.density_function = get_density_fit(parvo)

        print(self.pixels_per_degree)

        self.radial_pixel_positions = self.get_radial_positions()
        if right:
            self.angles = -np.pi/2 + np.linspace(0, np.pi, angle_steps)
        else:
            self.angles = 3*np.pi/2 - np.linspace(0, np.pi, angle_steps)

        centre_radius_fit = get_centre_radius_fit(parvo=parvo)
        self.centre_radii = self.pixels_per_degree \
            * centre_radius_fit(self.radial_pixel_positions / self.pixels_per_degree)

        surround_radius_fit = get_surround_radius_fit()
        self.surround_radii = self.pixels_per_degree \
            * surround_radius_fit(self.radial_pixel_positions / self.pixels_per_degree)

    def pixels_between_rfs(self, eccentricity):
        """
        :param eccentricity: distance from fovea in degrees visual angle
        :return: number of pixels between RF centres at that eccentricity
        """

        # This is basically unit book-keeping, as follows:
        # map_density(map_cells/degree) = scale(map_cells/actual_cell) * density(actual_cells/degree)
        # pixels/map_cell = (pixels/degree) / map_density(map_cells/degree)

        map_density = self.scale * self.density_function(eccentricity)
        print('density {} map-density {} pixel-step {}'.format(self.density_function(eccentricity), map_density, self.pixels_per_degree / map_density))
        return self.pixels_per_degree / map_density

    def rf_step_degrees(self, eccentricity):
        map_density = self.scale * self.density_function(eccentricity)
        return 1 / map_density

    def get_radial_positions(self):
        def get_euler_step(eccentricity):
            return self.rf_step_degrees(eccentricity)

        def get_trapezoidal_step(eccentricity):
            euler_step = get_euler_step(eccentricity)
            step_size_after_step = self.rf_step_degrees(eccentricity + euler_step)
            return (euler_step + step_size_after_step) / 2

        degrees = [get_trapezoidal_step(0)]
        while degrees[-1] <= self.source_degrees / 2:
            step = get_trapezoidal_step(degrees[-1])
            degrees.append(degrees[-1] + step)

        return self.pixels_per_degree * np.array(degrees[:-2]) #last one is overshoot

    def remap(self, image, pyramid=None, fast=True):
        pass

#TODO: pyramid class


def mean_centre_radius_over_eccentricity(parvo=True):
    eccentricities, radii = get_RCG_radii(parvo, centre=True)
    return np.mean(radii) / np.mean(eccentricities)




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


# print(image.shape)
rgcm = RGCMap(2500, 70, .1, 100)
# plt.plot(rgcm.radial_positions)
# plt.plot(rgcm.angles)
plt.plot(rgcm.radial_pixel_positions, rgcm.centre_radii)
plt.plot(rgcm.radial_pixel_positions, rgcm.surround_radii)
plt.show()


# rgcm = RGCMap(256, show_fit=False)
# degrees, pixels = rgcm.get_radial_positions()
# print(degrees)
# print(pixels.shape)

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
# # viewer = ImageViewer(image)
# viewer = ImageViewer(warped)
# viewer.show()
