import numpy as np
import skimage
from skimage.transform import warp_coords
from skimage.filters import gaussian
from scipy.ndimage import map_coordinates
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


class ImageSampler:
    def __init__(self, input_shape, angles, radial_positions, radii, n_steps=5, min_blur=1):
        #min_sigma to avoid aliasing at fine scale

        self.radial_positions = radial_positions
        self.radii = radii

        blurs = np.linspace(np.min(radii), np.max(radii), n_steps)
        blurs = np.maximum(min_blur, blurs)
        self.blurs = list(set(blurs))
        self.blurs.sort()

        self.sigmas = [min_blur]
        for i in range(1,len(blurs)):
            self.sigmas.append(np.sqrt(blurs[i]**2 - blurs[i-1]**2))

        # find index of blur stage closest to blur wanted at each radius
        self.blur_indices = np.interp(radii, self.blurs, range(len(self.blurs)))
        self.blur_indices = np.round(self.blur_indices).astype('int')

        self.coords = []
        for i in range(len(blurs)):
            rp = [radial_positions[j] for j in range(len(radial_positions)) if self.blur_indices[j] == i]
            map = AngleEccentricityMap(input_shape, angles, rp)
            if len(rp) > 0:
                wc = warp_coords(map, (len(angles), len(rp), 3))
            else:
                wc = None
            self.coords.append(wc)


    def __call__(self, image):
        images = []
        for sigma in self.sigmas:
            image = gaussian(image, sigma)
            images.append(image)

        result_parts = []
        for i in range(len(images)):
            if self.coords[i] is not None:
                result_part = map_coordinates(images[i], self.coords[i])
                print(result_part.shape)
                result_parts.append(result_part)

        return np.concatenate(result_parts, axis=1)


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

class AngleEccentricityMap:
    """
    For use with warp_coords.
    """

    def __init__(self, input_shape, angles, radial_pixel_positions):
        self.input_shape = input_shape
        self.angles = angles
        self.radial_pixel_positions = radial_pixel_positions
        print('# angles {};  # radial positions {}'.format(len(angles), len(radial_pixel_positions)))

    def __call__(self, xy):
        """
        :param xy: pixel locations in target image; each row (horizontal, vertical)
        :return: corresponding pixel locations in source image
        """
        # foo = [y for y in xy[:,1]]
        # print(foo)
        # print(len(self.radial_pixel_positions))
        # print(np.min(xy[:, 1]))
        # print(np.max(xy[:, 1]))
        angles = [self.angles[len(self.angles) - 1 - int(x)] for x in xy[:,1]]
        eccentricities = [self.radial_pixel_positions[int(y)] for y in xy[:, 0]]

        xy[:, 0] = self.input_shape[1] / 2 + eccentricities * np.cos(angles)
        xy[:, 1] = self.input_shape[0] / 2 - eccentricities * np.sin(angles)
        return xy

class DemoMap:
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


image = skimage.io.imread('../peggys-cove.jpg')
image = image[25:1525, 825:2325, :]

print(image.shape)
rgcm = RGCMap(np.min(image.shape[:2]), 70, .2, 100)
# plt.plot(rgcm.radial_positions)
# plt.plot(rgcm.angles)
# plt.plot(rgcm.radial_pixel_positions, rgcm.centre_radii)
# plt.plot(rgcm.radial_pixel_positions, rgcm.surround_radii)
# plt.show()

rfp = ImageSampler(image.shape[:2], rgcm.angles, rgcm.radial_pixel_positions, rgcm.centre_radii, n_steps=5, min_blur=.5)
warped = rfp(image)


# map = DemoMap(image.shape, (200, 200))
# coords = warp_coords(map, (200,200,3))
# warped = map_coordinates(image, coords)

# viewer = skimage.viewer.ImageViewer(image)
viewer = skimage.viewer.ImageViewer(warped)
viewer.show()
