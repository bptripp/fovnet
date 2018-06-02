import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import warp_coords
from skimage.filters import gaussian
from scipy.ndimage import map_coordinates
from fovnet.data.retina import get_density_fit, get_centre_radius_fit, get_surround_radius_fit

# TODO: if scale is too low, blur is insufficient for inter-pixel spacing; not clear whether this should be changed
# TODO: image pyramid for blurs to save run time

class RGCMap():
    """
    Defines a map that warps images to approximate retinal ganglion cell density and receptive
    field size.
    """

    def __init__(self, input_shape, source_degrees, scale, angle_steps=None, right=True, parvo=True):
        """
        :param input_shape: (height, width) of source image (pixels)
        :param source_degrees: corresponding degrees visual angle
        :param scale: linear density of result pixels (in radial direction) as a fraction of physiological value
        :param angle_steps: number of discrete angles to sample in one half of visual field
        :param right: right visual field if True, left otherwise
        :param parvo: based on parvocellular system if True, magnocellular otherwise
        """

        assert len(input_shape) == 2

        self.source_pixels = np.min(input_shape)
        self.source_degrees = source_degrees
        self.pixels_per_degree = self.source_pixels / source_degrees
        self.scale = scale
        self.density_function = get_density_fit(parvo)

        self.radial_pixel_positions = self.get_radial_positions()

        if angle_steps is None:
            # circumference of half circle is pi rad, so there should be pi time as many steps around
            angle_steps = int(np.round(np.pi * len(self.radial_pixel_positions)))

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

        self.centre_sampler = ImageSampler(
            input_shape,
            self.angles,
            self.radial_pixel_positions,
            self.centre_radii,
            n_steps=5,
            min_blur=.5)

        self.surround_sampler = ImageSampler(
            input_shape,
            self.angles,
            self.radial_pixel_positions,
            self.surround_radii,
            n_steps=15,
            min_blur=.5)

    def pixels_between_rfs(self, eccentricity):
        """
        :param eccentricity: distance from fovea in degrees visual angle
        :return: number of pixels between RF centres at that eccentricity
        """

        # This is basically unit book-keeping, as follows:
        # map_density(map_cells/degree) = scale(map_cells/actual_cell) * density(actual_cells/degree)
        # pixels/map_cell = (pixels/degree) / map_density(map_cells/degree)

        map_density = self.scale * self.density_function(eccentricity)
        return self.pixels_per_degree / map_density

    def rf_step_degrees(self, eccentricity):
        """
        :param eccentricity: distance from fovea in degrees visual angle
        :return: degrees visual angle between receptive-field centres in radial direction, at the given
            eccentricity
        """
        map_density = self.scale * self.density_function(eccentricity)
        return 1 / map_density

    def get_radial_positions(self):
        """
        :return: radial positions of receptive field centres (distance from image centre) in pixels
        """

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
    """
    Blurs and remaps images to model RF density and size.
    """

    def __init__(self, input_shape, angles, radial_positions, radii, n_steps=5, min_blur=.5):
        """
        :param input_shape: (height, width) of input images
        :param angles: list of angles of receptive field centres
        :param radial_positions: list of radial positions of receptive field centres (pixels from
            image centre)
        :param radii: list of radii of receptive fields; same length as radial_positions, as radius
            is mainly a function of eccentricity
        :param n_steps: receptive fields of various sizes are approximated in discrete steps by
            sampling from copies of the image with different degrees of blur; this is the number of
            different blurred images created; a larger number will result in less quantization error
            in the RF size, and longer run time
        :param min_blur: sigma of Gaussian blur of the sharpest image; some blur is needed to avoid
            moire due to aliasing
        """

        self.radial_positions = radial_positions
        self.radii = radii

        blurs = np.linspace(np.min(radii), np.max(radii), n_steps)
        blurs = np.maximum(min_blur, blurs)
        self.blurs = list(set(blurs))
        self.blurs.sort()

        self.sigmas = [blurs[0]]
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
        """
        :param image: input image
        :return: image warped to approximate retinal ganglion cell density, etc.
        """
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
    """
    :param shape: (height, width, channels) of image to be created
    :return: a target-like image of concentric rings
    """
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
        """
        :param input_shape: (height, width) of source images
        :param angles: list of angles at which to sample images
        :param radial_pixel_positions: list of eccentricities (in pixels) at which to sample images
        """
        self.input_shape = input_shape
        self.angles = angles
        self.radial_pixel_positions = radial_pixel_positions

    def __call__(self, xy):
        """
        :param xy: pixel locations in target image; each row (horizontal, vertical)
        :return: corresponding pixel locations in source image
        """
        angles = [self.angles[len(self.angles) - 1 - int(x)] for x in xy[:,1]]
        eccentricities = [self.radial_pixel_positions[int(y)] for y in xy[:, 0]]

        xy[:, 0] = self.input_shape[1] / 2 + eccentricities * np.cos(angles)
        xy[:, 1] = self.input_shape[0] / 2 - eccentricities * np.sin(angles)
        return xy


if __name__ == '__main__':
    image = imread('../peggys-cove.jpg')
    image = image[25:1525, 825:2325, :]

    rgcm = RGCMap(image.shape[:2], 70, .3, right=True)

    warped_faster = rgcm.centre_sampler(image)

    slower_sampler = ImageSampler(
        image.shape[:2],
        rgcm.angles,
        rgcm.radial_pixel_positions,
        rgcm.centre_radii,
        n_steps=50,
        min_blur=.5)
    warped_slower = slower_sampler(image)

    plt.subplot(1,3,1)
    plt.imshow(warped_faster)
    plt.axis('off')
    plt.title('5 blur steps')
    plt.subplot(1,3,2)
    plt.imshow(warped_slower)
    plt.axis('off')
    plt.title('50 blur steps')
    plt.subplot(1,3,3)
    plt.imshow(10*(warped_faster - warped_slower) + 0.5)
    plt.axis('off')
    plt.title('10x difference')
    plt.tight_layout()
    plt.show()

