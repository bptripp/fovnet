import timeit
import time
import csv
import itertools
import matplotlib.pyplot as plt
from skimage.transform import warp, warp_coords
from skimage.filters import gaussian
from skimage import data
from skimage.viewer import ImageViewer
from scipy.ndimage import map_coordinates
import numpy as np
from scipy.interpolate import interp1d
from skimage.transform import SimilarityTransform


# Assume 16mm from fovea on retinal surface corresponds to 90 degrees (~full range in nasal direction from
# Wassle et al., 1989, and ~full macaque field of view from Van Essen et al., 1984, Vision Research).
degrees_per_mm = 90 / 16


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
    filename = './data/croner_1995_4{}_{}'.format(figure, series)
    eccentricities, radii = _get_data(filename)
    return eccentricities, radii


def get_RCG_densities():
    filename = './data/wassle_1989_3'
    eccentricities, densities = _get_data(filename)
    eccentricities = eccentricities * degrees_per_mm
    return eccentricities, densities


def get_RGC_magno_fractions(direction):
    """
    :param direction: direction along retina from fovea: 'dorsal', 'nasal', 'temporal', or 'ventral'
    :return: eccentricities, fractions
    """

    filename = './data/silveira_1991_17_{}'.format(direction)
    eccentricities, percents = _get_data(filename)
    return eccentricities * degrees_per_mm, percents / 100


def get_RGC_dendritic_field_diameters(parvo, direction):
    """
    :param parvo: returns parvocellular data if true, magno if false
    :param direction: direction along retina from fovea: 'nasal', 'temporal', or 'vertical'
    :return: eccentricities (degrees visual angle), dendritic field diameters (micrometers)
    """

    figure = 'b' if parvo else 'a'
    filename = './data/perry_1984_6{}_{}'.format(figure, direction)
    eccentricities, diameters = _get_data(filename)
    return eccentricities * degrees_per_mm, diameters


def _get_data(filename):
    """
    :param filename: name of a comma-separated data file with two columns: eccentricity and some other
        quantity x
    :return: eccentricities, x
    """
    eccentricities = []
    x = []

    with open(filename) as file:
        r = csv.reader(file)
        for row in r:
            eccentricities.append(float(row[0]))
            x.append(float(row[1]))

    return np.array(eccentricities), np.array(x)


def get_RGC_scaled_diameters(parvo, peripheral_only=False):
    """
    :param parvo:
    :param peripheral_only:
    :return:
    """
    diameter_eccentricities, diameters = get_RGC_dendritic_field_diameters(parvo, 'temporal')
    if parvo:
        diameters = diameters / 1000 * 3.5
    else:
        diameters = diameters / 1000 * 2

    if peripheral_only:
        e, d = [], []
        for i in range(len(diameter_eccentricities)):
            if diameter_eccentricities[i] > 25:
                e.append(diameter_eccentricities[i])
                d.append(diameters[i])
        diameter_eccentricities = e
        diameters = d

    return diameter_eccentricities, diameters


def get_magno_fraction_fit(show=False):
    ne, nf = get_RGC_magno_fractions('nasal')
    te, tf = get_RGC_magno_fractions('temporal')
    de, df = get_RGC_magno_fractions('dorsal')
    ve, vf = get_RGC_magno_fractions('ventral')
    e = list(itertools.chain(*(ne, te, de, ve)))
    f = list(itertools.chain(*(nf, tf, df, vf)))
    fit = np.poly1d(np.polyfit(e, f, 2))

    if show:
        plt.plot(ne, nf, 'k.')
        plt.plot(te, tf, 'r.')
        plt.plot(de, df, 'g.')
        plt.plot(ve, vf, 'b.')
        xx = np.linspace(0, 90, 30)
        plt.plot(xx, fit(xx), 'k-')
        plt.show()

    return fit


def get_density_fit(parvo=True):
    class Fit:
        def __init__(self, parvo=True):
            self.parvo = parvo
            self.total_density_fit = get_total_density_fit()
            self.magno_fraction_fit = get_magno_fraction_fit()

        def __call__(self, eccentricities):
            total = self.total_density_fit(eccentricities)
            magno = self.magno_fraction_fit(eccentricities)

            if self.parvo:
                return np.multiply(total, 1 - magno)
            else:
                return np.multiply(total, magno)

    return Fit(parvo)


def get_total_density_fit(show=False):
    """
    TODO: using Wassle because they correct for RGC deviation; Perry shows nasal density is oddball
    :param parvo:
    :return:
    """
    eccentricities, densities = get_RCG_densities()

    ne, nd, te, td = [], [], [], []
    for i in range(len(densities)):
        if eccentricities[i] < 0: # nasal retina
            ne.append(-eccentricities[i])
            nd.append(densities[i])
        else: # temporal retina
            te.append(eccentricities[i])
            td.append(densities[i])

    # add point at periphery for extrapolation
    te.append(90)
    td.append(15)

    n_order = np.argsort(ne)
    ne = [ne[i] for i in n_order]
    nd = [nd[i] for i in n_order]

    t_order = np.argsort(te)
    te = [te[i] for i in t_order]
    td = [td[i] for i in t_order]

    class Fit:
        def __init__(self, ne, nd, te, td):
            self.ne = ne
            self.nd = nd
            self.te = te
            self.td = td

        def __call__(self, eccentricities):
            nasal_estimate = np.exp(np.interp(eccentricities, self.ne, np.log(self.nd)))
            temporal_estimate = np.exp(np.interp(eccentricities, self.te, np.log(self.td)))
            return .25*nasal_estimate + .75*temporal_estimate

    fit = Fit(ne, nd, te, td)

    if show:
        xx = np.linspace(0, 90, 40)
        plt.semilogy(ne, nd, 'k.')
        plt.semilogy(xx, np.exp(np.interp(xx, ne, np.log(nd))), 'k-')
        plt.semilogy(te, td, 'b.')
        plt.semilogy(xx, np.exp(np.interp(xx, te, np.log(td))), 'b-')
        plt.semilogy(xx, fit(xx), 'r-')
        plt.show()

    return fit


def get_centre_radius_fit(parvo=True):
    """
    TODO: incorporating some dendritic field information
    :param parvo:
    :return:
    """
    radius_eccentricities, radii = get_RCG_radii(parvo=parvo, centre=True)
    all_eccentricities = list(radius_eccentricities)
    all_data = list(radii)
    e, d = get_RGC_scaled_diameters(parvo, True)
    all_eccentricities.extend(e)
    all_data.extend(d)
    return np.poly1d(np.polyfit(all_eccentricities, all_data, 2))


def get_surround_radius_fit():
    """
    TODO: no difference parvo magno
    :return:
    """
    pe, pr = get_RCG_radii(parvo=True, centre=False)
    me, mr = get_RCG_radii(parvo=False, centre=False)
    e = list(itertools.chain(*(pe, me)))
    r = list(itertools.chain(*(pr, mr)))
    return np.poly1d(np.polyfit(e, r, 1))


def plot_RF_centre_radius_and_dendritic_field_diameter(parvo=True):
    """
    We would like to estimate the radii of retinal ganglion cell receptive field centres out to
    large eccentricities, but we only have data to about 35 degrees eccentricity. However, the
    centre radii (in degrees visual angle) seem to be similar to diameters of the dendritic fields
    (in mm), and we have data on these diameters to larger eccentricities. This function plots
    the data together to illustrate this.

    :param parvo: plot parvocellular data if True, otherwise magnocellular
    """

    xx = np.linspace(0, 90, 20)

    radius_eccentricities, radii = get_RCG_radii(parvo=parvo, centre=True)
    radius_fit = np.poly1d(np.polyfit(radius_eccentricities, radii, 2))
    plt.plot(radius_eccentricities, radii, 'k.')
    plt.plot(xx, radius_fit(xx), 'k')

    diameter_eccentricities, diameters = get_RGC_scaled_diameters(parvo, False)
    diameter_fit = np.poly1d(np.polyfit(diameter_eccentricities, diameters, 2))
    plt.plot(diameter_eccentricities, diameters, 'b.')
    plt.plot(xx, diameter_fit(xx), 'b')

    # all_eccentricities = list(radius_eccentricities)
    # all_data = list(radii)
    # e, d = get_RGC_scaled_diameters(parvo, True)
    # all_eccentricities.extend(e)
    # all_data.extend(d)
    # all_fit = np.poly1d(np.polyfit(all_eccentricities, all_data, 2))
    all_fit = get_centre_radius_fit(parvo)
    plt.plot(xx, all_fit(xx), 'r')

    plt.show()


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


# plot_RF_centre_radius_and_dendritic_field_diameter(parvo=False)

# e, d = get_RGC_scaled_diameters(True, peripheral_only=False)
# plt.plot(e, d, '.')
# plt.show()

# rgcm = RGCMap(256, show_fit=False)
# degrees, pixels = rgcm.get_radial_positions()
# print(degrees)
# print(pixels.shape)

# pe, pr = get_RCG_radii(parvo=True, centre=False)
# plt.plot(pe, pr, '.')
# me, mr = get_RCG_radii(parvo=False, centre=False)
# plt.plot(me, mr, 'o')
# surround_fit = get_surround_radius_fit()
# xx = np.linspace(0, 90, 20)
# plt.plot(xx, surround_fit(xx))
# plt.show()

# f = get_total_density_fit(show=False)
# get_magno_fraction_fit(show=True)

xx = np.linspace(0, 90, 30)
parvo_fit = get_density_fit(parvo=True)
magno_fit = get_density_fit(parvo=False)
plt.plot(xx, parvo_fit(xx), 'r')
plt.plot(xx, magno_fit(xx), 'b')
plt.show()

# eccentricities, densities = get_RCG_densities()
# plt.plot(eccentricities, densities, '.')
# plt.show()
#
# eccentricities, fractions = get_RGC_magno_fractions('ventral')
# plt.plot(eccentricities, fractions, '.')
# plt.show()

# eccentricities, diameters = get_RGC_dendritic_field_diameters(False, 'vertical')
# plt.plot(eccentricities, diameters, '.')
# plt.show()

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
# # viewer = ImageViewer(image)
# viewer = ImageViewer(warped)
# viewer.show()
