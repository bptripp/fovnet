import os
import inspect
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt

"""
This file contains functions for reading and processing physiological data on retinal ganglion cells
(RGCs; the output neurons of the retina). The main functions return curve fits for RGC density 
and receptive field size. These include: 

get_density_fit() -- linear density of RGCs per degree visual angle
get_centre_radius_fit() -- sigma of a Gaussian approximation of receptive field centre 
get_surround_radius_fit() -- sigma of a Gaussian approximation of receptive field surround

The return values are functions of eccentricity (in degress visual angle). Here is a usage example: 

surround_fit_function = get_surround_radius_fit()
eccentricities = numpy.linspace(0, 90, 10)
surround_radii = surround_fit_function(eccentricities)

There are also a number of supporting functions that load physiological data from text files and 
combine it in various ways to produce these fits. The text files (also in this folder) were 
generated from figures in the macaque literature using WebPlotDigitizer. The file data_sources.bib
has the references.

Running this file produces plots of the data and fits. 
"""


def data_folder():
    return os.path.dirname(inspect.stack()[0][1])

"""
Assume 16mm from fovea on retinal surface corresponds to 90 degrees (~full range in nasal direction from
Wassle et al., 1989, and ~full macaque field of view from Van Essen et al., 1984, Vision Research).
"""
degrees_per_mm = 90 / 16


def get_RCG_radii(parvo=True, centre=True):
    """
    :param parvo: parvocellular data if True, magnocellular otherwise
    :param centre: receptive field centre data if True, surround data otherwise
    :return: eccentricities (degrees visual angle from fovea); radii (sigma of Gaussian fit of RF)
    """
    figure = 'a' if centre else 'b'
    series = 'P' if parvo else 'M'
    filename = '{}/croner_1995_4{}_{}'.format(data_folder(), figure, series)
    eccentricities, radii = _get_data(filename)
    return eccentricities, radii


def get_RCG_densities():
    """
    :return: Linear retinal ganglion cell density per degree visual angle.
    """
    filename = '{}/wassle_1989_3'.format(data_folder())
    eccentricities, densities = _get_data(filename)
    eccentricities = eccentricities * degrees_per_mm
    return eccentricities, np.power(densities, .5) / degrees_per_mm


def get_RGC_magno_fractions(direction):
    """
    :param direction: direction along retina from fovea, one of 'dorsal', 'nasal', 'temporal', or 'ventral'
    :return: eccentricities, fractions of RGCs that are magnocellular
    """

    filename = '{}/silveira_1991_17_{}'.format(data_folder(), direction)
    eccentricities, percents = _get_data(filename)
    return eccentricities * degrees_per_mm, percents / 100


def get_RGC_dendritic_field_diameters(parvo, direction):
    """
    :param parvo: returns parvocellular data if true, magno if false
    :param direction: direction along retina from fovea: one of 'nasal', 'temporal', or 'vertical'
    :return: eccentricities (degrees visual angle), dendritic field diameters (micrometers)
    """

    figure = 'b' if parvo else 'a'
    filename = '{}/perry_1984_6{}_{}'.format(data_folder(), figure, direction)
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
    :param parvo: parvocellular if True, magnocellular otherwise
    :param peripheral_only: only data from >25 degrees from fovea returned if True
    :return: RGC dendritic field diameters, RESCALED so that the values are similar to
        radii of RGC receptive field centres in the central 25 degrees or so; we don't have
        RF radii more peripherally, and we will use the dendritic fields to extrapolate
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


def get_magno_fraction_fit(plot=False):
    """
    :param plot: if True, plot data and curve of best fit
    :return: a function that approximates fraction RGCs that are magnocellular given
        eccentricity from fovea; the fraction is different in nasal, temporal, dorsal, and
        ventral directions, but we combine data from all directions to get a radially
        symmetric estimate
    """

    ne, nf = get_RGC_magno_fractions('nasal')
    te, tf = get_RGC_magno_fractions('temporal')
    de, df = get_RGC_magno_fractions('dorsal')
    ve, vf = get_RGC_magno_fractions('ventral')
    e = list(itertools.chain(*(ne, te, de, ve)))
    f = list(itertools.chain(*(nf, tf, df, vf)))
    fit = np.poly1d(np.polyfit(e, f, 2))

    if plot:
        plt.plot(ne, nf, 'r.')
        plt.plot(te, tf, 'b.')
        plt.plot(de, df, 'c.')
        plt.plot(ve, vf, 'm.')
        x = np.linspace(0, 90, 30)
        plt.plot(x, fit(x), 'k-')
        plt.legend(('Nasal', 'Temporal', 'Dorsal', 'Ventral'), fontsize=9)

    return fit


def get_density_fit(parvo=True):
    """
    :param parvo: parvocellular if True, magnocellular otherwise
    :return: a function that approximates linear density of parvo or magnocellular RGCs per
        degree visual angle, given eccentricity from fovea as an arguments
    """
    class Fit:
        def __init__(self, parvo=True):
            self.parvo = parvo
            self.total_density_fit = get_total_density_fit()
            self.magno_fraction_fit = get_magno_fraction_fit()

        def __call__(self, eccentricities):
            eccentricities = np.clip(eccentricities, 0, 90)
            total = self.total_density_fit(eccentricities)
            magno = self.magno_fraction_fit(eccentricities)

            if self.parvo:
                return np.multiply(total, 1 - magno)
            else:
                return np.multiply(total, magno)

    return Fit(parvo)


def get_total_density_fit(plot=False):
    """
    We use estimates from Wassle et al., which are corrected for RGC deflection away from foveola.
    Wassle et al. provide data along the nasal-temporal axis, and density is different in each
    direction. Perry et al. show that density is similar in temporal, dorsal, and ventral directions,
    but higher in the nasal direction. However they do not correct for RGC deflection. Therefore, to
    calculate radially symmetric mean values, we sum nasal and temporal fits from Wassle et al
    corrected data, with weights 0.25 and 0.75 (to account for the fact that nasal density is atypical).

    :param plot: if True, plot data and curve fit
    :return: a function that approximates total linear density of RGCs per degree visual angle, given
        eccentricity from fovea (degrees) as an argument
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

    n_order = np.argsort(ne)
    ne = [ne[i] for i in n_order]
    nd = [nd[i] for i in n_order]

    t_order = np.argsort(te)
    te = [te[i] for i in t_order]
    td = [td[i] for i in t_order]

    class Fit:
        def __init__(self, ne, nd, te, td):
            self.ne = ne[:]
            self.nd = nd[:]
            self.te = te[:]
            self.td = td[:]

            # add point at periphery for extrapolation
            self.te.append(90)
            self.td.append(.7)

        def __call__(self, eccentricities):
            nasal_estimate = np.exp(np.interp(eccentricities, self.ne, np.log(self.nd)))
            temporal_estimate = np.exp(np.interp(eccentricities, self.te, np.log(self.td)))
            return .25*nasal_estimate + .75*temporal_estimate

    fit = Fit(ne, nd, te, td)

    if plot:
        nx = np.linspace(0, max(ne), 40)
        plt.semilogy(nx, np.exp(np.interp(nx, ne, np.log(nd))), 'r-')
        tx = np.linspace(0, max(te), 40)
        plt.semilogy(tx, np.exp(np.interp(tx, te, np.log(td))), 'b-')
        fx = np.linspace(0, 90, 40)
        plt.semilogy(fx, fit(fx), 'k-')
        plt.legend(('Nasal', 'Temporal', 'Weighted average'), fontsize=9)

    return fit


def get_centre_radius_fit(parvo=True, plot=False):
    """
    This estimate uses RGC receptive field data from the literature, but we only have data out
    to about 25 degrees eccentricity. To extrapolate to 90 degrees, we also incorporate data on
    RGC dendritic field sizes, which are correlated with RF size, and known farther into the periphery.

    :param parvo: deals with parvocellular data if True, magnocellular otherwise
    :param plot: if True, plots data and curve fit
    :return: a function that approximates the radius (sigma) of a Gaussian function that
        fits a typical RGC receptive field centre, in degrees visual angle, given eccentricity
        in degrees from fovea
    """
    radius_eccentricities, radii = get_RCG_radii(parvo=parvo, centre=True)
    all_eccentricities = list(radius_eccentricities)
    all_data = list(radii)
    e, d = get_RGC_scaled_diameters(parvo, True)
    all_eccentricities.extend(e)
    all_data.extend(d)
    fit = np.poly1d(np.polyfit(all_eccentricities, all_data, 2))

    if plot:
        diameter_color = (.75, .75, 1)
        radius_color = (.25, .25, 1)
        all_color = (0, 0, 0)

        plt.plot(e, d, 's', color=diameter_color)
        plt.plot(radius_eccentricities, radii, '.' if parvo else 'x', color=radius_color)
        x = np.linspace(0, 90, 20)
        plt.plot(x, fit(x), color=all_color)
        plt.legend(('Rescaled dendritic fields', 'RF radii'), fontsize=9)

    return fit


def get_surround_radius_fit(plot=False):
    """
    In contrast with RF centres, the surrounds are apparently similar for parvocellular and
    magnocellular systems. Also on contrast with the RF centres, we do not use dendritic field
    diameter to extrapolate into the periphery. This is because the surround is due to connections
    between horizontal cells and amacrine cells, so it is probably not tied closely to RGC dendritic
    fields.

    :param plot: if True, plots data and curve fit
    :return: a function that approximates the radius (sigma) of a Gaussian function that
        fits a typical RGC receptive field surround, in degrees visual angle, given eccentricity
        in degrees from fovea
    """
    pe, pr = get_RCG_radii(parvo=True, centre=False)
    me, mr = get_RCG_radii(parvo=False, centre=False)
    e = list(itertools.chain(*(pe, me)))
    r = list(itertools.chain(*(pr, mr)))
    fit = np.poly1d(np.polyfit(e, r, 1))

    if plot:
        plt.plot(pe, pr, '.', color='r')
        plt.plot(me, mr, 'x', color='r')
        x = np.linspace(0, 90, 20)
        plt.plot(x, fit(x), 'k')
        plt.legend(('Parvocellular', 'Magnocellular'))

    return fit



def plot_RF_centre_radius_and_dendritic_field_diameter(parvo=True):
    """
    We would like to estimate the radii of retinal ganglion cell receptive field centres out to
    large eccentricities, but we only have data to about 35 degrees eccentricity. However, the
    centre radii (in degrees visual angle) seem to be similar to diameters of the dendritic fields
    (in mm), and we have data on these diameters to larger eccentricities. This function plots
    the data together to illustrate this.

    :param parvo: plot parvocellular data if True, otherwise magnocellular
    """

    x = np.linspace(0, 90, 20)

    diameter_color = (.75, .75, 1)
    radius_color = (.25, .25, 1)
    all_color = (0, 0, 0)

    diameter_eccentricities, diameters = get_RGC_scaled_diameters(parvo, False)
    diameter_fit = np.poly1d(np.polyfit(diameter_eccentricities, diameters, 2))
    plt.plot(diameter_eccentricities, diameters, 's', color=diameter_color)

    radius_eccentricities, radii = get_RCG_radii(parvo=parvo, centre=True)
    radius_fit = np.poly1d(np.polyfit(radius_eccentricities, radii, 2))
    plt.plot(radius_eccentricities, radii, '.', color=radius_color)

    all_fit = get_centre_radius_fit(parvo)
    plt.plot(x, diameter_fit(x), color=diameter_color)
    plt.plot(x, radius_fit(x), color=radius_color)
    plt.plot(x, all_fit(x), color=all_color)

    plt.legend(('Rescaled dendritic fields', 'RF radii'), fontsize=9)


if __name__ == '__main__':
    """
    Running this plots all the data fitting figures. 
    """

    plt.figure(1, (6, 2.75))
    plt.subplot(1, 2, 1)
    f = get_total_density_fit(plot=True)
    plt.title('RGC density')
    plt.xlabel(r'Eccentricity ($\degree$)')
    plt.ylabel(r'Cells/$\degree$')
    plt.subplot(1, 2, 2)
    get_magno_fraction_fit(plot=True)
    plt.title('Magnocellular fraction')
    plt.xlabel(r'Eccentricity ($\degree$)')
    plt.ylabel('Fraction')
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.subplot(2,2,1)
    plot_RF_centre_radius_and_dendritic_field_diameter(parvo=True)
    plt.ylabel(r'Radius ($\degree$)')
    plt.title('Parvocellular centres')
    plt.subplot(2,2,2)
    get_centre_radius_fit(parvo=True, plot=True)
    plt.title('Parvocellular centres')
    plt.subplot(2,2,3)
    get_centre_radius_fit(parvo=False, plot=True)
    plt.xlabel(r'Eccentricity ($\degree$)')
    plt.ylabel(r'Radius ($\degree$)')
    plt.title('Magnocellular centres')
    plt.subplot(2,2,4)
    get_surround_radius_fit(plot=True)
    plt.xlabel(r'Eccentricity ($\degree$)')
    plt.title('Surrounds')
    plt.tight_layout()
    plt.show()

