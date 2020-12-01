# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:46:49 2020

@author: adeju
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


np.random.seed(19680801)


def rosechart(angles):

    # angles = segm_group_angles[:, 0]
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(angles, bin_edges)

    number_of_strikes[0] += number_of_strikes[-1]

    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves,
           width=np.deg2rad(10), bottom=0.0, color='red', edgecolor='k')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
    # ax.set_rgrids(np.arange(1, two_halves.max()+1, 2),angle=0,weight='black')
    ax.set_title("N = "+str(np.size(angles)), y=1.10, fontsize=25)
    # plt.show()
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.show()

    return plot


def inverse_cumulative_powerlaw(bins, alpha, cumulative=True):
    power_law = ((alpha-1)/(np.min(bins))*np.power(
        (bins/np.min(bins)), -alpha))
    if cumulative is True:
        power_law = power_law.cumsum()
        power_law /= power_law[-1]
    return power_law


def inverse_cumulative_lognormal(bins, mu, sigma, cumulative=True):
    log_normal = (1/(bins*sigma*np.sqrt(2 * np.pi))*np.exp(
        -np.power((np.log(bins)-mu), 2) / (2*np.power(sigma, 2))))
    if cumulative is True:
        log_normal = log_normal.cumsum()
        log_normal /= log_normal[-1]
    return log_normal


def histogram(x):

    mu = np.mean(x)
    sigma = np.std(x)
    n_bins = 251
    # x = segm_group_angles[:,1]

    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',
                               cumulative=True, label='Data')

    # Add a line showing the expected distribution.
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

    # Log-normal
    log_normal = inverse_cumulative_lognormal(bins, mu, sigma)
    ax.plot(bins, log_normal, 'k--', linewidth=1.5, label='Log-normal')

    # Power-law
    alpha = 2
    power_law = inverse_cumulative_powerlaw(bins, alpha)
    ax.plot(bins, power_law, 'b--', linewidth=1.5, label='Power law')

    # Overlay a reversed cumulative histogram.
    # ax.hist(x, bins=bins, density=True, histtype='step', cumulative=1,
    #        label='Reversed emp.')

    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative distribution')
    ax.set_xlabel('length')
    ax.set_ylabel('Probability')

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plot


def probability_plot(data, distribution_type, title, xlabel):
    # title = ''
    # distribution_type = 'gwer'
    # data = segm_group_angles[:,1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True)

    X = np.flip(np.sort(data))
    if distribution_type == 'power':
        alpha = 2
        Y = inverse_cumulative_powerlaw(np.sort(data), alpha)
    else:
        mu = np.mean(data)
        sigma = np.std(data)
        Y = inverse_cumulative_lognormal(np.sort(data), mu, sigma)

    Y = np.reshape(Y, (np.size(Y), 1))
    A = np.vstack([X, np.ones(np.size(X))]).T
    m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    ax.plot(X, Y, 'bx', label="Data")
    ax.plot(X, (m*X+c), 'r--', label="Least Squares")
    r2 = r2_score(Y, m*X+c)
    ax.plot([], [], ' ', label="R2 score: " + str("%.3f" % round(r2, 3)))
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative distribution function")
    plt.show()
    return
