# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:46:49 2020

@author: adeju
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

def histogram (x):

    mu = 200
    sigma = 25
    n_bins = 50
    #x = segm_group_angles[:,1]

    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',
                            cumulative=True, label='Dados')

    # Add a line showing the expected distribution.
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))


    y = (1/(bins*sigma*np.sqrt(2 * np.pi))*np.exp(-np.power((np.log(bins)-mu), 2) /
                                                (2*np.power(sigma, 2))))
    y = y.cumsum()
    y /= y[-1]
    ax.plot(bins, y, 'k--', linewidth=1.5, label='Log-normal')


    alpha = 2
    y = ((alpha-1)/(np.min(bins))*np.power((bins/np.min(bins)), -alpha))
    y = y.cumsum()
    y /= y[-1]
    ax.plot(bins, y, 'b--', linewidth=1.5, label='Lei da potência')


    # Overlay a reversed cumulative histogram.
    # ax.hist(x, bins=bins, density=True, histtype='step', cumulative=1,
    #        label='Reversed emp.')

    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Distribuição cumulativa de comprimentos')
    ax.set_xlabel('Distância em pixels')
    ax.set_ylabel('Probabilidade')

    plt.show()
