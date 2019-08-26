"""
This module contains our vizualization plotting methods and our helper functions
for visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics as stats
import pylab as pl
import pandas as pd

# Set specific parameters for the visualizations
large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


def create_sample_dists(cleaned_data, y_var=None, x_var=None, categories=[], samplesize=30, numsamples=400):
    np.random.seed(5)
    """
    This is our random sampling function to prep data for vizualisations
    :param cleaned_data: cleaned dataframe
    :param y_var: The numeric variable you are comparing
    :param categories: the categories whose means you are comparing
    :return: a list of sample distributions
    """
    df = cleaned_data

    dflist = []

    for cat in categories:
        dftemp = df.loc[df[x_var].str.contains(cat)][y_var]
        sampler = np.random.choice(dftemp, size=(samplesize, numsamples))
        sample_prop = sampler.mean(axis=0)
        dflist.append(sample_prop)

    return dflist


def overlapping_density(package=None, input_vars=None, target_vars=None, categories=None, output_image_name=None):
    """ 
    This plots our pairs of direct comparisons between categories.
    """

    # Set size of figure
    fig = plt.figure(figsize=(16, 10), dpi=80)
    sns.set(color_codes=True)

    # Starter code for figuring out which package to use
    if package == "sns":
        for counter, value in enumerate(input_vars):
            sns.kdeplot(value, label=categories[counter], shade=True)
            plt.xlabel('Means', fontsize=med)  # , figure = fig)
            plt.ylabel('Sample counts', fontsize=med)  # , figure = fig)
            plt.title('Overlapping mean desnsity',
                      fontsize=large)  # , figure = fig)
            plt.xticks(fontsize=med)
            plt.yticks(fontsize=med)
    elif package == 'matplotlib':
        for variable in input_vars:
            plt.plot(variable, label=None, linewidth=None,
                     color=None, figure=fig)

    plt.savefig(f'img/{output_image_name}.png', transparent=True, figure=fig)
    return fig


def color_plot(arr, categories=None, output_image_name=None):
    """ 
    This method plots bars for the average citation rate for each category in question
    """
    sns.set(color_codes=True)
    arr_list = np.asarray(arr).mean(axis=1)
    arr_lst = np.vstack((arr_list, 1-arr_list))
    df = pd.DataFrame(arr_lst.T)
    df.plot(kind='bar', stacked=True)
    plt.xlabel('Vehicle groups', fontsize=med)
    plt.ylabel('Citation rates', fontsize=med)
    plt.title('Ticketed vs Non-Ticketed vehicles')
    plt.xticks(np.arange(len(categories)), categories, rotation=0)
    plt.legend(labels=['Ticketed', 'Non-Ticketed'])

    plt.savefig(f'img/{output_image_name}.png', transparent=True)


def visualization_one(cleaned_data):
    # bar plot of color averages
    short_cleaned_data = cleaned_data[[
        'color', 'make', 'model', 'violation_type', 'year', 'ticket']]

    vizualization = create_sample_dists(short_cleaned_data,
                                        y_var='ticket',
                                        x_var='color',
                                        categories=['BLACK', 'WHITE', 'SILVER', 'BLUE', 'RED', 'GRAY'])
    color_plot(vizualization,
               categories=['BLACK', 'WHITE', 'SILVER', 'BLUE', 'RED', 'GRAY'],
               output_image_name='Ticketed_vs_Non-Ticketed')


def visualization_two(cleaned_data):
    # white vs red cars comparison
    wr_chart = create_sample_dists(cleaned_data,
                                   y_var='ticket',
                                   x_var='color',
                                         categories=['WHITE', 'RED'])

    overlapping_density('sns',
                        input_vars=wr_chart,
                        categories=['WHITE', 'RED'],
                        output_image_name='Red vs White')


def visualization_three(cleaned_data):
    # white vs blue cars comparison
    wr_chart = create_sample_dists(cleaned_data,
                                   y_var='ticket',
                                   x_var='color',
                                         categories=['WHITE', 'BLUE'])

    overlapping_density('sns',
                        input_vars=wr_chart,
                        categories=['WHITE', 'BLUE'],
                        output_image_name='White vs Blue')


def visualization_four(cleaned_data):
    # bar plot of make averages
    cleaned_data['ticket'] = cleaned_data['violation_type'].apply(
        lambda x: 1 if x == 'Citation' else 0)
    short_cleaned_data = cleaned_data[[
        'color', 'make', 'model', 'violation_type', 'year', 'ticket']]

    vizualization = create_sample_dists(short_cleaned_data,
                                        y_var='ticket',
                                        x_var='make',
                                        categories=['NISS', 'FORD', 'HOND', 'TOY'])
    color_plot(vizualization,
               categories=['NISS', 'FORD', 'HOND', 'TOY'],
               output_image_name='Ticketed_vs_Non-Ticketed')


def visualization_five(cleaned_data):
    # honda vs toyota comparison
    wr_chart = create_sample_dists(cleaned_data,
                                   y_var='ticket',
                                   x_var='make',
                                         categories=['HOND', 'TOY'])

    overlapping_density('sns',
                        input_vars=wr_chart,
                        categories=['HOND', 'TOY'],
                        output_image_name='Honda vs Toyota')
