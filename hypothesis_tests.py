"""
This module is for your final hypothesis tests.
Each hypothesis test should tie to a specific analysis question.

Each test should print out the results in a legible sentence
return either "Reject the null hypothesis" or "Fail to reject the null hypothes
is" depending on the specified alpha
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison


def create_sample_dists(cleaned_data, y_var=None, x_var=None, categories=[],
                        samplesize=30, numsamples=400, seed=5):
    """
    Each hypothesis test will require you to create a sample distribution from
    your data
    Best make a repeatable function

    :param cleaned_data:
    :param y_var: The numeric variable you are comparing
    :param categories: the categories whose means you are comparing
    :return: a list of sample distributions to be used in subsequent t-tests

    """
    np.random.seed(seed)
    df_clean = cleaned_data

    df_list = []

    for cat in categories:
        dftemp = df_clean.loc[df_clean[x_var].str.contains(cat)][y_var]
        sampler = np.random.choice(dftemp, size=(samplesize, numsamples))
        sample_prop = sampler.mean(axis=0)
        df_list.append(sample_prop)

    return df_list


def compare_pval_alpha(p_val, alpha):
    """ this functions tests p values vs our chosen alpha"""
    status = ''
    if p_val > alpha:
        status = "Fail to reject"
    else:
        status = 'Reject'
    return status

def compare_pval_alpha_tf(p_val, alpha=.05):
    """ this functions tests p values vs our chosen alpha returns bool"""
    status = None
    if p_val > alpha:
        status = False
    else:
        status = True
    return status


def hypothesis_test_one(cleaned_data, alpha=.05):
    """
    This function takes in cleaned data, then uses create sample dists to grab
    the required categories. From there the function perfoms an Anova on the
    groups and returns whether or not to reject he null hypothesis
    :param alpha: the critical value of choice
    :param cleaned_data: our cleaned dataset
    :return: sentece with outcome of test containing f and p value
    """
    # Get data for tests
    comparison_groups = create_sample_dists(cleaned_data, y_var='ticket', x_var='color', categories=['BLACK',
                                                                                                     'WHITE',
                                                                                                     'RED',
                                                                                                     'BLUE',
                                                                                                     'GRAY',
                                                                                                     'SILVER'])
    ###
    # perform f test (ANOVA) on the groups
    ###
    F_val, p_temp = stats.f_oneway(comparison_groups[0], comparison_groups[1],
                                   comparison_groups[2], comparison_groups[3],
                                   comparison_groups[4], comparison_groups[5])
    p_val = p_temp
    # starter code for return statement and printed results
    status = compare_pval_alpha(p_val, alpha)
    assertion = ''
    if status == 'Fail to reject':
        assertion = 'cannot'
    else:
        assertion = "can"


    print(f'Based on the p value of {p_val} and our aplha of {alpha} we {status.lower()}  the null hypothesis.'
          f'\n Due to these results, we  {assertion} state that there is a difference in citation rate between these colors')


def hypothesis_test_two(cleaned_data, alpha=.05):
    """
    This function takes in cleaned data, then uses create sample dists to grab
    the required categories. From there the function performs tukeys HSD
    analysis and displays a chart of all the pairwise compairisons and their
    signifigance
    :param alpha: the critical value of choice
    :param cleaned_data: our cleaned dataset
    :return:
    """
    # Get data for tests
    categories = ['BLACK', 'WHITE', 'RED', 'BLUE', 'GRAY', 'SILVER']
    comparison_groups = create_sample_dists(
        cleaned_data, y_var='ticket', x_var='color', categories=categories)
    list_for_tukey = []
    for i in range(len(categories)):
        cat_list = [categories[i]] * 50
        tk_temp = zip(list(comparison_groups[i]), cat_list)
        list_for_tukey += list(tk_temp)

    df_tukey = pd.DataFrame(list_for_tukey)

    ###
    # perform tukeys HSD for the groups
    ###
    mult_comp = MultiComparison(df_tukey[0], df_tukey[1])
    result = mult_comp.tukeyhsd(alpha)
    # need to convert table from the tukey result object into a dataframe
    result_summary = result.summary().as_html()
    tukey_df = pd.read_html(result_summary, header=0, index_col=0)[0]
    tukey_df.columns = ["Second Color", "Mean Difference",
                        "Min Difference", "Max Difference",
                        "Signifigant Difference?"]
    tukey_df.index.names = ['First Color']

    return tukey_df


def hypothesis_test_three(cleaned_data, alpha=.05):
    """
    This function takes in cleaned data, then uses create sample dists to grab
    the required categories. From there the function perfoms an Anova on the
    groups and returns whether or not to reject he null
    :param alpha: the critical value of choice
    :param cleaned_data: our cleaned dataset
    :return:
    """
    # Get data for tests
    comparison_groups = create_sample_dists(cleaned_data, y_var='ticket',
                                            x_var='make',
                                            categories=['NISS',
                                                        'FORD',
                                                        'HOND',
                                                        'TOY'],
                                            samplesize=30, numsamples=400, seed=4)

    ###
    # perform f test (ANOVA) on the groups
    ###
    F_val, p_temp = stats.f_oneway(
        comparison_groups[0], comparison_groups[1], comparison_groups[2],
        comparison_groups[3])
    p_val = p_temp
    # starter code for return statement and printed results
    status = compare_pval_alpha(p_val, alpha)
    assertion = ''
    if status == 'Fail to reject':
        assertion = 'cannot'
    else:
        assertion = "can"
        # calculations for effect size, power, etc here as well

    print(f'Based on the p value of {p_val} and our aplha of {alpha} we {status.lower()}  the null hypothesis.'
          f'\n Due to these results, we  {assertion} state that there is a difference in citation rate between these makes')


def hypothesis_test_four(cleaned_data):
    """
    This function takes in cleaned data, then uses create sample dists to grab
    the required categories. From there the function performs fishers lSD
    analysis and displays a chart of all the pairwise compairisons and the 
    p-values
    :param alpha: the critical value of choice
    :param cleaned_data: our cleaned dataset
    :return:
    """
    # Get data for tests
    categories = ['NISS', 'FORD', 'HOND', 'TOY']
    comparison_groups = create_sample_dists(
        cleaned_data, y_var='ticket', x_var='make', categories=categories, seed=4)
    list_for_lsd = []
    for i in range(len(categories)):
        cat_list = [categories[i]] * 50
        tk_lsd = zip(list(comparison_groups[i]), cat_list)
        list_for_lsd += list(tk_lsd)

    df_lsd = pd.DataFrame(list_for_lsd)
    
    # perform fisher LSD for the groups
    mult_comp = MultiComparison(df_lsd[0], df_lsd[1])
    result = mult_comp.allpairtest(stats.ttest_ind, method='Holm')
    # we need to convert the simpletable result object into a dataframe
    result_summary = result[0].as_html()
    lsd_df = pd.read_html(result_summary, header=0, index_col=0)[0]
    lsd_df = lsd_df.drop(columns=['stat', 'pval_corr'])
    lsd_df.reject=lsd_df.pval.apply(compare_pval_alpha_tf)
    lsd_df.columns = ["Second Make", "P Value", "Signifigant Difference?"]
    lsd_df.index.names = ['First Make']

    return lsd_df
