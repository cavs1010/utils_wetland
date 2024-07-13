import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

def calculate_outliers(data):
    """
    Calculate the number of outliers in a given data series using the IQR method.
    
    Parameters:
    data (pd.Series): The data series to analyze.
    
    Returns:
    int: The number of outliers in the data series.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers)

def add_sample_and_outliers_text(ax, df, value_column, category_column):
    """
    Add sample size and outliers count annotations to a boxplot.
    
    Parameters:
    ax (matplotlib.axes.Axes): The Axes object to annotate.
    df (pd.DataFrame): The dataframe containing the data.
    value_column (str): The name of the column containing the values.
    category_column (str): The name of the column containing the categories.
    """
    outlier_counts = df.groupby(category_column)[value_column].apply(calculate_outliers)
    sample_sizes = df[category_column].value_counts().sort_index()
    means = df.groupby(category_column)[value_column].mean()
    medians = df.groupby(category_column)[value_column].median()
    std_devs = df.groupby(category_column)[value_column].std()
    q1s = df.groupby(category_column)[value_column].quantile(0.25)
    q3s = df.groupby(category_column)[value_column].quantile(0.75)
    iqrs = q3s - q1s
    
    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15  # 15% of the y-axis range for spacing

    for i, category in enumerate(df[category_column].unique()):
        count = sample_sizes[category]
        outliers = outlier_counts[category]
        mean_val = means[category]
        median_val = medians[category]
        std_val = std_devs[category]
        iqr_val = iqrs[category]
        ax.text(i, df[value_column].max() if ax.get_ylim()[1] is None else ax.get_ylim()[1] - offset, 
                f'n={count}\nOutliers={outliers}\nMean={mean_val:.2f}\nMedian={median_val:.2f}\nStd={std_val:.2f}\nIQR={iqr_val:.2f}', 
                horizontalalignment='center', size='medium', color='black', weight='semibold')

def plot_boxplot(df, value_column, category_column, y_label, title_prefix, y_limits=None):
    """
    Plot a boxplot with sample size and outliers count annotations.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    value_column (str): The name of the column containing the values.
    category_column (str): The name of the column containing the categories.
    y_label (str): The label for the y-axis.
    title_prefix (str): The prefix for the plot title.
    y_limits (tuple): Optional. The limits for the y-axis in the format (y_min, y_max).
    """
    plt.figure(figsize=(15, 8))
    ax = sns.boxplot(x=category_column, y=value_column, data=df, hue=category_column, palette="Set2", legend=False)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    add_sample_and_outliers_text(ax, df, value_column, category_column)
    
    plt.title(f'{title_prefix} by {category_column}', pad=22)
    plt.xlabel(category_column)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def perform_kruskal_wallis_test(df, value_column, category_column, min_sample_size):
    """
    Perform the Kruskal-Wallis H test for independent samples, excluding groups with sample size below a threshold.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    value_column (str): The name of the column containing the values.
    category_column (str): The name of the column containing the categories.
    min_sample_size (int): The minimum sample size required to include a group in the analysis.
    
    Returns:
    kruskal_result: The result of the Kruskal-Wallis H test.
    """
    valid_groups = [category for category in df[category_column].unique() if len(df[df[category_column] == category]) >= min_sample_size]
    excluded_groups = [category for category in df[category_column].unique() if len(df[df[category_column] == category]) < min_sample_size]
    groups = [df[df[category_column] == category][value_column] for category in valid_groups]
    
    if excluded_groups:
        print(f'😔 Excluded from analysis due to small sample size: {excluded_groups}')
    
    kruskal_result = kruskal(*groups)
    print(f'Kruskal-Wallis H Test: Statistic={kruskal_result.statistic}, p-value={kruskal_result.pvalue}')
    return kruskal_result

def perform_posthoc_dunn_test(df, value_column, category_column, min_sample_size):
    """
    Perform Dunn's post-hoc test for multiple comparisons after the Kruskal-Wallis test, excluding groups with sample size below a threshold.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    value_column (str): The name of the column containing the values.
    category_column (str): The name of the column containing the categories.
    min_sample_size (int): The minimum sample size required to include a group in the analysis.
    
    Returns:
    dunn_result (pd.DataFrame): The result of Dunn's post-hoc test.
    """
    valid_df = df[df[category_column].isin([category for category in df[category_column].unique() if len(df[df[category_column] == category]) >= min_sample_size])]
    dunn_result = posthoc_dunn(valid_df, val_col=value_column, group_col=category_column, p_adjust='fdr_bh')
    #print(f'Dunn\'s Test Results:\n')
    return dunn_result

def plot_dunn_heatmap(dunn_result):
    """
    Plot a heatmap of Dunn's post-hoc test results, highlighting p-values less than 0.05.
    
    Parameters:
    dunn_result (pd.DataFrame): The result of Dunn's post-hoc test.
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(dunn_result, annot=True, fmt=".2f", cmap="coolwarm", cbar=False,
                     annot_kws={"size": 10, "weight": "bold"},
                     linewidths=.5)
    for text in ax.texts:
        if float(text.get_text()) < 0.05:
            text.set_text('★')
            text.set_size(14)
            text.set_weight('bold')
            text.set_color('white')
    
    cbar = ax.collections[0].colorbar
    #cbar.set_label('p-values')
    plt.title('Dunn\'s Post-hoc Test Results\n(p-values)')
    
    # Adding a custom legend for the star
    star_patch = plt.Line2D([0], [0], marker='*', color='w', label='Significant difference (p < 0.05)',
                            markerfacecolor='blue', markersize=15)
    plt.legend(handles=[star_patch], loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.show()

def create_boxplot_and_stats(df, value_column, category_column, y_label, title_prefix, y_limits=None, min_sample_size=1):
    """
    Create a boxplot and perform Kruskal-Wallis and Dunn's post-hoc tests, excluding groups with sample size below a threshold.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    value_column (str): The name of the column containing the values.
    category_column (str): The name of the column containing the categories.
    y_label (str): The label for the y-axis.
    title_prefix (str): The prefix for the plot title.
    y_limits (tuple): Optional. The limits for the y-axis in the format (y_min, y_max).
    min_sample_size (int): The minimum sample size required to include a group in the analysis.
    
    Returns:
    dunn_result (pd.DataFrame or None): The result of Dunn's post-hoc test if significant differences are found.
    """
    plot_boxplot(df, value_column, category_column, y_label, title_prefix, y_limits)
    kruskal_result = perform_kruskal_wallis_test(df, value_column, category_column, min_sample_size)
    if kruskal_result.pvalue < 0.05:
        dunn_result = perform_posthoc_dunn_test(df, value_column, category_column, min_sample_size)
        
        # Plot the heatmap of Dunn's test results
        plot_dunn_heatmap(dunn_result)
        
        #return dunn_result
    else:
        print("No significant difference found by Kruskal-Wallis test.")
        return None