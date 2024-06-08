def calculate_outliers(data):
    """
    Calculate the number of outliers in a dataset using the IQR method.

    An outlier is defined as a data point that lies below the lower bound or above the upper bound.
    The bounds are calculated as follows:
    - Lower Bound: Q1 - 1.5 * IQR
    - Upper Bound: Q3 + 1.5 * IQR
    where Q1 is the first quartile, Q3 is the third quartile, and IQR is the interquartile range (Q3 - Q1).

    Parameters:
    data (pandas Series): A pandas Series containing the dataset to analyze for outliers.

    Returns:
    int: The number of outliers in the dataset.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers)
