import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct
import scipy.optimize as opt
import errors as err


def read_data(file):
    """
    The function accepts a file and reads it into a pandas DataFrame and 
    cleans it and transposes it. It returns the cleaned original and 
    transposed DataFrame.

    Parameters
    ----------
    file : string
        The file name to be read into DataFrame.

    Returns
    -------
    df_clean : pandas DataFrame
        The cleaned version of the ingested DataFrame.
    df_t : pandas DataFrame
        The transposed version of the cleaned DataFrame.

    """

    # reads in an excel file
    if ".xlsx" in file:
        df = pd.read_excel(file, index_col=0)
    # reads in a csv file
    elif ".csv" in file:
        df = pd.read_csv(file, index_col=0)
    else:
        print("invalid filetype")
    # cleans the DataFrame
    df_clean = df.dropna(axis=1, how="all").dropna()
    # transposes the cleaned DataFrame
    df_t = df_clean.transpose()

    return df_clean, df_t


# for reproducibility
np.random.seed(10)


def kmeans_cluster(nclusters):
    """
    The function produces cluster centers and labels through kmeans
    clustering of given number of clusters and returns the cluster
    centers and the cluster labels.

    Parameters
    ----------
    nclusters : int
        The number of clusters.

    Returns
    -------
    labels : string 
        The labels of the clusters.
    cen : list of lists
        The coordinates of the cluster centres.

    """
    kmeans = cluster.KMeans(n_clusters=nclusters)
    # df_cluster is the dataframe in which clustering is performed
    kmeans.fit(df_cluster)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    return labels, cen


def poly(x, a, b, c):
    """
    The function which produces a polynomial curve for fitting the data.

    Parameters
    ----------
    x : int or float
        The variable of the polynomial.
    a : int or float
        The constant of the polynomial.
    b : int or float
        The coefficient of x.
    c : int or float
        The coefficient of x**2.

    Returns
    -------
    f : array
        The polynomial curve.

    """

    x = x - 2003
    f = a + b*x + c*x**2

    return f


# the csv files are read into dataframes
_, co2_df = read_data("co2_emissions.csv")
print(co2_df)

_, gdp_per_capita_df = read_data("gdp_per_capita.csv")
print(gdp_per_capita_df)

# Specific columns are extracted
co2_Poland = co2_df.loc[:, "Poland"].copy()
print(co2_Poland)

gdp_per_capita_Poland = gdp_per_capita_df.loc["1990":"2019", "Poland"].copy()
print(gdp_per_capita_Poland)

# The extracted columns are merged into a dataframe
df_Poland = pd.merge(co2_Poland, gdp_per_capita_Poland, on=co2_Poland.index,
                    how="outer")
df_Poland = df_Poland.rename(columns={'key_0': "Year",
                                    'Poland_x': "co2_emissions",
                                    'Poland_y': "gdp_per_capita"})
df_Poland = df_Poland.set_index("Year")
print(df_Poland)

# the scatter matrix of the dataframe is plotted
pd.plotting.scatter_matrix(df_Poland)

# The dataframe for clustering is created
df_cluster = df_Poland[["co2_emissions", "gdp_per_capita"]].copy()

# The data is normalized
df_cluster, df_min, df_max = ct.scaler(df_cluster)

# The number of clusters and respective silhouette scores are printed
print("n   score")
for ncluster in range(2, 10):
    lab, cent = kmeans_cluster(ncluster)
    print(ncluster, skmet.silhouette_score(df_cluster, lab))

# The cluster centers and labels are calculated using the function
label, center = kmeans_cluster(5)
xcen = center[:, 0]
ycen = center[:, 1]

# The clustering is plotted
plt.figure()
cm = plt.cm.get_cmap('Set1')
plt.scatter(df_cluster['gdp_per_capita'], df_cluster["co2_emissions"], s=10,
            c=label, marker='o', cmap=cm)
plt.scatter(xcen, ycen, s=20, c="k", marker="d")
plt.title("CO2 emission vs GDP per capita of Poland", fontsize=20)
plt.xlabel("GDP per capita", fontsize=18)
plt.ylabel("CO2 emissions", fontsize=18)
plt.show()

# The cluster centres are rescaled to original scale
centre = ct.backscale(center, df_min, df_max)
xcen = centre[:, 0]
ycen = centre[:, 1]


# The clustering is plotted with the original scale
plt.figure()
cm = plt.cm.get_cmap('Set1')
plt.scatter(df_Poland['gdp_per_capita'], df_Poland["co2_emissions"], 10,
            label, marker='o', cmap=cm)
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions")
plt.title("CO2 emission vs GDP per capita of Poland")
plt.show()


t = ['1990', '1995', '2000', '2005', '2010', '2015', '2020']

# The plot of CO2 Emissions(1990-2019) in Poland is plotted
plt.plot(df_Poland.index, df_Poland['co2_emissions'])
plt.xlabel("Years", fontsize=16)
plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
plt.title("CO2 Emissions (1990-2019)", fontsize=18)
plt.xticks(ticks=t, labels=t)
plt.show()

# The plot of GDP per capita (1990-2019) in Poland is plotted
plt.plot(df_Poland.index, df_Poland["gdp_per_capita"])
plt.xlabel("Years", fontsize=16)
plt.ylabel("GDP per capita", fontsize=14)
plt.title("GDP per capita (1990-2019)", fontsize=18)
plt.xticks(ticks=t, labels=t)
plt.show()

# The dataframe is prepared for fitting
df_Poland = df_Poland.reset_index()
df_Poland["gdp_per_capita"] = pd.to_numeric(df_Poland["gdp_per_capita"])
df_Poland["Year"] = pd.to_numeric(df_Poland["Year"])

# The fitting of the GDP per capita plot
# Calculates the parameters and covariance
param, covar = opt.curve_fit(poly, df_Poland["Year"],
                             df_Poland["gdp_per_capita"])
# Calculates the standard deviation
sigma = np.sqrt(np.diag(covar))
year = np.arange(1990, 2030)
# Calculates the fitting curve
forecast = poly(year, *param)
# Calculates the confidence range
low, up = err.err_ranges(year, poly, param, sigma)
df_Poland["fit1"] = poly(df_Poland["Year"], *param)
# Plots the graph with fitting and confidence range
plt.figure()
plt.plot(df_Poland["Year"], df_Poland["gdp_per_capita"], label="GDP", c='blue')
plt.plot(year, forecast, label="forecast", c='red')
plt.fill_between(year, low, up, color="yellow", alpha=0.8)
plt.xlabel("Year", fontsize=16)
plt.ylabel("GDP per capita", fontsize=14)
plt.title("GDP per capita forecast of Poland", fontsize=18)
plt.legend()
plt.show()

# The fitting of CO2 Emissions plot
# Calculates the parameters and covariance
param, covar = opt.curve_fit(poly, df_Poland["Year"], df_Poland["co2_emissions"])
# Calculates the standard deviation
sigma = np.sqrt(np.diag(covar))
# Calculates the fitting curve
forecast = poly(year, *param)
# Calculates the confidence range
low, up = err.err_ranges(year, poly, param, sigma)
df_Poland["fit2"] = poly(df_Poland["Year"], *param)
# Plots the graph with fitting and confidence range
plt.figure()
plt.plot(df_Poland["Year"], df_Poland["co2_emissions"], label="CO2 emissions",
         c='green')
plt.plot(year, forecast, label="forecast", c="red")
plt.fill_between(year, low, up, color="yellow", alpha=0.8)
plt.xlabel("Year", fontsize=16)
plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
plt.title("CO2 Emissions Forecast of Poland", fontsize=18)
plt.legend()
plt.show()