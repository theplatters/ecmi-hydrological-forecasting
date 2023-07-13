import os

import pandas as pd

os.environ["QT_API"] = "pyqt5"

import numpy as np
import matplotlib
import scipy
import matplotlib.widgets as mwidgets

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from matplotlib.widgets import RangeSlider


def pearson(dx, dy):
    return scipy.stats.pearsonr(dx, dy)


def spearman(dx, dy):
    return scipy.stats.spearmanr(dx, dy)


def distance_correlation(dx, dy):
    return dcor.distance_correlation(dx, dy), 0


def timewarping(dx, dy):
    distance, cost_matrix, acc_cost, path = dtw(np.array(dx).reshape(-1, 1), np.array(dy).reshape(-1, 1),
                                                dist=scipy.spatial.distance.euclidean)
    return distance


def corr(df, station, starting_point, interval, method, l, pv, p1):
    period2 = df[(df.index >= starting_point - interval) & (df.index <= starting_point + interval)]
    for col in df.columns.tolist():
        cor, p_value = method(p1[station], period2[col])
        l.append(cor)
        pv.append(p_value)


def calculate_correlations(df, starting_point, interval, method):
    cross_correlations = []
    for station in df.columns:
        day = pd.Timedelta(1, 'd')
        sp = starting_point
        l = []
        pv = []

        period1 = df[(df.index >= starting_point - interval) & (df.index <= starting_point + interval)]
        for i in np.linspace(0, 5, 5):
            starting_point = starting_point + day
            corr(df, station, starting_point, interval, method, l, pv, period1)
        starting_point = sp + day
        for i in np.linspace(0, 50, 50):
            starting_point = starting_point - day
            corr(df, station, starting_point, interval, method, l, pv, period1)
        cm = np.transpose(np.reshape(np.array(l), (-1, len(df.columns))))
        pv = np.transpose(np.reshape(np.array(pv), (-1, len(df.columns))))
        cross_correlations.append(cm)
    return cross_correlations

"""
data_training = pd.read_csv("../data/data_training.csv")
data_validation = pd.read_csv('../data/data_validation.csv')

data = pd.concat([data_training, data_validation])
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index('Date')
meta = pd.read_csv("../data/meta.csv")

meta = meta.set_index("reg_number")
meta_nans_removed = meta.loc[list(map(int,data.columns))]

cr = calculate_correlations(data, pd.Timestamp('2005-01-01'), pd.Timedelta(52, 'w'), pearson)

correlation_tensor = {}
for idx, station in enumerate(data.columns):
    correlation_tensor[station] = (pd.DataFrame(data=np.transpose(cr[idx]), columns=data.columns))

print(correlation_tensor['1515'])

correlation_tensor_max_corr = {k: v.idxmax() - 5 for k, v in correlation_tensor.items()}
"""

def rgb(val):
    return [[0.4 + min(val * 0.08, 0.6), 0.2, 0.00 + min(abs(val * 0.10), 1.0)]]


def draw():
    for index, row in meta_nans_removed.iterrows():
        a = plt.scatter(row['EOVx'], row['EOVy'], c=rgb(row['maximum_correlation']), marker=marker_dict[row['river']])
        if row['river'] not in rivers:
            rivers.append(row['river'])
            actors.append(a)
    plt.legend(actors, rivers)
    plt.show()


def update_plot(text):
    l = correlation_tensor_max_corr[text]
    # Parse the text and update the scatterplot accordingly
    # Here, I'm assuming the text contains comma-separated x, y coordinates

    # Split the text by commas and convert the coordinates to floats
    coordinates = [float(coord.strip()) for coord in text.split(',')]

    # Split the coordinates into x and y lists
    x = coordinates[::2]
    y = coordinates[1::2]

    # Update the scatterplot data
    scatter.set_offsets(list(zip(x, y)))

    # Redraw the plot
    plt.draw()


# Create a textbox widget
axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
textbox = mwidgets.TextBox(axbox, 'Enter coordinates:')

# Register the update function to be called when the textbox value changes
textbox.on_submit(update_plot)

# Show the plot
plt.show()
