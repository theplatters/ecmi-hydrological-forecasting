import os

import pandas as pd

import numpy as np
import matplotlib
import scipy
import matplotlib.widgets as mwidgets

import matplotlib.pyplot as plt

os.environ["QT_API"] = "pyqt5"
matplotlib.use('Qt5Agg')


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
        for i in np.linspace(0, 50, 31):
            starting_point = starting_point - day
            corr(df, station, starting_point, interval, method, l, pv, period1)
        cm = np.transpose(np.reshape(np.array(l), (-1, len(df.columns))))
        pv = np.transpose(np.reshape(np.array(pv), (-1, len(df.columns))))
        cross_correlations.append(cm)
    return cross_correlations


# %%
data_training = pd.read_csv("../data/data_training.csv")
data_validation = pd.read_csv('../data/data_validation.csv')

data = pd.concat([data_training, data_validation])
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index('Date')
meta = pd.read_csv("../data/meta.csv")

meta = meta.set_index("reg_number")
meta_nans_removed = meta.loc[list(map(int, data.columns))]
# %%
cr = calculate_correlations(data, pd.Timestamp('2005-01-01'), pd.Timedelta(52, 'w'), pearson)

correlation_tensor = {}
for idx, station in enumerate(data.columns):
    correlation_tensor[station] = (pd.DataFrame(data=np.transpose(cr[idx]), columns=data.columns))

print(correlation_tensor['1515'])

correlation_tensor_max_corr = {k: v.idxmax() - 5 for k, v in correlation_tensor.items()}

# %%
correlation_tensor_max_corr['2275']
# %%
marker_dict = {
    'Tisza': 'o',
    'Maros': 'v',
    'Kettős-Körös': '^',
    'Hármas-Körös': 'x',
    'Szamos': '+',
    'Sebes-Körös': 'D',
    'Bodrog': 'h',
    'Túr': 'd',
    'Sajó': 'X',
    'Kraszna': '1',
    'Hernád': '2',
    'Berettyó': '3',
    'Fekete-Körös': '4',
    'Fehér-Körös': ',',
    'Zagyva': '<'
}

x_max = meta_nans_removed['EOVx'].max() + 10000
x_min = meta_nans_removed['EOVx'].min() - 10000
y_min = meta_nans_removed['EOVy'].min() + 10000
y_max = meta_nans_removed['EOVy'].max() - 10000


# %%
def rgb(val):
    return [[0.4 + min(val * 0.08, 0.6), 0.2, 0.00 + min(abs(val * 0.10), 1.0)]]


def draw(station, slider_value):
    ax.clear()
    stations_to_draw = correlation_tensor_max_corr[station].loc[
        abs(correlation_tensor_max_corr[station]) <= slider_value]
    for other_station, delay in stations_to_draw.items():
        if station == other_station:
            ax.scatter(meta_nans_removed.loc[int(other_station), 'EOVx'],
                       meta_nans_removed.loc[int(other_station), 'EOVy'], c=[[0.0, 0.9, 0.9]],
                       marker=marker_dict[meta_nans_removed.loc[int(other_station), 'river']])
        else:
            ax.scatter(meta_nans_removed.loc[int(other_station), 'EOVx'],
                       meta_nans_removed.loc[int(other_station), 'EOVy'], c=rgb(delay),
                       marker=marker_dict[meta_nans_removed.loc[int(other_station), 'river']])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()


# %%
def update_plot(text):
    draw(text, slider.val)


def update_slider(val):
    draw(textbox.text, val)


def update(frame):
    draw('2275', frame)


# %%
import matplotlib.animation as animation

# %%
fig = plt.figure(figsize=(8, 6))
# gs = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[1, 1, 10])  # Create a GridSpec layout with 3 rows

# Create the plot
ax = plt.subplot()  # Assign the bottom row to the plot
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

# Create a textbox widget
# ax_textbox = plt.subplot(gs[0])  # Assign the top row to the textbox
# textbox = mwidgets.TextBox(ax_textbox, 'Enter coordinates:')

# Create a slider widget
# ax_slider = plt.subplot(gs[1])  # Assign the middle row to the slider
# slider = mwidgets.Slider(ax_slider, 'Slider', 0, 10, valinit=0,valstep=1)

# Register the update functions to be called when the textbox value or slider value changes
# textbox.on_submit(update_plot)
# slider.on_changed(update_slider)


# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.1)
Writer = animation.writers['imagemagick']
writer = Writer(fps=15, metadata=dict(artist='Me'))

ani = animation.FuncAnimation(fig=fig, func=update, frames=10, interval=500)
ani.save(filename='corr_animated.gif', writer='pillow',fps=1)

# plt.show()
