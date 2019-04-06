from __future__ import division
import pickle
import os
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_notebook, output_file, show, Figure, curdoc
from bokeh.models import HoverTool, ColumnDataSource, WMTSTileSource
from pyproj import Proj, transform

# Import the data
path = './fars_data/Accidents/'
accidents_list = []
for i, name in enumerate(os.listdir(path)):
    with open(os.path.join(path, name)) as inFile:
        accidents_list.append(pickle.load(inFile))
        
path = './fars_data/Person/'
person_list = []
for i, name in enumerate(os.listdir(path)):
    with open(os.path.join(path, name)) as inFile:
        person_list.append(pickle.load(inFile))
        
path = './fars_data/Vehicle/'
vehicle_list = []
for i, name in enumerate(os.listdir(path)):
    with open(os.path.join(path, name)) as inFile:
        vehicle_list.append(pickle.load(inFile))
        
with open('Pickle/id_to_state.pickle') as pickelObj:
    id_to_state = pickle.load(pickelObj)

with open('Pickle/us_states.pickle') as pickelObj:
    us_states = pickle.load(pickelObj)

# Clean the data
a_list = []
for i in xrange(len(accidents_list)):
    a, v = accidents_list[i], vehicle_list[i]
    
    # Remove unecissary columns
    v = v[["ST_CASE", "SPEEDREL"]]
    a = a[["ST_CASE", "STATE", "LATITUDE", "LONGITUD", "FATALS", "HOUR", "DAY", "MONTH", "YEAR", "DRUNK_DR"]]

    # drop null vals
    a["LONGITUD"] = a["LONGITUD"].replace([777.7777, 888.8888, 999.9999], np.nan)
    a["LATITUDE"] = a["LATITUDE"].replace([77.7777, 88.8888, 99.9999], np.nan)
    a = a.dropna()

    # Write state string
    a["STATE"] = a["STATE"].replace(0, 49)
    a["STATE"] = [id_to_state[x] for x in a["STATE"]]

    # Combine accedents and vehical dataFrames
    v["SPEEDREL"] = np.where((v["SPEEDREL"] >= 8), 0, v["SPEEDREL"])
    c = pd.merge(a, v, on="ST_CASE")

    # Create speeding column
    a["SPEEDING"] = c.groupby("ST_CASE").sum()["SPEEDREL"].values
    a["SPEEDING"] = a["SPEEDING"] != 0
    a["SPEEDING"] = a["SPEEDING"].astype(int)
    a_list.append(a)
    
accidents = pd.concat(a_list)

def convert(longitudes, latitudes):
    """Converts latlon coordinates to meters.
    Inputs:
    longitudes (array-like) : array of longitudes
    latitudes (array-like) : array of latitudes
    Example:
    x,y = convert(accidents.LONGITUD, accidents.LATITUDE)
    """
    from_proj = Proj(init="epsg:4326")
    to_proj = Proj(init="epsg:3857")
    
    x_vals = []
    y_vals = []
    for lon, lat in zip(longitudes, latitudes):
        x, y = transform(from_proj, to_proj, lon, lat)
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals

accidents["x"], accidents["y"] = convert(accidents.LONGITUD, accidents.LATITUDE)

d_list = []
for i in xrange(len(person_list)):
    p = person_list[i]
    v = vehicle_list[i]
    p = p[["ST_CASE", "VEH_NO", "PER_TYP", "AGE", "DRINKING"]]
    v = v[["SPEEDREL", "ST_CASE", "VEH_NO",]]
    d = pd.merge(p, v, on=["ST_CASE", "VEH_NO"])
    d["YEAR"] = 2010
    d_list.append(d)
    
drivers = pd.concat(d_list)

fig = Figure(plot_width=1100, plot_height=650,
    x_range=(-13000000, -7000000), y_range=(2750000, 6250000),
    tools=["wheel_zoom", "pan"], active_scroll="wheel_zoom", webgl=True)

fig.axis.visible = False

STAMEN_TONER_BACKGROUND = WMTSTileSource(
url='http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png',
attribution=(
'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.'
'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>')
)
fig.add_tile(STAMEN_TONER_BACKGROUND)

state_xs = [us_states[code]["lons"] for code in us_states]
state_ys = [us_states[code]["lats"] for code in us_states]

x_vals, y_vals = [], []
x_vals, y_vals = convert(state_xs, state_ys)

speeding_accidents = accidents[accidents["SPEEDING"] != 0]
drinking_accidents = accidents[accidents["DRUNK_DR"] != 0]
other_accidents = accidents[(accidents["DRUNK_DR"] == 0) & (accidents["SPEEDING"] == 0)]

total_acc = [len(accidents[accidents['STATE']==state_code]) for state_code in us_states]
total_drunk = [len(drinking_accidents[drinking_accidents["STATE"]==state_code]) for state_code in us_states]
total_speeding = [len(speeding_accidents[speeding_accidents["STATE"]==state_code]) for state_code in us_states]
total_other = [len(other_accidents[other_accidents["STATE"]==state_code]) for state_code in us_states]

perc_drunk = [a / b for a, b in zip(total_drunk, total_acc)]
perc_speeding = [a / b for a, b in zip(total_speeding, total_acc)]
perc_other = [a / b for a, b in zip(total_other, total_acc)]

# convert to strings for tooltip
total = [str(x) for x in total_acc]
perc_drunk = [str(x*100)+"%" for x in perc_drunk]
perc_speeding = [str(x*100)+"%" for x in perc_speeding]

border_source = ColumnDataSource(dict(
        xs=x_vals, 
        ys=y_vals,
        total=total_acc,
        state = us_states.keys(),
        perc_drunk = perc_drunk,
        perc_speeding = perc_speeding
    ))

states = fig.patches("xs", "ys", source=border_source, alpha=.5, line_color="red", hover_color="green", hover_alpha=.8, hover_line_color='black')

speeding_source = ColumnDataSource(dict(
    x=speeding_accidents['x'],
    y=speeding_accidents['y'] ))

drinking_source = ColumnDataSource(dict(
    x=drinking_accidents['x'],
    y=drinking_accidents['y'] ))

other_source = ColumnDataSource(dict(
    x=other_accidents['x'],
    y=other_accidents['y'] ))

fig.circle('x', 'y', source=speeding_source, fill_color="red", size=3)
fig.circle('x', 'y', source=drinking_source, fill_color="green", size=3)
fig.circle('x', 'y', source=other_source, fill_color="blue", size=3)

fig.add_tools(HoverTool(renderers=[states], tooltips=[("State", "@state"), ("Total", "@total"), ("Drunk Percent", "@perc_drunk"), ("Speeding Percent",  "@perc_speeding")]))

# -------------------------------------------------

from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models.widgets import Select
from bokeh.palettes import Reds9

output_file("final_bokeh.html")

select = Select(title="Option:", value="Default",
        options=["Other", "Speeding", "Drunk", "Default"])

border_source.add("white", name="color")

fig = Figure(plot_width=1100, plot_height=650,
x_range=(-13000000, -7000000), y_range=(2750000, 6250000),
tools=["wheel_zoom", "pan"], active_scroll="wheel_zoom", webgl=True)

fig.axis.visible = False

STAMEN_TONER_BACKGROUND = WMTSTileSource(
url='http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png',
attribution=(
'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY -3.0</a>.'
'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
)
)

fig.add_tile(STAMEN_TONER_BACKGROUND)

fig.patches("xs", "ys", source=border_source, fill_color="color", line_color="red", hover_color="green", hover_alpha=.8, hover_line_color='black')

#make the strings into numbers
drunk_percent = [float(x[:-1]) for x in perc_drunk]
speeding_percent = [float(x[:-1]) for x in perc_speeding]

Reds9 = Reds9[::-1]
no_colors = ['#FFFFFF']*len(us_states.keys())
other_colors = [Reds9[i] for i in pd.qcut(perc_other, len(Reds9)).codes]
drunk_colors = [Reds9[i] for i in pd.qcut(drunk_percent, len(Reds9)).codes]
speeding_colors = [Reds9[i] for i in pd.qcut(speeding_percent, len(Reds9)).codes]

def update_color(attrname, old, new):
    if new=="Speeding":
        border_source.data["color"] = speeding_colors
    if new=="Drunk":
        border_source.data["color"] = drunk_colors
    if new=="Other":
        border_source.data["color"] = other_colors
    if new=="Default":
        border_source.data["color"] = no_colors


select.on_change('value', update_color)
curdoc().add_root(column(fig, select))


