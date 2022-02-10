import pandas
import pdb
from datetime import datetime
import matplotlib
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import sys
from matplotlib.ticker import MultipleLocator
import json
import os
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=16)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def get_radar_data(rm_list, instances, batch='512'):
    all_lat = []
    min_lat = []
    all_picked = []

    for rm in rm_list:
        # for each testcase, draw a plot
        # to draw a plot, need a dict of latency for each instance
        latency = []
        testcase = rm + '_' + batch
        ins_list = []
        picked_lat = {}
        for instance in instances:
            if rm == 'rm1' and instance == 't3.2xlarge': # measurement outline
                instance_fix = 't3.xlarge'
            else:
                instance_fix = instance
            path = '../../'+instance_fix + '/' + testcase + '.json'
            path_load = '../../../logs_load/' + instance_fix + '/' + testcase + '.json'
            check_path = '../../'+instance_fix + '/ncf_128.json'
            if os.path.isfile(check_path):
                with open(path, 'r') as f:
                    lat_list = json.load(f)
                with open(path_load, 'r') as f_load:
                    load_list = np.asarray(json.load(f_load))
                if len(lat_list) != len(load_list):
                    print('error with instance', instance)
                    continue
                else:
                    total_lat = lat_list + load_list
                    filter_lat = np.delete(total_lat, total_lat.argmax())
                    latency.append(np.mean(filter_lat))
                    ins_list.append(instance)
                    picked_lat[instance] = np.mean(filter_lat)
        all_lat.append(latency)
        min_lat.append(min(latency))
        all_picked.append(picked_lat)
    
    all_picked_norm = {}
    for i in range(len(all_picked)):
        all_picked_norm[i] = {}
        for instance in instances:
            all_picked_norm[i][instance] = min_lat[i] / all_picked[i][instance]
    
    y1 = [round(all_picked_norm[0][ins],2) for ins in instances]
    y2 = [round(all_picked_norm[1][ins],2) for ins in instances]
    return [y1,y2]

