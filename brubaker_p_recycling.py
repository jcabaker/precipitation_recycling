#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
;;#############################################################################
;;
;; precipitation_recycling.py
;; Author: Jess Baker (j.c.baker@leeds.ac.uk)
;; University of Leeds, UK
;;
;;#############################################################################
;;
;; Description
;;    This script is used to calculate the Budyko recycling coefficient (B).
;;    For a given region, B is the ratio of locally evaporated to advected
;;    moisture.
;;
;;    Using the equation from Brukbaker et al. (1993):
;;
;;    B = 1 + EA/2Fin
;;
;;    where E is evap per unit area, A is the area of the region, and Fin is
;;    the total moisture flux into the region.
;;
;;    B can be used to estimate the precipitation recycling ratio (Rr), i.e. the
;;    proportion of precipitation within an area that originated from
;;    evaporation within that area.
;;
;;    Rr = 1 - (1/B)
;;
;; Requirements
;;    Need column-integrated water vapour and u and v winds to calculate
;;    mositure flux, as well as evaporation data (monthly means). Data should
;;    be formatted as Iris cubes, constrained to the same time period.
;;
;;#############################################################################
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
from datetime import datetime


def main(qu, qv, et, domain, plotting=True):

    """

    Program for calculating Budyko recycling coefficient (B) and precipitation
    recycling ratio (Rr) for a given domain.

    B = 1 + EA/2Fin (Brukbaker et al., 1993)

    Rr = 1 - (1/B)

    Takes harmonised Iris cubes constrained to the same timeframe as input.

    Arguments:
        qu = Iris cube of eastward water vapour flux.
        qv = Iris cube of northward water vapour flux.
        et = Iris cube of evapotranspiration.
        domain = list of box coordinates i.e. [latmin, latmax, lonmin, lonmax]
        plotting = Boolean. Plot output of metric.

    """

    # Calculate water vapour transport into grid box

    #      X---------X latmax
    #      |         |
    #      X---------X latmin
    #    lonmin     lonmax

    latmin = domain[0]
    latmax = domain[1]
    lonmin = domain[2]
    lonmax = domain[3]

    # Check if lats are ascending, if not then reverse
    qu = flip_lats(qu)
    qv = flip_lats(qv)
    et = flip_lats(et)

    lon = qu.coord('longitude').points
    lat = qu.coord('latitude').points
    cube_dates = qu.coord('time').units.num2date(qu.coord('time').points)
    dates = []
    for i in range(qu.shape[0]):
        year = cube_dates[i].year
        month = cube_dates[i].month
        dates.append(datetime(year, month, 1))

    # If needed convert domain lons to 0, 360
    print(lon.min())
    if any(i > 180 for i in lon) is True:
        if lonmin < 0:
            lonmin = lonmin+360
        if lonmax < 0:
            lonmax = lonmax+360
    print('lon min:', lon.min(), 'domain min:', lonmin)
    print('lon max:', lon.max(), 'domain max', lonmax)
    
    # Find average grid cell length on E, W, N and S sides of domain in m
    dl_e = find_dl(np.mean((latmin, latmax)), lon)
    print(dl_e)
    dl_w = find_dl(np.mean((latmin, latmax)), lon)
    print(dl_w)
    dl_n = find_dl(latmax, lon)
    print(dl_n)
    dl_s = find_dl(latmin, lon)
    print(dl_s)

    # Find indices for pixels on each side
    ii_e = np.where(lon >= (lonmax))[0][0]  # E lon index
    ii_w = np.where(lon >= (lonmin))[0][0]  # W lon index
    jj = np.where((lat >= latmin) & (lat <= latmax))[0]  # array of lats
    ii = np.where((lon >= (lonmin)) & (lon <= (lonmax)))[0]  # array of lons
    jj_n = np.where((lat <= latmax))[0][-1]  # N lat index
    jj_s = np.where((lat <= latmin))[0][-1]  # S lat index

    # Calculate inward moisture flux in kg s-1
    # Get water vapour flux for each pixel along transect (kg m-1 s-1) and
    # multiply by pixel lenth (m) to get units of kg s-1
    # i.e. for each pixel change units to kg s-1 then sum values
    NT = qu.shape[0]
    inflow_from_E = np.nan * np.zeros((NT))
    inflow_from_W = np.nan * np.zeros((NT))
    inflow_from_N = np.nan * np.zeros((NT))
    inflow_from_S = np.nan * np.zeros((NT))

    for n in range(NT):
        # NB) For E and N if +ve = negative inflow and if -ve = positive inflow
        inflow_from_E[n] = -dl_e*np.nansum(qu.data[n, jj, ii_e])
        inflow_from_W[n] = dl_w*np.nansum(qu.data[n, jj, ii_w])
        inflow_from_N[n] = -dl_n*np.nansum(qv.data[n, jj_n, ii])
        inflow_from_S[n] = dl_s*np.nansum(qv.data[n, jj_s, ii])

    if plotting is True:
        plt.figure()
        plt.plot(inflow_from_E/1e6)
        plt.plot(inflow_from_W/1e6)
        plt.plot(inflow_from_S/1e6)
        plt.plot(inflow_from_N/1e6)
        plt.legend(['inflow from E', 'inflow from W',
                    'inflow from S', 'inflow from N'], loc=(1.05, 0.4))
        plt.ylabel('10$^{6}$ kg s$^{-1}$')
        plt.tight_layout()
        plt.savefig('/nfs/see-fs-02_users/earjba/public_html/plots/python_plots/moisture_fluxes.png')

    # Calculate total water vapour flux into domain
    flux_in = np.zeros((NT))
    flux_out = np.zeros((NT))
    for nt in range(NT):
        if inflow_from_E[nt] > 0:
            flux_in[nt] += inflow_from_E[nt]
        if inflow_from_E[nt] < 0:
            flux_out[nt] += abs(inflow_from_E[nt])
            
        if inflow_from_W[nt] > 0:
            flux_in[nt] += inflow_from_W[nt]
        if inflow_from_W[nt] < 0:
            flux_out[nt] += abs(inflow_from_W[nt])
            
        if inflow_from_N[nt] > 0:
            flux_in[nt] += inflow_from_N[nt]
        if inflow_from_N[nt] < 0:
            flux_out[nt] += abs(inflow_from_N[nt])
            
        if inflow_from_S[nt] > 0:
            flux_in[nt] += inflow_from_S[nt]
        if inflow_from_S[nt] < 0:
            flux_out[nt] += abs(inflow_from_S[nt])

    domain_lats = [latmax, latmax, latmin, latmin, latmax]
    domain_lons = [lonmin, lonmax, lonmax, lonmin, lonmin]
    et_lat = et.coord('latitude').points
    et_lon = et.coord('longitude').points
    mask = get_box_mask(et_lat, et_lon, domain_lats, domain_lons)

    # Get grid of pixel areas
    pixel_size_grid = get_pixel_size(et_lat, et_lon)
    pixel_size_grid = np.array([pixel_size_grid]*len(et_lon)).transpose()

    # Average over mask for each timestep
    domain_et = np.nan * np.zeros((NT))
 
    # Mask invalid ET data
    et.data = np.ma.masked_invalid(et.data)
    for nt in range(NT):
        masked_data = np.ma.array(et.data[nt, :, :], mask=~mask)
        masked_weights = np.ma.array(pixel_size_grid, mask=~mask)
        domain_et[nt] = np.ma.average(masked_data, weights=masked_weights)

    # Calculate area of domain in m2
    A = np.nansum(mask * pixel_size_grid)
    print(A, 'm^2')

    # Calculate Budyko recycling coefficient
    EA = domain_et * A

    B = 1 + (EA/(2*flux_in))
    Rr = 1 - (1/B)

    # Make dataframe of results
    df = pd.DataFrame()
    df['dates'] = dates
    df = df.set_index('dates')
    df['ET'] = domain_et  # kg m-2 s-1
    df['A'] = A  # m2
    df['EA'] = EA  # kg s-1
    df['inflow_from_E'] = inflow_from_E
    df['inflow_from_W'] = inflow_from_W
    df['inflow_from_N'] = inflow_from_N
    df['inflow_from_S'] = inflow_from_S
    df['F_in'] = flux_in  # kg s-1
    df['F_out'] = flux_out  # kg s-1
    df['B'] = B  # Unitless
    df['Rr'] = Rr

    # Calculate seasonal cycle
    cycle = df.groupby([lambda x: x.month]).agg('mean')

    if plotting is True:
        bar_locations = np.arange(12)
        barwidth = 0.8
        plt.figure()
        plt.bar(bar_locations, cycle['Rr'], barwidth)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(411)
        ax.plot(range(len(B)), B)
        ax.set_ylabel('Budyko recycling coefficient')
        ax = fig.add_subplot(412)
        ax.plot(range(len(Rr)), Rr)
        ax.set_ylabel('Precipitation recycling ratio')
        ax = fig.add_subplot(413)
        ax.plot(range(len(Rr)), flux_in)
        ax.set_ylabel('Flux in')
        ax = fig.add_subplot(414)
        ax.plot(range(len(Rr)), flux_out)
        ax.set_ylabel('Flux out')
        plt.tight_layout()
        plt.savefig('/nfs/see-fs-02_users/earjba/public_html/plots/python_plots/p_recycling_plot.png')

    return(df)


def flip_lats(data_cube):

        lats = data_cube.coord('latitude').points
        
        # Check if lats need flipping
        if lats[0] < lats[-1]:
            print('Lats already ascending')
            return(data_cube)
        else:
            new_cube = data_cube.copy()
            new_lats = lats[::-1]
            new_data = data_cube.data[:, ::-1, :]
            new_cube.data = new_data
            new_cube.coord('latitude').points = new_lats
            print('Lats flipped')
            return(new_cube)


def find_dl(lat, lons):
        """
        Find average length of 1 grid cell - convert lat to radians, take cos +
        divide by Earth's circumference (m)
        """
        earth_circ = (40075*10**3)  # Earth's circumference in m
        lon_shape = len(lons)
        lat_rad = np.radians(abs(lat))
        dl = math.cos(lat_rad)*(earth_circ/lon_shape)  # length of pixel in m
        return(dl)


def get_box_mask(data_lat, data_lon, mask_lats, mask_lons):

    # Vertices of extraction domain
    coordlist = np.vstack((mask_lons, mask_lats)).T

    # Co-ordinates of every grid cell
    dat_x, dat_y = np.meshgrid(data_lon, data_lat)
    coord_map = np.vstack((dat_x.flatten(), dat_y.flatten())).T
    polypath = Path(coordlist)

    # Work out which coords are within the polygon
    mask = polypath.contains_points(coord_map).reshape(dat_x.shape)
    return(mask)


def get_pixel_size(lat, lon):
    if lat[0] > lat[-1]:
        temp_lat = lat[::-1]
    else:
        temp_lat = lat
    r = 6.371*1e6
    rad = (2*math.pi/360)  # (m)
    da = np.nan * np.zeros((len(temp_lat)))      # (m2)

    for i in range(len(temp_lat)-1):
        da[i] = (2*math.pi * (1/len(lon)) *
                 r**2*(math.sin(rad*temp_lat[i+1]) -
                       math.sin(rad*temp_lat[i])))

    # Check if top and bottom latitude are same
    if temp_lat[0] == -temp_lat[-1]:
        da[-1] = da[0]
    return(da)
