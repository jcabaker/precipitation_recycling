#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
;;#############################################################################
;;
;; eltahir_bras_p_recycling.py
;; Author: Jess Baker (j.c.baker@leeds.ac.uk)
;; University of Leeds, UK
;;
;;#############################################################################
;;
;; Description
;;    This script is used to calculate precipitation recycling following the 
;;    methods of Eltahir and Bras (1994).
;;
;; Requirements
;;    Need column-integrated water vapour flux fields and evapotranspiration
;;    data (monthly means), plus an array providing the length of every grid
;;    cell in metres (shape = (Nt, Nlat, Nlon), where Nt is number of time
;;    steps., Nlat is number of latitudes and Nlon is number of longitudes).
;;
;;#############################################################################
"""


import numpy as np
import math


def main(qu, qv, et, dl, initial_guess=0.05, threshold=0.025):

    # convert all to numpy arrays
    qu = np.array(qu)
    qv = np.array(qv)
    et = np.array(et)
    dl = np.array(dl)

    # area of cell in metres
    print('Calculating cell areas')
    A = dl * dl
    EA = et * A

    # get fluxes
    print('Getting fluxes')
    inflow_from_E = -dl*qu
    inflow_from_W = dl*qu
    inflow_from_N = -dl*qv
    inflow_from_S = dl*qv

    # for all flux values along N, E, S & W multiply by relevant reycling ratio
    initial_rr = np.zeros((qu.shape)) + initial_guess

    total_flux_in = np.zeros((qu.shape))  

    print('Sum influx values')
    for flux in [inflow_from_E, inflow_from_W, inflow_from_N, inflow_from_S]:
        # identify all grid cell with negative influx values - set to zero influx
        inflow = flux.copy()
        inflow[inflow < 0] = 0

        total_flux_in += inflow

    print('Getting recycling ratios')
    local_rr = get_local_rr(initial_rr, total_flux_in, inflow_from_E, inflow_from_W, inflow_from_N, inflow_from_S, EA, threshold=threshold)

    return(local_rr)


def get_local_rr(initial_rr, total_flux_in, inflow_from_E, inflow_from_W, inflow_from_N, inflow_from_S, EA, threshold=0.025, N_limit=500):

    N = 0
    previous_rr = None
    out_arr = np.zeros((total_flux_in.shape + (N_limit,)))
    
    while N < N_limit:

        out_arr[:, :, :, N] = initial_rr

        # moisture influx from local evaporation (estimated iteratively)
        initial_rr_edge = add_edge(initial_rr)
        
        local_inflow_from_E = inflow_from_E * initial_rr_edge[:, 1:-1, 2:]
        local_inflow_from_W = inflow_from_W * initial_rr_edge[:, 1:-1, 0:-2]
        local_inflow_from_N = inflow_from_N * initial_rr_edge[:, 2:, 1:-1]
        local_inflow_from_S = inflow_from_S * initial_rr_edge[:, 0:-2, 1:-1]

        # sum influx values
        flux_in_local = np.zeros((total_flux_in.shape))
        for nn, flux in enumerate([local_inflow_from_E, local_inflow_from_W, local_inflow_from_N, local_inflow_from_S]):
            # identify all grid cell with negative influx values - set to zero influx
            inflow = flux.copy()
            inflow[inflow < 0] = 0
            flux_in_local += inflow

        updated_rr = (flux_in_local + EA)/(total_flux_in + EA)
        previous_rr = initial_rr

        initial_rr = updated_rr

        N += 1

    out_arr[out_arr==0] = np.nan
    
    return(out_arr[:, :, :, -1])


def add_edge(array, value=0.05):

    nt = array.shape[0]
    ny = array.shape[1]
    nx = array.shape[2]
    
    new_array = np.zeros((nt, ny+2, nx+2))
    new_array[:, 1:-1, 1:-1] = array

    # outside edges of domain all set to 1 (none of the incoming moisture from local evaporation)
    new_array[:, 0, :] = value
    new_array[:, -1, :] = value
    new_array[:, :, 0] = value
    new_array[:, :, -1] = value

    return(new_array)

