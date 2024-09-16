"""
Script to create evaluation plots for COCOA models
either call plot_everything function from different script like analyse_cocoa_predictions.py
or run this script directly with the following command if analysisfile is already created:
python plot_cocoa.py /path/to/analysisfile /path/to/outputdir
"""

import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import plotly.express as px
import argparse
from scipy.stats import norm
from scipy.optimize import curve_fit


def calc_efficiencies(df, bins, mask=None):
    """
    Helper function to calculate the efficiencies, fake rates and classification accuracy for the photons and neutral hadrons
    """
    efficiencies = []
    efficiencies_err = []
    fake_rate = []
    fake_rate_err = []
    corr_class_prob = []
    corr_class_prob_err = []    
    
    mask_predicted = np.isnan(df['pred_energy']) == False
    mask_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    
    if mask == None:
        mask_PID_truth = np.ones(len(df), dtype=bool)
        mask_PID_pred = mask_PID_truth
    else:
        if(mask == 0):
            mask_PID_truth = df['truthHitAssignedPIDs'].isin([22])
        elif(mask == 1):
            mask_PID_truth = df['truthHitAssignedPIDs'].isin([130,310,311,2112,-2112,3122,-3122,3322,-3322])
        else:
            raise ValueError("mask must be 0 or 1")
        mask_PID_pred = df['pred_id_value'].isin([mask])
        
    mask_PID_matched = np.logical_and(mask_PID_truth, mask_PID_pred)
        
    
    for i in range(len(bins) - 1):
        mask_bintruth = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_binpred = np.logical_and(
            df['pred_energy'] >= bins[i],
            df['pred_energy'] < bins[i + 1])

        matched = np.logical_and(
            mask_predicted,
            mask_bintruth)
        faked = np.logical_and(
            mask_binpred,
            np.logical_not(mask_truth))
        
        eff = np.sum(matched[mask_PID_truth]) / np.sum(mask_bintruth[mask_PID_truth])
        eff_err = np.sqrt(eff * (1 - eff) / np.sum(matched[mask_PID_truth])) 
        efficiencies.append(eff)
        efficiencies_err.append(eff_err)
        
        fake = np.sum(faked[mask_PID_pred]) / np.sum(mask_binpred[mask_PID_pred])
        fake_err = np.sqrt(fake * (1 - fake) / np.sum(faked[mask_PID_pred]))
        fake_rate.append(fake)
        fake_rate_err.append(fake_err)
        
        cc_prob = np.sum(matched[mask_PID_matched]) / np.sum(matched[mask_PID_truth])
        cc_prob_err = np.sqrt(cc_prob * (1 - cc_prob) / np.sum(matched[mask_PID_truth]))
        corr_class_prob.append(cc_prob)
        corr_class_prob_err.append(cc_prob_err)
    
    return np.array(efficiencies), np.array(efficiencies_err), np.array(fake_rate), np.array(fake_rate_err), np.array(corr_class_prob), np.array(corr_class_prob_err)

def plot_efficencies(df, bins):
    """
    Function to plot the efficiencies, fake rates and classification accuracy for the photons and neutral hadrons similar to the HGPFlow paper
    """ 
    # Calculate the efficiencies and fake rates for the photons
    yeff_photon, yerr_eff_photon, yfake_photon, yerr_fake_photon, ycorr_photon, yerr_corr_photon = calc_efficiencies(df, bins, 0)
    yeff_photon, yerr_eff_photon, yfake_photon, yerr_fake_photon, ycorr_photon, yerr_corr_photon = \
        yeff_photon*100, yerr_eff_photon*100, yfake_photon*100, yerr_fake_photon*100, ycorr_photon*100, yerr_corr_photon*100
    
    # Calculate the efficiencies and fake rates for the neutral hadrons
    #130: "K0L", 310: "K0S", 311: "K0", 2112: "neutron",-2112: "antineutron",3122: "Lambda",-3122: "antilambda"3322: "Xi0",-3322: "antixi0"
    yeff_nh, yerr_eff_nh, yfake_nh, yerr_fake_nh, ycorr_nh, yerr_corr_nh = calc_efficiencies(df, bins, 1)
    yeff_nh, yerr_eff_nh, yfake_nh, yerr_fake_nh, ycorr_nh, yerr_corr_nh = \
        yeff_nh*100, yerr_eff_nh*100, yfake_nh*100, yerr_fake_nh*100, ycorr_nh*100, yerr_corr_nh*100
    
    # Calculate the bin positions and widths
    binwidth = bins[1:] - bins[:-1]
    x_pos = bins[:-1] + binwidth / 2
    x_err = binwidth / 2
    
    # Create the plots
    fig, ((ax2, ax3), (ax5,ax6), (ax8,ax9)) = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    ax2.errorbar(x_pos, yeff_photon, xerr=x_err, yerr=yerr_eff_photon, fmt='o', label='Efficiency')
    ax2.set_ylabel('efficiency gamma', fontsize=10)
    ax3.errorbar(x_pos, yeff_nh, xerr=x_err, yerr=yerr_eff_nh, fmt='o', label='Efficiency')
    ax3.set_ylabel('efficiency n.H.', fontsize=10)
    ytickseff = np.round(np.arange(40, 101, 20), 1)
    ax2.set_yticks(ytickseff)
    ax2.set_yticklabels([f"{y}%" for y in ytickseff], fontsize=10)
    ax3.set_yticks(ytickseff)
    ax3.set_yticklabels([f"{y}%" for y in ytickseff], fontsize=10)
    
    ax5.errorbar(x_pos, yfake_photon, xerr=x_err, yerr=yerr_fake_photon, fmt='o', label='Fake rate')
    ax5.set_ylabel('fake rate gamma', fontsize=10)
    ax6.errorbar(x_pos, yfake_nh, xerr=x_err, yerr=yerr_fake_nh, fmt='o', label='Fake rate')
    ax6.set_ylabel('fake rate n.H.', fontsize=10)
    yticksfake1 = np.round(np.arange(0, 50, 20), 1)
    ax5.set_yticks(yticksfake1)
    ax5.set_yticklabels([f"{y}%" for y in yticksfake1], fontsize=10)
    yticksfake2 = np.round(np.arange(0, 70, 20), 1)
    ax6.set_yticks(yticksfake2)
    ax6.set_yticklabels([f"{y}%" for y in yticksfake2], fontsize=10)
    
    ax8.errorbar(x_pos, ycorr_photon, xerr=x_err, yerr=yerr_corr_photon, fmt='o', label='probability of correct class')
    ax8.set_ylabel('p corr gamma', fontsize=10)
    ax9.errorbar(x_pos, ycorr_nh, xerr=x_err, yerr=yerr_corr_nh, fmt='o', label='probability of correct class')
    ax9.set_ylabel('p corr n.H.', fontsize=10)
    yticksp = np.round(np.arange(0, 101, 20), 1)
    ax8.set_yticks(yticksp)
    ax8.set_yticklabels([f"{y}%" for y in yticksp], fontsize=10)
    ax9.set_yticks(yticksp)
    ax9.set_yticklabels([f"{y}%" for y in yticksp], fontsize=10)

    for ax in [ax2,ax3, ax5, ax6, ax8, ax9]:    
        ax.set_xticks(bins)
        ax.set_xticklabels(bins, fontsize=10)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=0.4)
        ax.set_xlabel('Energy [GeV]', fontsize=10)
        #ax.set_ylim(20, 101)

    return fig
def plot_efficency_and_fakerate(df, bins, datasetname='Quark Jet'):       
    # Calculate the efficiencies and fake rates for the total
    efficiencies, efficiencies_err, fake_rate, fake_rate_err = [], [], [], []
    
    mask_predicted = np.isnan(df['pred_energy']) == False
    mask_truth = np.isnan(df['truthHitAssignedEnergies']) == False
        
    
    for i in range(len(bins) - 1):
        mask_bintruth = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_binpred = np.logical_and(
            df['pred_energy'] >= bins[i],
            df['pred_energy'] < bins[i + 1])

        matched = np.logical_and(
            mask_predicted,
            mask_bintruth)
        faked = np.logical_and(
            mask_binpred,
            np.logical_not(mask_truth))
        
        eff = np.sum(matched) / np.sum(mask_bintruth)
        eff_err = np.sqrt(eff * (1 - eff) / np.sum(matched)) 
        efficiencies.append(eff)
        efficiencies_err.append(eff_err)
        
        fake = np.sum(faked) / np.sum(mask_binpred)
        fake_err = np.sqrt(fake * (1 - fake) / np.sum(faked))
        fake_rate.append(fake)
        fake_rate_err.append(fake_err)

    yeff, yerr_eff, yfake, yerr_fake, = np.array(efficiencies), np.array(efficiencies_err), np.array(fake_rate), np.array(fake_rate_err)
    
    # Calculate the bin positions and widths
    binwidth = bins[1:] - bins[:-1]
    x_pos = bins[:-1] + binwidth / 2
    x_err = binwidth / 2
    
    fig_eff = plt.figure()
    plt.errorbar(x_pos, yeff, xerr=x_err, yerr=yerr_eff, fmt='o')    
    plt.xlim(bins[0], bins[-1])
    plt.xticks(np.array([1,5,10,20,30,50]))
    plt.xlabel('Truth Energy [GeV]')
    plt.ylabel('Efficiency')    
    plt.grid(alpha=0.4)
    plt.text(0.05, 0.95, datasetname, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10)   
    plt.close()
    
    fig_fake = plt.figure()
    plt.errorbar(x_pos, yfake, xerr=x_err, yerr=yerr_fake, fmt='o')
    plt.xlim(bins[0], bins[-1])
    plt.xticks(np.array([1,5,10,20,30,50]))
    plt.xlabel('Predicted Energy [GeV]')
    plt.ylim(bottom=0, top=max(0.1, np.max(yfake) + 0.01))
    plt.ylabel('Fake rate')    
    plt.grid(alpha=0.4)
    plt.text(0.05, 0.95, datasetname, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10) 
    plt.close()
    
    return fig_eff, fig_fake

def calc_classification_p(df, bins, mask):
    """
    Helper function to calculate the classification accuracy for the photons, neutral hadrons and charged particles
    """
    corr_class_prob = []
    corr_class_prob_err = []    
    
    mask_predicted = np.isnan(df['pred_energy']) == False
    mask_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    
    if(mask == 0):
        mask_PID_truth = df['truthHitAssignedPIDs'].isin([22])
    elif(mask == 1):
        mask_PID_truth = df['truthHitAssignedPIDs'].isin([130,310,311,2112,-2112,3122,-3122,3322,-3322])
    elif(mask == 2):
        mask_PID_truth = df['truthHitAssignedPIDs'].isin([11,-11,13,-13,211,-211,321,-321,2212,-2212,3112,-3112,3222,-3222,3312,-3312])
    else:
        raise ValueError("mask must be 0, 1 or 2")
    mask_PID_pred = df['pred_id_value'].isin([mask])        
    mask_PID_matched = np.logical_and(mask_PID_truth, mask_PID_pred)        
    
    for i in range(len(bins) - 1):
        mask_bintruth = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_binpred = np.logical_and(
            df['pred_energy'] >= bins[i],
            df['pred_energy'] < bins[i + 1])

        matched = np.logical_and(
            mask_predicted,
            mask_bintruth)    
        
        cc_prob = np.sum(matched[mask_PID_matched]) / np.sum(matched[mask_PID_truth])
        cc_prob_err = np.sqrt(cc_prob * (1 - cc_prob) / np.sum(matched[mask_PID_truth]))
        corr_class_prob.append(cc_prob)
        corr_class_prob_err.append(cc_prob_err)
    
    return np.array(corr_class_prob), np.array(corr_class_prob_err)

def plot_classification_p(df, bins, datasetname='Quark Jet'):
    ycorr_photon, yerr_corr_photon = calc_classification_p(df, bins, 0)    
    ycorr_nh, yerr_corr_nh = calc_classification_p(df, bins, 1)    
    ycorr_ch, yerr_corr_ch = calc_classification_p(df, bins, 2)
    
    # Calculate the bin positions and widths
    binwidth = bins[1:] - bins[:-1]
    x_pos = bins[:-1] + binwidth / 2
    x_err = binwidth / 2
    
    # Create the plots
    fig_classification = plt.figure()
    plt.errorbar(x_pos, ycorr_photon, xerr=x_err, yerr=yerr_corr_photon, fmt='o', label='Photon')
    plt.errorbar(x_pos, ycorr_nh, xerr=x_err, yerr=yerr_corr_nh, fmt='o', label='Neutral Hadron')
    plt.errorbar(x_pos, ycorr_ch, xerr=x_err, yerr=yerr_corr_ch, fmt='o', label='Charged Particle')
    
    plt.legend()
    plt.grid(alpha=0.4)  
    plt.xlim(bins[0], bins[-1])
    plt.xticks([1,5,10,20,30,50])
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Accuracy of classification')
    plt.ylim(0, 1)
    plt.text(0.05, 0.95, datasetname, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10) 
    plt.close()
    
    return fig_classification
    
def plot_jet_metrics_based_on_particles(df, datasetname='QuarkJet'):
    """
    plots the jet metrics under the assumption, that one event is one jet
    """
    
    reseta = []
    resphi = []
    relresE = []
    nParicles_pred = []
    nParicles_truth = []
    true_pt = []
    pred_pt = []
    
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(df['pred_energy']) == False

    for i in range(len(df['event_id'].unique())):
        event_mask = df['event_id'] == i
        truth_mask = np.logical_and(event_mask, has_truth)
        pred_mask = np.logical_and(event_mask, has_pred)
        
        jet_E_truth = np.sum(df[truth_mask]['truthHitAssignedEnergies'])
        jet_E_pred = np.sum(df[pred_mask]['pred_energy'])
        relresE.append((jet_E_truth-jet_E_pred)/jet_E_truth)
        
        nParicles_pred.append(np.sum(pred_mask))
        nParicles_truth.append(np.sum(truth_mask))
        
        t_eta = np.average(df[truth_mask]['truthHitAssignedZ'], weights=df[truth_mask]['truthHitAssignedEnergies'])
        pred_eta = np.average(df[pred_mask]['pred_pos'].apply(lambda x: x[2]), weights=df[pred_mask]['pred_energy'])
        reseta.append(t_eta - pred_eta)
        
        t_phi = np.average(np.arctan2(df[truth_mask]['truthHitAssignedY'], df[truth_mask]['truthHitAssignedX']), weights=df[truth_mask]['truthHitAssignedEnergies'])
        pred_phi = np.average(df[pred_mask]['pred_pos'].apply(lambda x: np.arctan2(x[1], x[0])), weights=df[pred_mask]['pred_energy'])
        delta_phi = calc_deltaphi(t_phi, pred_phi)        
        resphi.append(delta_phi)
        
        true_pt.append(np.sum(df[truth_mask]['truthHitAssignedEnergies'] / np.cosh(df[truth_mask]['truthHitAssignedZ'])))
        pred_pt.append(np.sum(df[pred_mask]['pred_energy'] / np.cosh(df[pred_mask]['pred_pos'].apply(lambda x: x[2]))))
        
    #Correct jet pt    
    jets_df = pd.DataFrame()
    jets_df['true_pt'] = true_pt
    jets_df['pred_pt'] = pred_pt
    jets_df['matched'] = 1
    jets_df = correct_jet_pt(jets_df, [0,5,10,20,30,50,100, 1000])
    relrespt = np.array((jets_df['true_pt'] - jets_df['pred_pt'])/ jets_df['true_pt'])
    
    #Start plotting
    nParticle_bins = np.linspace(0, 30, 31)
    fig_jet_nparticle = plt.figure()
    plt.hist(nParicles_truth, bins=nParticle_bins, label='Truth', color='orange', alpha=0.5)#color='#fee7ae')
    plt.grid(alpha=0.4)
    plt.hist(nParicles_pred, bins=nParticle_bins, label='Prediction', histtype='step')#color='#67c4ce', linestyle='--')
    plt.text(0.05, 0.95, datasetname, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel('Number of events', fontsize=20)
    plt.xlabel('Jet nConstituents', fontsize=20)
    plt.legend(fontsize=10)
    plt.xticks(np.linspace(0, 30, 7))
    plt.close()

    relres_bins = np.linspace(-1, 1, 51)
    fig_jet_relresE = plt.figure()
    plt.hist(np.clip(relresE, -1, 1), bins=relres_bins, histtype='step')#color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(relresE, relres_bins, datasetname)
    plt.ylabel('Number of events', fontsize=20)
    plt.xlabel('rel. res. Jet Energy', fontsize=20)
    plt.xticks(np.linspace(-1, 1, 5))
    plt.close()    
    
    res_bins = np.linspace(-0.2, 0.2, 51)
    fig_jet_reseta = plt.figure()
    plt.hist(np.clip(reseta,-0.2,0.2), bins=res_bins, histtype='step')# color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(reseta, res_bins, datasetname)
    plt.ylabel('Number of events', fontsize=20)
    plt.xlabel('Jet $\\Delta \\eta$', fontsize=20)
    plt.xticks(np.linspace(-0.2, 0.2, 5))
    plt.close()

    fig_jet_resphi = plt.figure()
    plt.hist(np.clip(resphi, -0.2, 0.2), bins=res_bins, histtype='step')#color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(resphi, res_bins, datasetname)
    plt.ylabel('Number of events', fontsize=20)
    plt.xlabel('Jet $\\Delta \\phi$', fontsize=20)
    plt.xticks(np.linspace(-0.2, 0.2, 5))
    plt.close()
    
    fig_jet_pt = plt.figure()
    plt.hist(np.clip(relrespt, -1, 1), bins=relres_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(relrespt, relres_bins, datasetname)
    plt.ylabel('Number of events', fontsize=20)
    plt.xlabel('rel. res. Jet $p_T$', fontsize=20)
    plt.xticks(np.linspace(-1, 1, 5))
    plt.close()
    
    return fig_jet_nparticle, fig_jet_relresE, fig_jet_reseta, fig_jet_resphi, fig_jet_pt

def correct_jet_pt(jets_df, bins, plotEachBin=False, outputDir='/work/friemer/hgcalml/testplots/'):
    """
    Function to calculate a correction factor for the jet pt in bins of pt, fit a correction function and apply it to the jets_df
    """
    
    corr_factors = []
    for i in range(len(bins) - 1):
        mask_binpred = np.logical_and(
            jets_df['pred_pt'] >= bins[i],
            jets_df['pred_pt'] < bins[i + 1])
        mask = np.logical_and(
            jets_df['matched'],
            mask_binpred)
        jet_pt_truth = jets_df[mask]['true_pt']
        jet_pt_pred = jets_df[mask]['pred_pt']
        respt = np.array((jet_pt_truth-jet_pt_pred))
        relrespt = np.array((jet_pt_truth-jet_pt_pred)/jet_pt_truth)
        
        if plotEachBin:
            relres_bins = np.linspace(-1, 1, 51)        
            fig_jet_pt = plt.figure()
            plt.hist(np.clip(relrespt, -1,1), bins=relres_bins, histtype='step')
            plt.grid(alpha=0.4)
            plt.text(0.05, 0.95, 'Quark Jet', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10)
            write_mean_std_displayed(relrespt, relres_bins)
            plt.ylabel('Number of jets', fontsize=20)
            plt.xlabel('rel. res. Cal. Jet $p_T$', fontsize=20)
            plt.xticks(np.round(np.linspace(-1, 1, 9), 2), fontsize=12)
            plt.savefig(os.path.join(outputDir, f'/jet_pt_{bins[i]}.png'))
            plt.close()

        corr_factor = np.mean(relrespt)
        corr_factors.append(corr_factor)
        
    #fit function to correction factor over bin middle    
    corr_factors = np.array(corr_factors)
    
    bins = np.array(bins)
    bin_middle = bins[:-1] + (bins[1:] - bins[:-1])/2
    
    mask = np.logical_and(~np.isnan(corr_factors), ~np.isinf(corr_factors)) #Should not be necessary, unless evaluation is performed on very small dataset
    
    def linearfunc(x, a, b):
        return a*x+b
    
    popt, pcov = curve_fit(linearfunc, bin_middle[mask], corr_factors[mask])
    
    corrected_jets_df = jets_df
    corrected_jets_df['pred_pt'] = corrected_jets_df['pred_pt']*(1 + linearfunc(corrected_jets_df['pred_pt'], *popt))
    return corrected_jets_df
    

def plot_jet_metrics_from_jets(jets_df, datasetname='Quark Jet'):
    """
    plots the jet metrics from the results of jet clustering algorithm
    """
    reseta = []
    resphi = []
    relrespt = []
    nParicles_pred = []
    nParicles_truth = []
    
    jets_df = correct_jet_pt(jets_df, [0,5,10,20,30,50,100, 1000])
    
    for i in range(len(jets_df['event_id'].unique())):
        matched = jets_df['matched']
        event_id_mask = jets_df['event_id'] == i
        mask = np.logical_and(event_id_mask,  matched)
        
        nParicles_truth.append(jets_df[mask]['true_n_constituents'])
        nParicles_pred.append(jets_df[mask]['pred_n_constituents'])
        
        t_eta = jets_df[mask]['true_eta']
        pred_eta = jets_df[mask]['pred_eta']
        reseta.append(t_eta-pred_eta)
        
        t_phi = jets_df[mask]['true_phi']
        pred_phi = jets_df[mask]['pred_phi']
        delta_phi = calc_deltaphi(t_phi, pred_phi)        
        resphi.append(delta_phi)
        
        jet_pt_truth = jets_df[mask]['true_pt']
        jet_pt_pred = jets_df[mask]['pred_pt']
        relrespt.append((jet_pt_truth-jet_pt_pred)/jet_pt_truth)

    print('Total unmatched jets:', np.sum([jets_df['matched'] == False]))
    
    nParicles_truth = np.concatenate(nParicles_truth)
    nParicles_pred = np.concatenate(nParicles_pred)
    reseta = np.concatenate(reseta)
    resphi = np.concatenate(resphi)
    relrespt = np.concatenate(relrespt)

    nParticle_bins = np.linspace(0, 30, 31)
    fig_jet_nparticle = plt.figure()
    plt.hist(nParicles_truth, bins=nParticle_bins, label='Truth', color='orange', alpha=0.5)
    plt.grid(alpha=0.4)
    plt.hist(nParicles_pred, bins=nParticle_bins, label='Prediction', histtype='step')
    plt.text(0.05, 0.95, datasetname, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel('Number of jets', fontsize=20)
    plt.xlabel('Jet nConstituents', fontsize=20)
    plt.legend(fontsize=10)
    plt.xticks(np.linspace(0, 30, 7))
    plt.close()
    
    res_bins = np.linspace(-0.2, 0.2, 51)
    fig_jet_reseta = plt.figure()
    plt.hist(np.clip(reseta,-0.2,0.2), bins=res_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(reseta, res_bins, datasetname)
    plt.ylabel('Number of jets', fontsize=20)
    plt.xlabel('Jet $\\Delta \\eta$', fontsize=20)
    plt.xticks(np.linspace(-0.2, 0.2, 5))
    plt.close()

    fig_jet_resphi = plt.figure()
    plt.hist(np.clip(resphi, -0.2, 0.2), bins=res_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(resphi, res_bins, datasetname)
    plt.ylabel('Number of jets', fontsize=20)
    plt.xlabel('Jet $\\Delta \\phi$', fontsize=20)
    plt.xticks(np.linspace(-0.2, 0.2, 5))
    plt.close()
    
    relres_bins = np.linspace(-1, 1, 51)
    fig_jet_pt = plt.figure()
    plt.hist(np.clip(relrespt, -1,1), bins=relres_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(relrespt, relres_bins, datasetname)
    plt.ylabel('Number of jets', fontsize=20)
    plt.xlabel('rel. res. Cal. Jet $p_T$', fontsize=20)
    plt.xticks(np.round(np.linspace(-1, 1, 9), 2), fontsize=12)
    plt.close()
    
    return fig_jet_nparticle, fig_jet_reseta, fig_jet_resphi, fig_jet_pt


def write_mean_std_displayed(data, bins, datasetname='Quark Jet'):
    """
    Helper function to write the mean, std and displayed fraction of the data into the plot
    """
    #remove inf and Nan
    data = np.array(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    data = data[np.isnan(data) == False]
    
    displayed = np.sum(np.logical_and(data >= bins[0], data < bins[-1]))/len(data)
    mean= np.mean(data)
    std = np.std(data)
    
    #Fit a normal distribution to the data
    data = data[np.logical_and(data >= bins[0], data < bins[-1])]
    mu, sigma = norm.fit(data)    
    
    textstr = '\n'.join((
    # r'$\mathrm{Mean}=%.2f$' % (mean, ),
    # r'$\mathrm{Std\ }=%.2f$' % (std, ),
    r'$\mu=%.2f$' % (mu, ),
    r'$\sigma=%.2f$' % (sigma, ),
    r'$\mathrm{Displayed}=%.2f$' % (displayed, )))

    # Add the text box
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
    plt.text(0.05, 0.95, datasetname, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10)
    

def calc_matched_mask(df):
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(df['pred_energy']) == False
    matched = np.logical_and(has_truth, has_pred)
    return matched

def calc_deltaphi(t_phi, pred_phi):
    delta_phi = t_phi-pred_phi
    delta_phi = np.where(delta_phi > np.pi, delta_phi-2*np.pi, delta_phi)
    delta_phi = np.where(delta_phi < -np.pi, delta_phi+2*np.pi, delta_phi)
    return delta_phi

def plot_particle_metrics(df, datasetname='Quark Jet'):
        
    matched = calc_matched_mask(df)
    
    reseta = []
    resphi = []
    relresE = []
    relrespt = []

    for i in range(len(df['event_id'].unique())):
        mask = np.logical_and(df['event_id'] == i, matched)
        
        truth_e = df[mask]['truthHitAssignedEnergies']
        pred_e = df[mask]['pred_energy']
        newrelresE = ((truth_e-pred_e)/truth_e).to_numpy()
        relresE = np.concatenate((relresE, newrelresE))
        
        t_eta = df[mask]['truthHitAssignedZ']
        pred_eta = df[mask]['pred_pos'].apply(lambda x: x[2])
        delta_eta = (t_eta-pred_eta).to_numpy()
        reseta = np.concatenate((reseta, delta_eta))
        
        t_phi = np.arctan2(df[mask]['truthHitAssignedY'], df[mask]['truthHitAssignedX'])
        pred_phi = df[mask]['pred_pos'].apply(lambda x: np.arctan2(x[1], x[0]))
        delta_phi = calc_deltaphi(t_phi, pred_phi)
        resphi = np.concatenate((resphi, delta_phi))
        
        truth_pt = truth_e/np.cosh(t_eta)
        pred_pt = pred_e/np.cosh(pred_eta)
        newrelrespt = ((truth_pt-pred_pt)/truth_pt).to_numpy()
        relrespt = np.concatenate((relrespt, newrelrespt))

    relres_bins = np.linspace(-1, 1, 51)
    fig_relresE = plt.figure()
    plt.hist(np.clip(relresE, -1,1), bins=relres_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(relresE, relres_bins, datasetname)
    plt.xlabel('rel. res. Neutral Particle Energy', fontsize=20)
    plt.ylabel('Number of neutral particles', fontsize=20)
    plt.xticks(np.linspace(-1, 1, 5))
    plt.close()

    res_bins = np.linspace(-0.4, 0.4, 51)
    fig_reseta = plt.figure()
    plt.hist(np.clip(reseta, -0.4, 0.4), bins=res_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(reseta, res_bins, datasetname)
    plt.xlabel('Neutral Particle $\\Delta \\eta$', fontsize=20)
    plt.ylabel('Number of neutral particles', fontsize=20)
    plt.xticks(np.linspace(-0.4, 0.4, 5))
    plt.close()

    fig_resphi = plt.figure()
    plt.hist(np.clip(resphi, -0.4, 0.4), bins=res_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(resphi, res_bins, datasetname)
    plt.xlabel('Neutral Particle $\\Delta \\phi$', fontsize=20)
    plt.ylabel('Number of neutral particles', fontsize=20)
    plt.xticks(np.linspace(-0.4, 0.4, 5))
    plt.close()
    
    fig_pt = plt.figure()
    plt.hist(np.clip(relrespt, -1, 1), bins=relres_bins, histtype='step')
    plt.grid(alpha=0.4)
    write_mean_std_displayed(relrespt, relres_bins, datasetname)
    plt.xlabel('rel. res. Neutral Particle $p_T$', fontsize=20)
    plt.ylabel('Number of neutral particles', fontsize=20)
    plt.xticks(np.linspace(-1, 1, 5))
    plt.close()

    return fig_relresE, fig_reseta, fig_resphi, fig_pt

def calc_energy_resolution(df, bins=None):
    matched = calc_matched_mask(df)

    ratios = [] #Relative deviation of predicted energy from truth energy
    ratios_err= []
    diffs = [] #Absolute deviation of predicted energy from truth energy
    diffs_err = []

    for i in range(len(bins) - 1):
        mask_truth_bin = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_bin = np.logical_and(mask_truth_bin, matched)
        diff = np.abs(df['pred_energy'][mask_bin] - df['truthHitAssignedEnergies'][mask_bin])
        diffs.append(np.mean(diff))
        diffs_err.append(np.std(diff) / np.sqrt(len(diff)))
        
        ratio = df['pred_energy'][mask_bin] / df['truthHitAssignedEnergies'][mask_bin]
        ratios.append(np.mean(ratio))
        ratios_err.append(np.std(ratio) / np.sqrt(len(ratio)))
        
    diffs, diffs_err, ratios, ratios_err = np.array(diffs), np.array(diffs_err), np.array(ratios), np.array(ratios_err)
    return diffs, diffs_err, ratios, ratios_err
def plot_energy_resolution(df, bins=None, datasetname='Quark Jet'):
    diffs, diffs_err, ratios, ratios_err = calc_energy_resolution(df, bins)
    
    neutral_mask = df['truthHitAssignedPIDs'].isin([22,130,310,311,2112,-2112,3122,-3122,3322,-3322])
    diffs_neutral, diffs_err_neutral, ratios_neutral, ratios_err_neutral = calc_energy_resolution(df[neutral_mask], bins)
    
    charged_mask = df['truthHitAssignedPIDs'].isin([11,-11,13,-13,211,-211,321,-321,2212,-2212,3112,-3112,3222,-3222,3312,-3312])
    diffs_charged, diffs_err_charged, ratios_charged, ratios_err_charged = calc_energy_resolution(df[charged_mask], bins)
    
    # Calculate the bin positions and widths
    binwidth = bins[1:] - bins[:-1]
    x_pos = bins[:-1] + binwidth / 2
    x_err = binwidth / 2
    # Create the stacked plots of diff and ratio with one errorbar for total, neutral and charged each
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.errorbar(x_pos, diffs, xerr=x_err, yerr=diffs_err, fmt='o', label='Total')
    ax1.errorbar(x_pos, diffs_neutral, xerr=x_err, yerr=diffs_err_neutral, fmt='o', label='Neutral')
    ax1.errorbar(x_pos, diffs_charged, xerr=x_err, yerr=diffs_err_charged, fmt='o', label='Charged')
    ax1.set_ylabel('$\Delta p \ [GeV]$')
    ax1.grid(alpha=0.4)    
    #write datasetname in top right corner
    ax1.text(0.05, 0.65, datasetname, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, fontsize=10)
    ax1.legend()

    ax2.errorbar(x_pos, ratios, xerr=x_err, yerr=ratios_err, fmt='o', label='Total')
    ax2.errorbar(x_pos, ratios_neutral, xerr=x_err, yerr=ratios_err_neutral, fmt='o', label='Neutral')
    ax2.errorbar(x_pos, ratios_charged, xerr=x_err, yerr=ratios_err_charged, fmt='o', label='Charged')
    ax2.set_ylabel('$p_{pred} / p_{truth}$')
    ax2.grid(alpha=0.4)
    
    ydelta = np.round(max(1 - ax2.get_ylim()[0], ax2.get_ylim()[1] - 1),1)
    ax2.set_ylim(1 - ydelta, 1 + ydelta)

    ax2.set_xlim(bins[0], bins[-1])
    ax2.set_xscale('log')
    ax2.set_xlabel('Truth Energy [GeV]')
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    xticks = np.round(bins, 0).astype(int)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks)

    fig.subplots_adjust(hspace=0.1)
    
    return fig

def plot_condensation(pred_dict, t_dict, feature_dict, coordspace='condensation'):
    """
    Plots the condensation space of a event (coordspace='condensation") or the input space (coordspace='input')
    """
    
    if(coordspace == 'condensation'):
        coords = pred_dict['pred_ccoords']

        if coords.shape[1] < 3: #add a zero
            coords = tf.concat([coords, tf.zeros_like(coords)[:,0:1]], axis=-1)
        X = coords[:,0]
        Y = coords[:,1]
        Z = coords[:,2]
        sizefield = 'beta'
    elif(coordspace == 'input'):
        X = feature_dict['recHitX'][:,0]
        Y = feature_dict['recHitY'][:,0]
        Z = feature_dict['recHitZ'][:,0]
        sizefield = 'rechit_energy_scaled'
    else:
        raise ValueError('coordspace must be "condensation" or "input"')

    data={
        'X': X,
        'Y': Y,
        'Z': Z,
        't_idx': t_dict['truthHitAssignementIdx'][:,0],
        'beta': pred_dict['pred_beta'][:,0],
        'pdgid': t_dict['truthHitAssignedPIDs'][:,0],
        'recHitID': feature_dict['recHitID'][:,0],
        'rechit_energy': feature_dict['recHitEnergy'][:,0],        
        'rechit_energy_scaled': np.log(feature_dict['recHitEnergy'][:,0]+1),
        }
    eventdf = pd.DataFrame(data)

    hover_data = {'X': True, 'Y': True, 'Z': True, 't_idx': True,'beta': True, 'pdgid': True,'recHitID': True, 'rechit_energy': True}

    fig = px.scatter_3d(eventdf, x="X", y="Y", z="Z", 
                        color="t_idx",
                        size=sizefield,
                        template='plotly_white',
                        hover_data=hover_data,
            color_continuous_scale=px.colors.sequential.Rainbow)
    fig.update_traces(marker=dict(line=dict(width=0)))
    
    return fig

def plot_distribution(truth_list, feature_list):
    """
    Makes boxplots of the number of particles, tracks, hits, etc. in the events
    """
    n_tracks = []
    n_hits = []
    n_particles = []
    n_charged_had = []
    n_lepton = []
    n_photon = []
    n_neutral_had = []
    for i in range(len(feature_list)):
        n_tracks.append(np.sum(np.abs(feature_list[i]['recHitID'])))
        n_hits.append(len(feature_list[i]['recHitID'])-n_tracks[-1])
        truth_data ={
            'truthHitAssignementIdx': truth_list[i]['truthHitAssignementIdx'][:,0],
            'truthHitAssignedPIDs': truth_list[i]['truthHitAssignedPIDs'][:,0],            
        }
        truth_df = pd.DataFrame(truth_data)
        n_particles.append(len(truth_df['truthHitAssignementIdx'].unique()))
        
        charged_hd_mask = truth_df['truthHitAssignedPIDs'].isin([211,-211,321,-321,2212,-2212,3112,-3112,3222,-3222,3312,-3312])
        lepton_mask = truth_df['truthHitAssignedPIDs'].isin([11,-11,13,-13])
        photon_mask = truth_df['truthHitAssignedPIDs'].isin([22])
        neutral_mask = truth_df['truthHitAssignedPIDs'].isin([130,310,311,2112,-2112,3122,-3122,3322,-3322])
        n_charged_had.append(len(truth_df[charged_hd_mask]['truthHitAssignementIdx'].unique()))
        n_lepton.append(len(truth_df[lepton_mask]['truthHitAssignementIdx'].unique()))
        n_photon.append(len(truth_df[photon_mask]['truthHitAssignementIdx'].unique()))
        n_neutral_had.append(len(truth_df[neutral_mask]['truthHitAssignementIdx'].unique()))
    #convert to numpy
    n_tracks, n_hits, n_particles, n_charged_had, n_lepton, n_photon, n_neutral_had = \
        np.array(n_tracks), np.array(n_hits), np.array(n_particles), np.array(n_charged_had), np.array(n_lepton), np.array(n_photon), np.array(n_neutral_had)
    
    #create horizontal boxplot
    fig = plt.figure()
    values = [n_lepton, n_charged_had, n_neutral_had, n_photon, n_particles, n_tracks,n_hits/100]
    labels = ['leptons', 'ch. hadrons','nu. hadrons','photons','total particles', 'tracks','cells [$10^2$]']

    plt.boxplot(values, labels=labels, vert=False, patch_artist=True, 
                boxprops=dict(facecolor="limegreen"), medianprops=dict(color="black"),
                whis =  (0, 100))
    
    plt.grid(alpha=0.4, axis='x')
    # Add mean values as text
    for pos, data in zip(np.arange(len(values)), values):
        mean = np.mean(data)
        plt.text(21.2, pos+1, f'({mean:.1f})', va='center', ha='left')
    
    plt.close()
    return fig
def plot_truth_jet_nconstrituents(jets_df, df, datasetname='Quark Jet'):
    nParicles_truth_jet = []
    
    jets_df = correct_jet_pt(jets_df, [0,5,10,20,30,50,100, 1000])
    matched = jets_df['matched']
    for i in range(len(jets_df['event_id'].unique())):        
        event_id_mask = jets_df['event_id'] == i
        mask = np.logical_and(event_id_mask,  matched)
        
        nParicles_truth_jet.append(jets_df[mask]['true_n_constituents'])    
    nParicles_truth_jet = np.concatenate(nParicles_truth_jet)
    
    nParicles_truth_particle = []    
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False

    for i in range(len(df['event_id'].unique())):
        event_mask = df['event_id'] == i
        truth_mask = np.logical_and(event_mask, has_truth)
        nParicles_truth_particle.append(np.sum(truth_mask))
    nParicles_truth_particle = np.array(nParicles_truth_particle)

    nParticle_bins = np.linspace(0, 30, 31)
    fig_nparticle_truth_compare = plt.figure()
    plt.hist(nParicles_truth_jet, bins=nParticle_bins, label='Anti-$k_T$ based', color='blue', alpha=0.5)
    plt.hist(nParicles_truth_particle, bins=nParticle_bins, label='Event based', color='orange', alpha=0.5)
    plt.grid(alpha=0.4)
    plt.text(0.05, 0.95, datasetname, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel('Number of jets', fontsize=20)
    plt.xlabel('Jet nConstituents', fontsize=20)
    plt.legend(fontsize=12)
    plt.xticks(np.linspace(0, 30, 7))
    plt.close()
    return fig_nparticle_truth_compare

def plot_energydistribution(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['truthHitAssignedEnergies'], histtype='step', color='blue', label='Truth')
    ax.hist(df['pred_energy'], histtype='step', label='Prediction')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy [GeV]')
    ax.set_ylabel('Number of showers')
    ax.legend()
    ax.grid(alpha=0.4)
    return fig
    

def plot_everything(df, pred_list, t_list, feature_list, jets_df, outputpath='/work/friemer/hgcalml/testplots/', datasetname='Quark Jet'):
    
    #Create Output directory  
    if not os.path.exists(outputpath):
        print("Creating output directory", outputpath)
        os.mkdir(outputpath)
    else:
        var = input(\
            'Working directory exists, are you sure you want to continue, please type "yes/y"\n')
        var = var.lower()
        if not var in ('yes', 'y'):
            sys.exit()
    os.makedirs(os.path.join(outputpath, 'condensation'), exist_ok=True)   
      
    matplotlib.rcParams.update({'font.size': 20})
    
    print('Plotting efficiencies')
    energy_bins_neutral = np.array([1,2,3,4,5,10,20,30,50])
    fig_eff_overview=plot_efficencies(df, bins=energy_bins_neutral)
    fig_eff_overview.savefig(os.path.join(outputpath,'efficienciesOverview_{}.png'.format(datasetname)), bbox_inches='tight')
    
    fig_eff, fig_fake = plot_efficency_and_fakerate(df, bins=energy_bins_neutral, datasetname=datasetname)
    fig_eff.savefig(os.path.join(outputpath,'efficiencies_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_fake.savefig(os.path.join(outputpath,'fake_rate_{}.png'.format(datasetname)), bbox_inches='tight')
    
    print('Plotting jet metrics')
    fig_jet_nparticle_all, fig_jet_relresE_all, fig_jet_reseta_all, fig_jet_resphi_all, fig_jet_pt_all = plot_jet_metrics_based_on_particles(df, datasetname=datasetname)
    fig_jet_nparticle_all.savefig(os.path.join(outputpath,'jet_nparticle_all_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_jet_relresE_all.savefig(os.path.join(outputpath,'jet_relresE_all_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_jet_reseta_all.savefig(os.path.join(outputpath,'jet_reseta_all_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_jet_resphi_all.savefig(os.path.join(outputpath,'jet_resphi_all_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_jet_pt_all.savefig(os.path.join(outputpath,'jet_pt_all_{}.png'.format(datasetname)), bbox_inches='tight')
    
    fig_jet_nparticle_clustered,  fig_jet_reseta_clustered, fig_jet_resphi_clustered, fig_jet_pt_clustered = plot_jet_metrics_from_jets(jets_df, datasetname=datasetname)
    fig_jet_nparticle_clustered.savefig(os.path.join(outputpath,'jet_nparticle_clustered_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_jet_reseta_clustered.savefig(os.path.join(outputpath,'jet_reseta_clustered_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_jet_resphi_clustered.savefig(os.path.join(outputpath,'jet_resphi_clustered_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_jet_pt_clustered.savefig(os.path.join(outputpath,'jet_pt_clustered_{}.png'.format(datasetname)), bbox_inches='tight')
        
    print('Plotting particle metrics')
    mask_neutral = df['truthHitAssignedPIDs'].isin([22,130, 310, 311, 2112, -2112, 3122, -3122, 3322, -3322])    
    fig_neutral_relresE, fig_neutral_reseta, fig_neutral_resphi, fig_neutral_pt = plot_particle_metrics(df[mask_neutral], datasetname=datasetname)
    fig_neutral_relresE.savefig(os.path.join(outputpath,'neutral_relresE_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_neutral_reseta.savefig(os.path.join(outputpath,'neutral_reseta_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_neutral_resphi.savefig(os.path.join(outputpath,'neutral_resphi_{}.png'.format(datasetname)), bbox_inches='tight')
    fig_neutral_pt.savefig(os.path.join(outputpath,'neutral_pt_{}.png'.format(datasetname)), bbox_inches='tight')
    
    fig_class = plot_classification_p(df, energy_bins_neutral, datasetname=datasetname)
    fig_class.savefig(os.path.join(outputpath,'classification_p_{}.png'.format(datasetname)), bbox_inches='tight')
    
    print('Plotting energy resolution')    
    energy_bins_combined = np.array([1,5,10,20,30,50,200])
    fig_new_res = plot_energy_resolution(df, bins=energy_bins_combined, datasetname=datasetname)
    fig_new_res.savefig(os.path.join(outputpath,'energy_resolution_{}.png'.format(datasetname)), bbox_inches='tight')

    print('Plotting condensation and input') 
    for event_id in range(10):
        fig = plot_condensation(pred_list[event_id], t_list[event_id], feature_list[event_id], 'condensation')
        fig.write_html(os.path.join(outputpath, 'condensation' ,'condensation{}_{}.html'.format(event_id, datasetname)))
        fig = plot_condensation(pred_list[event_id], t_list[event_id], feature_list[event_id], 'input')
        fig.write_html(os.path.join(outputpath, 'condensation' ,'input{}_{}.html'.format(event_id, datasetname)))
        
    # print('Plotting truth compare')
    # fig_nparticle_truth_compare = plot_truth_jet_nconstrituents(jets_df, df, datasetname=datasetname)
    # fig_nparticle_truth_compare.savefig(os.path.join(outputpath,'nparticle_truth_compare_{}.png'.format(datasetname)), bbox_inches='tight')
    
    # print('Plotting distribution')
    # fig_dist = plot_distribution(t_list, feature_list)
    # fig_dist.savefig(os.path.join(outputpath, 'distribution_{}.png'.format(datasetname)), bbox_inches='tight')   
    
def plot_everything_from_file(analysisfilepath, outputpath='/work/friemer/hgcalml/testplots/', datasetname='Quark Jet'):
    #Load analysis file created by analyse_cocoa_predictions.py
    with gzip.open(analysisfilepath, 'rb') as input_file:
        analysis_data = pickle.load(input_file)
    
    df = analysis_data['showers_dataframe']
    pred_list = analysis_data['prediction']
    t_list = analysis_data['truth']
    feature_list =  analysis_data['features']
    jets_df = analysis_data['jets_dataframe']
    
    plot_everything(df, pred_list, t_list, feature_list, jets_df, outputpath, datasetname=datasetname)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Create plots for the COCOA analysis')
    parser.add_argument('analysisfile',
        help='Filepath to analysis file created by analyse_cocoa_predictions.py containing shower dataframe')
    parser.add_argument('outputlocation',
        help="Output directory for the plots",
        default='')
    parser.add_argument('--gluon', action='store_true', help='Write Gluon dataset instead of Quark dataset in top left corner')
    args = parser.parse_args()
    
    if args.gluon:
        datasetname='Gluon Jet'
    else:
        datasetname='Quark Jet'
    plot_everything_from_file(args.analysisfile, args.outputlocation, datasetname=datasetname)