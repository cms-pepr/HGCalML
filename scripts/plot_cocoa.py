import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import plotly.express as px
import argparse


def calc_efficiencies(df, bins, mask=None):
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
    # Calculate the bin positions and widths
    binwidth = bins[1:] - bins[:-1]
    x_pos = bins[:-1] + binwidth / 2
    x_err = binwidth / 2
    
    # Calculate the efficiencies and fake rates for the total
    yeff, yerr_eff, yfake, yerr_fake, ycorr, yerr_corr = calc_efficiencies(df, bins)
    yeff, yerr_eff, yfake, yerr_fake, ycorr, yerr_corr = \
        yeff*100, yerr_eff*100, yfake*100, yerr_fake*100, ycorr*100, yerr_corr*100
    
    # Calculate the efficiencies and fake rates for the photons
    yeff_photon, yerr_eff_photon, yfake_photon, yerr_fake_photon, ycorr_photon, yerr_corr_photon = calc_efficiencies(df, bins, 0)
    yeff_photon, yerr_eff_photon, yfake_photon, yerr_fake_photon, ycorr_photon, yerr_corr_photon = \
        yeff_photon*100, yerr_eff_photon*100, yfake_photon*100, yerr_fake_photon*100, ycorr_photon*100, yerr_corr_photon*100
    
    # Calculate the efficiencies and fake rates for the neutral hadrons
    #130: "K0L", 310: "K0S", 311: "K0", 2112: "neutron",-2112: "antineutron",3122: "Lambda",-3122: "antilambda"3322: "Xi0",-3322: "antixi0"
    yeff_nh, yerr_eff_nh, yfake_nh, yerr_fake_nh, ycorr_nh, yerr_corr_nh = calc_efficiencies(df, bins, 1)
    yeff_nh, yerr_eff_nh, yfake_nh, yerr_fake_nh, ycorr_nh, yerr_corr_nh = \
        yeff_nh*100, yerr_eff_nh*100, yfake_nh*100, yerr_fake_nh*100, ycorr_nh*100, yerr_corr_nh*100
    
    # Create the plots
    fig, ((ax1, ax2, ax3), (ax4, ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
    ax1.errorbar(x_pos, yeff, xerr=x_err, yerr=yerr_eff, fmt='o', color='red', label='Efficiency')
    ax1.set_ylabel('efficiency total', fontsize=10)
    ax2.errorbar(x_pos, yeff_photon, xerr=x_err, yerr=yerr_eff_photon, fmt='o', color='red', label='Efficiency')
    ax2.set_ylabel('efficiency gamma', fontsize=10)
    ax3.errorbar(x_pos, yeff_nh, xerr=x_err, yerr=yerr_eff_nh, fmt='o', color='red', label='Efficiency')
    ax3.set_ylabel('efficiency n.H.', fontsize=10)
    
    ax4.errorbar(x_pos, yfake, xerr=x_err, yerr=yerr_fake, fmt='o', color='red', label='Fake rate')
    ax4.set_ylabel('fake rate total', fontsize=10)
    ax5.errorbar(x_pos, yfake_photon, xerr=x_err, yerr=yerr_fake_photon, fmt='o', color='red', label='Fake rate')
    ax5.set_ylabel('fake rate gamma', fontsize=10)
    ax6.errorbar(x_pos, yfake_nh, xerr=x_err, yerr=yerr_fake_nh, fmt='o', color='red', label='Fake rate')
    ax6.set_ylabel('fake rate n.H.', fontsize=10)
    
    ax7.errorbar(x_pos, ycorr, xerr=x_err, yerr=yerr_corr, fmt='o', color='red', label='probability of correct class')
    ax7.set_ylabel('p corr total', fontsize=10)
    ax8.errorbar(x_pos, ycorr_photon, xerr=x_err, yerr=yerr_corr_photon, fmt='o', color='red', label='probability of correct class')
    ax8.set_ylabel('p corr gamma', fontsize=10)
    ax9.errorbar(x_pos, ycorr_nh, xerr=x_err, yerr=yerr_corr_nh, fmt='o', color='red', label='probability of correct class')
    ax9.set_ylabel('p corr n.H.', fontsize=10)

    for ax in [ax1, ax2,ax3, ax4, ax5, ax6, ax7, ax8, ax9]:    
        ax.set_xticks(bins)
        ax.set_xticklabels(bins, fontsize=10)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)
        ax.set_xlabel('Energy [GeV]', fontsize=10)
    
        yticks1 = np.round(np.arange(40, 101, 20), 1)
        ax.set_yticks(yticks1)
        ax.set_yticklabels([f"{y}%" for y in yticks1], fontsize=10)
        #ax.set_ylim(20, 101)

    return fig
def plot_jet_metrics(df):
    #Jet metrics
    reseta = []
    resphi = []
    relresE = []
    nParicles_pred = []
    nParicles_truth = []
    
    matched = calc_matched_mask(df)

    for i in range(len(df['event_id'].unique())):
        mask = np.logical_and(df['event_id'] == i, matched)
        
        jet_E_truth = np.sum(df[mask]['truthHitAssignedEnergies'])
        jet_E_pred = np.sum(df[mask]['pred_energy'])
        relresE.append((jet_E_truth-jet_E_pred)/jet_E_truth)
        
        nParicles_pred.append(len(df[mask]['pred_sid']))
        nParicles_truth.append(len(df[mask]['truthHitAssignementIdx']))
        
        t_eta = np.mean(df[mask]['truthHitAssignedZ'])
        pred_eta = df[mask]['pred_pos'].apply(lambda x: x[2]).mean()
        reseta.append(t_eta-pred_eta)
        
        t_phi = np.mean(np.arctan2(df[mask]['truthHitAssignedY'], df[mask]['truthHitAssignedX']))
        pred_phi = df[mask]['pred_pos'].apply(lambda x: np.arctan2(x[1], x[0])).mean()
        delta_phi = calc_deltaphi(t_phi, pred_phi)        
        resphi.append(delta_phi)

    nParticle_bins = np.linspace(0, 30, 31)
    fig_jet_nparticle = plt.figure()
    plt.hist(nParicles_truth, bins=nParticle_bins, label='Truth', color='#fee7ae')
    plt.grid(alpha=0.5)
    plt.hist(nParicles_pred, bins=nParticle_bins, label='Prediction', histtype='step', color='#67c4ce', linestyle='--')
    plt.text(0.02, 0.98, 'Quark Jet', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)    
    plt.ylabel('Events')
    plt.xlabel('Jet nConstituents')
    plt.legend()
    plt.close()

    relres_bins = np.linspace(-1, 1, 51)
    fig_jet_relresE = plt.figure()
    plt.hist(relresE, bins=relres_bins, histtype='step', color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.5)
    plt.text(0.02, 0.98, 'Quark Jet', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.ylabel('Number of events')
    plt.xlabel('rel. res. Cal. Jet Energy')
    plt.close()
    
    
    res_bins = np.linspace(-0.2, 0.2, 51)
    fig_jet_reseta = plt.figure()
    plt.hist(reseta, bins=res_bins, histtype='step', color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.5)
    plt.text(0.02, 0.98, 'Quark Jet', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.ylabel('Number of events')
    plt.xlabel('Jet $\\Delta \\eta$')
    plt.close()

    fig_jet_resphi = plt.figure()
    plt.hist(resphi, bins=res_bins, histtype='step', color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.5)
    plt.text(0.02, 0.98, 'Quark Jet', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.ylabel('Number of events')
    plt.xlabel('Jet $\\Delta \\phi$')
    plt.close()
    
    return fig_jet_nparticle, fig_jet_relresE, fig_jet_reseta, fig_jet_resphi

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

def plot_particle_metrics(df):
    
    matched = calc_matched_mask(df)
    
    reseta = []
    resphi = []
    relresE = []

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

    relres_bins = np.linspace(-1, 1, 51)
    fig_relresE = plt.figure()
    plt.hist(relresE, bins=relres_bins, histtype='step', color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.5)
    plt.xlabel('res. res. Neutral Particle Energy')
    plt.ylabel('Number of neutral particles')
    plt.close()

    res_bins = np.linspace(-0.2, 0.2, 51)
    fig_reseta = plt.figure()
    plt.hist(reseta, bins=res_bins, histtype='step', color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.5)
    plt.xlabel('Neutral Particle $\\Delta \\eta$')
    plt.ylabel('Number of neutral particles')
    plt.close()

    fig_resphi = plt.figure()
    plt.hist(resphi, bins=res_bins, histtype='step', color='#67c4ce', linestyle='--')
    plt.grid(alpha=0.5)
    plt.xlabel('Neutral Particle $\\Delta \\phi$')
    plt.ylabel('Number of neutral particles')
    plt.close()

    return fig_relresE, fig_reseta, fig_resphi
    
def plt_energy_resolution(df, bins=None):
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(df['pred_energy']) == False
    matched = np.logical_and(has_truth, has_pred)

    means = []
    std_error = []

    for i in range(len(bins) - 1):
        mask_truth_bin = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_bin = np.logical_and(mask_truth_bin, matched)
        diff = np.abs(df['pred_energy'][mask_bin] - df['truthHitAssignedEnergies'][mask_bin])
        #ratios = df['pred_energy'][mask_bin] / df['truthHitAssignedEnergies'][mask_bin]
        means.append(np.mean(diff))
        std_error.append(np.std(diff) / np.sqrt(len(diff)))
        
    means = np.array(means)
    std_error = np.array(std_error)
    
    # make boxplots of the ratios for each bin
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    binwidth = bins[1:] - bins[:-1]
    x = bins[:-1] + binwidth / 2
    xerr = binwidth / 2    
    y = means
    yerr = std_error
    
    #Convert MeV to GeV
    x = x/1000
    xerr = xerr/1000
    y = y/1000
    yerr = yerr/1000
    bins = bins/1000
    
    # plot x, y with error bars
    ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color='#67c4ce')
    
    # set xticks
    xticks = np.round(bins, 0).astype(int)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=10)
    
    # ymin = np.round(ax1.get_ylim()[0], 1)
    # ymax = np.round(ax1.get_ylim()[1], 1)
    # ydelta = max(1 - ymin, ymax - 1)
    # ax1.set_ylim(1 - ydelta, 1 + ydelta)    
    # # set yticks to be evenly spaced around 1
    # yticks = np.round(np.arange(1 - ydelta, 1 + ydelta, 0.02), 2)
    # yticks = np.round(np.linspace(1 - ydelta, 1 + ydelta, 11), 2)
    # ax1.set_yticks(yticks)
    # ax1.set_yticklabels(yticks, fontsize=20)
    
    ax1.grid(alpha=0.5, linestyle='--')
    ax1.set_ylabel('$\sigma E [GeV]$', fontsize=20)
    ax1.set_xlabel('Energy [GeV]', fontsize=20)
    
    return fig

def plot_condensation(event_id, pred_list, t_list, feature_list, location='/work/friemer/hgcalml/condensationplots/condensation'):
    coords = pred_list[event_id]['pred_ccoords']

    if coords.shape[1] < 3: #add a zero
        coords = tf.concat([coords, tf.zeros_like(coords)[:,0:1]], axis=-1)

    data={
        'X':  coords[:,0],
        'Y':  coords[:,1],
        'Z':  coords[:,2],
        'hyper_idx': t_list[event_id]['truthHitAssignementIdx'][:,0],
        'features': pred_list[event_id]['pred_beta'][:,0],
        'pdgid': t_list[event_id]['truthHitAssignedPIDs'][:,0],
        'recHitID': feature_list[event_id]['recHitID'][:,0],
        }
    eventdf = pd.DataFrame(data)

    hover_data = {'X': True, 'Y': True, 'Z': True, 'hyper_idx': True,'features': True, 'pdgid': True,'recHitID': True}

    fig = px.scatter_3d(eventdf, x="X", y="Y", z="Z", 
                        color="hyper_idx",
                        size='features',
                        template='plotly_dark',
                        hover_data=hover_data,
            color_continuous_scale=px.colors.sequential.Rainbow)
    fig.update_traces(marker=dict(line=dict(width=0)))

    fig.write_html(location+str(event_id)+".html")
    return fig

def plot_everything(df, outputpath='/work/friemer/hgcalml/testplots/'):
    
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
            
    #Create everything            
    energy_bins_neutral = np.array([1,2,3,4,5,10,20,30,50])*1000
    fig_eff=plot_efficencies(df, bins=energy_bins_neutral)
    
    fig_jet_nparticle, fig_jet_relresE, fig_jet_reseta, fig_jet_resphi = plot_jet_metrics(df)
    
    mask_neutral = df['truthHitAssignedPIDs'].isin([22,130, 310, 311, 2112, -2112, 3122, -3122, 3322, -3322])    
    fig_neutral_relresE, fig_neutral_reseta, fig_neutral_resphi = plot_particle_metrics(df[mask_neutral])
    
    energy_bins_charged = np.array([15,20,30,50,200])*1000
    mask_charged = df['truthHitAssignedPIDs'].isin([11,-11,13,-13,211,-211,321,-321,2212,-2212,3112,-3112,3222,-3222,3312,-3312])    
    fig_charged_res = plt_energy_resolution(df[mask_charged], bins=energy_bins_charged)
    
    #Save everything
    fig_eff.savefig(os.path.join(outputpath,'efficiencies.png'))
    fig_jet_nparticle.savefig(os.path.join(outputpath,'jet_nparticle.png'))
    fig_jet_relresE.savefig(os.path.join(outputpath,'jet_relresE.png'))
    fig_jet_reseta.savefig(os.path.join(outputpath,'jet_reseta.png'))
    fig_jet_resphi.savefig(os.path.join(outputpath,'jet_resphi.png'))
    fig_neutral_relresE.savefig(os.path.join(outputpath,'neutral_relresE.png'))
    fig_neutral_reseta.savefig(os.path.join(outputpath,'neutral_reseta.png'))
    fig_neutral_resphi.savefig(os.path.join(outputpath,'neutral_resphi.png'))
    fig_charged_res.savefig(os.path.join(outputpath,'charged_res.png'))
    
    
    
def plot_everything_from_file(analysisfilepath, outputpath='/work/friemer/hgcalml/testplots/'):
    with gzip.open(analysisfilepath, 'rb') as input_file:
        analysis_data = pickle.load(input_file)
    df = analysis_data['showers_dataframe']
    plot_everything(df, outputpath)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Create Plots for the COCOA analysis')
    parser.add_argument('analysisfile',
        help='Filepath to analysis file created by analyse_cocoa_predictions.py containing shower dataframe')
    parser.add_argument('outputlocation',
        help="Output directory for the plots",
        default='')

    args = parser.parse_args()

    plot_everything_from_file(args.analysisfile, args.outputlocation)    
