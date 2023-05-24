#!/usr/bin/env python3
import os
import gzip
import pickle
import numpy as np
import mgzip
import argparse
import time
import pandas as pd
from globals import pu
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from OCHits2Showers import OCHits2ShowersLayer, process_endcap, OCGatherEnergyCorrFac
from OCHits2Showers import process_endcap2, OCGatherEnergyCorrFac2
from ShowersMatcher2 import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter


def calc_efficiencies(df, bins):
    mask_predicted = np.isnan(df['pred_energy']) == False
    mask_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    n_trues = []
    n_pred = []
    n_fakes = []
    n_matched = []
    efficiencies = []
    fake_rate = []
    for i in range(len(bins) - 1):
        mask_bintruth = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        n_trues.append(np.sum(mask_bintruth))
        mask_binpred = np.logical_and(
            df['pred_energy'] >= bins[i],
            df['pred_energy'] < bins[i + 1])
        n_pred.append(np.sum(mask_binpred))

        matched = np.logical_and(
            mask_predicted,
            mask_bintruth)
        n_matched.append(np.sum(matched))
        faked = np.logical_and(
            mask_binpred,
            np.logical_not(mask_truth))
        n_fakes.append(np.sum(faked))

        efficiencies.append(np.sum(matched) / np.sum(mask_bintruth))
        fake_rate.append(np.sum(faked) / np.sum(mask_binpred))

    data = pd.DataFrame({
        "n_trues": np.array(n_trues),
        "n_pred": np.array(n_pred),
        "n_matched": np.array(n_matched),
        "n_fakes": np.array(n_fakes),
        "efficiencies": np.array(efficiencies),
        "fake_rate": np.array(fake_rate),
    })

    return data


def calc_resolution(df, bins):
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(df['pred_energy']) == False
    mask = np.logical_and(has_truth, has_pred)

    ratios = []
    means = []
    stddevs = []
    sigmaOverE = []
    std_error = []
    entries = []

    for i in range(len(bins) - 1):
        mask_bin = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_bin = np.logical_and(mask_bin, mask)
        ratios.append(df['pred_energy'][mask_bin] / df['truthHitAssignedEnergies'][mask_bin])
        means.append(np.mean(ratios[-1]))
        stddevs.append(np.std(ratios[-1]))
        sigmaOverE.append(stddevs[-1] / means[-1])
        std_error.append(stddevs[-1] / np.sqrt(len(ratios[-1])))
        entries.append(len(ratios[-1]))

    data = pd.DataFrame({
        "means": np.array(means),
        "stddevs": np.array(stddevs),
        "sigmaOverE": np.array(sigmaOverE),
        "std_error": np.array(std_error),
        "entries": np.array(entries),
    })

    return data, ratios


def resolution_func(energy, a, b, c):
    return np.sqrt(a**2 + b**2 / energy + c**2 / energy**2)


def energy_resolution(df, bins=None, binwidth=10., addfit=False):
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(df['pred_energy']) == False
    mask = np.logical_and(has_truth, has_pred)
    if bins is None:
        binmax = df['truthHitAssignedEnergies'].max()
        bins = calc_energy_bins(binmax, binwidth)


    data, ratios = calc_resolution(df, bins)
    means = data['means']
    stddevs = data['stddevs']
    sigma_over_e = data['sigma_over_e']
    std_error = data['std_error']
    entries = data['entries']

    # make boxplots of the ratios for each bin
    fig, (ax1, ax3) = plt.subplots(nrows=2, figsize=(20, 20))
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    # ax1 and ax2 for histogram + mean
    # ax3 and ax4 for boxplot and stdddev
    x = bins[:-1] + binwidth / 2
    xerr = np.ones_like(x) * float(binwidth / 2.)
    y = means
    yerr = std_error
    # plot x, y with error bars
    ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color='blue', zorder=2, label='Mean Response')
    # set xticks
    xticks = np.round(bins, 0).astype(int)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=20)
    ymin = np.round(ax1.get_ylim()[0], 1)
    ymax = np.round(ax1.get_ylim()[1], 1)
    ydelta = max(1 - ymin, ymax - 1)
    ax1.set_ylim(1 - ydelta, 1 + ydelta)
    # set yticks to be evenlyt spaced around 1
    yticks = np.round(np.arange(1 - ydelta, 1 + ydelta, 0.02), 2)
    yticks = np.round(np.linspace(1 - ydelta, 1 + ydelta, 11), 2)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=20)
    ax1.grid(alpha=0.5, linestyle='--')
    ax1.set_ylabel('Mean Response', fontsize=40)

    # plot histograms with entries in ax2 in the background
    # ax2.hist(bins[1:], weights=entries, histtype='stepfilled', color='grey', alpha=0.2, zorder=1)
    ax2.hist(bins[1:], weights=entries, color='grey', alpha=0.2, zorder=1, label='Counts')
    yticks = ax2.get_yticks()
    ax2.set_yticklabels(yticks, fontsize=20)
    ax2.set_title("Respone and Counts", fontsize=40)
    ax2.set_ylabel("Counts", fontsize=40)

    # combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, fontsize=20)

    # plot the standard deviation in ax3
    fit_mask = sigma_over_e > 0
    ax3.errorbar(x[fit_mask], sigma_over_e[fit_mask], xerr=xerr[fit_mask], fmt='o', color='blue', zorder=2, label='Standard Deviation')
    xticks = np.round(bins, 0).astype(int)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticks, fontsize=20)
    yticks = ax3.get_yticks()
    ax3.set_yticklabels(np.round(yticks, 3), fontsize=20)
    ax3.grid(alpha=0.5, linestyle='--')
    # fit resolution function to sigmaOverE
    print(sigma_over_e)
    if addfit:
        popt, pcov = curve_fit(resolution_func, x[fit_mask], sigma_over_e[fit_mask], p0=[0.1, 0.1, 0.1])
        # plot fit
        xfit = np.linspace(0, bins[-1], 1000)
        yfit = resolution_func(xfit, *popt)
        plot_mask = yfit < 0.9 * ax3.get_ylim()[1]
        # add fit parameters to label
        label = f"Fit: {popt[0]:.3f} + {popt[1]:.3f} / sqrt(E) + {popt[2]:.3f} / E"
        ax3.plot(xfit[plot_mask], yfit[plot_mask], color='red', label=label)
    ylabel = r"$\sigma / E$"
    ax3.set_ylabel(ylabel, fontsize=40)

    ax4.boxplot(ratios, positions=bins[:-1] + binwidth / 2, widths=binwidth * 0.8, showfliers=False, patch_artist=True, zorder=2,
                boxprops={
                    'color': 'grey',
                    'linewidth': 3,
                    'alpha': 0.2,
                    # fill boxplots with grey
                    'facecolor': 'grey',
                    'edgecolor': 'grey',
                    },
                whiskerprops={
                    'color': 'grey',
                    'linewidth': 3,
                    'alpha': 0.2,
                    },
                capprops={
                    'color': 'grey',
                    'linewidth': 3,
                    'alpha': 0.2,
                    },
                medianprops={
                    'color': 'black',
                    'linewidth': 3,
                    'alpha': 0.2,
                    },
                )
    xticks = np.round(bins, 0).astype(int)
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xticks, fontsize=20)
    yticks = np.arange(np.round(ax4.get_ylim()[0], 1), np.round(ax4.get_ylim()[1], 1), 0.1)
    yticks = np.round(yticks, 1)
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticks, fontsize=20)
    ax4.set_xlabel('Energy [GeV]', fontsize=40)
    ax4.set_ylabel('Predicted / Truth', fontsize=40)
    ax4.set_title('Resolution', fontsize=40)
    # ax4.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
    # ax4.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)

    # combine legends
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()
    handles = handles3 + handles4
    labels = labels3 + labels4
    ax3.legend(handles, labels, fontsize=20)

    # set zorder to move boxplots in background
    ax3.set_zorder(1)
    ax3.patch.set_visible(False)
    ax4.set_zorder(0)
    return fig


def efficiency_plot(df, bins=None, binwidth=10):
    if bins is None:
        binmax = df['truthHitAssignedEnergies'].max()
        bins = calc_energy_bins(binmax, binwidth)

    data = calc_efficiencies(df, bins)
    n_trues = data['n_trues']
    n_preds = data['n_pred']
    efficiencies = data['efficiencies']
    fake_rates = data['fake_rate']

    x_pos = bins[:-1] + binwidth / 2
    x_err = binwidth / 2
    y_eff = efficiencies
    yerr_eff = np.sqrt(efficiencies * (1 - efficiencies) / n_trues)
    y_fake = fake_rates
    yerr_fake = np.sqrt(fake_rates * (1 - fake_rates) / n_preds)

    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()

    ax2.hist([bins[:-1], bins[:-1]], bins=bins, weights=[n_trues, n_preds], histtype='bar',
         color=['lightgrey', 'lightblue'], alpha=1., label=['n_true', 'n_pred'])
    ax1.errorbar(x_pos, y_fake, xerr=x_err, yerr=yerr_fake, fmt='o', color='red', label='Fake rate', zorder=2)
    ax1.errorbar(x_pos, y_eff, xerr=x_err, yerr=yerr_eff, fmt='o', color='blue', label='Efficiency', zorder=3)

    ax1.set_xticks(bins)
    ax1.set_xticklabels(bins, fontsize=20)
    yticks1 = np.round(np.arange(0, 1.1, 0.1), 1)
    ax1.set_yticks(yticks1)
    ax1.set_yticklabels(yticks1, fontsize=20)
    yticks2 = np.round(np.arange(0, max(n_trues.max(), n_preds.max()) * 1.1, 25), 0).astype(int)
    ax2.set_yticks(yticks2)
    ax2.set_yticklabels(yticks2, fontsize=20)
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, max(n_trues.max(), n_preds.max()) * 1.1)

    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
    ax1.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)

    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    ax2.set_zorder(1)

    ax1.set_xlabel('Energy [GeV]', fontsize=40)
    ax1.set_ylabel('Fraction', fontsize=40)
    ax2.set_ylabel('Number of showers', fontsize=40)
    ax1.set_title('Efficiency and Fake Rate', fontsize=40)

    # combine the two legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    # make the legend's background transparent and position it precisely

    legend = ax1.legend(handles, labels, loc=(0.8, 0.6), fontsize=20, framealpha=0.8)
    return fig


def calc_energy_bins(binmax, binwidth):
    """Calculate energy bins for a given binwidth and maximum energy."""
    if binmax % binwidth == 0:
        extra = 0
    else:
        extra = binwidth
    binmax = int(binmax + binwidth - binmax % binwidth)
    return np.arange(0, binmax + extra, binwidth)


def calc_resolution(showers, bins, predstring='pred_energy'):
    """Calculate resolution for a given dataframe and energy bins."""
    has_truth = np.isnan(showers['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(showers[predstring]) == False
    matched = np.logical_and(has_truth, has_pred)

    all_ratios = []
    means = []
    stddevs = []
    sigma_over_e = []
    std_error = []
    entries = []

    for i in range(len(bins) - 1):
        mask_truth_bin = np.logical_and(
            showers['truthHitAssignedEnergies'] >= bins[i],
            showers['truthHitAssignedEnergies'] < bins[i + 1])
        mask_bin = np.logical_and(mask_truth_bin, matched)
        ratios = showers[predstring][mask_bin] / showers['truthHitAssignedEnergies'][mask_bin]
        all_ratios.append(ratios)
        means.append(np.mean(all_ratios[-1]))
        stddevs.append(np.std(all_ratios[-1]))
        sigma_over_e.append(stddevs[-1] / means[-1])
        std_error.append(stddevs[-1] / np.sqrt(len(all_ratios[-1])))
        entries.append(len(all_ratios[-1]))

    data = pd.DataFrame({
        "means": np.array(means),
        "stddevs": np.array(stddevs),
        "sigma_over_e": np.array(sigma_over_e),
        "std_error": np.array(std_error),
        "entries": np.array(entries),
    })

    return data, all_ratios


def dictlist_to_dataframe(dictlist, masks=None):
    full_df = pd.DataFrame()
    for i in range(len(dictlist)):
        df = pd.DataFrame()
        for key, value in dictlist[i].items():
            if key in ['row_splits']:
                continue
            if len(value.shape) == 1:
                continue
            if value.shape[1] > 1:
                for j in range(value.shape[1]):
                    df[key + '_' + str(j)] = value[:, j]
            else:
                df[key] = value[:, 0]
        df['event_id'] = i * np.ones_like(df.shape[0])
        if masks is not None:
            mask = masks[i].reshape(-1)
            df = df.iloc[mask]
        full_df = pd.concat([full_df, df])
    return full_df


def filter_truth_dict(truth_dict, mask):
    n_orig = truth_dict['truthHitAssignementIdx'].shape[0]
    filtered = {}
    for key, item in truth_dict.items():
        if item.shape[0] == n_orig:
            item_filtered = item[mask]
            filtered[key] = item_filtered.reshape(-1, 1)
        else:
            print(key, " untouched")
    return filtered


def filter_features_dict(features_dict, mask):
    n_orig = features_dict['recHitEnergy'].shape[0]
    filtered = {}
    for key, item in features_dict.items():
        if item.shape[0] == n_orig:
            item_filtered = item[mask]
            filtered[key] = item_filtered.reshape(-1,1)
        else:
            print(key, " untouched")
    return filtered


def analyse(preddir, pdfpath, beta_threshold, distance_threshold, iou_threshold,
        matching_mode, analysisoutpath, nfiles, nevents,
        local_distance_scaling, is_soft, de_e_cut, angle_cut,
        kill_pu=False, filter_pu=False, toydata=False, energy_mode='hits', raw=False):

    hits2showers = OCHits2ShowersLayer(
            beta_threshold, distance_threshold, local_distance_scaling)
    showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

    energy_gatherer = OCGatherEnergyCorrFac2()

    files_to_be_tested = [
        os.path.join(preddir, x)
        for x in os.listdir(preddir)
        if (x.endswith('.bin.gz') and x.startswith('pred'))]
    if toydata:
        extra_files = [
            os.path.join(preddir, x)
            for x in os.listdir(preddir)
            if ( x.endswith('.pkl') and x.startswith('pred') )]
    if nfiles!=-1:
        files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

    showers_dataframe = pd.DataFrame()
    features = []
    truth = []
    prediction = []
    processed = []
    alpha_ids = []
    noise_masks = []
    event_id = 0
    matched = []

    ###############################################################################################
    ### Loop over all events ######################################################################
    ###############################################################################################

    for i, file in enumerate(files_to_be_tested):
        print("Analysing file %d/%d"% (i, len(files_to_be_tested)))
        with mgzip.open(file, 'rb') as f:
            file_data = pickle.load(f)
            if toydata:
                with open(extra_files[i], 'rb') as xf:
                    xfile = pickle.load(xf)
            for j, endcap_data in enumerate(file_data):
                if (nevents != -1) and (j > nevents):
                    continue
                if toydata:
                    t_min_bias = xfile[j][0][0]
                print("Analysing endcap %d/%d" % (j, len(file_data)))
                features_dict, truth_dict, predictions_dict = endcap_data
                features.append(features_dict)
                prediction.append(predictions_dict)
                truth.append(truth_dict)
                if filter_pu and not toydata:
                    print("Filter PU only possible if t_min_bias is provided.\
                            currently this only exists for toydata")
                if filter_pu and toydata:
                    pu_filter = np.array(t_min_bias == 0).flatten()
                    for key in predictions_dict.keys():
                        try:
                            predictions_dict[key] = predictions_dict[key][pu_filter]
                        except:
                            print(key, "no filter applied")
                    for key in features_dict.keys():
                        try:
                            features_dict[key] = features_dict[key][pu_filter]
                        except:
                            print(key, "no filter applied")
                    for key in truth_dict.keys():
                        try:
                            truth_dict[key] = truth_dict[key][pu_filter]
                        except:
                            print(key, "no filter applied")
                else:
                    pu_filter = None

                noise_mask = predictions_dict['no_noise_sel']
                noise_masks.append(noise_mask)
                filtered_features = filter_features_dict(features_dict, noise_mask)
                filtered_truth = filter_truth_dict(truth_dict, noise_mask)

                processed_pred_dict, pred_shower_alpha_idx = process_endcap2(
                        hits2showers,
                        energy_gatherer,
                        filtered_features,
                        predictions_dict,
                        energy_mode=energy_mode)

                alpha_ids.append(pred_shower_alpha_idx)
                processed.append(processed_pred_dict)
                showers_matcher.set_inputs(
                    features_dict=filtered_features,
                    truth_dict=filtered_truth,
                    predictions_dict=processed_pred_dict,
                    pred_alpha_idx=pred_shower_alpha_idx
                )
                showers_matcher.process()
                dataframe = showers_matcher.get_result_as_dataframe()
                matched_truth_sid, matched_pred_sid = showers_matcher.get_matched_hit_sids()
                matched.append((matched_truth_sid, matched_pred_sid))

                dataframe['event_id'] = event_id
                event_id += 1
                if kill_pu:
                    if len(dataframe[dataframe['truthHitAssignementIdx']>=pu.t_idx_offset]):
                        print('\nWARNING REMOVING PU TRUTH MATCHED SHOWERS, HACK.\n')
                        dataframe = dataframe[dataframe['truthHitAssignementIdx']<pu.t_idx_offset]
                showers_dataframe = pd.concat((showers_dataframe, dataframe))
                processed_dataframe = dictlist_to_dataframe(processed)

    ###############################################################################################
    ### New plotting stuff ########################################################################
    ###############################################################################################

    ### Tracks versus hits ########################################################################
    bins = calc_energy_bins(200,10)
    data_track_raw, ratios_track_raw, = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_tracks_raw')
    medians_track_raw = [np.median(r) for r in ratios_track_raw]

    data_track, ratios_track = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_tracks')
    medians_track = [np.median(r) for r in ratios_track]

    data_hits_raw, ratios_hits_raw = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_hits_raw')
    medians_hits_raw = [np.median(r) for r in ratios_hits_raw]

    data_hits, ratios_hits = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_hits')
    medians_hits = [np.median(r) for r in ratios_hits]
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
    ax_comp.plot(bins[:-1], medians_track_raw, '--', color='red', lw=3,
            label='Median of Ratio, Tracks Raw')
    ax_comp.plot(bins[:-1], medians_track, color='red',
            label='Median of Ratio, Tracks Corrected')
    ax_comp.plot(bins[:-1], medians_hits_raw, '--', color='blue', lw=3,
            label='Median of Ratio, Hits Raw')
    ax_comp.plot(bins[:-1], medians_hits, color='blue', lw=3,
            label='Median of Ratio, Hits Corrected')
    ax_comp.legend()
    ax_comp.grid()
    ax_comp.set_title("Median of Ratios 'True' / Predicted'", fontsize=20)
    ax_comp.set_xlabel("True Energy [GeV]", fontsize=20)
    ax_comp.set_ylabel("Ratio", fontsize=20)
    # save figure
    fig_comp.savefig(os.path.join('.', 'median_ratios.jpg'))

    ### Efficiency plots ##########################################################################
    fig_eff = efficiency_plot(showers_dataframe)
    fig_eff.savefig(os.path.join('.', 'efficiency.jpg'))

    ### Resolution plots ##########################################################################
    fig_res = energy_resolution(showers_dataframe)
    fig_res.savefig(os.path.join('.', 'energy_resolution.jpg'))

    # This is only to write to pdf files
    scalar_variables = {
        'beta_threshold': str(beta_threshold),
        'distance_threshold': str(distance_threshold),
        'iou_threshold': str(iou_threshold),
        'matching_mode': str(matching_mode),
        'is_soft': str(is_soft),
        'de_e_cut': str(de_e_cut),
        'angle_cut': str(angle_cut),
    }

    if len(analysisoutpath) > 0:
        analysis_data = {
            'showers_dataframe' : showers_dataframe,
            'events_dataframe' : None,
            'scalar_variables' : scalar_variables,
            'alpha_ids'        : alpha_ids,
            'noise_masks': noise_masks,
            'matched': matched,
        }
        if not args.slim:
            analysis_data['processed_dataframe'] = processed_dataframe
            analysis_data['features'] = features
            analysis_data['truth'] = truth
            analysis_data['prediction'] = prediction
        with gzip.open(analysisoutpath, 'wb') as f:
            print("Writing dataframes to pickled file",analysisoutpath)
            pickle.dump(analysis_data,f)

    if len(pdfpath)>0:
        plotter = HGCalAnalysisPlotter()
        plotter.set_data(showers_dataframe, None, '', pdfpath, scalar_variables=scalar_variables)
        plotter.process()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('preddir',
        help='Directory with .bin.gz files or a txt file with full paths of the \
            bin-gz files from the prediction.')
    parser.add_argument('-p',
        help="Output directory for the final analysis pdf file (otherwise, it won't be produced)",
        default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-m', help='Matching mode', default='iou_max')
    parser.add_argument('--analysisoutpath',
        help='Will dump analysis data to a file to remake plots without re-running everything.',
        default='')
    parser.add_argument('--nfiles',
        help='Maximum number of files. -1 for everything in the preddir',
        default=-1)
    parser.add_argument('--no_local_distance_scaling', help='With local distance scaling',
        action='store_true')
    parser.add_argument('--de_e_cut', help='dE/E threshold to allow match.', default=-1)
    parser.add_argument('--angle_cut', help='Angle cut for angle based matching', default=-1)
    parser.add_argument('--no_soft', help='Use condensate op', action='store_true')
    parser.add_argument('--filter_pu', help='Filter PU', action='store_true')
    parser.add_argument('--toydata', help='Use toy detector', action='store_true')
    parser.add_argument('--nevents', help='Maximum number of events (per file)', default=-1)
    parser.add_argument('--emode', help='Mode how energy is calculated', default='hits')
    parser.add_argument('--raw', help="Ignore energy correction factor", action='store_true')
    parser.add_argument('--slim',
        help="Produce only a small analysis.bin.gz file. \
            Only applicable if --analysisoutpath is set",
        action='store_true')


    args = parser.parse_args()

    analyse(preddir=args.preddir,
            pdfpath=args.p,
            beta_threshold=float(args.b),
            distance_threshold=float(args.d),
            iou_threshold=float(args.i),
            matching_mode=args.m,
            analysisoutpath=args.analysisoutpath,
            nfiles=int(args.nfiles),
            local_distance_scaling=not args.no_local_distance_scaling,
            is_soft=not args.no_soft,
            de_e_cut=float(args.de_e_cut),
            angle_cut=float(args.angle_cut),
            filter_pu=bool(args.filter_pu),
            toydata=bool(args.toydata),
            nevents=int(args.nevents),
            energy_mode=str(args.emode),
            raw=bool(args.raw),)
