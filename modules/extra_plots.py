# Plotting tools used when analysing model predictions
import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import confusion_matrix


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


def calc_within_uncertainty(df, bins, predstring='pred_energy'):
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(df[predstring]) == False
    mask = np.logical_and(has_truth, has_pred)
    fraction_within_1sigma = []
    fraction_within_2sigma = []
    fraction_within_3sigma = []

    for i in range(len(bins) - 1):
        mask_bin = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_bin = np.logical_and(mask_bin, mask)
        e_true = df['truthHitAssignedEnergies'][mask_bin]
        e_pred = df[predstring][mask_bin]
        uncertainty = df['pred_energy_high_quantile'][mask_bin]
        fraction_within_1sigma.append(
            np.sum(np.abs(e_true - e_pred) < uncertainty) / len(e_true))
        fraction_within_2sigma.append(
            np.sum(np.abs(e_true - e_pred) < 2 * uncertainty) / len(e_true))
        fraction_within_3sigma.append(
            np.sum(np.abs(e_true - e_pred) < 3 * uncertainty) / len(e_true))

    fraction_within_1sigma = np.array(fraction_within_1sigma)
    fraction_within_2sigma = np.array(fraction_within_2sigma)
    fraction_within_3sigma = np.array(fraction_within_3sigma)
    return fraction_within_1sigma, fraction_within_2sigma, fraction_within_3sigma


def bin_uncertainty(df, bins, predstring='pred_energy'):
    has_truth = np.isnan(df['truthHitAssignedEnergies']) == False
    has_pred = np.isnan(df[predstring]) == False
    mask = np.logical_and(has_truth, has_pred)
    mean = []
    std = []

    for i in range(len(bins) - 1):
        mask_bin = np.logical_and(
            df['truthHitAssignedEnergies'] >= bins[i],
            df['truthHitAssignedEnergies'] < bins[i + 1])
        mask_bin = np.logical_and(mask_bin, mask)
        uncertainty = df['pred_energy_high_quantile'][mask_bin]
        mean.append(np.mean(uncertainty))
        std.append(np.std(uncertainty))
    mean = np.array(mean)
    std = np.array(std)
    return mean, std


def within_uncertainty(df, bins=None, binwidth=10.):
    if bins is None:
        binmax = df['truthHitAssignedEnergies'].max()
        bins = calc_energy_bins(binmax, binwidth)

    within_1sigma, within_2sigma, within_3sigma = calc_within_uncertainty(df, bins)
    mean, std = bin_uncertainty(df, bins)

    fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
    x = (bins[1:] + bins[:-1]) / 2
    xerr = (bins[1:] - bins[:-1]) / 2
    ax[0].errorbar(x, within_1sigma, xerr=xerr, fmt='o', label='within 1 sigma')
    ax[0].errorbar(x, within_2sigma, xerr=xerr, fmt='o', label='within 2 sigma')
    ax[0].errorbar(x, within_3sigma, xerr=xerr, fmt='o', label='within 3 sigma')
    ax[1].errorbar(x, mean, xerr=xerr, yerr=std, fmt='o', label='uncertainty')
    ax[0].legend()
    ax[1].legend()
    return fig


def energy_resolution(df, bins=None, binwidth=10., addfit=False):
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


def dictlist_to_dataframe(dictlist, masks=None, add_event_id=True):
    if isinstance(dictlist, dict):
        dictlist = [dictlist]
    full_df = pd.DataFrame()
    for i in range(len(dictlist)):
        df = pd.DataFrame()
        for key, value in dictlist[i].items():
            # print(key)
            if key in ['row_splits', 'recHitXY']:
                continue
            if len(value.shape) == 1:
                continue
            if value.shape[1] > 1:
                for j in range(value.shape[1]):
                    df[key + '_' + str(j)] = value[:, j]
            else:
                df[key] = value[:, 0]
        if add_event_id:
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


def tracks_vs_hits(showers_dataframe):
    bins = calc_energy_bins(200, 10)
    _, ratios_track_raw, = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_tracks_raw')
    medians_track_raw = [np.median(r) for r in ratios_track_raw]

    _, ratios_track = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_tracks')
    medians_track = [np.median(r) for r in ratios_track]

    _, ratios_hits_raw = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_hits_raw')
    medians_hits_raw = [np.median(r) for r in ratios_hits_raw]

    _, ratios_hits = calc_resolution(
        showers_dataframe, bins, predstring='pred_energy_hits')
    medians_hits = [np.median(r) for r in ratios_hits]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bins[:-1], medians_track_raw, '--', color='red', lw=3,
            label='Median of Ratio, Tracks Raw')
    ax.plot(bins[:-1], medians_track, color='red',
            label='Median of Ratio, Tracks Corrected')
    ax.plot(bins[:-1], medians_hits_raw, '--', color='blue', lw=3,
            label='Median of Ratio, Hits Raw')
    ax.plot(bins[:-1], medians_hits, color='blue', lw=3,
            label='Median of Ratio, Hits Corrected')
    ax.legend()
    ax.grid()
    ax.set_title("Median of Ratios 'True' / Predicted'", fontsize=20)
    ax.set_xlabel("True Energy [GeV]", fontsize=20)
    ax.set_ylabel("Ratio", fontsize=20)

    return fig


def prediction_overview(prediction_dictlist):
    prediction = dictlist_to_dataframe(prediction_dictlist)
    fig, ax = plt.subplots(nrows=7, ncols=3, figsize=(40, 50))
    ax = ax.flatten()
    skip = ['row_splits']
    # print(prediction.keys())
    for i, key in enumerate(prediction.keys()):
        if key in skip:
            continue
        if i == len(ax):
            print("Not enough space in histogram")
            break
        N = len(prediction[key])
        ax[i].set_yscale('log')
        ax[i].hist(prediction[key], bins=100)
        ax[i].set_title(f"{key} - {N} entries", fontsize=20)
    fig.tight_layout()
    return fig


def noise_performance(noise_df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set_title("Noise Filter", fontsize=20)
    noise_reduction_hit = 1. - noise_df.n_noise_filtered / noise_df.n_noise_orig
    noise_reduction_energy = 1. - noise_df.e_noise_filtered / noise_df.e_noise_orig
    hit_retention_hit = noise_df.n_hits_filtered / noise_df.n_hits_orig
    hit_retention_energy = noise_df.e_hits_filtered / noise_df.e_hits_orig
    # make boxplot of noise_reduction_hit, noise_reduction_energy, hit_retention_hit, hit_retention_energy
    ax.boxplot(
        [noise_reduction_hit, noise_reduction_energy, hit_retention_hit, hit_retention_energy],
        labels=['Noise Reduction (Hits)', 'Noise Reduction (Energy)', 'Hit Retention (Hits)', 'Hit Retention (Energy)'])
    ax.set_ylabel("Fraction", fontsize=40)
    # set label size to 20
    ax.tick_params(axis='both', which='major', labelsize=20)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    return fig


def map_pid_to_classes(truth_pids):
    """
    0.  Muon
    1.  Electron
    2.  Photon
    3.  Charged Hadron
    4.  Neutral Hadron
    5.  Ambiguous
    """
    map_dict = {
        13: 0,
        -13: 0,
        11: 1,
        -11: 1,
        22: 2,
        211: 3,
        -211: 3,
        312: 3,
        -312: 3,
        2212: 3,
        -2212: 3,
        130: 4,
        2112: 4,
    }

    mapped = truth_pids.map(map_dict)
    return mapped


def classification_plot(showers_df):
    has_pred = np.logical_not(showers_df.pred_pos.isna())
    has_truth = np.logical_not(showers_df.truth_mean_x.isna())
    matched = showers_df[np.logical_and(has_pred, has_truth)]
    matched_truthPID = matched.truthHitAssignedPIDs
    matched_predPID = matched.pred_id
    mapped_truthClasses = map_pid_to_classes(matched_truthPID)

    confusion_matrix(mapped_truthClasses, matched_predPID, labels=[0,1,2,3,4,5])

    classes = [
        "Muon",
        "Electron",
        "Photon",
        "Charged Hadron",
        "Neutral Hadron",
        "Ambiguous",
    ]

    df_cm = pd.DataFrame(
        confusion_matrix(mapped_truthClasses, matched_predPID, labels=[0,1,2,3,4,5], normalize=None),
        index = classes,
        columns = classes)

    fig = plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel(r'Predicted label', fontsize=20)
    plt.show()

    return fig


def plot_high_low_difference(prediction):
    prediction = dictlist_to_dataframe(prediction)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    low = prediction.pred_energy_low_quantile
    high = prediction.pred_energy_high_quantile
    distance = high - low
    ax.hist(distance, bins=100)
    ax.set_title("Difference between high and low quantile", fontsize=20)
    return fig
