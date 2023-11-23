import gzip
import pickle
import fastjet
import awkward as ak
import vector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

vector.register_awkward()
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.6)



def pred_to_4momentum(pred):
    pred_energy = pred['pred_energy_hits'].values
    pred_x = pred['pred_mean_x'].values
    pred_y = pred['pred_mean_y'].values
    pred_z = pred['pred_mean_z'].values
    nan_mask = np.isnan(pred_x)
    if np.any(nan_mask):
        pred_x = pred_x[~nan_mask]
        pred_y = pred_y[~nan_mask]
        pred_z = pred_z[~nan_mask]
        e_lost = np.sum(pred_energy[nan_mask])
        pred_energy = pred_energy[~nan_mask]
        print(f"Found {np.sum(nan_mask)} NaNs in predicted mean values")
        print(f"Energy: {e_lost}")
    pred_px = pred_x * pred_energy / np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    pred_py = pred_y * pred_energy / np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    pred_pz = pred_z * pred_energy / np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    # use px, py, pz and e to create an awkard Momentum4D array
    pred_momentum = ak.zip(
        {
            "px": pred_px,
            "py": pred_py,
            "pz": pred_pz,
            "E": pred_energy,
        },
        with_name="Momentum4D",
    )
    return pred_momentum


def truth_to_4momentum(truth):
    truth_energy = truth['truthHitAssignedEnergies'].values
    truth_x = truth['truthHitAssignedX'].values
    truth_y = truth['truthHitAssignedY'].values
    truth_z = truth['truthHitAssignedZ'].values
    truth_px = truth_x * truth_energy / np.sqrt(truth_x**2 + truth_y**2 + truth_z**2)
    truth_py = truth_y * truth_energy / np.sqrt(truth_x**2 + truth_y**2 + truth_z**2)
    truth_pz = truth_z * truth_energy / np.sqrt(truth_x**2 + truth_y**2 + truth_z**2)
    truth_momentum = ak.zip(
        {
            "px": truth_px,
            "py": truth_py,
            "pz": truth_pz,
            "E": truth_energy,
        },
        with_name="Momentum4D",
    )
    return truth_momentum


def awkardjets_to_pandas(jets):
    jets_df = pd.DataFrame()
    jets_df['px'] = [jet.px for jet in jets]
    jets_df['py'] = [jet.py for jet in jets]
    jets_df['pz'] = [jet.pz for jet in jets]
    jets_df['E'] = [jet.E for jet in jets]
    jets_df['pt'] = [jet.pt for jet in jets]
    jets_df['eta'] = [jet.eta for jet in jets]
    jets_df['phi'] = [jet.phi for jet in jets]
    jets_df['mass'] = [jet.mass for jet in jets]
    # sort by E
    jets_df = jets_df.sort_values(by=['E'], ascending=False)
    return jets_df


def get_jets_from_event(showers_df, i=0, R=0.4, verbose=False, filter_pu=False):
    truth_df = showers_df[~np.isnan(showers_df['truthHitAssignedEnergies'])]
    pred_df = showers_df[~np.isnan(showers_df['pred_energy_hits'])]
    truth_i = truth_df[truth_df['event_id'] == i]
    pred_i = pred_df[pred_df['event_id'] == i]
    if filter_pu:
        truth_i = truth_i[truth_i.t_only_minbias == 0]
        pred_i = pred_i[pred_i.t_only_minbias == 0]

    pred_4momentum = pred_to_4momentum(pred_i)
    truth_4momentum = truth_to_4momentum(truth_i)

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    cluster_truth = fastjet.ClusterSequence(truth_4momentum, jetdef)
    cluster_pred = fastjet.ClusterSequence(pred_4momentum, jetdef)

    truth_jets = cluster_truth.inclusive_jets()
    pred_jets = cluster_pred.inclusive_jets()

    truth_jets_df = awkardjets_to_pandas(truth_jets)
    pred_jets_df = awkardjets_to_pandas(pred_jets)
    if verbose:
        print(f"Event {i} using R={R}")
        print(f"Truth: found {len(truth_jets)} jets")
        print(f"Prediction: found {len(pred_jets)} jets")
        print()

    return truth_jets_df, pred_jets_df


def calculate_matched_df(truth_jets_df, pred_jets_df, df=None,
                         R_matching_cut=0.3, pt_matching_cut=0.5, pt_cut=0.,
                         event_id=0, verbose=False):
    matched_df = pd.DataFrame()
    matched_predicted_ids = []
    keys = ['px', 'py', 'pz', 'E', 'pt', 'eta', 'phi', 'mass']

    # apply pt cut
    truth_jets_df = truth_jets_df[truth_jets_df.pt > pt_cut]
    pred_jets_df = pred_jets_df[pred_jets_df.pt > pt_cut]

    for i in range(len(truth_jets_df)):
        t_jet = truth_jets_df.iloc[i]
        delta_R = np.sqrt((pred_jets_df.eta - t_jet.eta)**2 + (pred_jets_df.phi - t_jet.phi)**2)
        ratio_pt = pred_jets_df.pt / t_jet.pt
        mask_R = delta_R < R_matching_cut
        mask_pt = ratio_pt > pt_matching_cut
        not_yet_matched_mask = ~pred_jets_df.index.isin(matched_predicted_ids)
        mask = mask_R & mask_pt & not_yet_matched_mask
        has_match = np.any(mask)
        d  = {}
        # for key in t_jet.keys():
        for key in keys:
            d[f"truth_{key}"] = t_jet[key]
        if has_match:
            # matched_index = np.argmin(delta_R[mask])
            matched_index = np.argmin(np.where(mask, delta_R, 100.))
            if verbose:
                print(f"Truth shower {i} matched to pred shower {matched_index}")
                print(matched_index)
            matched_predicted_ids.append(matched_index)
            for key in keys:
                d[f"pred_{key}"] = pred_jets_df.iloc[matched_index][key]
        else:
            if verbose:
                print(f"Truth shower {i} not matched")
            for key in keys:
                d[f"pred_{key}"] = np.nan
        matched_df = matched_df.append(d, ignore_index=True)

    # not_matched_indices = pred_jets_df.index[pred_jets_df.index.isin(matched_predicted_ids)]
    not_matched_indices = [i for i in range(len(pred_jets_df)) if i not in matched_predicted_ids]

    for nim in not_matched_indices:
        if verbose:
            print(f"Pred shower {nim} not matched")
        d = {}
        for key in keys:
            d[f"truth_{key}"] = np.nan
        for key in keys:
            d[f"pred_{key}"] = pred_jets_df.iloc[nim][key]
        matched_df = matched_df.append(d, ignore_index=True)
    matched_df['event_id'] = event_id

    if df is None:
        return matched_df
    else: # concatenate to df
        return pd.concat([df, matched_df], ignore_index=True)


def matched_df_loop(showers_df, n_max=-1):

    if n_max == -1:
        n_events = np.max(showers_df['event_id'])
    else:
        n_events = n_max

    matched_df = None
    for i in range(n_events):
        if i % 100 == 0:
            print(f"Event {i} / {n_events}")
        truth_jets_df, pred_jets_df = get_jets_from_event(showers_df, i=i, R=0.4)
        matched_df = calculate_matched_df(truth_jets_df, pred_jets_df, df=matched_df, event_id=i)
    return matched_df


def jet_efficiency_plot(matched_df, binwidth=20, pt_min=0, title=None, return_figure=True):
    pt_bin_mins = np.arange(pt_min, 150, binwidth)

    efficiencies = []
    eff_errors = []
    fake_rates = []
    fake_errors = []
    for pt_bin in pt_bin_mins:
        truth_pt_mask = (matched_df.truth_pt >= pt_bin) & (matched_df.truth_pt < pt_bin + binwidth)
        n_truth = np.sum(truth_pt_mask)
        n_matched = np.sum(truth_pt_mask & ~matched_df.pred_px.isna())
        efficiency = n_matched / n_truth
        efficiencies.append(efficiency)
        eff_errors.append(np.sqrt(efficiency * (1 - efficiency) / n_truth))

        pred_pt_mask = (matched_df.pred_pt >= pt_bin) & (matched_df.pred_pt < pt_bin + binwidth)
        n_pred = np.sum(pred_pt_mask)
        n_fake = np.sum(pred_pt_mask & matched_df.truth_px.isna())
        fake_rate = n_fake / n_pred
        fake_error = np.sqrt(fake_rate * (1 - fake_rate) / n_pred)
        fake_rates.append(fake_rate)
        fake_errors.append(fake_error)

    efficiencies = np.array(efficiencies)
    eff_errors = np.array(eff_errors)
    fake_rates = np.array(fake_rates)
    fake_errors = np.array(fake_errors)
    pt_centers = pt_bin_mins + binwidth / 2
    pt_error = binwidth / 2
    data = {
        'pt_centers': pt_centers,
        'pt_error': pt_error,
        'efficiencies': efficiencies,
        'eff_errors': eff_errors,
        'fake_rates': fake_rates,
        'fake_errors': fake_errors,
    }
    if not return_figure:
        return data

    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        ax.errorbar(pt_centers, efficiencies, xerr=pt_error, yerr=eff_errors, fmt='o', color='blue', label='Efficiency')
        ax.errorbar(pt_centers, fake_rates, xerr=pt_error, yerr=fake_errors, fmt='o', color='red', label='Fake rate')

        ax.set_xlim((0, np.max(pt_bin_mins) + 2 * binwidth))
        ax.set_xticks(pt_bin_mins + binwidth / 2)
        ax.set_xticklabels(np.array(pt_bin_mins + binwidth / 2).astype(int))
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_xlabel('Jet $p_T$ [GeV]', fontsize=30)

        ax.set_ylim((-0.1, 1.1))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        # set yticklabels as percentage
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

        if title is None:
            title = f"Jet efficiency and fake rate"
        ax.set_title(title, fontsize=40)
        ax.legend(fontsize=20, loc='best')
        ax.grid()
        fig.tight_layout()

        return fig




def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def jet_resolution_hists(matched_df, binwidth=20, pt_min=10, title=None, return_figure=True):
    pt_bin_mins = np.arange(pt_min, 150, binwidth)

    n_plots = len(pt_bin_mins)
    n_rows = int(np.sqrt(n_plots))
    n_cols = int(np.ceil(n_plots / n_rows))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 5*n_rows))
    ax = ax.flatten()

    responses = []
    response_errors = []
    resolutions = []
    resolution_errors = []
    for i, pt_bin in enumerate(pt_bin_mins):
        truth_pt_mask = (matched_df.truth_pt >= pt_bin) & (matched_df.truth_pt < pt_bin + binwidth)
        matched_mask = truth_pt_mask & ~matched_df.pred_px.isna()

        pt_true = matched_df[matched_mask].truth_pt
        pt_pred = matched_df[matched_mask].pred_pt

        ax[i].set_title(f"Truth $p_T \; \in$ ({pt_bin}, {pt_bin + binwidth}) GeV")
        if i % n_cols == 0:
            ax[i].set_ylabel('Number of jets', fontsize=20)
        n, bins, patches = ax[i].hist(pt_pred / pt_true, range=(0, 2))

        bin_centers = (bins[1:] + bins[:-1]) / 2
        has_fit = False
        try:
            popt, pcov = curve_fit(gauss, bin_centers, n, p0=[1, 1, 1])
            has_fit = True
        except RuntimeError:
            popt = [1, 1, 1]



        if has_fit:
            ax[i].plot(bin_centers, gauss(bin_centers, *popt), 'r-', label='fit')
            ax[i].text(0.05, 0.95, f"$\mu = {popt[1]:.2f}$\n$\sigma = {popt[2]:.2f}$")

            responses.append(popt[1])
            response_errors.append(np.sqrt(pcov[1, 1]))
            resolutions.append(popt[2])
            resolution_errors.append(np.sqrt(pcov[2, 2]))
        else:
            ax[i].text(0.05, 0.95, f"Fit failed")
            responses.append(np.nan)
            response_errors.append(np.nan)
            resolutions.append(np.nan)
            resolution_errors.append(np.nan)


        vals = ax[i].get_yticks()
        ax[i].set_yticklabels(['{:,.0f}'.format(x) for x in vals])

    fig.tight_layout()

    if return_figure:
        return fig
    else:
        plt.close(fig)
        data = {
            'pt_bin_centers': np.abs(pt_bin_mins + binwidth / 2),
            'responses': np.abs(responses),
            'response_errors': np.abs(response_errors),
            'resolutions': np.abs(resolutions),
            'resolution_errors': np.abs(resolution_errors),
        }
        return pd.DataFrame(data)


def response_resolution_plot(df, fig=None, ax=None, color='blue', label=None):
    columns = ['pt_bin_centers', 'responses', 'response_errors', 'resolutions', 'resolution_errors']
    assert all([c in df.columns for c in columns])
    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    # plot responses in first subplot, deal with nan values
    binwidth = df.pt_bin_centers[1] - df.pt_bin_centers[0]
    ax[0].errorbar(
        df.pt_bin_centers,
        df.responses,
        xerr=binwidth / 2,
        yerr=df.response_errors,
        fmt='o',
        color=color,
        label=label)

    ax[0].set_xlim((0, np.max(df.pt_bin_centers) + 2 * binwidth))
    diff = np.max(np.abs(df.responses - 1)) + np.max(df.response_errors)
    ax[0].set_ylim((1 - 1.1 * diff, 1 + 1.1 * diff))
    ax[0].axhline(y=1, color='black', linestyle='--')

    ax[1].errorbar(
        df.pt_bin_centers,
        df.resolutions,
        xerr=binwidth / 2,
        yerr=df.resolution_errors,
        fmt='o',
        color=color,
        label=label)
    diff = np.max(df.resolutions) + np.max(df.resolution_errors)
    ax[1].set_ylim((0, 1.1 * diff))
    ax[1].set_xlim((0, np.max(df.pt_bin_centers) + 2 * binwidth))

    for a in ax:
        a.grid()
        a.tick_params(axis='both', which='major', labelsize=30)
        a.legend(fontsize=20, loc='best')

    return fig, ax


if __name__ == "__main__":

    import os
    import sys


    path = sys.argv[1]
    dirname = os.path.dirname(path)

    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    showers_df = data['showers_dataframe']
    truth_jets_df, pred_jets_df = get_jets_from_event(showers_df, i=0, R=0.4)

    matched_df = matched_df_loop(showers_df, n_max=100)
    # save matched_df
    matched_df.to_pickle(os.path.join(dirname, 'matched_df.pkl'))

    try:
        fig = jet_efficiency_plot(matched_df, binwidth=20, pt_min=0)
        fig.savefig(os.path.join(dirname, 'jet_efficiency.png'))
        fig = jet_resolution_hists(matched_df, binwidth=25, pt_min=0, return_figure=True)
        fig.savefig(os.path.join(dirname, 'jet_resolution.png'))

        data_true = jet_resolution_hists(matched_df, binwidth=30, pt_min=20, return_figure=False, use_truth=True)
        data_pred = jet_resolution_hists(matched_df, binwidth=30, pt_min=20, return_figure=False, use_truth=False)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
        fig, ax = response_resolution_plot(data_pred, fig=fig, ax=ax, color='blue', label='Prediction')
        fig, ax = response_resolution_plot(data_true, fig=fig, ax=ax, color='green', label='Truth')
        fig.savefig(os.path.join(dirname, 'response_resolution.png'))
    except Exception as e:
        print(e)
        print("Could not make plots")
