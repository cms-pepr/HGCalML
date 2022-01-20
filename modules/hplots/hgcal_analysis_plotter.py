import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hplots.general_2d_plot_extensions_2 import EfficiencyFakeRatePlot, ResponsePlot, ResolutionPlot
from hplots.general_hist_extensions import ResponseHisto, Multi4HistEnergy

def eta_transform(eta):
    eta = np.abs(eta)
    eta[eta > 3] = 3.01
    return eta

class HGCalAnalysisPlotter:
    def __init__(self):
        self.energy_bins = np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200])
        self.local_shower_fraction_bins = np.array([0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        self.eta_bins = np.array([1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.25,2.5,2.75,3,3.1])
        self.pt_bins = np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80])

    def set_energy_bins(self, energy_bins):
        self.energy_bins = energy_bins

    def set_local_shower_fraction_bins(self, local_shower_fraction_bins):
        self.local_shower_fraction_bins = local_shower_fraction_bins

    def set_eta_bins(self, eta_bins):
        self.eta_bins = eta_bins

    def set_pt_bins(self, pt_bins):
        self.pt_bins = pt_bins


    def _make_pdfs(self, ):
        if os.path.exists(self.pdf_path):
            if os.path.isdir(self.pdf_path):
                shutil.rmtree(self.pdf_path)
            else:
                os.unlink(self.pdf_path)

        os.mkdir(self.pdf_path)

        self.pdf_efficiency = PdfPages(os.path.join(self.pdf_path,'efficiency.pdf'))
        self.pdf_response = PdfPages(os.path.join(self.pdf_path,'response.pdf'))
        # self.pdf_pid = PdfPages(os.path.join(self.pdf_path,'pid.pdf'))
        self.pdf_fake_rate = PdfPages(os.path.join(self.pdf_path,'fake_rate.pdf'))
        self.pdf_others = PdfPages(os.path.join(self.pdf_path,'others.pdf'))
        self.pdf_resolution = PdfPages(os.path.join(self.pdf_path,'resolution.pdf'))
        self.pdf_response_histos = PdfPages(os.path.join(self.pdf_path,'response_histos.pdf'))

    def _close_pdfs(self):
        self.pdf_efficiency.close()
        self.pdf_response.close()
        self.pdf_fake_rate.close()
        self.pdf_others.close()
        # self.pdf_pid.close()
        self.pdf_resolution.close()
        self.pdf_response_histos.close()
        plt.close('all')

    def _make_efficiency_plots(self):
        filter_has_truth = self.showers_dataframe['truthHitAssignementIdx'].notnull()
        found = self.showers_dataframe['pred_sid'][filter_has_truth].notnull().to_numpy()

        # Efficiency fo true energy
        plot = EfficiencyFakeRatePlot(bins=self.energy_bins, x_label='True Energy [GeV]', y_label='Efficiency')
        plot.add_raw_values(self.showers_dataframe['truthHitAssignedEnergies'][filter_has_truth].to_numpy(), found)
        self.pdf_efficiency.savefig(plot.draw())

        # Efficiency fo pT
        plot = EfficiencyFakeRatePlot(bins=self.pt_bins, x_label='pT [GeV]', y_label='Efficiency')
        plot.add_raw_values(self.showers_dataframe['truth_pt'][filter_has_truth].to_numpy(), found)
        self.pdf_efficiency.savefig(plot.draw())

        # Efficiency fo local shower energy fraction
        plot = EfficiencyFakeRatePlot(bins=self.local_shower_fraction_bins, x_label='Local shower energy fraction', y_label='Efficiency')
        x = self.showers_dataframe['truth_local_shower_energy_fraction'][filter_has_truth].to_numpy()
        plot.add_raw_values(self.showers_dataframe['truth_local_shower_energy_fraction'][filter_has_truth].to_numpy(), found)
        self.pdf_efficiency.savefig(plot.draw())

        # Efficiency fo eta
        plot = EfficiencyFakeRatePlot(bins=self.eta_bins, x_label='$|\\eta_{true}|$', y_label='Efficiency')
        plot.add_raw_values(eta_transform(self.showers_dataframe['truthHitAssignedEta'][filter_has_truth].to_numpy()), found)
        self.pdf_efficiency.savefig(plot.draw())

    def _make_fake_rate_plots(self):
        filter_has_pred = self.showers_dataframe['pred_sid'].notnull()
        fake = np.logical_not(self.showers_dataframe['truthHitAssignementIdx'][filter_has_pred].notnull().to_numpy())

        # Efficiency fo true energy
        plot = EfficiencyFakeRatePlot(bins=self.energy_bins, x_label='Pred Energy [GeV]', y_label='Fake rate')
        plot.add_raw_values(self.showers_dataframe['pred_energy'][filter_has_pred].to_numpy(), fake)
        self.pdf_fake_rate.savefig(plot.draw())

        # Fake rate fo pT
        plot = EfficiencyFakeRatePlot(bins=self.pt_bins, x_label='pT [GeV]', y_label='Fake rate')
        plot.add_raw_values(self.showers_dataframe['pred_energy'][filter_has_pred].to_numpy(), fake)
        self.pdf_fake_rate.savefig(plot.draw())

        # Fake rate fo eta
        plot = EfficiencyFakeRatePlot(bins=self.eta_bins, x_label='$|\\eta_{true}|$', y_label='Fake rate')
        plot.add_raw_values(
            eta_transform(self.showers_dataframe['pred_energy'][filter_has_pred].to_numpy()), fake)
        self.pdf_fake_rate.savefig(plot.draw())

    def _make_resolution_plots(self):
        filter_has_truth = self.showers_dataframe['truthHitAssignementIdx'].notnull()
        filter_has_pred = self.showers_dataframe['pred_sid'].notnull()
        filter = np.logical_and(filter_has_truth, filter_has_pred)
        response = self.showers_dataframe['truthHitAssignedEnergies'][filter].to_numpy() \
                   / self.showers_dataframe['pred_energy'][filter].to_numpy()

        # Resolution fo true energy
        plot = ResolutionPlot(bins=self.energy_bins, x_label='True Energy [GeV]', y_label='Resolution')
        plot.add_raw_values(self.showers_dataframe['truthHitAssignedEnergies'][filter].to_numpy(), response)
        self.pdf_resolution.savefig(plot.draw())

        # Resolution fo pT
        plot = ResolutionPlot(bins=self.pt_bins, x_label='pT [GeV]', y_label='Resolution')
        plot.add_raw_values(self.showers_dataframe['truth_pt'][filter].to_numpy(), response)
        self.pdf_resolution.savefig(plot.draw())

        # Resolution fo local shower energy fraction
        plot = ResolutionPlot(bins=self.local_shower_fraction_bins, x_label='Local shower energy fraction',
                                      y_label='Resolution')
        plot.add_raw_values(self.showers_dataframe['truth_local_shower_energy_fraction'][filter].to_numpy(),
                            response)
        self.pdf_resolution.savefig(plot.draw())

        # Resolution fo e other
        plot = ResolutionPlot(bins=self.energy_bins, x_label='$E_{other}$ [GeV]', y_label='Resolution')
        plot.add_raw_values(self.showers_dataframe['truth_e_other'][filter].to_numpy(),
                            response)
        self.pdf_resolution.savefig(plot.draw())

    def _make_response_plots(self):
        filter_has_truth = self.showers_dataframe['truthHitAssignementIdx'].notnull()
        filter_has_pred = self.showers_dataframe['pred_sid'].notnull()
        filter = np.logical_and(filter_has_truth, filter_has_pred)
        response = self.showers_dataframe['truthHitAssignedEnergies'][filter].to_numpy() \
                   / self.showers_dataframe['pred_energy'][filter].to_numpy()

        # Response fo true energy
        plot = ResponsePlot(bins=self.energy_bins, x_label='True Energy [GeV]', y_label='Response')
        plot.add_raw_values(self.showers_dataframe['truthHitAssignedEnergies'][filter].to_numpy(), response)
        self.pdf_response.savefig(plot.draw())

        # Response fo pT
        plot = ResponsePlot(bins=self.pt_bins, x_label='pT [GeV]', y_label='Response')
        plot.add_raw_values(self.showers_dataframe['truth_pt'][filter].to_numpy(), response)
        self.pdf_response.savefig(plot.draw())

        # Response fo local shower energy fraction
        plot = ResponsePlot(bins=self.local_shower_fraction_bins, x_label='Local shower energy fraction',
                                      y_label='Response')
        plot.add_raw_values(self.showers_dataframe['truth_local_shower_energy_fraction'][filter].to_numpy(),
                            response)
        self.pdf_response.savefig(plot.draw())

        # Response fo eta
        plot = ResponsePlot(bins=self.eta_bins, x_label='$|\\eta_{true}|$', y_label='Response')
        plot.add_raw_values(eta_transform(self.showers_dataframe['truthHitAssignedEta'][filter].to_numpy()),
                            response)
        self.pdf_response.savefig(plot.draw())

        # Response fo e other
        plot = ResponsePlot(bins=self.energy_bins, x_label='$E_{other}$ [GeV]', y_label='Response')
        plot.add_raw_values(self.showers_dataframe['truth_e_other'][filter].to_numpy(),
                            response)
        self.pdf_response.savefig(plot.draw())

    def _write_scalar_properties(self):
        if self.scalar_variables is None:
            return
        text_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal',
                     'verticalalignment': 'bottom'}
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_visible(False)
        ax.axis('off')
        s = ''
        for k,v in self.scalar_variables.items():
            s += '%s = %s\n'%(k,v)
        fig.text(0, 1, s, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                 fontdict=text_font)
        self.pdf_others.savefig(fig)


    def _make_response_histograms(self):
        filter_has_truth = self.showers_dataframe['truthHitAssignementIdx'].notnull()
        filter_has_pred = self.showers_dataframe['pred_sid'].notnull()
        filter = np.logical_and(filter_has_truth, filter_has_pred)
        response = self.showers_dataframe['truthHitAssignedEnergies'][filter].to_numpy() \
                   / self.showers_dataframe['pred_energy'][filter].to_numpy()

        # Response histo
        plot = ResponseHisto()
        plot.add_raw_values(response)
        self.pdf_response_histos.savefig(plot.draw())

        plot = Multi4HistEnergy()
        plot.add_raw_values(self.showers_dataframe['truthHitAssignedEnergies'][filter].to_numpy(), response)
        self.pdf_response_histos.savefig(plot.draw())



    def set_data(self, showers_dataframe, events_dataframe, model_name, pdf_path, scalar_variables=None):
        self.pdf_path = pdf_path
        self.showers_dataframe = showers_dataframe
        self.events_dataframe = events_dataframe
        self.model_name = model_name
        self.scalar_variables = scalar_variables

    def _add_additional_columns(self):
        filter_has_truth = self.showers_dataframe['truthHitAssignementIdx'].notnull().to_numpy()
        true_energy = self.showers_dataframe['truthHitAssignedEnergies'][filter_has_truth].to_numpy()
        phi = self.showers_dataframe['truthHitAssignedPhi'][filter_has_truth].to_numpy()
        eta = self.showers_dataframe['truthHitAssignedEta'][filter_has_truth].to_numpy()
        pT = true_energy / eta
        # found = self.showers_dataframe['pred_sid'][filter_has_truth].notnull().to_numpy()

        _pT = np.zeros(filter_has_truth.shape, np.float)*np.NAN
        _pT[filter_has_truth] = pT
        self.showers_dataframe['truth_pt'] = _pT

        local_shower_energy_fraction = []
        e_other_event = []


        event_ids = self.showers_dataframe['event_id'][filter_has_truth].to_numpy()
        unique_events = np.unique(event_ids)
        for u in unique_events:
            event_filter = event_ids == u
            energy_event = true_energy[event_filter]

            eta_event = eta[event_filter]
            phi_event = phi[event_filter]
            d_eta_phi = np.sqrt((eta_event[..., np.newaxis] - eta_event[np.newaxis, ...]) ** 2 + (
                        phi_event[..., np.newaxis] - phi_event[np.newaxis, ...]) ** 2)
            lsf = energy_event / np.sum(np.less_equal(d_eta_phi, 0.5) * energy_event[np.newaxis, ...], axis=1)
            e_other = np.sum(np.less_equal(d_eta_phi, 0.3) * (1 - np.eye(N=len(d_eta_phi))) * energy_event[np.newaxis, ...],
                             axis=1)
            local_shower_energy_fraction += lsf.tolist()
            e_other_event += e_other.tolist()

        _local_shower_energy_fraction = np.zeros(filter_has_truth.shape, np.float)*np.NAN
        _local_shower_energy_fraction[filter_has_truth] = np.array(local_shower_energy_fraction)
        self.showers_dataframe['truth_local_shower_energy_fraction'] = _local_shower_energy_fraction

        _truth_e_other = np.zeros(filter_has_truth.shape, np.float)*np.NAN
        _truth_e_other[filter_has_truth] = np.array(e_other_event)
        self.showers_dataframe['truth_e_other'] = _truth_e_other


    def process(self):
        self._make_pdfs()

        self._add_additional_columns()
        self._write_scalar_properties()
        self._make_efficiency_plots()
        self._make_fake_rate_plots()
        self._make_response_plots()
        self._make_resolution_plots()
        self._make_response_histograms()

        self._close_pdfs()


