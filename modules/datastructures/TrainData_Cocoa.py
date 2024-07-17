from datastructures.TrainData_NanoML import TrainData_NanoML
import uproot
import pandas as pd
import awkward as ak
import numpy as np
from DeepJetCore import SimpleArray


class TrainData_Cocoa(TrainData_NanoML):

    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        #open File
        with uproot.open(filename) as file:
            events = file[file.keys()[0]]
        data = events.arrays(library='ak')#[0:10]#JUST FOR TESTING REMOVE SLICING LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #convert data to training data
        trainData,rs = self.converttotrainingdfvec(data)

        #make simpleArray features
        features = trainData[['recHitEnergy', 'recHitEta', 'isTrack','recHitTheta', 'recHitR', 'recHitX', 'recHitY', 'recHitZ', 'recHitTime', 'recHitHitR']]
        farr = SimpleArray(np.float32(features.to_numpy()),rs,name="recHitFeatures")

        #make simpleArray truth
        t={}
        #First the float32 fields
        for field in ['t_energy','t_spectator', 't_time','t_rec_energy']:         
            t[field] = SimpleArray(np.float32(trainData[field].to_numpy().reshape(-1, 1)),rs,name=field)
        #Then the int32 fields
        for field in ['t_idx', 't_pid',  't_fully_contained', 't_is_unique']:         
            t[field] = SimpleArray(np.int32(trainData[field].to_numpy().reshape(-1, 1)),rs,name=field)
        #Then the pos field
        t['t_pos'] = SimpleArray(np.float32(np.array(trainData['t_pos'].tolist())),rs,name='t_pos')

        #return result
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'],
                t['t_rec_energy'], t['t_is_unique'] ],[], []

    def convertevent(self, event):
        #convert cells to df
        df_cell = pd.DataFrame()
        df_cell['recHitEnergy'] =  ak.to_dataframe(event['cell_e'], how='outer')
        df_cell['recHitEta'] = ak.to_dataframe(event['cell_eta'], how='outer')
        df_cell['isTrack'] = 0
        df_cell['recHitTheta'] =ak.to_dataframe(event['cell_phi'], how='outer')
        df_cell['recHitR'] = 0 #will be calculated later
        df_cell['recHitX']= ak.to_dataframe(event['cell_x'], how='outer')
        df_cell['recHitY']= ak.to_dataframe(event['cell_y'], how='outer')
        df_cell['recHitZ']= ak.to_dataframe(event['cell_z'], how='outer')
        df_cell['recHitTime'] = 0
        df_cell['recHitHitR'] = 0

        df_cell['t_idx'] = ak.to_dataframe(event['cell_parent_idx'], how='outer')

        #Calculate Distance from 0,0,0
        df_cell['recHitR']= np.sqrt(df_cell['recHitX']**2+df_cell['recHitY']**2+df_cell['recHitZ']**2)


        #convert particle information to df
        df_particle= pd.DataFrame()

        df_particle['particle_idx'] = np.arange(len(event['particle_e']))
        df_particle['t_energy'] = ak.to_dataframe(event['particle_e'], how='outer')
        df_particle['t_time'] = 0
        df_particle['t_pid'] = ak.to_dataframe(event['particle_pdgid'], how='outer')
        df_particle['t_spectator'] = 0
        df_particle['t_fully_contained'] = 1
        df_particle['t_rec_energy'] = ak.to_dataframe(event['particle_dep_energy'], how='outer')
        df_particle['particle_eta'] = ak.to_dataframe(event['particle_eta_lay0'], how='outer')
        df_particle['particle_phi'] = ak.to_dataframe(event['particle_phi_lay0'], how='outer')
        
        #convert track information to df
        df_track = pd.DataFrame()
        df_track['t_idx'] = ak.to_dataframe(event['track_parent_idx'], how='outer')
        df_track['recHitX'] = ak.to_dataframe(event['track_x_layer_0'], how='outer')
        df_track['recHitY'] = ak.to_dataframe(event['track_y_layer_0'], how='outer')
        df_track['recHitZ'] = ak.to_dataframe(event['track_z_layer_0'], how='outer')
        df_track['recHitR']= np.sqrt(df_track['recHitX']**2+df_track['recHitY']**2+df_track['recHitZ']**2)

        df_track['recHitTheta'] =ak.to_dataframe(event['track_theta'], how='outer')
        df_track['recHitEta'] = -np.log(np.tan(df_track['recHitTheta']/2))

        df_track['recHitTime'] = 0
        df_track['recHitHitR'] = 0

        #join track information to particle information
        df_track = df_track.join(df_particle, on='t_idx')
        
        #set is Track to -1 or 1 depending on sign of t_pid
        df_track['isTrack'] = pdgid_to_charge(df_track['t_pid'])
        #raise Error if any istrack is zero
        if 0 in df_track['isTrack'].values:
            raise ValueError("Error in isTrack")
        
        
        #Smear true energy for track
        df_track['recHitEnergy'] = df_track['t_energy'].apply(lambda x: np.random.normal(x, 0.01 * x))

        #join cell information to truth information
        df_cell = df_cell.join(df_particle, on='t_idx')

        df_training = pd.concat([df_cell, df_track])

        #set NaN values (particle information for hits made from background noise to -1)
        df_training.fillna(-1, inplace=True)
        
        #encode pos as cos(phi), sin(phi), eta
        df_training['t_pos'] = df_training.apply(lambda row: [np.cos(row['particle_phi']), np.sin(row['particle_phi']), row['particle_eta']], axis=1)

        #set t_is_unique
        df_training['t_is_unique'] = 0
        first_occurrence_mask = (df_training['t_idx'] != -1) & (df_training.groupby('t_idx').cumcount() == 0)
        df_training.loc[first_occurrence_mask, 't_is_unique'] = 1

        return df_training
    
    def converttotrainingdfvec(self, data):
        #For some reason, there is an outlier event with a track at -1e12, which is not physical
        data = data[ak.all(data.track_x_layer_0 > -2000, axis=1)]
        
        #Convert events one by one
        traindata = np.array([self.convertevent(data[eventnumber]) for eventnumber in np.arange(len(data))])

        #Remove all events with an energy lower than 15GeV
        E_cutoff = 15000
        E = np.sqrt(data['true_jet_pt']**2 * np.cos(data['true_jet_phi'])**2 + data['true_jet_m']**2)
        mask = ak.to_numpy(ak.all(E >= E_cutoff, axis=1))
        traindata = traindata[mask]

        #find the row splits
        rs = np.cumsum([len(df) for df in traindata])
        rs = np.insert(rs, 0, 0)

        #concat all events
        traindata = pd.concat(traindata)
        return traindata, rs
    
#HelpFunction to convert PDGIDs to the charge of the particle
def pdgid_to_charge(pdgid):
    charge_dict = {
        11: -1, # electron
        -11: 1, # positron
        13: -1, # muon
        -13: 1, # antimuon
        211: 1, # pi+
        -211: -1, # pi-
        321: 1, # K+
        -321: -1, # K-
        2212: 1, # proton
        -2212: -1, # antiproton
        3112: -1, # Sigma-
        -3112: 1, # antilambda_c+
        3222: 1, # Sigma+
        -3222: -1, # antilambda_c-
        3312: -1, # Xi-
        -3312: 1, # antixi+
    }
    # Default to 0, check if valid later
    return np.vectorize(charge_dict.get)(pdgid, 0)
