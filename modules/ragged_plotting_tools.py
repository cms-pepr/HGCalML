
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from plotting_tools import base_plotter, plotter_3d
from index_dicts import create_index_dict
# from numba import jit
import math
from matplotlib.backends.backend_pdf import PdfPages
from obc_data import convert_dataset_dict_elements_to_numpy
import index_dicts
from matplotlib.patches import Patch


'''
Everything here assumes non flattened format:

B x V x F

'''
# tools before making the ccoords plot working on all events
# @jit(nopython=True)
def c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, in_distance_threshold,
                          n_ccoords):
    
    distance_threshold = in_distance_threshold**2
    for e in range(len(betasel)):
        selected = []
        for si in range(len(sorting[e])):
            i = sorting[e][si]
            use=True
            for s in selected:
                distance=0
                for cci in range(n_ccoords):
                    distance +=  (s[cci]-ccoords[e][i][cci])**2
                if distance  < distance_threshold:
                    use=False
                    break
            if not use:
                betasel[e][i] = False
                continue
            else:
                selected.append(ccoords[e][i])
             
    return betasel
    
def collectoverthresholds(data, 
                          beta_threshold, distance_threshold):
    
    betas   = np.reshape(data['predBeta'], [data['predBeta'].shape[0], -1])
    ccoords = np.reshape(data['predCCoords'], [data['predCCoords'].shape[0], -1, data['predCCoords'].shape[2]])
    
    sorting = np.argsort(-betas, axis=1)
    
    betasel = betas > beta_threshold
    
    
    bsel =  c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold,
                          data['predCCoords'].shape[2]
                          )
    
    
    return np.reshape(bsel , [data['predBeta'].shape[0], data['predBeta'].shape[1], data['predBeta'].shape[2]])

#alredy selected for one event here!


def selectEvent(rs, feat, truth, event):
    rs = np.array(rs , dtype='int')
    rs = rs[:rs[-1]]

    feat = feat[rs[event]:rs[event+1],...]
  
    return feat, truth[rs[event]:rs[event+1],...]

def createRandomizedColors(basemap,seed=0):
    cmap = plt.get_cmap(basemap)
    vals = np.linspace(0,1,256)
    np.random.seed(seed)
    np.random.shuffle(vals)
    return plt.cm.colors.ListedColormap(cmap(vals))


def make_cluster_coordinates_plot(plt, ax, 
                                  truthHitAssignementIdx, #[ V ] or [ V x 1 ]
                                  predBeta,               #[ V ] or [ V x 1 ]
                                  predCCoords,            #[ V x 2 ]
                                  identified_coords=None,
                                  beta_threshold=0.2, distance_threshold=0.8,
                                  cmap=None,
                                  noalpha=False,
                                  direct_color=False
                                ):
    
    #data = create_index_dict(truth,pred,usetf=False)
    
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(predBeta.shape)>1:
        predBeta = np.array(predBeta[:,0])
    
    if np.max(predBeta)>1.:
        raise ValueError("make_cluster_coordinates_plot: at least one beta value is above 1. Check your model!")
    
    if predCCoords.shape[1] == 2:
        ax.set_aspect(aspect=1.)
    #print(truthHitAssignementIdx)
    if cmap is None:
        rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    else:
        rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    rgbcolor[truthHitAssignementIdx<0]=[0.92,0.92,0.92]
    #print(rgbcolor)
    #print(rgbcolor.shape)
    alphas = predBeta
    alphas = np.clip(alphas, a_min=1e-2,a_max=1.-1e-2)
    #alphas = np.arctanh(alphas)/np.arctanh(1.-1e-2)
    #alphas *= alphas
    alphas[alphas<0.01] = 0.01
    alphas = np.expand_dims(alphas, axis=1)
    if noalpha:
        alphas = np.ones_like(alphas)
    
    rgba_cols = np.concatenate([rgbcolor,alphas],axis=-1)
    rgb_cols = np.concatenate([rgbcolor,np.zeros_like(alphas+1.)],axis=-1)
    
    if direct_color:
        rgba_cols = truthHitAssignementIdx
        
    if np.max(rgba_cols) >= 1.:
        rgba_cols /= np.max(rgba_cols)+1e-3

    sorting = np.reshape(np.argsort(alphas, axis=0), [-1])
    
    if predCCoords.shape[1] == 2:
        ax.scatter(predCCoords[:,0][sorting],
                  predCCoords[:,1][sorting],
                  s=.25*matplotlib.rcParams['lines.markersize'] ** 2,
                  c=rgba_cols[sorting])
    elif predCCoords.shape[1] == 3:
        ax.scatter(predCCoords[:,0][sorting],
                  predCCoords[:,1][sorting],
                  predCCoords[:,2][sorting],
                  s=.25*matplotlib.rcParams['lines.markersize'] ** 2,
                  c=rgba_cols[sorting])
        
        
    
    if beta_threshold < 0. or beta_threshold > 1 or distance_threshold<0:
        return
    
    data = {'predBeta': np.expand_dims(np.expand_dims(predBeta,axis=-1),axis=0),
            'predCCoords': np.expand_dims(predCCoords,axis=0)}
    

    if identified_coords is None:
        #run the inference part
        identified = collectoverthresholds(data,beta_threshold,distance_threshold)[0,:,0] #V

        if predCCoords.shape[1] == 2:
            ax.scatter(predCCoords[:,0][identified],
                  predCCoords[:,1][identified],
                  s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
                  c='#000000',#rgba_cols[identified],
                  marker='+')
        elif predCCoords.shape[1] == 3:
            ax.scatter(predCCoords[:,0][identified],
                  predCCoords[:,1][identified],
                  predCCoords[:,2][identified],
                  s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
                  c='#000000',#rgba_cols[identified],
                  marker='+')

        return identified
    else:
        if predCCoords.shape[1] == 2:
            ax.scatter(identified_coords[:, 0],
                   identified_coords[:, 1],
                  s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
                  c='#000000',#rgba_cols[identified],
                  marker='+')
        elif predCCoords.shape[1] == 3:
            ax.scatter(identified_coords[:, 0],
                   identified_coords[:, 1],
                   identified_coords[:, 3],
                  s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
                  c='#000000',#rgba_cols[identified],
                  marker='+')


def make_original_truth_shower_plot(plt, ax,
                                    truthHitAssignementIdx,                      
                                    recHitEnergy, 
                                    recHitX,
                                    recHitY,
                                    recHitZ,
                                    cmap=None,
                                    rgbcolor=None,
                                    alpha=0.5,
                                    predBeta=None):
    
    
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(recHitEnergy.shape)>1:
        recHitEnergy = np.array(recHitEnergy[:,0])
    if len(recHitX.shape)>1:
        recHitX = np.array(recHitX[:,0])
    if len(recHitY.shape)>1:
        recHitY = np.array(recHitY[:,0])
    if len(recHitZ.shape)>1:
        recHitZ = np.array(recHitZ[:,0])
        
        
    pl = plotter_3d(output_file="/tmp/plot", colorscheme=None)#will be ignored
    if rgbcolor is None:
        if cmap is None:
            rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
        else:
            rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    rgbcolor[truthHitAssignementIdx<0]=[0.92,0.92,0.92]
    
    if predBeta is not None:
        alpha=None #use beta instead
        if len(predBeta.shape)>1:
            predBeta = np.array(predBeta[:,0])
            
        alphas = predBeta
        alphas = np.clip(alphas, a_min=5e-1,a_max=1.-1e-2)
        alphas = np.arctanh(alphas)/np.arctanh(1.-1e-2)
        #alphas *= alphas
        alphas[alphas<0.05] = 0.05
        alphas = np.expand_dims(alphas, axis=1)
        
        rgbcolor = np.concatenate([rgbcolor,alphas],axis=-1)
            
    if np.max(rgbcolor) >= 1.:
        rgbcolor /= np.max(rgbcolor) 

    pl.set_data(x = recHitX , y=recHitY   , z=recHitZ, e=recHitEnergy , c =rgbcolor)
    pl.marker_scale=2.
    pl.plot3d(ax=ax,alpha=alpha)
    
    
def make_eta_phi_projection_truth_plot(plt, ax,
                                    truthHitAssignementIdx,                      
                                    recHitEnergy, 
                                    recHitEta,
                                    recHitPhi,
                                    predEta,
                                    predPhi,
                                    truthEta,
                                    truthPhi,
                                    truthEnergy,
                                    predBeta,               #[ V ] or [ V x 1 ]
                                    predCCoords,            #[ V x 2 ]
                                    beta_threshold=0.2, distance_threshold=0.8,
                                    cmap=None,
                                    identified=None,
                                    predEnergy=None):
    
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(recHitEnergy.shape)>1:
        recHitEnergy = np.array(recHitEnergy[:,0])
    if len(recHitEta.shape)>1:
        recHitEta = np.array(recHitEta[:,0])
    if len(recHitPhi.shape)>1:
        recHitPhi = np.array(recHitPhi[:,0])
    if len(predEta.shape)>1:
        predEta = np.array(predEta[:,0])
    if len(predPhi.shape)>1:
        predPhi = np.array(predPhi[:,0])
    if len(truthEta.shape)>1:
        truthEta = np.array(truthEta[:,0])
    if len(truthPhi.shape)>1:
        truthPhi = np.array(truthPhi[:,0])
    if len(truthEnergy.shape)>1:
        truthEnergy = np.array(truthEnergy[:,0])
        
        
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(predBeta.shape)>1:
        predBeta = np.array(predBeta[:,0])
        
    

    ax.set_aspect(aspect=1.)
    #print(truthHitAssignementIdx)
    if cmap is None:
        rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    else:
        rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]

    rgbcolor[truthHitAssignementIdx<0]=[0.92,0.92,0.92]
    size_scaling = np.log(recHitEnergy+1)+0.1
    size_scaling /=  np.max(size_scaling)

    ax.scatter(recHitPhi,
              recHitEta,
              s=.25*size_scaling,
              c=rgbcolor)
     
    _, truth_idxs = np.unique(truthHitAssignementIdx,return_index=True)
    
    truth_size_scaling=np.log(truthEnergy[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0] +1.)+0.1
    truth_size_scaling /=  np.max(truth_size_scaling)
    
    true_sel_phi = truthPhi[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0]
    true_sel_eta = truthEta[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0]
    true_sel_col = rgbcolor[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0]
    ax.scatter(true_sel_phi,
              true_sel_eta,
              s=100.*truth_size_scaling,
              c=true_sel_col,
              marker='x')
    
    
    if beta_threshold < 0. or beta_threshold > 1 or distance_threshold<0:
        return
    
    data = {'predBeta': np.expand_dims(np.expand_dims(predBeta,axis=-1),axis=0),
            'predCCoords': np.expand_dims(predCCoords,axis=0)}
       
    #run the inference part
    if identified is None:
        identified = collectoverthresholds(data,beta_threshold,distance_threshold)[0,:,0] #V
    
    
    ax.scatter(predPhi[identified],
              predEta[identified],
              s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
              c='#000000',#rgba_cols[identified],
              marker='+')
    
    if predEnergy is not None:
        if len(predEnergy.shape)>1:
            predEnergy = np.array(predEnergy[:,0])
            predE = predEnergy[identified]
        for i in range(len(predE)):
            
            #predicted
            ax.text(predPhi[identified][i],
                    predEta[identified][i],
                    s = str(predE[i])[:4],
                    verticalalignment='bottom', horizontalalignment='right',
                    rotation=30,
                    fontsize='small')
            
            #truth
        true_sel_en = truthEnergy[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0]
        for i in range(len(true_sel_en)):
            ax.text(true_sel_phi[i],true_sel_eta[i], 
                    s=str(true_sel_en[i])[:4],
                    color = true_sel_col[i]/1.2,
                    verticalalignment='top', horizontalalignment='left',
                    rotation=30,
                    fontsize='small')


def make_truth_energy_histogram(plt, ax, truth_energies):
    plt.figure()
    plt.hist(truth_energies, bins=50, histtype='step')
    plt.xlabel("Truth shower energy")
    plt.ylabel("Frequency")
    plt.title('Truth energies')




def histogram_total_window_resolution(plt, ax, total_truth_energies, total_obc_energies, total_ticl_energies, energy_filter=0):

    if energy_filter > 0:
        filter = total_truth_energies > energy_filter

        total_truth_energies = total_truth_energies[filter]
        total_obc_energies = total_obc_energies[filter]
        total_ticl_energies = total_ticl_energies[filter]

    response_ticl = total_ticl_energies / total_truth_energies
    response_obc = total_obc_energies / total_truth_energies

    bins = np.linspace(0,3.001,40)

    response_ticl[response_ticl>3] = 3
    response_obc[response_obc>3] = 3

    plt.figure()
    plt.hist(response_obc, bins=bins, histtype='step', label='Object condensation')
    plt.hist(response_ticl, bins=bins, histtype='step', label='ticl')
    plt.xlabel("Response")
    plt.ylabel("Frequency")

    sx = '' if energy_filter==0 else ' - truth energy > %.2f GeV' % energy_filter

    plt.title('Total energy response'+sx)
    plt.legend()



def make_fake_energy_regressed_histogram(plt, ax, regressed_energy, ticl=False):
    plt.figure()
    plt.hist(regressed_energy, bins=50, histtype='step')
    plt.xlabel("Fake shower energy regressed")
    plt.ylabel("Frequency")
    plt.title('Fakes energy histogram' + (' - ticl' if ticl else ''))

def make_fake_energy_sum_histogram(plt, ax, predicted_energy_sum, ticl=False):
    predicted_energy_sum[predicted_energy_sum>60]=60
    plt.figure()
    plt.hist(predicted_energy_sum, bins=50, histtype='step')
    plt.xlabel("Fake shower energy rechit sum")
    plt.ylabel("Frequency")
    plt.title('Fakes energy histogram' + (' - ticl' if ticl else ''))







def make_response_histograms(plt, ax, found_showers_predicted_sum, found_showers_truth_sum, found_showers_predicted_energies, found_showers_target_energies, ticl=False):
    found_showers_predicted_sum = np.array(found_showers_predicted_sum)
    found_showers_truth_sum = np.array(found_showers_truth_sum)
    found_showers_target_energies = np.array(found_showers_target_energies)
    found_showers_predicted_energies = np.array(found_showers_predicted_energies)

    response_rechit_sum_energy = found_showers_predicted_sum/found_showers_truth_sum
    response_rechit_sum_energy[response_rechit_sum_energy > 3] = 3

    response_energy_predicted = found_showers_predicted_energies/found_showers_target_energies
    response_energy_predicted[response_energy_predicted > 3] = 3
    response_energy_predicted[response_energy_predicted < 0.1] = 0.1


    data_dict = {}
    plt.figure()
    plt.hist(response_rechit_sum_energy, bins=20, histtype='step')
    plt.hist(response_energy_predicted, bins=20, histtype='step')
    plt.legend(['predicted shower sum / truth shower sum', 'predicted energy / target energy'])
    plt.xlabel("Predicted/truth")
    plt.ylabel("Frequency")
    plt.title('Response curves' + (' - ticl' if ticl else ''))


def make_response_histograms_energy_segmented(plt, ax, _found_showers_predicted_sum, _found_showers_truth_sum, _found_showers_predicted_energies, _found_showers_target_energies, ticl=False):
    energy_segments = [0,5,10,20,30,50,100,200,300,3000]
    names = ['0-5', '5-10','10-20', '20-30','30-50', '50-100', '100-200','200-300', '300+']
    if ticl:
        names = ['Energy = %s Gev - ticl' % s for s in names]
    else:
        names = ['Energy = %s Gev' % s for s in names]

    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(3, 3)

    ax = [[fig.add_subplot(gs[0, 0]),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[0, 2]),],

          [fig.add_subplot(gs[1, 0]),
          fig.add_subplot(gs[1, 1]),
          fig.add_subplot(gs[1, 2]),],

          [fig.add_subplot(gs[2, 0]),
          fig.add_subplot(gs[2, 1]),
          fig.add_subplot(gs[2, 2]),]]


    _found_showers_predicted_sum = np.array(_found_showers_predicted_sum)
    _found_showers_truth_sum = np.array(_found_showers_truth_sum)
    _found_showers_target_energies = np.array(_found_showers_target_energies)
    _found_showers_predicted_energies = np.array(_found_showers_predicted_energies)

    for i in range(9):
        c = int(i/3)
        r = i%3
        n = names[i]
        l = energy_segments[i]
        h = energy_segments[i+1]

        condition = np.logical_and(_found_showers_truth_sum>l, _found_showers_truth_sum<h)

        found_showers_predicted_sum = _found_showers_predicted_sum[condition]
        found_showers_truth_sum = _found_showers_truth_sum[condition]
        found_showers_target_energies = _found_showers_target_energies[condition]
        found_showers_predicted_energies = _found_showers_predicted_energies[condition]

        response_rechit_sum_energy = found_showers_predicted_sum/found_showers_truth_sum
        response_rechit_sum_energy[response_rechit_sum_energy > 3] = 3

        response_energy_predicted = found_showers_predicted_energies/found_showers_target_energies
        response_energy_predicted[response_energy_predicted > 3] = 3
        response_energy_predicted[response_energy_predicted < 0.1] = 0.1

        # spread_sum = np.sqrt(np.var(response_rechit_sum_energy)/np.alen(response_rechit_sum_energy)).round(5)
        # spread_pred = np.sqrt(np.var(response_energy_predicted)/np.alen(response_energy_predicted)).round(5)
        spread_sum = np.std(response_rechit_sum_energy).round(3)
        spread_pred = np.std(response_energy_predicted).round(3)
        mean_sum = np.mean(response_rechit_sum_energy).round(3)
        mean_pred = np.mean(response_energy_predicted).round(3)
        error_sum = np.sqrt(np.var(response_rechit_sum_energy)/np.alen(response_rechit_sum_energy)).round(5)
        error_pred = np.sqrt(np.var(response_energy_predicted)/np.alen(response_energy_predicted)).round(5)

        data_dict = {}
        # plt.figure()
        ax[c][r].hist(response_rechit_sum_energy, bins=20, histtype='step')
        ax[c][r].hist(response_energy_predicted, bins=20, histtype='step')
        ax[c][r].legend(['predicted shower sum / truth shower sum\nmean ' + str(mean_sum) +' ± '+str(error_sum)+' spread '+str(spread_sum), 'predicted energy / target energy\nmean ' + str(mean_pred) +' ± '+str(error_pred)+' spread '+str(spread_pred)])
        ax[c][r].set_xlabel("Predicted/truth")
        ax[c][r].set_ylabel("Frequency")
        ax[c][r].set_title(n)
        # ax[c][r].text(n)


def make_truth_predicted_rotational_distance_histogram(plt, ax, rotational_distance_data):
    rotational_distance_data = np.array(rotational_distance_data)
    rotational_distance_data[rotational_distance_data > 0.2] = 0.2


    plt.figure()
    plt.hist(rotational_distance_data, bins=20, histtype='step')
    plt.xlabel("Rotational distance between true and predicted eta/phi coordinates")
    plt.ylabel("Frequency")
    plt.title('Positional performance')



def make_truth_predicted_rotational_distance_histogram(plt, ax, eta_predicted, eta_truth, phi_predicted, phi_truth):
    eta_predicted = np.array(eta_predicted)
    eta_truth = np.array(eta_truth)
    phi_predicted = np.array(phi_predicted)
    phi_truth = np.array(phi_truth)

    rotational_distance_data = np.sqrt((eta_predicted - eta_truth)**2 + (phi_predicted - phi_truth)**2)

    rotational_distance_data = np.array(rotational_distance_data)
    rotational_distance_data[rotational_distance_data > 0.2] = 0.2


    plt.figure()
    plt.hist(rotational_distance_data, bins=20, histtype='step')
    plt.xlabel("Rotational distance between true and predicted eta/phi coordinates")
    plt.ylabel("Frequency")
    plt.title('Positional performance')


def make_found_showers_plot_as_function_of_energy(plt, ax, energies, found_or_not, ticl=False):
    e_bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]

    centers = []
    mean = []
    std = []

    energies = np.array(energies)
    found_or_not = np.array(found_or_not)

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_energies = np.argwhere(np.logical_and(energies > l, energies < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append(l+5)


    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Shower energy')
    plt.ylabel('% found')
    plt.title('Function of energy' + (' - ticl' if ticl else ''))


def make_energy_response_curve_as_a_function_of_truth_energy(plt, ax, energies, predicted_energies, ticl=False):
    e_bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]

    centers = []
    mean = []
    std = []

    energies = np.array(energies)
    predicted_energies = np.array(predicted_energies)

    energies = energies[predicted_energies!=-1]
    predicted_energies = predicted_energies[predicted_energies!=-1]

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_energies_indices = np.argwhere(np.logical_and(energies > l, energies < h))

        this_energies = energies[this_energies_indices]
        this_predicted = predicted_energies[this_energies_indices].astype(np.float)
        response = this_predicted/this_energies

        m = np.mean(response)
        mean.append(m)
        std.append(np.std(response))
        centers.append(l+5)


    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Shower energy')
    plt.ylabel('Response')
    plt.title('Function of energy' + (' - ticl' if ticl else ''))


#
# def make_energy_response_curve_as_a_function_of_local_energy_density(plt, ax, local_energy_densities, predicted_energies, ticl=False):
#     e_bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#
#     centers = []
#     mean = []
#     std = []
#
#     local_energy_densities = np.array(local_energy_densities)
#     predicted_energies = np.array(predicted_energies)
#
#     local_energy_densities = local_energy_densities[predicted_energies != -1]
#     predicted_energies = predicted_energies[predicted_energies!=-1]
#
#     for i in range(len(e_bins)-1):
#         l = e_bins[i]
#         h = e_bins[i+1]
#
#         this_energies_indices = np.argwhere(np.logical_and(local_energy_densities > l, local_energy_densities < h))
#
#         this_energies = local_energy_densities[this_energies_indices]
#         this_predicted = predicted_energies[this_energies_indices].astype(np.float)
#         response = this_predicted/this_energies
#
#         m = np.mean(response)
#         mean.append(m)
#         std.append(np.std(response))
#         centers.append((l+h)/2)
#
#
#     plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
#     plt.xticks(centers)
#     plt.xlabel('Local energy densities')
#     plt.ylabel('Response')
#     plt.title('Function of energy' + (' - ticl' if ticl else ''))



def make_energy_response_curve_as_a_function_of_local_energy_density(plt, ax, local_energy_densities, truth_energies, predicted_energies, ticl=False):
    e_bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    centers = []
    mean = []
    std = []

    local_energy_densities = np.array(local_energy_densities)
    predicted_energies = np.array(predicted_energies)
    truth_energies = np.array(truth_energies)


    truth_energies = truth_energies[predicted_energies != -1]
    local_energy_densities = local_energy_densities[predicted_energies != -1]
    predicted_energies = predicted_energies[predicted_energies!=-1]

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_density_indices = np.argwhere(np.logical_and(local_energy_densities > l, local_energy_densities < h))

        this_energies = truth_energies[this_density_indices]
        this_predicted = predicted_energies[this_density_indices].astype(np.float)
        response = this_predicted/this_energies

        m = np.mean(response)
        mean.append(m)
        std.append(np.std(response))
        centers.append((l+h)/2)


    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Local energy densities')
    plt.ylabel('Response')
    plt.title('Function of local energy density' + (' - ticl' if ticl else ''))



def make_energy_response_curve_as_a_function_of_closest_particle_distance(plt, ax, closest_particle_distance, truth_energies, predicted_energies, ticl=False):
    e_bins = [0.,     0.0625 ,0.125,  0.1875, 0.25,   0.3125, 0.375 , 0.4375, 0.5 ,10  ]

    centers = []
    mean = []
    std = []

    closest_particle_distance = np.array(closest_particle_distance)
    predicted_energies = np.array(predicted_energies)
    truth_energies = np.array(truth_energies)


    truth_energies = truth_energies[predicted_energies != -1]
    closest_particle_distance = closest_particle_distance[predicted_energies != -1]
    predicted_energies = predicted_energies[predicted_energies!=-1]

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_density_indices = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))

        this_energies = truth_energies[this_density_indices]
        this_predicted = predicted_energies[this_density_indices].astype(np.float)
        response = this_predicted/this_energies

        m = np.mean(response)
        mean.append(m)
        std.append(np.std(response))
        centers.append(l+0.3)


    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Closest particle distance')
    plt.ylabel('Response')
    plt.title('Function of closest particle distance' + (' - ticl' if ticl else ''))




def make_found_showers_plot_as_function_of_closest_particle_distance(plt, ax, cloest_particle_distance, found_or_not, ticl=False):
    e_bins = [0.,     0.0625 ,0.125,  0.1875, 0.25,   0.3125, 0.375 , 0.4375, 0.5 ,10  ]

    centers = []
    mean = []
    std = []

    cloest_particle_distance = np.array(cloest_particle_distance)
    found_or_not = np.array(found_or_not)

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_energies = np.argwhere(np.logical_and(cloest_particle_distance > l, cloest_particle_distance < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append(l+0.03)


    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Closest particle distance (eta, phi, sqrt)')
    plt.ylabel('% found')
    plt.title('Function of closest particle distance' + (' - ticl' if ticl else ''))


def make_found_showers_plot_as_function_of_local_density(plt, ax, local_densities, found_or_not, ticl=False):
    e_bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    centers = []
    mean = []
    std = []

    local_densities = np.array(local_densities)
    found_or_not = np.array(found_or_not)

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_energies = np.argwhere(np.logical_and(local_densities > l, local_densities < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append((l+h)/2)


    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Local particle density')
    plt.ylabel('% found')
    plt.title('Function of local particle density' + (' - ticl' if ticl else ''))



def efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, ax, _local_densities, _found_or_not, _found_or_not_ticl, _truth_energies, energy_filter=0, make_segments=False):
    segments_low = [0,5,10,30]
    segments_high = [5,10,30,300]

    count_segments = 4 if make_segments else 1

    if make_segments:
        ax_array = [0,0,0,0]
        fig, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2,2, figsize=(16,10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)


    else:
        fig, ax_array = plt.subplots(1, 1, figsize=(8, 6))

    if energy_filter > 0:
        _local_densities = _local_densities[_truth_energies > energy_filter]
        _found_or_not = _found_or_not[_truth_energies > energy_filter]
        _found_or_not_ticl = _found_or_not_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        local_densities = _local_densities
        found_or_not = _found_or_not
        found_or_not_ticl = _found_or_not_ticl
        truth_energies = _truth_energies

    if energy_filter!=0:
        assert make_segments==False

    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax_array


        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number], _truth_energies<segments_high[segment_number])
            local_densities = _local_densities[filter]
            found_or_not = _found_or_not[filter]
            found_or_not_ticl = _found_or_not_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]
        e_bins = [0, .01, .02, .03, .04, .06, .08, .10, .15, .20, .30, .40, .50, .60, .70, .80, .90, 1]
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        mean_ticl = []
        std = []

        local_densities = np.array(local_densities)
        found_or_not = np.array(found_or_not)


        for i in range(len(e_bins)-1):
            l = e_bins[i]
            h = e_bins[i+1]

            filter = np.argwhere(np.logical_and(local_densities > l, local_densities < h))
            filtered_found = found_or_not[filter].astype(np.float)
            filtered_found_ticl = found_or_not_ticl[filter].astype(np.float)


            m = np.mean(filtered_found)
            mt = np.mean(filtered_found_ticl)
            mean.append(m)
            mean_ticl.append(mt)
            std.append(np.std(filtered_found))
            centers.append((l+h)/2)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        #
        # print(len(mean), len(centers))
        #
        # 0/0


        ax2 = ax1.twinx()
        hist_values,_ = np.histogram(local_densities, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Number of showers')
        ax2.set_ylim(0,np.max(hist_values)*1.3)
        ax1.set_title('Efficiency comparison '+sx)


        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Local shower energy fraction ($\\frac{e_s}{\\sum_{i}^{} e_i \mid \Delta R(s, i) < 0.5 }$)')
        ax1.set_ylabel('Reconstruction efficiency')
        ax1.legend(loc = 'center right')


def efficiency_comparison_plot_with_distribution_fo_truth_energy(plt, ax, _found_or_not, _found_or_not_ticl, _truth_energies, energy_filter=0):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))


    if energy_filter > 0:
        _found_or_not = _found_or_not[_truth_energies > energy_filter]
        _found_or_not_ticl = _found_or_not_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    found_or_not = _found_or_not
    found_or_not_ticl = _found_or_not_ticl
    truth_energies = _truth_energies


    e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())


    centers = []
    mean = []
    mean_ticl = []
    std = []

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        filter = np.argwhere(np.logical_and(truth_energies > l, truth_energies < h))
        filtered_found = found_or_not[filter].astype(np.float)
        filtered_found_ticl = found_or_not_ticl[filter].astype(np.float)


        m = np.mean(filtered_found)
        mt = np.mean(filtered_found_ticl)
        mean.append(m)
        mean_ticl.append(mt)
        std.append(np.std(filtered_found))
        centers.append((l+h)/2)


    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter        #
    # print(len(mean), len(centers))
    #
    # 0/0


    ax2 = ax1.twinx()
    hist_values,_ = np.histogram(truth_energies, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.set_ylabel('Number of showers')
    ax2.set_ylim(0,np.max(hist_values)*1.3)
    ax1.set_title('Efficiency comparison '+sx)


    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Truth energy')
    ax1.set_ylabel('Reconstruction efficiency')
    ax1.legend(loc = 'center right')


def efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, ax, _closest_particle_distance, _found_or_not, _found_or_not_ticl, _truth_energies, energy_filter=0, make_segments=False):
    segments_low = [0,5,10,30]
    segments_high = [5,10,30,300]

    count_segments = 4 if make_segments else 1

    if make_segments:
        ax_array = [0,0,0,0]
        fig, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2,2, figsize=(16,10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
    else:
        fig, ax_array = plt.subplots(1, 1, figsize=(8, 6))


    if energy_filter > 0:
        _closest_particle_distance = _closest_particle_distance[_truth_energies > energy_filter]
        _found_or_not = _found_or_not[_truth_energies > energy_filter]
        _found_or_not_ticl = _found_or_not_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        closest_particle_distance = _closest_particle_distance
        found_or_not = _found_or_not
        found_or_not_ticl = _found_or_not_ticl
        truth_energies = _truth_energies


    if energy_filter!=0:
        assert make_segments==False



    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax_array


        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number], _truth_energies<segments_high[segment_number])
            closest_particle_distance = _closest_particle_distance[filter]
            found_or_not = _found_or_not[filter]
            found_or_not_ticl = _found_or_not_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.07, 0.14, 0.21, 0.28, 0.35, .42, .49, .56, .63, .7]
        e_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.055, 0.07, 0.09, 0.11, 0.13, 0.14, 0.175, 0.21, 0.28, 0.35, .42, .49, .56,
                  .63, .7]
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        mean_ticl = []
        std = []

        closest_particle_distance = np.array(closest_particle_distance)
        found_or_not = np.array(found_or_not)


        for i in range(len(e_bins)-1):
            l = e_bins[i]
            h = e_bins[i+1]

            filter = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))
            filtered_found = found_or_not[filter].astype(np.float)
            filtered_found_ticl = found_or_not_ticl[filter].astype(np.float)


            m = np.mean(filtered_found)
            mt = np.mean(filtered_found_ticl)
            mean.append(m)
            mean_ticl.append(mt)
            std.append(np.std(filtered_found))
            centers.append((l+h)/2)


        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter        #
        # print(len(mean), len(centers))
        #
        # 0/0


        ax2 = ax1.twinx()
        hist_values,_ = np.histogram(closest_particle_distance, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Number of showers')
        ax2.set_ylim(0,np.max(hist_values)*1.3)
        ax1.set_title('Efficiency comparison '+sx)


        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Closest particle distance')
        ax1.set_ylabel('Reconstruction efficiency')
        ax1.legend(loc = 'center right')



def response_comparison_plot_with_distribution_fo_local_fraction(plt, ax, _local_densities, _energy_predicted, _energy_predicted_ticl, _truth_energies, energy_filter=0, make_segments=False):
    segments_low = [0,5,10,30]
    segments_high = [5,10,30,300]

    count_segments = 4 if make_segments else 1

    if make_segments:
        ax_array = [0,0,0,0]
        ax_array_res = [0,0,0,0]
        fig1, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2,2, figsize=(16,10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)

        fig2, ((ax_array_res[0], ax_array_res[1]), (ax_array_res[2], ax_array_res[3])) = plt.subplots(2,2, figsize=(16,10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)


    else:
        fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(16, 6))
        plt.subplots_adjust(wspace=0.4)



    if energy_filter > 0:
        _local_densities = _local_densities[_truth_energies > energy_filter]
        _energy_predicted = _energy_predicted[_truth_energies > energy_filter]
        _energy_predicted_ticl = _energy_predicted_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        local_densities = _local_densities
        energy_predicted = _energy_predicted
        energy_predicted_ticl = _energy_predicted_ticl
        truth_energies = _truth_energies

    if energy_filter!=0:
        assert make_segments==False



    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax1
        ax22 = ax_array_res[segment_number] if make_segments else ax22

        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number], _truth_energies<segments_high[segment_number])
            local_densities = _local_densities[filter]
            energy_predicted = _energy_predicted[filter]
            energy_predicted_ticl = _energy_predicted_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]
        e_bins = [0, .01, .02, .03, .04, .06, .08, .10, .15, .20, .30, .40, .50, .60, .70, .80, .90, 1]
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())


        centers = []
        mean = []
        mean_ticl = []
        var = []
        var_ticl = []

        local_densities = np.array(local_densities)
        energy_predicted = np.array(energy_predicted)


        for i in range(len(e_bins)-1):
            l = e_bins[i]
            h = e_bins[i+1]

            filter = np.argwhere(np.logical_and(local_densities > l, local_densities < h))

            filtered_truth_energy = truth_energies[filter].astype(np.float)
            filtered_predicted_energy = energy_predicted[filter].astype(np.float)

            second_filter = filtered_predicted_energy >= 0
            filtered_truth_energy = filtered_truth_energy[second_filter]
            filtered_predicted_energy = filtered_predicted_energy[second_filter]



            m = np.mean(filtered_predicted_energy / filtered_truth_energy)
            mean.append(m)

            var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)


            centers.append((l+h)/2)



        for i in range(len(e_bins)-1):
            l = e_bins[i]
            h = e_bins[i+1]

            filter = np.argwhere(np.logical_and(local_densities > l, local_densities < h))

            filtered_truth_energy = truth_energies[filter].astype(np.float)
            filtered_predicted_energy_ticl = energy_predicted_ticl[filter].astype(np.float)


            second_filter = filtered_predicted_energy_ticl >= 0
            filtered_truth_energy = filtered_truth_energy[second_filter]
            filtered_predicted_energy_ticl = filtered_predicted_energy_ticl[second_filter]


            mt = np.mean(filtered_predicted_energy_ticl / filtered_truth_energy)
            mean_ticl.append(mt)

            var_ticl.append(np.std(filtered_predicted_energy_ticl / filtered_truth_energy - m) / m)


        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax1.set_title('Response comparison '+sx)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax22.set_title('Response comparison '+sx)
        #
        # print(len(mean), len(centers))
        #
        # 0/0


        ax2 = ax1.twinx()
        hist_values,_ = np.histogram(local_densities, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Number of showers')
        ax2.set_ylim(0,np.max(hist_values)*1.3)


        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Local shower energy fraction ($\\frac{e_s}{\\sum_{i}^{} e_i \mid \Delta R(s, i) < 0.5 }$)')
        ax1.set_ylabel('$<e_{pred} / e_{true}>$')
        ax1.legend(loc = 'center right')

        ax22_twin = ax22.twinx()
        hist_values,_ = np.histogram(local_densities, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

        ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax22_twin.set_ylabel('Number of showers')

        ax22_twin.set_ylim(0,np.max(hist_values)*1.3)


        ax22.step(e_bins, [var[0]] + var, label='Object condensation')
        ax22.step(e_bins, [var_ticl[0]] + var_ticl, label='ticl')
        ax22.set_xticks(e_bins_ticks)
        ax22.set_xlabel('Local shower energy fraction ($\\frac{e_s}{\\sum_{i}^{} e_i \mid \Delta R(s, i) < 0.5 }$)')
        ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
        ax22.legend(loc = 'center right')

    if make_segments:
        return fig1, fig2





def response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, ax, _closest_particle_distance, _energy_predicted, _energy_predicted_ticl, _truth_energies, energy_filter=0, make_segments=False):
    segments_low = [0,5,10,30]
    segments_high = [5,10,30,300]

    count_segments = 4 if make_segments else 1


    if make_segments:
        ax_array = [0,0,0,0]
        ax_array_res = [0,0,0,0]
        fig1, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2,2, figsize=(16,10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)

        fig2, ((ax_array_res[0], ax_array_res[1]), (ax_array_res[2], ax_array_res[3])) = plt.subplots(2,2, figsize=(16,10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
    else:
        fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(16, 6))
        plt.subplots_adjust(wspace=0.4)

    if energy_filter > 0:
        _closest_particle_distance = _closest_particle_distance[_truth_energies > energy_filter]
        _energy_predicted = _energy_predicted[_truth_energies > energy_filter]
        _energy_predicted_ticl = _energy_predicted_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        closest_particle_distance = _closest_particle_distance
        energy_predicted = _energy_predicted
        energy_predicted_ticl = _energy_predicted_ticl
        truth_energies = _truth_energies

    if energy_filter!=0:
        assert make_segments==False


    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax1
        ax22 = ax_array_res[segment_number] if make_segments else ax22

        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number], _truth_energies<segments_high[segment_number])
            closest_particle_distance = _closest_particle_distance[filter]
            energy_predicted = _energy_predicted[filter]
            energy_predicted_ticl = _energy_predicted_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.07, 0.14, 0.21, 0.28, 0.35, .42, .49, .56, .63, .7]
        e_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.055, 0.07, 0.1, 0.13, 0.21, 0.28, 0.35, 0.49, .7]
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())


        centers = []
        mean = []
        mean_ticl = []

        var = []
        var_ticl = []

        closest_particle_distance = np.array(closest_particle_distance)
        energy_predicted = np.array(energy_predicted)

        for i in range(len(e_bins)-1):
            l = e_bins[i]
            h = e_bins[i+1]

            filter = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))

            filtered_truth_energy = truth_energies[filter].astype(np.float)
            filtered_predicted_energy = energy_predicted[filter].astype(np.float)

            second_filter = filtered_predicted_energy >= 0
            filtered_truth_energy = filtered_truth_energy[second_filter]
            filtered_predicted_energy = filtered_predicted_energy[second_filter]



            m = np.mean(filtered_predicted_energy / filtered_truth_energy)
            mean.append(m)
            var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)




        for i in range(len(e_bins)-1):
            l = e_bins[i]
            h = e_bins[i+1]

            filter = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))

            filtered_truth_energy = truth_energies[filter].astype(np.float)
            filtered_predicted_energy_ticl = energy_predicted_ticl[filter].astype(np.float)


            second_filter = filtered_predicted_energy_ticl >= 0
            filtered_truth_energy = filtered_truth_energy[second_filter]
            filtered_predicted_energy_ticl = filtered_predicted_energy_ticl[second_filter]


            mt = np.mean(filtered_predicted_energy_ticl / filtered_truth_energy)
            mean_ticl.append(mt)
            var_ticl.append(np.std(filtered_predicted_energy_ticl / filtered_truth_energy - mt) / mt)


        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax1.set_title('Response comparison '+sx)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax22.set_title('Response comparison '+sx)
        #
        # print(len(mean), len(centers))
        #
        # 0/0


        ax2 = ax1.twinx()
        hist_values,_ = np.histogram(closest_particle_distance, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Number of showers')

        ax2.set_ylim(0,np.max(hist_values)*1.3)


        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Closest particle distance')
        ax1.set_ylabel('$<e_{pred} / e_{true}>$')
        ax1.legend(loc = 'center right')

        ax22_twin = ax22.twinx()
        hist_values,_ = np.histogram(closest_particle_distance, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

        ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax22_twin.set_ylabel('Number of showers')

        ax22_twin.set_ylim(0,np.max(hist_values)*1.3)


        ax22.step(e_bins, [var[0]] + var, label='Object condensation')
        ax22.step(e_bins, [var_ticl[0]] + var_ticl, label='ticl')
        ax22.set_xticks(e_bins_ticks)
        ax22.set_xlabel('Closest particle distance')
        ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
        ax22.legend(loc = 'center right')

    if make_segments:
        return fig1, fig2



def response_curve_comparison_with_distribution_fo_energy(plt, ax, energy_truth, energy_predicted, energy_predicted_ticl, energy_filter=0):
    e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())



    if energy_filter > 0:
        energy_predicted = energy_predicted[energy_truth > energy_filter]
        energy_predicted_ticl = energy_predicted_ticl[energy_truth > energy_filter]
        energy_truth = energy_truth[energy_truth > energy_filter]


    centers = []
    mean = []
    mean_ticl = []
    var = []
    var_ticl = []

    energy_truth = np.array(energy_truth)
    energy_predicted = np.array(energy_predicted)

    fig, (ax1, ax22) = plt.subplots(1,2, figsize=(16,6))
    plt.subplots_adjust(wspace=0.4)

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))

        filtered_truth_energy = energy_truth[filter].astype(np.float)
        filtered_predicted_energy = energy_predicted[filter].astype(np.float)

        second_filter = filtered_predicted_energy >= 0
        filtered_truth_energy = filtered_truth_energy[second_filter]
        filtered_predicted_energy = filtered_predicted_energy[second_filter]



        m = np.mean(filtered_predicted_energy / filtered_truth_energy)
        mean.append(m)
        var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)




    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))

        filtered_truth_energy = energy_truth[filter].astype(np.float)
        filtered_predicted_energy_ticl = energy_predicted_ticl[filter].astype(np.float)


        second_filter = filtered_predicted_energy_ticl >= 0
        filtered_truth_energy = filtered_truth_energy[second_filter]
        filtered_predicted_energy_ticl = filtered_predicted_energy_ticl[second_filter]


        mt = np.mean(filtered_predicted_energy_ticl / filtered_truth_energy)
        mean_ticl.append(mt)

        var_ticl.append(np.std(filtered_predicted_energy_ticl / filtered_truth_energy - mt) / mt)


    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
    ax1.set_title('Response comparison '+sx)


    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
    ax22.set_title('Response comparison '+sx)
    #
    # print(len(mean), len(centers))
    #
    # 0/0


    ax2 = ax1.twinx()
    hist_values,_ = np.histogram(energy_truth, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.set_ylabel('Number of showers')
    ax2.set_ylim(0,np.max(hist_values)*1.3)



    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Truth energy')
    ax1.set_ylabel('$<e_{pred} / e_{true}>$')
    ax1.legend(loc = 'center right')



    ax22_twin = ax22.twinx()
    hist_values,_ = np.histogram(energy_truth, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()

    ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)
    ax22_twin.set_ylim(0,np.max(hist_values)*1.3)


    ax22_twin.set_ylabel('Number of showers')


    ax22.step(e_bins, [var[0]] + var, label='Object condensation')
    ax22.step(e_bins, [var_ticl[0]] + var_ticl, label='ticl')
    ax22.set_xticks(e_bins_ticks)
    ax22.set_xlabel('Truth energy')
    ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
    ax22.legend(loc = 'center right')





def fake_rate_comparison_with_distribution_fo_energy(plt, ax, predicted_energies, matched_energies, predicted_energies_ticl, matched_energies_ticl, energy_filter=0):
    e_bins_ticks = [0,10,20,30,40,50,60,70,80,90,100]
    e_bins = [0,1.,2,3,4,6,8,10,15,20,30,40,50,60,70,80,90,100]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())


    if energy_filter > 0:
        matched_energies = matched_energies[predicted_energies > energy_filter]
        predicted_energies = predicted_energies[predicted_energies > energy_filter]

        matched_energies_ticl = matched_energies_ticl[predicted_energies_ticl > energy_filter]
        predicted_energies_ticl = predicted_energies_ticl[predicted_energies_ticl > energy_filter]


    centers = []
    mean = []
    mean_ticl = []
    std = []

    predicted_energies = np.array(predicted_energies)
    matched_energies = np.array(matched_energies)

    fig, ax1 = plt.subplots(figsize=(8,6))

    fake_energies = predicted_energies[matched_energies==-1]
    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]


        fake_energies_interval = np.argwhere(np.logical_and(fake_energies > l, fake_energies < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies > l, predicted_energies < h))


        try:
            m = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            m = 0


        mean.append(m)



    fake_energies_ticl = predicted_energies_ticl[matched_energies_ticl==-1]
    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        fake_energies_interval = np.argwhere(np.logical_and(fake_energies_ticl > l, fake_energies_ticl < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies_ticl > l, predicted_energies_ticl < h))

        try:
            mt = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            mt = 0


        mean_ticl.append(mt)

    # mean_ticl = mean_ticl / (e_bins[1:] - e_bins[:-1])

    sx = '' if energy_filter == 0 else ' - predicted energy > %.2f GeV' % energy_filter
    plt.title('Fake rate comparison'+sx)
    #
    # print(len(mean), len(centers))
    #
    # 0/0



    ax2 = ax1.twinx()
    hist_values,_ = np.histogram(predicted_energies, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()


    hist_values_ticl,_ = np.histogram(predicted_energies_ticl, bins=e_bins)
    hist_values_ticl = (hist_values_ticl / (e_bins_n[1:] - e_bins_n[:-1])).tolist()


    ax2.set_ylim(0,max(np.max(hist_values_ticl), np.max(hist_values))*1.3)
    #
    # hist_values[hist_values == 0] = 10
    # hist_values[hist_values_ticl == 0] = 10

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.step(e_bins, [hist_values_ticl[0]] + hist_values_ticl, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values_ticl[0]] + hist_values_ticl, step="pre", alpha=0.2)

    legend_elements = [Patch(facecolor='#1f77b4', label='Object condensation', alpha=0.2),
                       Patch(facecolor='#ff7f0e', label='ticl', alpha=0.2)]

    ax2.set_ylabel('Number of showers')

    ax2.legend(handles=legend_elements, loc=(0.675,0.34))






    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Predicted energy')
    ax1.set_ylabel('Fake rate')
    ax1.legend(loc = 'center right')

def make_fake_rate_plot_as_function_of_fake_energy(plt, ax, predicted_energies, matched_energies, is_sum, ticl=False):
    predicted_energies = np.array(predicted_energies)
    matched_energies = np.array(matched_energies)
    e_bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]

    centers = []
    mean = []
    std = []

    fake_energies = predicted_energies[matched_energies==-1]

    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        fake_energies_interval = np.argwhere(np.logical_and(fake_energies > l, fake_energies < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies > l, predicted_energies < h))

        try:
            m = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            m = 0
        mean.append(m)
        # std.append(np.std(filtered_found))
        centers.append(l+5)


    # plt.errorbar(centers, mean, std, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Fake energy sum' if is_sum else 'Fake energy regressed')
    plt.ylabel('% fake')
    plt.title('Function of fake energy' + (' - ticl' if ticl else ''))


def make_energy_hists(plt, ax, predicted_energies, matched_energies, truth_shower_energies,truth_showers_found_or_not, is_sum, ticl=False):
    predicted_energies = np.array(predicted_energies)
    truth_energies = np.array(truth_shower_energies)
    fake_energies = predicted_energies[matched_energies == -1]
    missed_energies = truth_shower_energies[truth_showers_found_or_not == 0]

    bins = np.linspace(0, 200, 30)

    plt.hist(predicted_energies, bins=bins, histtype='step', log=True)
    plt.hist(truth_energies, bins=bins, histtype='step', log=True)
    plt.hist(fake_energies, bins=bins, histtype='step', log=True)
    plt.hist(missed_energies, bins=bins, histtype='step', log=True)

    plt.xlabel('Energy (GeV')
    plt.ylabel('Frequency')

    plt.legend(['Predicted', 'Truth', 'Fake', 'Missed'])



    # plt.errorbar(centers, mean, std, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    # plt.xlabel('Fake energy sum' if is_sum else 'Fake energy regressed')
    # plt.ylabel('% fake')
    plt.title('Energy histograms' + (' - ticl' if ticl else ''))


def make_found_showers_plot_as_function_of_pt(plt, ax, energies, eta, found_or_not, ticl=False):
    pt_bins = [0,10,30,70,100,150,250,700,800,900,1000,1100,1200,1300,1400,1500]
    pt_bins = np.linspace(0, 800,15)
    pt_bins = [0,50,100,150,200,250,300,350,400,450,500,600,700,800]

    centers = []
    mean = []
    std = []

    energies = np.array(energies)
    eta = np.array(eta)
    found_or_not = np.array(found_or_not)

    pt = np.cosh(eta) * energies

    for i in range(len(pt_bins)-1):
        l = pt_bins[i]
        h = pt_bins[i+1]

        this_energies = np.argwhere(np.logical_and(pt > l, pt < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append(l+5)


    plt.errorbar(centers, mean, std, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    plt.xticks(centers)
    plt.xlabel('Shower pT')
    plt.ylabel('% found')
    plt.title('Function of pT' + (' - ticl' if ticl else ''))

def make_real_predicted_number_of_showers_histogram(plt, ax, num_real_showers, num_predicted_showers, ticl=False):
    plt.hist(num_real_showers, bins=np.arange(0,50), histtype='step')
    plt.hist(num_predicted_showers, bins=np.arange(0,70), histtype='step')
    plt.xlabel('Num showers')
    plt.ylabel('Frequency')
    plt.legend(['Real showers','Predicted showers'])
    plt.title('Histogram of predicted/real number of showers' +(' - ticl' if ticl else ''))


def make_histogram_of_number_of_rechits_per_shower(plt, ax, num_rechits_per_shower):
    plt.hist(num_rechits_per_shower, histtype='step')
    plt.xlabel('Num rechits per shower')
    plt.ylabel('Frequency')
    # plt.legend(['Num rechits per window','Num rechits per shower'])
    plt.title('Distribution of number of rechits')


def make_histogram_of_number_of_rechits_per_segment(plt, ax, num_rechits_per_segment):
    plt.hist(num_rechits_per_segment, histtype='step')
    plt.xlabel('Num rechits per segment')
    plt.ylabel('Frequency')
    # plt.legend(['Num rechits per window','Num rechits per shower'])
    plt.title('Distribution of number of rechits')


def make_histogram_of_number_of_showers_per_segment(plt, ax, num_showers_per_segment):
    plt.hist(num_showers_per_segment, bins=np.arange(0, 50), histtype='step')
    # plt.hist(num_predicted_showers, bins=np.arange(0,70), histtype='step')
    plt.xlabel('Num showers per window cut')
    plt.ylabel('Frequency')
    # plt.legend(['Real showers','Predicted showers'])
    plt.title('Distribution of number of showers')


def visualize_the_segment(plt, truth_showers_this_segment, truth_and_pred_dict, feature_dict, ticl_showers, labels,
                          coords_representative_predicted_showers, distance_threshold):
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(3, 2)

    ax = [fig.add_subplot(gs[0, 0], projection='3d'),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 0], projection='3d'),
          fig.add_subplot(gs[1, 1], projection='3d'),
          fig.add_subplot(gs[2, 0], projection='3d'), ]

    # wrt ground truth colors

    ax[0].set_xlabel('z (cm)')
    ax[0].set_ylabel('y (cm)')
    ax[0].set_zlabel('x (cm)')
    ax[0].set_title('Input data')

    ax[1].set_xlabel('Clustering dimension 1')
    ax[1].set_ylabel('Clustering dimension 2')
    ax[1].set_title('Colors = truth showers')

    ax[2].set_xlabel('z (cm)')
    ax[2].set_ylabel('y (cm)')
    ax[2].set_zlabel('x (cm)')
    ax[2].set_title('Colors = truth showers')

    ax[3].set_xlabel('z (cm)')
    ax[3].set_ylabel('y (cm)')
    ax[3].set_zlabel('x (cm)')
    ax[3].set_title('Colors = predicted showers')

    ax[4].set_xlabel('z (cm)')
    ax[4].set_ylabel('y (cm)')
    ax[4].set_zlabel('x (cm)')
    ax[4].set_title('Colors = ticl showers')

    cmap = createRandomizedColors('jet')

    make_original_truth_shower_plot(plt, ax[0], truth_showers_this_segment * 0, feature_dict['recHitEnergy'][:,0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:,0], feature_dict['recHitZ'][:,0],
                                    cmap=plt.get_cmap('Wistia'))
    make_cluster_coordinates_plot(plt, ax[1], truth_showers_this_segment, truth_and_pred_dict['predBeta'][:, 0],
                                  truth_and_pred_dict['predCCoords'],
                                  identified_coords=coords_representative_predicted_showers, cmap=cmap,
                                  distance_threshold=distance_threshold)
    #
    # make_original_truth_shower_plot(plt, ax[4], 1-identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
    # make_original_truth_shower_plot(plt, ax[5], identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))

    # wrt predicted colors

    np.set_printoptions(threshold=np.inf)

    # print(np.array(truth_showers_this_segment))
    # print(np.array(labels))

    xmax = max(np.max(truth_showers_this_segment), np.max(labels))
    rgbcolor_truth = cmap(truth_showers_this_segment/xmax)[:,:-1]
    rgbcolor_labels = cmap(labels/xmax)[:,:-1]
    rgbcolor_ticl = cmap(ticl_showers/xmax)[:,:-1]


    make_original_truth_shower_plot(plt, ax[2], truth_showers_this_segment, feature_dict['recHitEnergy'][:,0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:,0], feature_dict['recHitZ'][:,0],
                                    cmap=cmap, rgbcolor=rgbcolor_truth)
    make_original_truth_shower_plot(plt, ax[3], labels, feature_dict['recHitEnergy'][:,0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:,0], feature_dict['recHitZ'][:,0],
                                    cmap=cmap, rgbcolor=rgbcolor_labels)
    make_original_truth_shower_plot(plt, ax[4], ticl_showers, feature_dict['recHitEnergy'][:,0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:,0], feature_dict['recHitZ'][:,0],
                                    cmap=cmap, rgbcolor=rgbcolor_ticl)

    # make_cluster_coordinates_plot(plt, ax[3], labels, pred_this_segment[:, -6], pred_this_segment[:, -2:], identified_coords=coords_representative_predicted_showers, cmap=cmap)


# def visualize_the_segment_separate(pdf2, plt, truth_showers_this_segment, truth_and_pred_dict, feature_dict, ticl_showers, labels,
#                           coords_representative_predicted_showers, distance_threshold):
#
#     # gs = plt.GridSpec(3, 2)
#
#     # ax = [fig.add_subplot(gs[0, 0], projection='3d'),
#     #       fig.add_subplot(gs[0, 1]),
#     #       fig.add_subplot(gs[1, 0], projection='3d'),
#     #       fig.add_subplot(gs[1, 1], projection='3d'),
#     #       fig.add_subplot(gs[2, 0], projection='3d'), ]
#     ax = []
#     fig1 = plt.figure()
#     ax.append(fig1.add_subplot(111, projection='3d'))
#     fig2 = plt.figure()
#     ax.append(fig2.gca())
#     fig3 = plt.figure()
#     ax.append(fig3.add_subplot(111, projection='3d'))
#     fig4 = plt.figure()
#     ax.append(fig4.add_subplot(111, projection='3d'))
#     fig5 = plt.figure()
#     ax.append(fig5.add_subplot(111, projection='3d'))
#
#
#     # wrt ground truth colors
#
#     ax[0].set_xlabel('z (cm)')
#     ax[0].set_ylabel('y (cm)')
#     ax[0].set_zlabel('x (cm)')
#     ax[0].set_title('Input data')
#
#     font = {'family': 'sans-serif',
#             'color': 'black',
#             'weight': 'bold',
#             'size': 10,
#             }
#
#     font2 = {'family': 'sans-serif',
#             'color': 'black',
#             'weight': 'normal',
#             'size': 10,
#             }
#
#     fig1.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig1.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig2.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig2.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig3.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig3.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig4.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig4.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig5.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig5.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#
#
#     ax[1].set_xlabel('Clustering dimension 1')
#     ax[1].set_ylabel('Clustering dimension 2')
#     # ax[1].set_title('Colors = truth showers')
#
#
#
#     ax[2].set_xlabel('z (cm)')
#     ax[2].set_ylabel('y (cm)')
#     ax[2].set_zlabel('x (cm)')
#     ax[2].set_title('Colors = truth showers')
#
#     ax[3].set_xlabel('z (cm)')
#     ax[3].set_ylabel('y (cm)')
#     ax[3].set_zlabel('x (cm)')
#     ax[3].set_title('Colors = predicted showers')
#
#     ax[4].set_xlabel('z (cm)')
#     ax[4].set_ylabel('y (cm)')
#     ax[4].set_zlabel('x (cm)')
#     ax[4].set_title('Colors = ticl showers')
#
#     cmap = createRandomizedColors('jet')
#
#     make_original_truth_shower_plot(plt, ax[0], truth_showers_this_segment * 0, x_this_segment[:, 0],
#                                     x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7],
#                                     cmap=plt.get_cmap('Wistia'))
#     make_cluster_coordinates_plot(plt, ax[1], truth_showers_this_segment, pred_this_segment[:, -6],
#                                   pred_this_segment[:, -2:],
#                                   identified_coords=coords_representative_predicted_showers, cmap=cmap,
#                                   distance_threshold=distance_threshold)
#     #
#     # make_original_truth_shower_plot(plt, ax[4], 1-identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
#     # make_original_truth_shower_plot(plt, ax[5], identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
#
#     # wrt predicted colors
#
#     np.set_printoptions(threshold=np.inf)
#
#     # print(np.array(truth_showers_this_segment))
#     # print(np.array(labels))
#
#     xmax = max(np.max(truth_showers_this_segment), np.max(labels))
#     rgbcolor_truth = cmap(truth_showers_this_segment/xmax)[:,:-1]
#     rgbcolor_labels = cmap(labels/xmax)[:,:-1]
#     rgbcolor_ticl = cmap(ticl_showers/xmax)[:,:-1]
#
#
#     make_original_truth_shower_plot(plt, ax[2], truth_showers_this_segment, x_this_segment[:, 0], x_this_segment[:, 5],
#                                     x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap, rgbcolor=rgbcolor_truth)
#     make_original_truth_shower_plot(plt, ax[3], labels, x_this_segment[:, 0], x_this_segment[:, 5],
#                                     x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap, rgbcolor=rgbcolor_labels)
#     make_original_truth_shower_plot(plt, ax[4], ticl_showers, x_this_segment[:, 0], x_this_segment[:, 5],
#                                     x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap, rgbcolor=rgbcolor_ticl)
#
#     # make_cluster_coordinates_plot(plt, ax[3], labels, pred_this_segment[:, -6], pred_this_segment[:, -2:], identified_coords=coords_representative_predicted_showers, cmap=cmap)
#
#     pdf2.savefig(fig1)
#     pdf2.savefig(fig2)
#     pdf2.savefig(fig3)
#     pdf2.savefig(fig4)
#     pdf2.savefig(fig5)




def make_found_showers_plot_as_function_of_number_of_truth_showers(plt, ax, num_real, num_found, num_missed, num_fakes, num_predicted, ticl=False):

    num_real = np.array(num_real)
    num_found = np.array(num_found)
    num_missed = np.array(num_missed)
    num_fakes = np.array(num_fakes)
    num_predicted = np.array(num_predicted)

    x_num_real = []
    mean_fraction_found = []
    mean_fraction_missed = []
    mean_fraction_fakes = []


    var_fraction_found = []
    var_fraction_missed = []
    var_fraction_fakes = []


    fraction_found_by_real = num_found / num_real
    fraction_missed_by_real = num_missed / num_real
    fraction_fake_by_predicted = num_fakes / num_predicted


    for i in np.sort(np.unique(num_real)):
        if i<=0:
            continue


        x_num_real.append(i)
        mean_fraction_found.append(np.mean(fraction_found_by_real[num_real==i]))
        mean_fraction_missed.append(np.mean(fraction_missed_by_real[num_real==i]))
        mean_fraction_fakes.append(np.mean(fraction_fake_by_predicted[num_real==i]))


        var_fraction_found.append(np.std(fraction_found_by_real[num_real==i]))
        var_fraction_missed.append(np.std(fraction_missed_by_real[num_real==i]))
        var_fraction_fakes.append(np.std(fraction_fake_by_predicted[num_real==i]))

    x_num_real = np.array(x_num_real)
    mean_fraction_found = np.array(mean_fraction_found)
    mean_fraction_missed = np.array(mean_fraction_missed)
    mean_fraction_fakes = np.array(mean_fraction_fakes)

    var_fraction_found = np.array(var_fraction_found)
    var_fraction_missed = np.array(var_fraction_missed)
    var_fraction_fakes = np.array(var_fraction_fakes)


    plt.errorbar(x_num_real, mean_fraction_found, var_fraction_found, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    plt.errorbar(x_num_real, mean_fraction_fakes, var_fraction_fakes, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    plt.xlabel('Num showers')
    plt.ylabel('Fraction')
    plt.legend(['Found / Truth','Fakes / Predicted'])

    plt.title('Found/Missed' +(' - ticl' if ticl else ''))

# def make_found_showers_plot_as_function_of_number_of_truth_showers_2(plt, ax, predicted_showers_sample_ids, predicted_showers_energies_sum, predicted_showers_matched_energies_sum, predicted_showers_regressed_energies, predicted_showers_matched_energies, truth_showers_sample_ids, truth_showers_energies, truth_showers_found_or_not, ticl=False, filter=False):
#     unique_sample_ids = np.unique(predicted_showers_sample_ids)
#
#     num_real = []
#     num_found = []
#     num_missed = []
#     num_fakes = []
#     num_predicted = []
#
#     for u in unique_sample_ids:
#         sample_predicted_showers_energies_sum = predicted_showers_energies_sum[predicted_showers_sample_ids==u]
#         sample_predicted_showers_matched_energies_sum = predicted_showers_matched_energies_sum[predicted_showers_sample_ids==u]
#         sample_predicted_showers_regressed_energies = predicted_showers_regressed_energies[predicted_showers_sample_ids==u]
#         sample_predicted_showers_matched_energies = predicted_showers_matched_energies[predicted_showers_sample_ids==u]
#
#         sample_truth_showers_energies = truth_showers_energies[truth_showers_sample_ids==u]
#         sample_truth_showers_found_or_not = truth_showers_found_or_not[truth_showers_sample_ids==u]
#
#         if filter:
#             pass



def hist_2d_found_efficiency_vs_local_fraction_and_truth_shower_energy(plt, ax, truth_energies, local_fraction, found_or_not, found_or_not_ticl):

    bins_l = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    bins_e = [0,10,20,30,40,50,60,70,80,90,100]

    H = np.zeros((len(truth_energies), 2), np.float)

    H[:, 0] = np.digitize(truth_energies, bins_e)
    H[:, 1] = np.digitize(local_fraction, bins_l)

    values_obc = np.zeros((10,10), np.float)
    values_ticl = np.zeros((10,10), np.float)
    values_count = np.zeros((10,10), np.float)

    for i, l in enumerate(bins_l):
        if i==10:
            continue
        for j, e in enumerate(bins_e):
            if j==10:
                continue
            filter = np.logical_and(H[:, 0] == j, H[:, 1] == i)
            found_or_not_filtered = found_or_not[filter]
            found_or_not_filtered_ticl = found_or_not_ticl[filter]

            values_count[i,j] = len(found_or_not_filtered)
            values_obc[i,j] = np.mean(found_or_not_filtered) if values_count[i,j] !=0 else 0
            values_ticl[i,j] = np.mean(found_or_not_filtered_ticl) if values_count[i,j] !=0 else 0


    pos = plt.imshow(values_ticl, vmax=1, vmin=0)

    # # Major ticks
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))

    # Labels for major ticks
    ax.set_xticklabels([str(x) for x in np.arange(0., 1., 0.1)])
    ax.set_yticklabels(np.arange(0, 100, 10))
    plt.colorbar(pos)


def draw_text_page(plt, s):
    text_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    fig,ax = plt.subplots(figsize=(8,3))
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.text(0.5, 0.5, s, horizontalalignment='center',verticalalignment = 'center', transform = ax.transAxes, fontdict=text_font)


def make_plots_from_object_condensation_clustering_analysis(pdfpath, dataset_analysis_dict):
    global truth_energies, found_truth_energies_is_it_found, found_truth_energies_energies_truth
    global  num_real_showers, num_predicted_showers
    global  num_found_g, num_missed_g, num_fakes_g, pdf
    global found_2_truth_energies_predicted_sum, found_2_truth_energies_truth_sum
    global found_2_truth_rotational_distance
    global found_2_truth_predicted_energies
    global found_2_truth_target_energies
    global num_rechits_per_shower, num_rechits_per_segment

    dataset_analysis_dict = convert_dataset_dict_elements_to_numpy(dataset_analysis_dict)
    pdf = PdfPages(pdfpath)


    #################################################################################
    draw_text_page(plt, 'Total energy response')
    pdf.savefig()

    fig = plt.figure()
    histogram_total_window_resolution(plt, fig.axes, dataset_analysis_dict['predicted_total_truth'], dataset_analysis_dict['predicted_total_obc'], dataset_analysis_dict['predicted_total_ticl'])
    pdf.savefig()


    fig = plt.figure()
    histogram_total_window_resolution(plt, fig.axes, dataset_analysis_dict['predicted_total_truth'], dataset_analysis_dict['predicted_total_obc'], dataset_analysis_dict['predicted_total_ticl'], energy_filter=2)
    pdf.savefig()
    #################################################################################



    #################################################################################
    draw_text_page(plt, 'Fake rate comparison')
    pdf.savefig()

    fix, ax = plt.subplots()
    fake_rate_comparison_with_distribution_fo_energy(plt, ax,dataset_analysis_dict['predicted_showers_regressed_energy'], dataset_analysis_dict['predicted_showers_matched_energy'], dataset_analysis_dict['ticl_showers_regressed_energy'], dataset_analysis_dict['ticl_showers_matched_energy'])
    pdf.savefig()

    fix, ax = plt.subplots()
    fake_rate_comparison_with_distribution_fo_energy(plt, ax,dataset_analysis_dict['predicted_showers_regressed_energy'], dataset_analysis_dict['predicted_showers_matched_energy'], dataset_analysis_dict['ticl_showers_regressed_energy'], dataset_analysis_dict['ticl_showers_matched_energy'], energy_filter=2)
    pdf.savefig()
    #################################################################################


    # fig = plt.figure()
    # make_truth_energy_histogram(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'])
    # pdf.savefig()
    #
    #
    # print("XYZ", len(dataset_analysis_dict['predicted_showers_predicted_energy_sum']), len(dataset_analysis_dict['predicted_showers_matched_energy']))
    #
    # a = np.array(dataset_analysis_dict['predicted_showers_predicted_energy_sum'])
    # b =  np.array(dataset_analysis_dict['predicted_showers_regressed_energy'])
    # c =  np.array(dataset_analysis_dict['predicted_showers_matched_energy'])
    #
    # a = a[c==-1]
    # b = b[c==-1]
    # fig = plt.figure()
    # make_fake_energy_sum_histogram(plt, fig.axes, a)
    # pdf.savefig()
    # fig = plt.figure()
    # make_fake_energy_regressed_histogram(plt, fig.axes, b)
    # pdf.savefig()
    #
    #
    # a = np.array(dataset_analysis_dict['ticl_showers_predicted_energy_sum'])
    # b =  np.array(dataset_analysis_dict['ticl_showers_regressed_energy'])
    # c =  np.array(dataset_analysis_dict['ticl_showers_matched_energy'])
    #
    # a = a[c==-1]
    # b = b[c==-1]
    # fig = plt.figure()
    # make_fake_energy_sum_histogram(plt, fig.axes, a, ticl=True)
    # pdf.savefig()
    # fig = plt.figure()
    # make_fake_energy_regressed_histogram(plt, fig.axes, b, ticl=True)
    # pdf.savefig()
    #
    #
    #
    # fig = plt.figure()
    # make_fake_rate_plot_as_function_of_fake_energy(plt, fig.axes, dataset_analysis_dict['predicted_showers_regressed_energy'], dataset_analysis_dict['predicted_showers_matched_energy'], False)
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_fake_rate_plot_as_function_of_fake_energy(plt, fig.axes, dataset_analysis_dict['ticl_showers_regressed_energy'], dataset_analysis_dict['ticl_showers_matched_energy'], False, ticl=True)
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_fake_rate_plot_as_function_of_fake_energy(plt, fig.axes, dataset_analysis_dict['predicted_showers_predicted_energy_sum'], dataset_analysis_dict['predicted_showers_matched_energy_sum'], True)
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_fake_rate_plot_as_function_of_fake_energy(plt, fig.axes, dataset_analysis_dict['ticl_showers_predicted_energy_sum'], dataset_analysis_dict['ticl_showers_matched_energy_sum'], True, ticl=True)
    # pdf.savefig()
    #
    #
    #
    # fig = plt.figure()
    # make_energy_hists(plt, fig.axes, dataset_analysis_dict['predicted_showers_predicted_energy_sum'], dataset_analysis_dict['predicted_showers_matched_energy_sum'], dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_showers_found_or_not'], True)
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_energy_hists(plt, fig.axes, dataset_analysis_dict['ticl_showers_predicted_energy_sum'], dataset_analysis_dict['ticl_showers_matched_energy_sum'], dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], True, ticl=True)
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_response_histograms(plt, fig.axes, dataset_analysis_dict['predicted_showers_predicted_energy_sum'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['predicted_showers_matched_energy_sum'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['predicted_showers_regressed_energy'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['predicted_showers_matched_energy'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1])
    # pdf.savefig()
    #
    #
    #
    #
    # fig = plt.figure()
    # make_response_histograms(plt, fig.axes, dataset_analysis_dict['ticl_showers_predicted_energy_sum'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['ticl_showers_matched_energy_sum'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['ticl_showers_regressed_energy'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['ticl_showers_matched_energy'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1], ticl=True)
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_response_histograms_energy_segmented(plt, fig.axes, dataset_analysis_dict['predicted_showers_predicted_energy_sum'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['predicted_showers_matched_energy_sum'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['predicted_showers_regressed_energy'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['predicted_showers_matched_energy'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1])
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_response_histograms_energy_segmented(plt, fig.axes, dataset_analysis_dict['ticl_showers_predicted_energy_sum'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['ticl_showers_matched_energy_sum'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['ticl_showers_regressed_energy'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1],
    #                          dataset_analysis_dict['ticl_showers_matched_energy'][dataset_analysis_dict['ticl_showers_matched_energy_sum']!=-1], ticl=True)
    # pdf.savefig()
    #
    #
    #
    # fig = plt.figure()
    # # make_truth_predicted_rotational_distance_histogram(plt, fig.axes, dataset_analysis_dict['found_showers_predicted_truth_rotational_difference'])
    # make_truth_predicted_rotational_distance_histogram(
    #     plt, fig.axes, dataset_analysis_dict['predicted_showers_regressed_eta'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                    dataset_analysis_dict['predicted_showers_matched_eta'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                    dataset_analysis_dict['predicted_showers_regressed_phi'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    #                    dataset_analysis_dict['predicted_showers_matched_phi'][dataset_analysis_dict['predicted_showers_matched_energy_sum']!=-1],
    # )
    # pdf.savefig()
    #
    #
    #
    # fig = plt.figure()
    # # make_truth_predicted_rotational_distance_histogram(plt, fig.axes, dataset_analysis_dict['found_showers_predicted_truth_rotational_difference'])
    # make_truth_predicted_rotational_distance_histogram(
    #     plt, fig.axes, dataset_analysis_dict['ticl_showers_regressed_eta'][dataset_analysis_dict['ticl_showers_matched_eta']!=-1],
    #                    dataset_analysis_dict['ticl_showers_matched_eta'][dataset_analysis_dict['ticl_showers_matched_eta']!=-1],
    #                    dataset_analysis_dict['ticl_showers_regressed_phi'][dataset_analysis_dict['ticl_showers_matched_eta']!=-1],
    #                    dataset_analysis_dict['ticl_showers_matched_phi'][dataset_analysis_dict['ticl_showers_matched_eta']!=-1],
    # )
    # pdf.savefig()
    #
    #
    # for vis_dict in dataset_analysis_dict['visualized_segments']:
    #     visualize_the_segment(plt, vis_dict['truth_showers'], vis_dict['pred_and_truth_dict'], vis_dict['feature_dict'],
    #         vis_dict['ticl_showers'], vis_dict['predicted_showers'], vis_dict['coords_representatives'], dataset_analysis_dict['distance_threshold'])
    #     pdf.savefig()


    # pdf2 = PdfPages("separate.pdf")
    # for vis_dict in dataset_analysis_dict['visualized_segments']:
    #     visualize_the_segment_separate(pdf2, plt, vis_dict['truth_showers'], vis_dict['x'], vis_dict['y'],
    #         vis_dict['prediction_all'], vis_dict['ticl_showers'], vis_dict['predicted_showers'], vis_dict['coords_representatives'], dataset_analysis_dict['distance_threshold'])
    #     # pdf.savefig()
    # pdf2.close()
    #
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_energy(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_showers_found_or_not'])
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_energy(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], ticl=True)
    # pdf.savefig()
    #
    #
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_local_density(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_showers_found_or_not'])
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_local_density(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], ticl=True)
    # pdf.savefig()



    #################################################################################
    draw_text_page(plt, 'Efficiency comparison - local fraction')
    pdf.savefig()


    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], dataset_analysis_dict['truth_shower_energies'])
    pdf.savefig()


    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], dataset_analysis_dict['truth_shower_energies'], energy_filter=2)
    pdf.savefig()


    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], dataset_analysis_dict['truth_shower_energies'], make_segments=True)
    pdf.savefig()
    #################################################################################



    #################################################################################
    draw_text_page(plt, 'Efficiency comparison - closest particle distance')
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], dataset_analysis_dict['truth_shower_energies'])
    pdf.savefig()


    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], dataset_analysis_dict['truth_shower_energies'], energy_filter=2)
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], dataset_analysis_dict['truth_shower_energies'], make_segments=True)
    pdf.savefig()
    #################################################################################



    #################################################################################
    draw_text_page(plt, 'Efficiency comparison - truth energy')
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_truth_energy(plt, fig.axes, dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], dataset_analysis_dict['truth_shower_energies'])
    pdf.savefig()

    #################################################################################



    #################################################################################
    draw_text_page(plt, 'Response comparison - local fraction')
    pdf.savefig()


    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], dataset_analysis_dict['truth_shower_energies'])
    pdf.savefig()


    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], dataset_analysis_dict['truth_shower_energies'], energy_filter=2)
    pdf.savefig()

    fig1, fig2 = response_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes, dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], dataset_analysis_dict['truth_shower_energies'], make_segments=True)
    pdf.savefig(fig1)
    pdf.savefig(fig2)

    #################################################################################
    draw_text_page(plt, 'Response comparison - closest particle distance')
    pdf.savefig()

    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], dataset_analysis_dict['truth_shower_energies'])
    pdf.savefig()


    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], dataset_analysis_dict['truth_shower_energies'], energy_filter=2)
    pdf.savefig()


    fig1, fig2 = response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], dataset_analysis_dict['truth_shower_energies'], make_segments=True)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    #################################################################################


    #################################################################################
    draw_text_page(plt, 'Response comparison - function of truth energy')
    pdf.savefig()


    fig = plt.figure()
    response_curve_comparison_with_distribution_fo_energy(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'])
    pdf.savefig()

    fig = plt.figure()
    response_curve_comparison_with_distribution_fo_energy(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_shower_matched_energies_regressed'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], energy_filter=2)
    pdf.savefig()

    #################################################################################

    #
    # fig, ax = plt.subplots()
    # hist_2d_found_efficiency_vs_local_fraction_and_truth_shower_energy(plt, ax, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_shower_local_density'], dataset_analysis_dict['truth_showers_found_or_not'], dataset_analysis_dict['truth_showers_found_or_not_ticl'])
    # pdf.savefig()



    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_closest_particle_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_showers_found_or_not'])
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_closest_particle_distance(plt, fig.axes, dataset_analysis_dict['truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], ticl=True)
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_energy_response_curve_as_a_function_of_truth_energy(plt, fig.axes, dataset_analysis_dict[
    #     'truth_shower_energies'], dataset_analysis_dict['truth_shower_matched_energies_regressed'])
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_energy_response_curve_as_a_function_of_truth_energy(plt, fig.axes, dataset_analysis_dict[
    #     'truth_shower_energies'], dataset_analysis_dict['truth_shower_matched_energies_regressed_ticl'], ticl=True)
    # pdf.savefig()

    # fig = plt.figure()
    # make_energy_response_curve_as_a_function_of_closest_particle_distance(plt, fig.axes, dataset_analysis_dict[
    #     'truth_shower_closest_particle_distance'], dataset_analysis_dict[
    #     'truth_shower_energies_sum'], dataset_analysis_dict['truth_shower_matched_energies_sum'])
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_energy_response_curve_as_a_function_of_closest_particle_distance(plt, fig.axes, dataset_analysis_dict[
    #     'truth_shower_closest_particle_distance'], dataset_analysis_dict[
    #     'truth_shower_energies_sum'], dataset_analysis_dict['truth_shower_matched_energies_sum_ticl'], ticl=True)
    # pdf.savefig()
    #
    #
    #
    # fig = plt.figure()
    # make_energy_response_curve_as_a_function_of_local_energy_density(plt, fig.axes, dataset_analysis_dict[
    #     'truth_shower_local_density'], dataset_analysis_dict[
    #     'truth_shower_energies_sum'], dataset_analysis_dict['truth_shower_matched_energies_sum'])
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_energy_response_curve_as_a_function_of_local_energy_density(plt, fig.axes, dataset_analysis_dict[
    #     'truth_shower_local_density'], dataset_analysis_dict[
    #     'truth_shower_energies_sum'], dataset_analysis_dict['truth_shower_matched_energies_sum_ticl'], ticl=True)
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_pt(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_shower_etas'], dataset_analysis_dict['truth_showers_found_or_not'])
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_pt(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_shower_etas'], dataset_analysis_dict['truth_showers_found_or_not_ticl'], ticl=True)
    # pdf.savefig()
    #
    #
    #
    #
    # fig = plt.figure()
    # make_real_predicted_number_of_showers_histogram(plt, fig.axes, dataset_analysis_dict['num_real_showers'], dataset_analysis_dict['num_predicted_showers'])
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_real_predicted_number_of_showers_histogram(plt, fig.axes, dataset_analysis_dict['num_real_showers'], dataset_analysis_dict['num_predicted_showers_ticl'], ticl=True)
    # pdf.savefig()
    #
    #
    #
    # fig = plt.figure()
    # make_histogram_of_number_of_rechits_per_shower(plt, fig.axes, dataset_analysis_dict['num_rechits_per_truth_shower'])
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_histogram_of_number_of_rechits_per_segment(plt, fig.axes, dataset_analysis_dict['num_rechits_per_window'])
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_histogram_of_number_of_showers_per_segment(plt, fig.axes, dataset_analysis_dict['num_real_showers'])
    # pdf.savefig()
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_number_of_truth_showers(plt, fig.axes, dataset_analysis_dict['num_real_showers'], dataset_analysis_dict['num_found_showers'], dataset_analysis_dict['num_missed_showers'], dataset_analysis_dict['num_fake_showers'], dataset_analysis_dict['num_predicted_showers'])
    # pdf.savefig()
    #
    #
    # fig = plt.figure()
    # make_found_showers_plot_as_function_of_number_of_truth_showers(plt, fig.axes, dataset_analysis_dict['num_real_showers'], dataset_analysis_dict['num_found_showers_ticl'], dataset_analysis_dict['num_missed_showers_ticl'], dataset_analysis_dict['num_fake_showers_ticl'], dataset_analysis_dict['num_predicted_showers_ticl'], ticl=True)
    # pdf.savefig()
    #
    # # fig = plt.figure()
    # # make_found_showers_plot_as_function_of_number_of_truth_showers_2(plt, fig.axes, dataset_analysis_dict['num_real_showers'], dataset_analysis_dict['num_found_showers_ticl'], dataset_analysis_dict['num_missed_showers_ticl'], dataset_analysis_dict['num_fake_showers_ticl'], dataset_analysis_dict['num_predicted_showers_ticl'], ticl=True)
    # # pdf.savefig()
    #


    pdf.close()
