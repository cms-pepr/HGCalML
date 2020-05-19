
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from plotting_tools import base_plotter, plotter_3d
from index_dicts import create_index_dict
from numba import jit
import math
from matplotlib.backends.backend_pdf import PdfPages


'''
Everything here assumes non flattened format:

B x V x F

'''
# tools before making the ccoords plot working on all events
@jit(nopython=True)        
def c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold):
    

    for e in range(len(betasel)):
        selected = []
        for si in range(len(sorting[e])):
            i = sorting[e][si]
            use=True
            for s in selected:
                distance = math.sqrt( (s[0]-ccoords[e][i][0])**2 +  (s[1]-ccoords[e][i][1])**2 )
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
                          beta_threshold, distance_threshold)
    
    
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
                                  cmap=None
                                ):
    
    #data = create_index_dict(truth,pred,usetf=False)
    
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(predBeta.shape)>1:
        predBeta = np.array(predBeta[:,0])
    
    if np.max(predBeta)>1.:
        raise ValueError("make_cluster_coordinates_plot: at least one beta value is above 1. Check your model!")
    
    ax.set_aspect(aspect=1.)
    #print(truthHitAssignementIdx)
    if cmap is None:
        rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    else:
        rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    rgbcolor[truthHitAssignementIdx<0]=[0.98,0.98,0.98]
    #print(rgbcolor)
    #print(rgbcolor.shape)
    alphas = predBeta
    alphas[alphas<0.01] = 0.01
    alphas = np.expand_dims(alphas, axis=1)
    
    rgba_cols = np.concatenate([rgbcolor,alphas],axis=-1)
    rgb_cols = np.concatenate([rgbcolor,np.zeros_like(alphas+1.)],axis=-1)
    
    sorting = np.reshape(np.argsort(alphas, axis=0), [-1])
    
    ax.scatter(predCCoords[:,0][sorting],
              predCCoords[:,1][sorting],
              s=.25*matplotlib.rcParams['lines.markersize'] ** 2,
              c=rgba_cols[sorting])
    
    
    if beta_threshold < 0. or beta_threshold > 1 or distance_threshold<0:
        return
    
    data = {'predBeta': np.expand_dims(np.expand_dims(predBeta,axis=-1),axis=0),
            'predCCoords': np.expand_dims(predCCoords,axis=0)}
    

    if identified_coords is None:
        #run the inference part
        identified = collectoverthresholds(data,beta_threshold,distance_threshold)[0,:,0] #V


        ax.scatter(predCCoords[:,0][identified],
                  predCCoords[:,1][identified],
                  s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
                  c='#000000',#rgba_cols[identified],
                  marker='+')

        return identified
    else:
        ax.scatter(identified_coords[:, 0],
                   identified_coords[:, 1],
                  s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
                  c='#000000',#rgba_cols[identified],
                  marker='+')
        for plus in identified_coords:
            ax.add_artist(plt.Circle((plus[0], plus[1]), distance_threshold,  color='black', fill=False))


def make_original_truth_shower_plot(plt, ax,
                                    truthHitAssignementIdx,                      
                                    recHitEnergy, 
                                    recHitX,
                                    recHitY,
                                    recHitZ,
                                    cmap=None):
    
    
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
        
        
    pl = plotter_3d(output_file="/tmp/plot")#will be ignored
    if cmap is None:
        rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    else:
        rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    rgbcolor[truthHitAssignementIdx<0]=[0.92,0.92,0.92]
    
    pl.set_data(x = recHitX , y=recHitY   , z=recHitZ, e=recHitEnergy , c =rgbcolor)
    pl.marker_scale=2.
    pl.plot3d(ax=ax)
    
    
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
                                    identified=None):
    
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

    size_scaling = np.log(recHitEnergy+1)+0.1
    size_scaling /=  np.max(size_scaling)

    ax.scatter(recHitPhi,
              recHitEta,
              s=.25*size_scaling,
              c=rgbcolor)
     
    _, truth_idxs = np.unique(truthHitAssignementIdx,return_index=True)
    
    truth_size_scaling=np.log(truthEnergy[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0] +1.)+0.1
    truth_size_scaling /=  np.max(truth_size_scaling)
    
    ax.scatter(truthPhi[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0],
              truthEta[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0],
              s=100.*truth_size_scaling,
              c=rgbcolor[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0],
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
    
    


def make_truth_energy_histogram(plt, ax, truth_energies):
    plt.figure()
    plt.hist(truth_energies, bins=50, histtype='step')
    plt.xlabel("Truth shower energy")
    plt.ylabel("Frequency")
    plt.title('Truth energies')


def make_response_histograms(plt, ax, found_showers_predicted_sum, found_showers_truth_sum, found_showers_predicted_energies, found_showers_target_energies):
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
    plt.title('Response curves')


def make_truth_predicted_rotational_distance_histogram(plt, ax, rotational_distance_data):
    rotational_distance_data = np.array(rotational_distance_data)
    rotational_distance_data[rotational_distance_data > 0.2] = 0.2


    plt.figure()
    plt.hist(rotational_distance_data, bins=20, histtype='step')
    plt.xlabel("Rotational distance between true and predicted eta/phi coordinates")
    plt.ylabel("Frequency")
    plt.title('Positional performance')


def make_found_showers_plot_as_function_of_energy(plt, ax, energies, found_or_not):
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


    plt.errorbar(centers, mean, std, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    plt.xticks(centers)
    plt.xlabel('Shower energy')
    plt.ylabel('% found')
    plt.title('Function of energy')

def make_real_predicted_number_of_showers_histogram(plt, ax, num_real_showers, num_predicted_showers):
    plt.hist(num_real_showers, bins=np.arange(0,50), histtype='step')
    plt.hist(num_predicted_showers, bins=np.arange(0,70), histtype='step')
    plt.xlabel('Num showers')
    plt.ylabel('Frequency')
    plt.legend(['Real showers','Predicted showers'])
    plt.title('Histogram of predicted/real number of showers')


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


def visualize_the_segment(plt, classes_this_segment, x_this_segment, y_this_segment, pred_this_segment, labels,
                          coords_representative_predicted_showers, distance_threshold):
    fig = plt.figure(figsize=(16, 12))
    gs = plt.GridSpec(3, 2)

    ax = [fig.add_subplot(gs[0, 0], projection='3d'),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 0], projection='3d'),
          fig.add_subplot(gs[1, 1], projection='3d'), ]

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

    cmap = createRandomizedColors('jet')

    make_original_truth_shower_plot(plt, ax[0], classes_this_segment * 0, x_this_segment[:, 0],
                                    x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7],
                                    cmap=plt.get_cmap('Wistia'))
    make_cluster_coordinates_plot(plt, ax[1], classes_this_segment, pred_this_segment[:, -6],
                                  pred_this_segment[:, -2:],
                                  identified_coords=coords_representative_predicted_showers, cmap=cmap,
                                  distance_threshold=distance_threshold)
    #
    # make_original_truth_shower_plot(plt, ax[4], 1-identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
    # make_original_truth_shower_plot(plt, ax[5], identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))

    # wrt predicted colors
    make_original_truth_shower_plot(plt, ax[2], classes_this_segment, x_this_segment[:, 0], x_this_segment[:, 5],
                                    x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap)
    make_original_truth_shower_plot(plt, ax[3], labels, x_this_segment[:, 0], x_this_segment[:, 5],
                                    x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap)

    # make_cluster_coordinates_plot(plt, ax[3], labels, pred_this_segment[:, -6], pred_this_segment[:, -2:], identified_coords=coords_representative_predicted_showers, cmap=cmap)



def make_found_showers_plot_as_function_of_number_of_truth_showers(plt, ax, num_real, num_found, num_missed, num_fakes, num_predicted):

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


def make_plots_from_object_condensation_clustering_analysis(pdfpath, dataset_analysis_dict):
    global truth_energies, found_truth_energies_is_it_found, found_truth_energies_energies_truth
    global  num_real_showers, num_predicted_showers
    global  num_found_g, num_missed_g, num_fakes_g, pdf
    global found_2_truth_energies_predicted_sum, found_2_truth_energies_truth_sum
    global found_2_truth_rotational_distance
    global found_2_truth_predicted_energies
    global found_2_truth_target_energies
    global num_rechits_per_shower, num_rechits_per_segment

    pdf = PdfPages(pdfpath)

    fig = plt.figure()
    make_truth_energy_histogram(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'])
    pdf.savefig()



    # fig = plt.figure()
    # make_response_histograms(plt, fig.axes, dataset_analysis_dict['found_showers_predicted_sum'],
    #                          dataset_analysis_dict['found_showers_truth_sum'],
    #                          dataset_analysis_dict['found_showers_predicted_energies'],
    #                          dataset_analysis_dict['found_showers_target_energies'])

    # fig = plt.figure()
    # make_truth_predicted_rotational_distance_histogram(plt, fig.axes, dataset_analysis_dict['found_showers_predicted_truth_rotational_difference'])
    # pdf.savefig()
    for vis_dict in dataset_analysis_dict['visualized_segments']:
        visualize_the_segment(plt, vis_dict['truth_showers'], vis_dict['x'], vis_dict['y'],
            vis_dict['prediction_all'], vis_dict['predicted_showers'], vis_dict['coords_representatives'], dataset_analysis_dict['distance_threshold'])
        pdf.savefig()


    fig = plt.figure()
    make_found_showers_plot_as_function_of_energy(plt, fig.axes, dataset_analysis_dict['truth_shower_energies'], dataset_analysis_dict['truth_showers_found_or_not'])
    pdf.savefig()



    fig = plt.figure()
    make_real_predicted_number_of_showers_histogram(plt, fig.axes, dataset_analysis_dict['num_real_showers'], dataset_analysis_dict['num_predicted_showers'])
    pdf.savefig()




    fig = plt.figure()
    make_histogram_of_number_of_rechits_per_shower(plt, fig.axes, dataset_analysis_dict['num_rechits_per_truth_shower'])
    pdf.savefig()


    fig = plt.figure()
    make_histogram_of_number_of_rechits_per_segment(plt, fig.axes, dataset_analysis_dict['num_rechits_per_window'])
    pdf.savefig()


    fig = plt.figure()
    make_histogram_of_number_of_showers_per_segment(plt, fig.axes, dataset_analysis_dict['num_real_showers'])
    pdf.savefig()

    fig = plt.figure()
    make_found_showers_plot_as_function_of_number_of_truth_showers(plt, fig.axes, dataset_analysis_dict['num_real_showers'], dataset_analysis_dict['num_found_showers'], dataset_analysis_dict['num_missed_showers'], dataset_analysis_dict['num_fake_showers'], dataset_analysis_dict['num_predicted_showers'])
    pdf.savefig()



    pdf.close()
