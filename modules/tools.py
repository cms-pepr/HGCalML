
from plotting_tools import plotter_fraction_colors, snapshot_movie_maker_Nplots, plotter_2d
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
import multiprocessing
from multiprocessing import Process
import numpy as np
import gc
import copy
import ctypes
import matplotlib.pyplot as plt

def make_shared(arr):
    fulldim=1
    shapeinfo=arr.shape
    flattened = np.reshape(arr,[-1])
    shared_array_base = multiprocessing.RawArray(ctypes.c_float, flattened)
    shared_array = np.ctypeslib.as_array(shared_array_base)#.get_obj())
    #print('giving shape',shapeinfo)
    shared_array = shared_array.reshape(shapeinfo)
    return shared_array

class plot_pred_during_training(object):
    def __init__(self, 
               samplefile,
               output_file,
               use_event=0,
               x_index = 5,
               y_index = 6,
               z_index = 7,
               e_index = 0,
               cut_z=None,
               plotter=None,
               plotfunc=None,
               afternbatches=-1,
               on_epoch_end=True,
               decay_function=None
                 ):
        
        self.x_index = x_index 
        self.y_index = y_index 
        self.z_index = z_index 
        self.e_index = e_index 
        self.cut_z=cut_z
        if self.cut_z is not None:
            if 'pos' in self.cut_z:
                self.cut_z = 1.
            elif 'neg' in self.cut_z:
                self.cut_z = -1.
        
        self.decay_function=decay_function
        self.callback = PredictCallback(
            samplefile=samplefile,
            function_to_apply=self.make_plot, #needs to be function(counter,[model_input], [predict_output], [truth])
                 after_n_batches=afternbatches,
                 on_epoch_end=on_epoch_end,
                 use_event=use_event,
                 decay_function=self.decay_function)
        
        self.output_file=output_file
        if plotter is not None:
            self.plotter = plotter
        else:
            self.plotter = plotter_fraction_colors(output_file=output_file)
            self.plotter.gray_noise=False
        if plotfunc is not None:
            self.plotfunc=plotfunc
        else:
            self.plotfunc=None
        
    
    def make_plot(self,call_counter,feat,predicted,truth):
        # self.call_counter,self.td.x,predicted,self.td.y
        f = predicted[0][0] #list entry 0, 0th event
        if call_counter==0:
            f = truth[0][0] # make the first epoch be the truth plot
        feat = feat[0][0] #list entry 0, 0th event
        x = feat[:,self.x_index]
        y = feat[:,self.y_index]
        z = feat[:,self.z_index]
        e = feat[:,self.e_index]
        
        if self.cut_z is not None:
            x = x[z > self.cut_z]
            y = y[z > self.cut_z]
            e = e[z > self.cut_z]
            f = f[z > self.cut_z]
            z = z[z > self.cut_z]
            
        #send this to another fork so it does not prevent the training to continue
        def worker():
        
            outfile = self.output_file+str(call_counter)
            self.plotter.set_data(x,y,z,e,f)
            if self.plotfunc is not None:
                self.plotfunc()
            else:
                self.plotter.output_file=outfile
                self.plotter.plot3d()
                self.plotter.save_image()
    
        p = Process(target=worker)#, args=(,))
        p.start()
        
        
        

class plot_truth_pred_plus_coords_during_training(plot_pred_during_training):
    def __init__(self, 
               samplefile,
               output_file,
               use_event=0,
               x_index = 5,
               y_index = 6,
               z_index = 7,
               e_index = 0,
               pred_fraction_end = 20,
               transformed_x_index = 21,
               transformed_y_index = 22,
               transformed_z_index = 23,
               transformed_e_index = 24,
               cut_z=None,
               afternbatches=-1,
               on_epoch_end=True,
               only_truth_and_pred=False,
               plottertype=plotter_fraction_colors ,
               **kwargs
                 ):
        plot_pred_during_training.__init__(self,samplefile,output_file,use_event,
                                           x_index,y_index,z_index,
                                           e_index=e_index,
                                           cut_z=cut_z,
                                           plotter=None,
                                           plotfunc=None,
                                           afternbatches=afternbatches,
                                           on_epoch_end=on_epoch_end, **kwargs)
        if only_truth_and_pred:
            self.snapshot_maker = snapshot_movie_maker_Nplots(output_file, nplots=2, plottertype=plottertype)
        else:
            self.snapshot_maker = snapshot_movie_maker_Nplots(output_file, nplots=4, plottertype=plottertype)
        
        self.transformed_x_index = transformed_x_index
        self.transformed_y_index = transformed_y_index
        self.transformed_z_index = transformed_z_index
        self.transformed_e_index = transformed_e_index
        self.only_truth_and_pred = only_truth_and_pred
        
        self.pred_fraction_end=pred_fraction_end
        self.threadlist=[]
        self.usenfeatureslist=1
        
        
        
        
    def end_job(self):
        self.snapshot_maker.end_job()
    
    def _make_e_zsel(self,z,e):
        if self.cut_z is not None:
            zsel = z > self.cut_z
            esel = e > 0.
            zsel = np.expand_dims(zsel,axis=1)
            esel = np.expand_dims(esel,axis=1)
            esel = np.concatenate([esel,zsel],axis=-1)
            esel = np.all(esel,axis=-1)
            return esel
        else: return e>0.

    def _make_plot(self,call_counter,feat,predicted,truth):
        self.snapshot_maker.glob_counter=call_counter
        
        pred  = copy.deepcopy(predicted[0]) #not a list anymore, 0th event
        truth = copy.deepcopy(truth[0]) # make the first epoch be the truth plot
        feat  = copy.deepcopy(feat[0]) #not a list anymore 0th event
        
        e = feat[:,self.e_index]
        z = feat[:,self.z_index]
        x = feat[:,self.x_index]
        y = feat[:,self.y_index]
        
        esel = e>0
        ez_sel = self._make_e_zsel(z,e)
        
        tx,ty,tz,te,dummy=None,None,None,None,None
        
        if not self.only_truth_and_pred:
            tx = pred[:,self.transformed_x_index]
            dummy = np.zeros_like(tx)
            if self.transformed_y_index is not None:
                ty = pred[:,self.transformed_y_index]
            else:
                ty=dummy
            if self.transformed_z_index is not None:
                tz = pred[:,self.transformed_z_index]
            else:
                tz=dummy
            if self.transformed_e_index is not None:
                te = pred[:,self.transformed_e_index]
            else:
                te=dummy
            
        truth_fracs = truth[:,0:-1] #last one is energy
        pred_fracs = pred[:,:self.pred_fraction_end]
 
        sel_truth_fracs = truth_fracs[ez_sel]
 
        self.snapshot_maker.reset()
        self.snapshot_maker.set_plot_data(0, x[ez_sel], y[ez_sel], z[ez_sel], e[ez_sel], sel_truth_fracs) #just the truth plot
        self.snapshot_maker.set_plot_data(1, x[ez_sel], y[ez_sel], z[ez_sel], e[ez_sel], pred_fracs[ez_sel]) #just the predicted plot
        
        if not self.only_truth_and_pred:
            self.snapshot_maker.set_plot_data(2,  tz[ez_sel], ty[ez_sel], tx[ez_sel], e[ez_sel], sel_truth_fracs) #just the predicted plot
            self.snapshot_maker.set_plot_data(3,  te[ez_sel], ty[ez_sel], tx[ez_sel], e[ez_sel], sel_truth_fracs) #just the predicted plot
        
        self.snapshot_maker.make_snapshot()
        
        
    def make_plot(self,call_counter,feat,predicted,truth):
        
        #clean up
        t_alive=[]
        for t in self.threadlist:
            if not t.is_alive():
                t.join()
            else:
                t_alive.append(t)
        self.threadlist = t_alive  
        
        s_feat = make_shared(feat[0])
        if self.usenfeatureslist>1:
            s_feat=[s_feat]
        for i in range(self.usenfeatureslist-1):
            s_feat.append(make_shared(feat[i+1]))
            
        s_predicted = make_shared(predicted[0])
        s_truth = make_shared(truth[0])
        
        del feat,truth,predicted
        
        #send this directly to a fork so it does not interrupt training too much
        p = Process(target=self._make_plot, args=(call_counter,s_feat,s_predicted,s_truth))
        self.threadlist.append(p)
        gcisenabled = gc.isenabled()
        gc.disable()
        p.start()
        if gcisenabled:
            gc.enable()
        #del feat,predicted,truth
        
        
class plot_preclustering_during_training(plot_truth_pred_plus_coords_during_training):
    
    def __init__(self, **kwargs):
        plot_truth_pred_plus_coords_during_training.__init__(self,**kwargs)
        self.usenfeatureslist=2
    #    if confidenceidx is None:
    #        self.confidenceidx = self.pred_fraction_end
        
    def _make_plot(self,call_counter,feat,predicted,truth):
        self.snapshot_maker.glob_counter=call_counter
        
        pred  = copy.deepcopy(predicted[0]) #not a list anymore, 0th event
        if self.usenfeatureslist>1:
            truth = copy.deepcopy(feat[1][0]) # make the first epoch be the truth plot
            feat  = copy.deepcopy(feat[0][0]) #not a list anymore 0th event
        else:
            truth = copy.deepcopy(truth[0]) 
            feat  = copy.deepcopy(feat[0])
        
        x,y,z,e,dummy=None,None,None,None,None
        
        e = feat[:,self.e_index]
        dummy = np.zeros_like(e)
        if self.z_index is not None:
            z = feat[:,self.z_index]
        else:
            z = dummy
        if self.x_index is not None:
            x = feat[:,self.x_index]
        else:
            x = dummy
        if self.y_index is not None:
            y = feat[:,self.y_index]
        else:
            y = dummy
        
        
        esel = e>0
        ez_sel = self._make_e_zsel(z,e)
        
        tx,ty,tz,te,dummy=None,None,None,None,None
        
        if not self.only_truth_and_pred:
            tx = pred[:,self.transformed_x_index]
            dummy = np.zeros_like(tx)
            if self.transformed_y_index is not None:
                ty = pred[:,self.transformed_y_index]
            else:
                ty=dummy
            if self.transformed_z_index is not None:
                tz = pred[:,self.transformed_z_index]
            else:
                tz=dummy
            if self.transformed_e_index is not None:
                te = pred[:,self.transformed_e_index]
            else:
                te=dummy
            
        truth_fracs = truth  #[:,0:-1] #last one is energy
        if self.usenfeatureslist<=1:
            truth_fracs = truth[:,0:-1]
        
        #confidence=pred[:,:,self.confidenceidx]
        
        sel_truth_fracs = truth_fracs[ez_sel]
 
        self.snapshot_maker.reset()
        self.snapshot_maker.set_plot_data(0, x[ez_sel], y[ez_sel], z[ez_sel], e[ez_sel], sel_truth_fracs) #just the truth plot
        self.snapshot_maker.set_plot_data(1, tx[ez_sel], ty[ez_sel], tz[ez_sel], e[ez_sel], sel_truth_fracs) #just the predicted plot
        
        if not self.only_truth_and_pred:
            self.snapshot_maker.set_plot_data(2,  te[ez_sel], tz[ez_sel], ty[ez_sel] , e[ez_sel], sel_truth_fracs) #just the predicted plot
            self.snapshot_maker.set_plot_data(3,  tx[ez_sel], tz[ez_sel], ty[ez_sel], e[ez_sel], sel_truth_fracs) #just the predicted plot
        
        self.snapshot_maker.make_snapshot()
        
        
        
      
class plot_pixel_clustering_during_training(plot_truth_pred_plus_coords_during_training):
    
    def __init__(self, **kwargs):
        from plotting_tools import plotter_3d
        plot_truth_pred_plus_coords_during_training.__init__(self,
                                                             plottertype=plotter_3d,
                                                             **kwargs)
        self.usenfeatureslist=1
    #    if confidenceidx is None:
    #        self.confidenceidx = self.pred_fraction_end
    
        for pm in self.snapshot_maker.plotters:
            pm.marker_scale=0.3
        
    def _make_plot(self,call_counter,feat,predicted,truth):
        self.snapshot_maker.glob_counter=call_counter
        
        pred  = copy.deepcopy(predicted[0]) #not a list anymore, 0th event
        if self.usenfeatureslist>1:
            truth = copy.deepcopy(feat[1][0]) # make the first epoch be the truth plot
            feat  = copy.deepcopy(feat[0][0]) #not a list anymore 0th event
        else:
            truth = copy.deepcopy(truth[0]) 
            feat  = copy.deepcopy(feat[0])
            
        print('truthtruth',truth.shape)
        npixels = truth.shape[0]
        reshaping = [truth.shape[0]*truth.shape[1], -1]
        truth = np.reshape(truth, reshaping)
        feat = np.reshape(feat, reshaping)
        pred = np.reshape(pred, reshaping)
        
        x,y,z,e,dummy=None,None,None,None,None
        
        e = feat[:,self.e_index]
        dummy = np.zeros_like(e)
        if self.z_index is not None:
            z = feat[:,self.z_index]
        else:
            z = dummy
        if self.x_index is not None:
            x = feat[:,self.x_index]
        else:
            x = dummy
        if self.y_index is not None:
            y = feat[:,self.y_index]
        else:
            y = dummy
        
        
        ez_sel = truth[:,0]>0 #mask
        
        tx,ty,tz,te,dummy=None,None,None,None,None
        
        if not self.only_truth_and_pred:
            tx = pred[:,self.transformed_x_index]
            dummy = np.zeros_like(tx)
            if self.transformed_y_index is not None:
                ty = pred[:,self.transformed_y_index]
            else:
                ty=dummy
            if self.transformed_z_index is not None:
                tz = pred[:,self.transformed_z_index]
            else:
                tz=dummy
            if self.transformed_e_index is not None:
                te = pred[:,self.transformed_e_index]
            else:
                te=dummy
                
        pixrange = np.arange(npixels)
        y = np.tile(np.expand_dims(pixrange,axis=0), [npixels,1])
        y = np.reshape(y, [npixels*npixels])
        z = np.tile(np.expand_dims(pixrange,axis=1), [1,npixels])
        z = np.reshape(z, [npixels*npixels])
        
            
        truth_fracs = feat[:,0:3]  #[:,0:-1] #last one is energy
        
        #confidence=pred[:,:,self.confidenceidx]
        
        sel_truth_fracs = truth_fracs[ez_sel]
        sel_truth_fracs = sel_truth_fracs/ np.expand_dims(sel_truth_fracs.max(axis=-1),axis=1)
 
        self.snapshot_maker.reset()
        self.snapshot_maker.set_plot_data(0, x[ez_sel], y[ez_sel], z[ez_sel], e[ez_sel], sel_truth_fracs) #just the truth plot
        self.snapshot_maker.set_plot_data(1, tx[ez_sel], ty[ez_sel], tz[ez_sel], e[ez_sel], sel_truth_fracs) #just the predicted plot
        
        if not self.only_truth_and_pred:
            self.snapshot_maker.set_plot_data(2,  te[ez_sel], tz[ez_sel], ty[ez_sel] , e[ez_sel], sel_truth_fracs) #just the predicted plot
            self.snapshot_maker.set_plot_data(3,  tx[ez_sel], tz[ez_sel], ty[ez_sel], e[ez_sel], sel_truth_fracs) #just the predicted plot
        
        self.snapshot_maker.make_snapshot()
        
        

class plot_pixel_1D_clustering_during_training(plot_pred_during_training):
    def __init__(self, 
               samplefile,
               output_file,
               use_event=0,
               afternbatches=-1,
               on_epoch_end=False,
               **kwargs
                 ):
        plot_pred_during_training.__init__(self,samplefile,output_file,use_event,
                                           afternbatches=afternbatches,
                                           on_epoch_end=on_epoch_end, 
                                           plotter=-1,
                                           **kwargs)
        import os
        os.system('mkdir -p '+output_file)
        self.tmp_out_prefix=output_file+'/tmp/'
        os.system('mkdir -p '+self.tmp_out_prefix)
        self.firstcall=True
        self.pix_x=None
        self.pix_y=None
        self.npixels = 0
        self.glob_counter = 0
        self.offset_counter = 0
        self.plotter_left  = plotter_2d()
        self.plotter_right = plotter_2d()
        self.threadlist = []
        self.rjust=10
        
    def _make_plot(self,call_counter,feat,predicted,truth):
        self.glob_counter = call_counter
        pred  = copy.deepcopy(predicted[0]) #not a list anymore, 0th event
        truth = copy.deepcopy(truth[0]) # make the first epoch be the truth plot
        feat  = copy.deepcopy(feat[0]) #not a list anymore 0th event
        
        '''
        t_mask =    tf.reshape(truth[:,:,:,0:1], reshaping) 
    true_pos =  tf.reshape(truth[:,:,:,1:3], reshaping) 
    true_ID =   tf.reshape(truth[:,:,:,3:6], reshaping) 
    true_dim =  tf.reshape(truth[:,:,:,6:8], reshaping) 
    n_objects = truth[:,0,0,8]
    # B x P x P x N
    
    
    #make it all lists
    p_beta   =  tf.reshape(pred[:,:,:,0:1], reshaping)
    p_tpos   =  tf.reshape(pred[:,:,:,1:3], reshaping)
    p_ID     =  tf.reshape(pred[:,:,:,3:6], reshaping)
    p_dim    =  tf.reshape(pred[:,:,:,6:8], reshaping)
    p_object  = pred[:,0,0,8]
    p_ccoords = tf.reshape(pred[:,:,:,9:10], reshaping)
        '''
        
        
        if self.firstcall:
            self.npixels = truth.shape[0]
            pixrange = np.arange(self.npixels)
            self.pix_x = np.tile(np.expand_dims(pixrange,axis=0), [self.npixels,1])
            self.pix_x = np.reshape(self.pix_x, [self.npixels*self.npixels])
            self.pix_y = np.tile(np.expand_dims(pixrange,axis=1), [1,self.npixels])
            self.pix_y = np.reshape(self.pix_y, [self.npixels*self.npixels])
            
        
        reshaping = [self.npixels*self.npixels, -1]
        truth = np.reshape(truth, reshaping)
        feat = np.reshape(feat, reshaping)
        pred = np.reshape(pred, reshaping)
        
        colours = feat[:,0:3]
        colours /= np.expand_dims(colours.max(axis=-1),axis=1)
        
        ccorrds = pred[:,9]
        betas = pred[:,0] + 1e-3
        
        mask = truth[:,0]>0
        
        fig, axs = plt.subplots(1, 2)# subplot_kw=dict(projection='2d'))
        for i in range(len(axs)):
            axs[i].clear()
            axs[i].autoscale(True)
            axs[i].set_aspect('auto')
            axs[i].relim() 
            
        self.plotter_left.set_data(x=self.pix_x[mask], y=self.pix_y[mask], z=colours[mask])
        self.plotter_right.set_data(x=ccorrds[mask], y=betas[mask], z=colours[mask])
     
        self.plotter_left.plot2d(ax=axs[0])
        self.plotter_right.plot2d(ax=axs[1])
        #axs[1].set_yscale("log", nonposy='clip')
        for i in range(len(axs)):
            axs[i].set_aspect('auto')
            axs[i].relim() 
            
        outputname = self.tmp_out_prefix+str(self.glob_counter+self.offset_counter).rjust(self.rjust, '0')+'.png'
        fig.savefig(outputname, dpi=300)
        fig.clear()
        plt.close(fig)
        plt.clf()
        plt.cla()
        plt.close() 
        
        
    def increment_counter(self):
        self.glob_counter+=1 
        
    def end_job(self):
        os.system('ffmpeg -r 20 -f image2  -i '+ self.tmp_out_prefix +'$w%10d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 20 '+ self.output_file +'_movie.mp4')
        os.system('rm -f '+ self.tmp_out_prefix +'*.png')   
        
    def make_plot(self,call_counter,feat,predicted,truth):
        
        #clean up
        t_alive=[]
        for t in self.threadlist:
            if not t.is_alive():
                t.join()
            else:
                t_alive.append(t)
        self.threadlist = t_alive  
        
        s_feat = make_shared(feat[0])
            
        s_predicted = make_shared(predicted[0])
        s_truth = make_shared(truth[0])
        
        del feat,truth,predicted
        
        #send this directly to a fork so it does not interrupt training too much
        p = Process(target=self._make_plot, args=(call_counter,s_feat,s_predicted,s_truth))
        self.threadlist.append(p)
        gcisenabled = gc.isenabled()
        gc.disable()
        p.start()
        if gcisenabled:
            gc.enable()      
        