
from plotting_tools import plotter_fraction_colors
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from multiprocessing import Process

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
               plotfunc=None
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
        
        
        self.callback = PredictCallback(
            samplefile=samplefile,
            function_to_apply=self.make_plot, #needs to be function(counter,[model_input], [predict_output], [truth])
                 after_n_batches=-1,
                 on_epoch_end=True,
                 use_event=use_event)
        
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
        