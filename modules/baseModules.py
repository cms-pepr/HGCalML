print(__name__,">>> move to DeepJetCore soon")

import tensorflow as tf

#for metric layers
class PromptMetric(tf.keras.metrics.Mean):   
    def __init__(self, **kwargs):
        super(PromptMetric, self).__init__(**kwargs)
    
    def update_state(self,*args,**kwargs):
        self.reset_states()#reset and only take last state
        super(PromptMetric, self).update_state(*args,**kwargs)
        
class LayerWithMetrics(tf.keras.layers.Layer):
    def __init__(self, 
                 record_metrics=False,
                 _promptnames=None, **kwargs):
        super(LayerWithMetrics, self).__init__(**kwargs)
        self.prompt_metrics = {}
        self.record_metrics = record_metrics
        if _promptnames is not None:
            for n in _promptnames:
                if not n in self.prompt_metrics.keys():
                    with tf.name_scope(self.name+"/sub/"+n):
                        self.prompt_metrics[n]=PromptMetric(name=n)
        
    def get_config(self):
        config = {'_promptnames': [m[1].name for m in self.prompt_metrics.items()],
                  'record_metrics': self.record_metrics}
        base_config = super(LayerWithMetrics, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
    
    def add_prompt_metric(self,x,name):
        if not self.record_metrics:
            return
        if not name in self.prompt_metrics.keys():
            with tf.name_scope(self.name+"/sub/"+name):
                self.prompt_metrics[name]=PromptMetric(name=name)
        self.add_metric(self.prompt_metrics[name](x))

#class LayerWithMetrics(tf.keras.layers.Layer):
#    def __init__(self, _promptnames=None, **kwargs):
#        super(LayerWithMetrics, self).__init__(**kwargs)
#        self.prompt_metrics={}
#        
#    def register_prompt_metric(self,name): 
#        if name in self.prompt_metrics.keys():
#            return
#        with tf.name_scope(self.name+"/sub/"+name):
#            self.prompt_metrics[name]=PromptMetric(name=name)
#            
#    def add_prompt_metric(self,x,name):
#        if not name in self.prompt_metrics.keys():
#            raise ValueError("Metric with name "+str(name)+" not registred. Register in layers or model constructor.")
#        self.add_metric(self.prompt_metrics[name](x))
