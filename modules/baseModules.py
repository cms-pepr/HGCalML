print(__name__,"deprecated, import directly from DeepJetCore.DJCLayers")
from DeepJetCore.DJCLayers import LayerWithMetrics

@classmethod
def from_config(cls, config):
    # Handle backward compatibility with older configurations
    if '_promptnames' in config:
        # Remove or handle the _promptnames keyword argument
        del config['_promptnames']
    return super(LayerWithMetrics, cls).from_config(config)

LayerWithMetrics.from_config = from_config
