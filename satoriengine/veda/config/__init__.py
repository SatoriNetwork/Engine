import os
import yaml

def get():
    ''' gets configuration out of the yaml file '''
    path = '/Satori/Neuron/config/config.yaml'
    if os.path.exists(path):
        with open(path, mode='r') as f:
            try:
                return yaml.load(f, Loader=yaml.FullLoader) or {}
            except AttributeError:
                try:
                    return yaml.load(f) or {}
                except AttributeError:
                    pass
    return {}
