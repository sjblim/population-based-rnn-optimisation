"""
data_loader_factory.py


Created by limsi on 25/10/2018
"""

from data import mg_loader, ar_loader, ou_data_loaders, vol_data_loaders

from configs import mg_configs, ar_configs, ou_configs, vol_configs


def _set_data_loader():
    _data_loader_class_map = {
        'mackey': mg_loader.MackeyGlassDataLoader,
        'short_mackey': mg_loader.ShortMackeyGlassDataLoader,
        'simple_mackey': mg_loader.SimpleMackeyGlassDataLoader,
        'short_simple': mg_loader.ShortSimpleMackeyGlassDataLoader,
        'ar': ar_loader.AutoregressiveDataLoader,
        'ou': ou_data_loaders.OrnsteinUhlenbeckSimDataLoader,
        'vol': vol_data_loaders.VolDataLoader,
        'noisy': mg_loader.NoisyMackeyGlassDataLoader,
        'norm_vol': vol_data_loaders.NormalisedVolDataLoader
    }

    return _data_loader_class_map


# -----------------------------------------------------------------------------
class DataLoaderFactory:

    _data_loader_class_map = _set_data_loader()
    
    @classmethod
    def get_valid_loaders(cls):
        loader_names = list(cls._data_loader_class_map.keys())
        loader_names.sort()
        return loader_names

    @classmethod
    def _check_loader_name(cls, loader_name):
        if loader_name not in cls._data_loader_class_map:
            raise ValueError("Unrecognised data loader: {}! Valid loaders=[{}]".format(loader_name,
                                                                                        ",".join(
                                                                                            cls.get_valid_loaders())))

    @classmethod
    def make_data_loader(cls, loader_name):

        cls._check_loader_name(loader_name)

        return cls._data_loader_class_map[loader_name]()

    @classmethod
    def get_default_config(cls, loader_name):

        cls._check_loader_name(loader_name)

        loader_class = cls._data_loader_class_map[loader_name]
        data_loader = loader_class()

        if issubclass(loader_class, mg_loader.MackeyGlassDataLoader):
            return mg_configs

        elif issubclass(loader_class, ar_loader.AutoregressiveDataLoader):
            return ar_configs
        elif issubclass(loader_class, ou_data_loaders.OrnsteinUhlenbeckSimDataLoader):
            return ou_configs

        elif issubclass(loader_class, vol_data_loaders.VolDataLoader):
            return vol_configs

        else:
            raise ValueError("Unrecognised argument loader type: {}".format(loader_name))


