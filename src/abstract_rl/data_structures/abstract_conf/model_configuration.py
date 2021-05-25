from os.path import join
import time
from os.path import exists
from os import mkdir

import json

import numpy as np
import torch


from abstract_rl.src.data_structures.abstract_conf.hierarchical_conf import SharedConf
from abstract_rl.src.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.src.env.mcmc_env_wrapper import MCMCEnvWrapper


class ModelConfiguration:
    """
    Represents a model configuration. Contains important elements like the environment or policy. It also
    integrated a syncing mechanism due to the need of target and original data structures for more stable evaluations.
    """

    def __init__(self, conf, root='data'):
        """
        Create a new model configuration with no syncables no data and no environment.
        """

        # settings for the script
        reload = conf['reload']
        conf['reload'] = True

        self.syncables = {}
        self.data = {}
        if not reload:

            # create folder structure
            self.env_name = conf['env_name']
            self.env_dir = join(root, self.env_name)
            if not exists(self.env_dir): mkdir(self.env_dir)
            self.alg_name = conf['alg']
            self.env_dir = join(self.env_dir, self.alg_name)
            if not exists(self.env_dir): mkdir(self.env_dir)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            self.run_root = join(self.env_dir, timestr)
            if not exists(self.run_root): mkdir(self.run_root)

            self.overall_step = 0
            self.env = None
            conf['reload_dir'] = self.run_root
            conf = SharedConf(conf)
            self.add_main('conf', conf)

            self.model_root = join(self.run_root, 'model')
            if not exists(self.model_root): mkdir(self.model_root)

            self.plot_root = join(self.run_root, 'plot')
            if not exists(self.plot_root): mkdir(self.plot_root)

            self.data_dir = join(self.run_root, 'data')
            if not exists(self.data_dir): mkdir(self.data_dir)

        else:
            conf_dir = conf['reload_dir']
            self.run_root = conf_dir

            # load the config as a json file
            with open(join(self.run_root, "run.conf"), "r") as file_conf:
                file_str = file_conf.read()

                conf = json.loads(file_str)
            conf = SharedConf(conf)
            self.add_main('conf', conf)

    def __getitem__(self, key):
        """
        Access elements using operator overloading.
        :param key: The key to extract.
        :return: Calls get with the same key and target to False.
        """
        return self.get(key)

    def load(self):
        """Load the model configuration."""

        conf = self['conf']
        self.overall_step = conf['overall_step']

        for k in self.syncables.keys():
            model = self.syncables[k][1]
            model.load_state_dict(torch.load(join(self.run_root, "model/last", f"{k}.data")))

        return self.overall_step

    def store(self, name):
        """Store the model configuration."""

        # store the config as a json file
        dconf = self['conf'].to_dict()
        dconf['seed'] = np.random.randint(1000000)
        dconf['overall_step'] = self.overall_step

        r = json.dumps(dconf, indent=4, sort_keys=True)
        with open(join(self.run_root, "run.conf"), "w") as file_conf:
            file_conf.write(r)

        for k in self.syncables.keys():
            model = self.syncables[k][1]
            torch_path = join(self.model_root, name)
            if not exists(torch_path): mkdir(torch_path)
            torch.save(model.state_dict(), join(torch_path, f"{k}.data"))

    def get(self, key, target=False):
        """
        Access a specific key. Solves internally for env, main and sync able keys to give access using only this
        function.
        :param key: The key to extract.
        :param target: If this is True, a prefix 't_' is appended to the key.
        :return: The extracted object.
        """
        if key is 'env': return self.get_environment()
        ti = 1 if not target else 2
        return self.data[key] if key in self.data else self.syncables[key][ti]

    def add_environment(self, env):
        """
        Adds an environment.
        :param env: The environment to add internally.
        """
        assert isinstance(env, MCMCEnvWrapper)
        self.env = env

    def get_environment(self):
        """
        Pass back an environment.
        :return: Return the internal instance of environment.
        """
        return self.env

    def add_main(self, key, obj):
        """
        Adds an object as a main element.
        :param key: The key to use for this component.
        :param obj: The component to add.
        """
        assert key not in self.syncables
        self.data[key] = obj

    def add_syncable(self, key, syncable, step):
        """
        Adds a syncable object to the model configuration.
        :param key: The key for this sync able.
        :param syncable: The sync able itself..
        """
        assert key not in self.data
        assert isinstance(syncable, Syncable)

        obj = syncable
        self.syncables[key] = (step, obj, obj.clone())

    def sync(self):
        """
        Calls copy to method on all sync ables.
        """

        for s, o, to in self.syncables.values():
            if self.overall_step % s == 0:
                o.sync(to)

        self.overall_step += 1
