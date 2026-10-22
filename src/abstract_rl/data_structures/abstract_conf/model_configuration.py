import json
from datetime import datetime
from os import makedirs, mkdir
from os.path import exists, join
from shutil import copy, copytree
from typing import Any

import numpy as np
import torch

from abstract_rl.data_structures.abstract_conf.hierarchical_conf import SharedConf
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper


class ModelConfiguration:
    """
    Represents a model configuration. Contains important elements like the environment or policy. It also
    integrated a syncing mechanism due to the need of target and original data structures for more stable evaluations.
    """

    def __init__(self, conf: dict[str, Any], root: str | None = None) -> None:
        """
        Create a new model configuration with no syncables no data and no environment.
        """

        # settings for the script
        root = root if root is not None else conf.get("data_root", "data")
        reload = conf.get("reload", False)
        conf["reload"] = True

        self.syncables: dict[str, tuple[int, Any, Any]] = {}
        self.data: dict[str, Any] = {}
        if not reload:
            # create folder structure
            self.env_name = conf["env_name"]
            self.env_dir = join(root, self.env_name)
            self.alg_name = conf["alg"]
            self.env_dir = join(self.env_dir, self.alg_name)
            # parallel runs may create the same directories at the same time, so never check before creating
            makedirs(self.env_dir, exist_ok=True)
            while True:
                timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                self.run_root = join(self.env_dir, timestr)
                try:
                    makedirs(self.run_root)
                    break
                except FileExistsError:
                    # another run started within the same microsecond, take a new time stamp
                    continue

            self.overall_step = 0
            self.best_return = -np.inf
            self.env: Any = None
            conf["reload_dir"] = self.run_root
            self.add_main("conf", SharedConf(conf))

            self.model_root = join(self.run_root, "model")
            if not exists(self.model_root):
                mkdir(self.model_root)

            self.plot_root = join(self.run_root, "plot")
            if not exists(self.plot_root):
                mkdir(self.plot_root)

            self.data_dir = join(self.run_root, "data")
            if not exists(self.data_dir):
                mkdir(self.data_dir)

        else:
            conf_dir = conf["reload_dir"]
            self.run_root = conf_dir

            # load the config as a json file
            with open(join(self.run_root, "run.conf")) as file_conf:
                file_str = file_conf.read()
            self.add_main("conf", SharedConf(json.loads(file_str)))

    def __getitem__(self, key: str) -> Any:
        """
        Access elements using operator overloading.
        :param key: The key to extract.
        :return: Calls get with the same key and target to False.
        """
        return self.get(key)

    def load(self) -> int:
        """Load the model configuration."""

        conf = self["conf"]
        self.overall_step = conf["overall_step"]

        for k in self.syncables:
            model = self.syncables[k][1]
            model.load_state_dict(torch.load(join(self.run_root, "model/last", f"{k}.data")))

        return self.overall_step

    def store(self, name: str) -> None:
        """Store the model configuration."""

        # store the config as a json file
        dconf = self["conf"].to_dict()
        dconf["seed"] = np.random.randint(1000000)  # noqa: NPY002
        dconf["overall_step"] = self.overall_step

        r = json.dumps(dconf, indent=4, sort_keys=True)
        with open(join(self.run_root, "run.conf"), "w") as file_conf:
            file_conf.write(r)

        for k in self.syncables:
            model = self.syncables[k][1]
            torch_path = join(self.model_root, name)
            if not exists(torch_path):
                mkdir(torch_path)
            torch.save(model.state_dict(), join(torch_path, f"{k}.data"))

    def store_best(self, ret: float, epoch: int) -> None:
        """Copy the stored ``last`` models (and observation statistics) to ``best`` if ``ret`` is a new best return."""

        if ret <= self.best_return:
            return
        self.best_return = ret
        best_path = join(self.model_root, "best")
        copytree(join(self.model_root, "last"), best_path, dirs_exist_ok=True)
        if exists(join(self.model_root, "obs_rms.npz")):
            copy(join(self.model_root, "obs_rms.npz"), best_path)
        with open(join(self.model_root, "best.json"), "w") as file_best:
            json.dump({"epoch": epoch, "return": ret}, file_best)

    def get(self, key: str, target: bool = False) -> Any:
        """
        Access a specific key. Solves internally for env, main and sync able keys to give access using only this
        function.
        :param key: The key to extract.
        :param target: If this is True, a prefix 't_' is appended to the key.
        :return: The extracted object.
        """
        if key == "env":
            return self.get_environment()
        ti = 1 if not target else 2
        return self.data[key] if key in self.data else self.syncables[key][ti]

    def add_environment(self, env: MCMCEnvWrapper) -> None:
        """
        Adds an environment.
        :param env: The environment to add internally.
        """
        assert isinstance(env, MCMCEnvWrapper)
        self.env = env

    def get_environment(self) -> MCMCEnvWrapper:
        """
        Pass back an environment.
        :return: Return the internal instance of environment.
        """
        return self.env

    def add_main(self, key: str, obj: Any) -> None:
        """
        Adds an object as a main element.
        :param key: The key to use for this component.
        :param obj: The component to add.
        """
        assert key not in self.syncables
        self.data[key] = obj

    def add_syncable(self, key: str, syncable: Syncable, step: int) -> None:
        """
        Adds a syncable object to the model configuration.
        :param key: The key for this sync able.
        :param syncable: The sync able itself..
        """
        assert key not in self.data
        assert isinstance(syncable, Syncable)

        obj = syncable
        self.syncables[key] = (step, obj, obj.clone())

    def sync(self) -> None:
        """
        Calls copy to method on all sync ables.
        """

        for s, o, to in self.syncables.values():
            if self.overall_step % s == 0:
                o.sync(to)

        self.overall_step += 1
