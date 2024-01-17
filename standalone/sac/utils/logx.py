"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import json
import joblib
import numpy as np
import torch
import os.path as osp
import time
import atexit
import os
import warnings
from sac.utils.mpi_tools import proc_id, mpi_statistics_scalar
from sac.utils.serialization_utils import convert_json

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if proc_id() == 0:
            self.output_dir = output_dir or osp.join(
                os.getcwd(), "experiments", exp_name or "experiment_%i" % int(time.time()))
            if osp.exists(self.output_dir):
                print(colorize(
                    "Warning: Log dir %s already exists!" % self.output_dir, "yellow", bold=True))
                if input(colorize("Continue? [y/n]")) != "y":
                    exit(-1)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(
                osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s" %
                  self.output_file.name, 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpisodeLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id() == 0:
            output = json.dumps(config_json, separators=(
                ',', ':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config', color='cyan', bold=True))
            # print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def load_config(self):
        """
        Load an experiment configuration from a saved JSON file.

        This function reads the "config.json" file saved by save_config
        and returns the configuration as a Python dictionary.

        Returns:
            dict: Experiment configuration loaded from the file.
        """
        config_path = osp.join(self.output_dir, "config.json")
        if osp.exists(config_path):
            with open(config_path, 'r') as infile:
                config_json = json.load(infile)

            # Remove the 'exp_name' key if it was added during saving
            config = config_json.copy()
            if 'exp_name' in config:
                del config['exp_name']

            print(colorize('Loaded config:\n', color='cyan', bold=True))
            print(json.dumps(config, separators=(
                ',', ':\t'), indent=4, sort_keys=True))

            return config
        else:
            print(colorize('Config file not found.', color='red'))
            return None

    def save_state(self, state_dict, save_env=False, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        if proc_id() == 0:
            if save_env:
                fname = 'env.pkl' if itr is None else 'env%d.pkl' % itr
                try:
                    joblib.dump(state_dict, osp.join(self.output_dir, fname))
                except:
                    self.log('Warning: could not pickle state_dict.', color='red')
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        if proc_id() == 0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
            fname = osp.join(self.output_dir, fname)
            os.makedirs(self.output_dir, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving) as opposed to
                # just saving network weights. This works sufficiently well
                # for the purposes of Spinning Up, but you may want to do
                # something different for your personal PyTorch project.
                # We use a catch_warnings() context to avoid the warnings about
                # not being able to save the source code.
                torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%'+'%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-"*n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-"*n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers)+"\n")
                self.output_file.write("\t".join(map(str, vals))+"\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpisodeLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over episodes.

    Typical use case: there is some quantity which is calculated many times
    throughout an episode, and at the end of the episode, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpisodeLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        episode_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpisodeLogger's state. Then at the end of the episode, you 
    would use 

    .. code-block:: python

        episode_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the episode_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.episode_dict.keys()):
                self.episode_dict[k] = []
            self.episode_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the episode.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the episode.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.episode_dict[key]
            vals = np.concatenate(v) if isinstance(
                v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi_statistics_scalar(
                vals, with_min_and_max=with_min_and_max)
            super().log_tabular(
                key if average_only else 'Average' + key, stats[0])
            if not (average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
        self.episode_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.episode_dict[key]
        vals = np.concatenate(v) if isinstance(
            v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi_statistics_scalar(vals)
