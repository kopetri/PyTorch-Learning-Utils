import sys
import random
import torch
from argparse import ArgumentParser
import lightning as L
import lightning.pytorch as pl

class Trainer(pl.Trainer):
    def __init__(self, project_name):
        self.project_name = project_name
        self.parser = ArgumentParser("Training of {}".format(project_name))
        self.parser.add_argument('--seed', default=None, type=int, help='Random Seed')
        self.parser.add_argument('--accelerator', default='gpu', type=str, help='chose the accelerator to use. cpu or gpu')
        self.parser.add_argument('--devices', default=-1, nargs='+', type=int, help='Number of gpus to use. default -1 using all gpus.')
        self.parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
        self.parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
        self.parser.add_argument('--overfit', default=0, type=int, help='Set this to a number greater 0 for overfitting on few batches.')
        self.parser.add_argument('--profiler', default=None, type=str, help='Enable profiling by setting this to "simple" or "advanced"')
        self.parser.add_argument('--min_epochs', default=10, type=int, help='Minimum number of epochs.')
        self.parser.add_argument('--max_epochs', default=50, type=int, help='Maximum number ob epochs to train')
        self.parser.add_argument('--worker', default=8, type=int, help='Number of workers for data loader')
        self.parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')
        self.parser.add_argument('--learning_rate_decay', default=0.99999, type=float, help='Add learning rate decay.')
        self.parser.add_argument('--early_stop_patience', default=0, type=int, help='Stop training after n epochs with ne val_loss improvement.')
        self.parser.add_argument('--name', default=None, help='Name of the training run.')
        self.parser.add_argument('--log_every_n_steps', default=50, type=int, help='Interval for logging.')
        self.parser.add_argument('--ckpt_every_n_epochs', default=None, type=int, help='Interval for saving checkpoints.')
        self.parser.add_argument('--ckpt_every_n_steps', default=None, type=int, help='Interval for saving checkpoints.')
        self.parser.add_argument('--save_code_base', default=1, type=int, help='Enable saving code base.')
        self.parser.add_argument('--checkpoint_metric', default=['valid_loss'], nargs='+', type=str, help='Metric to use for saving checkpoints.')
        self.parser.add_argument('--mode', default=['min'], nargs='+', type=str, help='If the checkpoint_metric needs to me minimized or maximized.')
        self.__initialized__ = False
        self.__args__ = None

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def setup(self, train=True):
        self.__args__ = self.parser.parse_args()
    
        if self.__args__.detect_anomaly:
            print("Enabling anomaly detection")
            torch.autograd.set_detect_anomaly(True)
        
        # windows safe
        if sys.platform in ["win32"]:
            self.__args__.worker = 6

        # Manage Random Seed
        if self.__args__.seed is None: # Generate random seed if none is given
            self.__args__.seed = random.randrange(4294967295) # Make sure it's logged
        L.seed_everything(self.__args__.seed)
        

        # append stats to hparameter file
        yaml = self.__args__.__dict__
        yaml.update({
                'random_seed': self.__args__.seed,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_capability': torch.cuda.get_device_capability(0)
                })


        #################### ADD LOGGING   ########################################
        if self.__args__.name is None or self.__args__.dev:
            logger = None
        else:
            logger = pl.loggers.WandbLogger(project=self.project_name, name=self.__args__.name, log_model=self.__args__.save_code_base)
        ###########################################################################

        #################### ADD CALLBACKS ########################################
        callbacks = []

        if self.__args__.learning_rate_decay and logger:
            callbacks += [pl.callbacks.lr_monitor.LearningRateMonitor()]

        callbacks += [pl.callbacks.ModelCheckpoint(
            verbose=True,
            save_top_k=1,
            filename='{epoch}-'+'{'+metric+'}',
            monitor=metric,
            mode=mode
        ) for metric, mode in zip(self.__args__.checkpoint_metric, self.__args__.mode)]
        
        if self.__args__.ckpt_every_n_epochs:
            callbacks += [pl.callbacks.ModelCheckpoint(
                verbose=True,
                save_top_k=-1,
                every_n_epochs=self.__args__.ckpt_every_n_epochs,
                filename='{epoch}'
            )]
            
        if self.__args__.ckpt_every_n_steps:
            callbacks += [pl.callbacks.ModelCheckpoint(
                verbose=True,
                save_top_k=-1,
                every_n_train_steps=self.__args__.ckpt_every_n_steps,
                filename='{step}'
            )]

        if self.__args__.early_stop_patience > 0:
            callbacks += [pl.callbacks.EarlyStopping(
                monitor=self.__args__.checkpoint_metric,
                min_delta=0.00,
                patience=self.__args__.early_stop_patience,
                verbose=True,
                mode=self.__args__.mode
            )]
        ###########################################################################

        if train:
            super().__init__(
                fast_dev_run=self.__args__.dev,
                accelerator=self.__args__.accelerator,
                devices=self.__args__.devices,
                log_every_n_steps=self.__args__.log_every_n_steps,
                overfit_batches=self.__args__.overfit,
                precision=self.__args__.precision,
                min_epochs=self.__args__.min_epochs,
                max_epochs=self.__args__.max_epochs,
                logger=logger,
                callbacks=callbacks,
                deterministic=True,
                profiler=self.__args__.profiler
            )
        else:
            super().__init__(accelerator=self.__args__.accelerator, devices=self.__args__.devices, deterministic=True)
        return self.__args__


