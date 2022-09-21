import sys
import random
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl

class Trainer(object):
    def __init__(self, project_name, log_every_n_steps=50, *args, **kwargs):
        self.project_name = project_name
        self.parser = ArgumentParser("Training of {}".format(project_name))
        self.parser.add_argument('--seed', default=None, type=int, help='Random Seed')
        self.parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
        self.parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
        self.parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
        self.parser.add_argument('--min_epochs', default=10, type=int, help='Minimum number of epochs.')
        self.parser.add_argument('--max_epochs', default=50, type=int, help='Maximum number ob epochs to train')
        self.parser.add_argument('--worker', default=8, type=int, help='Number of workers for data loader')
        self.parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')
        self.parser.add_argument('--learning_rate_decay', default=0.99999, type=float, help='Add learning rate decay.')
        self.parser.add_argument('--early_stop_patience', default=0, type=int, help='Stop training after n epochs with ne val_loss improvement.')
        self.parser.add_argument('--name', default=None, help='Name of the training run.')
        self.trainer = None

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def setup(self):
        self.args = self.parser.parse_args()
    

        if self.args.detect_anomaly:
            print("Enabling anomaly detection")
            torch.autograd.set_detect_anomaly(True)
        
        # windows safe
        if sys.platform in ["win32"]:
            self.args.worker = 0

        # Manage Random Seed
        if self.args.seed is None: # Generate random seed if none is given
            self.args.seed = random.randrange(4294967295) # Make sure it's logged
        pl.utilities.seed.seed_everything(self.args.seed)

        # append stats to hparameter file
        yaml = self.args.__dict__
        yaml.update({
                'random_seed': self.args.seed,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_capability': torch.cuda.get_device_capability(0)
                })


        #################### ADD LOGGING   ########################################
        if self.args.name is None or self.args.dev:
            logger = None
        else:
            logger = pl.loggers.WandbLogger(project=self.project_name, name=self.args.name)
        ###########################################################################

        #################### ADD CALLBACKS ########################################
        callbacks = []

        if self.args.learning_rate_decay and logger:
            callbacks += [pl.callbacks.lr_monitor.LearningRateMonitor()]

        callbacks += [pl.callbacks.ModelCheckpoint(
            verbose=True,
            save_top_k=1,
            filename='{epoch}-{valid_loss}',
            monitor='valid_loss',
            mode='min'
        )]

        if self.args.early_stop_patience > 0:
            callbacks += [pl.callbacks.EarlyStopping(
                monitor='valid_loss',
                min_delta=0.00,
                patience=self.args.early_stop_patience,
                verbose=True,
                mode='min'
            )]
        ###########################################################################

        self.trainer = pl.Trainer(
            fast_dev_run=self.args.dev,
            accelerator='gpu',
            devices=1,
            log_every_n_steps=log_every_n_steps,
            overfit_batches=1 if self.args.overfit else 0,
            precision=self.args.precision,
            min_epochs=self.args.min_epochs,
            max_epochs=self.args.max_epochs,
            logger=logger,
            callbacks=callbacks,
            *args, **kwargs
        )
        return self.args

    def fit(self, *args, **kwargs):
        if self.trainer is None: self.setup()
        return self.trainer.fit(*args, **kwargs)
