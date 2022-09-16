from argparse import Namespace
import pytorch_lightning as pl

class LightningModule(pl.LightningModule):
    def __init__(self, opt=None, **kwargs):
        super().__init__()
        if opt is None:
            # loaded from checkpoint
            self.opt = Namespace(**kwargs)
        else:
            self.opt = opt
        # save hyperparameters to checkpoint
        self.save_hyperparameters(vars(self.opt))

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, "test")

    def log_value(self, key, value, split, batch_size):
        assert split in ["train", "valid", "test"]
        self.log("{split}_{key}".format(split=split, key=key), value, prog_bar=True, on_epoch=not split=="train", on_step=split=="train", batch_size=batch_size)

    def log_image(self, key, images, **kwargs):
        self.logger.log_image(key=key, images=images, **kwargs)

    def forward(self, batch, batch_idx, split):
        raise NotImplementedError()