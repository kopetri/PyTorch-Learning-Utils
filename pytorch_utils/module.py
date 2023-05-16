from argparse import Namespace
import lightning.pytorch as pl
from pathlib import Path
from zipfile import ZipFile

class LightningModule(pl.LightningModule):
    def __init__(self, opt=None, **kwargs):
        super().__init__()
        if opt is None:
            # loaded from checkpoint
            self.opt = Namespace(**kwargs)
            if "hparams" in self.opt: self.opt = self.opt.hparams # make compatible with old ckpts
        else:
            self.opt = opt
        # save hyperparameters to checkpoint
        self.save_hyperparameters(vars(self.opt))
        self.learning_rate = self.opt.learning_rate

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, "test")

    def log_value(self, key, value, split, batch_size):
        if split in ["train", "valid", "test"]:
            self.log("{split}_{key}".format(split=split, key=key), value, prog_bar=True, on_epoch=not split=="train", on_step=split=="train", batch_size=batch_size)

    def log_image(self, key, images, **kwargs):
        if self.logger:
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.log_image(key=key, images=images, **kwargs)
            else:
                print("Warning - cannot log image. Please use a WandbLogger!")

    def on_save_checkpoint(self, checkpoint) -> None:
        if not (isinstance(self.logger, pl.loggers.WandbLogger) and self.opt.save_code_base): return
        path = Path(".", self.logger.experiment.project, self.logger.experiment.id, "code")
        zipfile = path/"code.zip"
        if not zipfile.exists():
            path.mkdir(parents=True, exist_ok=True)
            code_base = [file for file in Path(".").glob("**/*") if file.suffix == ".py" and not any([f in file.as_posix() for f in ["venv", "wandb", "lightning_log"]])]
            with ZipFile(zipfile.as_posix(), "w") as codezip:
                for code in code_base:
                    codezip.write(code)
            print("Saved code base to ", zipfile.as_posix())

    def forward(self, batch, batch_idx, split):
        raise NotImplementedError()
