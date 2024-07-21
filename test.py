import argparse
import importlib
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datasets.base_dataset import DataModule



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='CheckPoints')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='COCO_masks')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pl.utilities.seed.seed_everything(0)

    model = importlib.import_module('models.' + args.model).Model.load_from_checkpoint(os.path.join('checkpoints', 'model.ckpt'))

    trainer = pl.Trainer(accelerator='gpu', gpus=1,
                         precision=16, sync_batchnorm=True,
                         logger=WandbLogger(name='test_32_vis', project="UniPointNet_test"),
                         )
    datamodule = DataModule(args.dataset, args.data_root, model.hparams.image_size, 1, model.hparams.num_workers)
    out = trainer.test(model, datamodule=datamodule)
    # print(out[0]['keypoints'])
