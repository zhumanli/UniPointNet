import argparse
import importlib
import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datasets.base_dataset import DataModule

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

## 1722_95
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--project', type=str, default='UniPointNet')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--encoder', type=str, default='encoder')
    parser.add_argument('--decoder', type=str, default='decoder')
    parser.add_argument('--n_parts', type=int, default=4, help='number of keypoints')
    parser.add_argument('--n_clusters', type=int, default=100, help='number of clusters')
    parser.add_argument('--missing', type=float, default=0.8, help='ratio of the image masking')
    parser.add_argument('--block', type=int, default=16, help='number of patches to divide the image in one dimension')
    parser.add_argument('--thick', type=float, default=5e-2, help='thickness of the deges')
    parser.add_argument('--sklr', type=float, default=512, help='the learning rate of the edge weights')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='COCO_masks')
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=1)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    args.log = 'Project_{0}_k{1}'.format(args.project, args.n_parts)
    wandb_logger = WandbLogger(name=args.log, project=args.project)


    model = importlib.import_module('models.' + args.model).Model(**vars(args))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                       dirpath=os.path.join('CheckPoints', model.hparams.log),
                                                       filename='model')
    datamodule = DataModule(model.hparams.dataset, model.hparams.data_root, model.hparams.image_size, model.hparams.batch_size,
                            model.hparams.num_workers)

    trainer = pl.Trainer(accelerator='gpu', gpus=model.hparams.gpus, num_nodes=model.hparams.num_nodes,
                         fast_dev_run=model.hparams.debug,
                         max_steps=30001, precision=16, auto_scale_batch_size='binsearch', sync_batchnorm=True, strategy='ddp',
                        #  limit_val_batches=1,
                         val_check_interval=100,
                         check_val_every_n_epoch=1,
                         callbacks=checkpoint_callback, logger=wandb_logger,
                         )

    trainer.fit(model, datamodule=datamodule)
