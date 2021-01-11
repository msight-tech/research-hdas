# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import torch.distributed as dist
from utils.logging_util import get_std_logging
from config.augmentStage_config import AugmentStageConfig
from trainer.augmentStage_trainer import AugmentStageTrainer


def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger

    if config.dist:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        config.world_size = world_size
        config.rank = rank
        config.local_rank = local_rank

        if config.local_rank == 0:
            config.print_params(logger.info)
        
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        logger.info(f'world_size {world_size}, gpu {local_rank}, rank {rank} init done.')
    else:
        config.world_size, config.rank, config.local_rank = 1, 0, 0
        config.print_params(logger.info)
    
    trainer = AugmentStageTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch

    best_top1 = 0.
    for epoch in range(start_epoch, trainer.total_epochs):
        drop_prob = config.drop_path_prob * epoch / trainer.total_epochs
        trainer.model.module.drop_path_prob(drop_prob)
        trainer.train_epoch(epoch, printer=logger.info)
        top1 = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()

        if best_top1 < top1:
            best_top1, is_best = top1, True
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best)
        if config.local_rank == 0:
            logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))
    if config.local_rank == 0:
        logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def main():
    config = AugmentStageConfig()
    run_task(config)


if __name__ == "__main__":
    # print('*****************************************')
    # print(f'pytorch version: {torch.__version__}')
    # print(f'os environ: {os.environ}')
    main()
