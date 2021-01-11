# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
from utils.logging_util import get_std_logging
from config.searchStage_config import SearchStageConfig
from trainer.searchStage_trainer import SearchStageTrainer
from utils.visualize import plot2


def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger

    config.print_params(logger.info)

    trainer = SearchStageTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch

    best_top1 = 0.
    for epoch in range(start_epoch, trainer.total_epochs):
        trainer.train_epoch(epoch, printer=logger.info)
        top1 = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()

        # plot macro architecture
        macro_arch = trainer.model.DAG()
        logger.info("DAG = {}".format(macro_arch))

        plot_path = os.path.join(config.DAG_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot2(macro_arch.DAG1, plot_path + '-DAG1', caption)
        plot2(macro_arch.DAG2, plot_path + '-DAG2', caption)
        plot2(macro_arch.DAG3, plot_path + '-DAG3', caption)

        if best_top1 < top1:
            best_top1, is_best = top1, True
            best_macro = macro_arch
        else:
            is_best = False
        # trainer.save_checkpoint(epoch, is_best=is_best)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Final Best Genotype = {}".format(best_macro))


def main():
    config = SearchStageConfig()
    run_task(config)


if __name__ == "__main__":
    main()
