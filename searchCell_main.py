# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

'''
H^c-DAS
search specific cells of different stages.
'''
import os
from config.searchCell_config import SearchCellConfig
from trainer.searchCell_trainer import SearchCellTrainer
from utils.logging_util import get_std_logging
from utils.visualize import plot


def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger

    config.print_params(logger.info)
    trainer = SearchCellTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch

    best_top1 = 0.
    for epoch in range(start_epoch, trainer.total_epochs):
        trainer.train_epoch(epoch, printer=logger.info)
        top1 = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()

        genotype = trainer.model.genotype()
        logger.info("genotype = {}".format(genotype))
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot(genotype.normal1, plot_path + "-normal1", caption)
        plot(genotype.reduce1, plot_path + "-reduce1", caption)
        plot(genotype.normal2, plot_path + "-normal2", caption)
        plot(genotype.reduce2, plot_path + "-reduce2", caption)
        plot(genotype.normal3, plot_path + "-normal3", caption)
        if best_top1 < top1:
            best_top1, is_best = top1, True
            best_genotype = genotype
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best=is_best)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))
    
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Final Best Genotype = {}".format(best_genotype))


def main():
    config = SearchCellConfig()
    run_task(config)


if __name__ == "__main__":
    main()
