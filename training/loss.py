# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

from torch_utils import training_stats
from R3GAN.Trainer import AdversarialTraining
import torch

#----------------------------------------------------------------------------

class R3GANLoss:
    def __init__(self, G, D, augment_pipe=None, reg_interval=1):
        self.trainer = AdversarialTraining(G, D)
        self.reg_interval = reg_interval
        if augment_pipe is not None:
            self.preprocessor = lambda x: augment_pipe(x.to(torch.float32)).to(x.dtype)
        else:
            self.preprocessor = lambda x: x

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gamma, gain, cur_nimg=0, batch_size=1, do_gap_stuff=False):
        if phase == 'G':
            AdversarialLoss, RelativisticLogits, fake_logits, real_logits = self.trainer.AccumulateGeneratorGradients(gen_z, real_img, real_c, gain, self.preprocessor, do_gap_stuff=do_gap_stuff)

            training_stats.report('Loss/scores/fake', RelativisticLogits)
            training_stats.report('Loss/signs/fake', RelativisticLogits.sign())
            training_stats.report('Loss/G/loss', AdversarialLoss)

        if phase == 'D':
            do_reg = (cur_nimg % (self.reg_interval * batch_size) < batch_size)
            AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty, fake_logits, real_logits = self.trainer.AccumulateDiscriminatorGradients(
                gen_z, real_img, real_c, gamma, gain, self.preprocessor,
                do_reg=do_reg, reg_interval=self.reg_interval, do_gap_stuff=do_gap_stuff
            )

            training_stats.report('Loss/scores/real', RelativisticLogits)
            training_stats.report('Loss/signs/real', RelativisticLogits.sign())
            training_stats.report('Loss/D/loss', AdversarialLoss)
            training_stats.report('Loss/r1_penalty', R1Penalty)
            training_stats.report('Loss/r2_penalty', R2Penalty)
#----------------------------------------------------------------------------