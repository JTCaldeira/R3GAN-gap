import torch
from torch.nn.functional import relu

from torch_utils import training_stats, persistence


@persistence.persistent_class
class Gap(torch.nn.Module):
    def __init__(self, n_classes, effective_num_samples, ema_decay, classes_to_log):
        super().__init__()
        self.effective_num_samples = effective_num_samples
        self.ema_decay = ema_decay
        self.classes_to_log = classes_to_log

        self.register_buffer("ema_fake", -torch.ones(n_classes))
        self.register_buffer("ema_real", torch.ones(n_classes))
        self.started = False

    def update_fake_ema(self, gen_logits, gen_classes):
        for logit, c in zip(gen_logits, gen_classes.astype(bool)):
            # Update EMA of discriminator outputs for fake images
            self.ema_fake[c] = logit[0] + self.ema_decay * (self.ema_fake[c] - logit[0])

        n_digits = len(str(max(self.classes_to_log)))
        for c in self.classes_to_log:
            training_stats.report(f'Gap-fake/class{str(c).zfill(n_digits)}', self.ema_fake[c])

    def update_real_ema(self, real_logits, real_classes):
        for logit, c in zip(real_logits, real_classes.astype(bool)):
            # Update EMA of discriminator outputs for real images
            self.ema_real[c] = logit[0] + self.ema_decay * (self.ema_real[c] - logit[0])

        n_digits = len(str(max(self.classes_to_log)))
        for c in self.classes_to_log:
            training_stats.report(f'Gap-real/class{str(c).zfill(n_digits)}', self.ema_real[c])
        training_stats.report(f'Gap-real/class-average', self.ema_real.mean())

    def loss_fake(self, gen_logits, gen_classes):
        if not self.started:
            return 0.0

        loss = torch.empty(gen_logits.size(), dtype=gen_logits.dtype, device=gen_logits.device)

        for i, (logit, c) in enumerate(zip(gen_logits, gen_classes)):
            c = c.argmax()
            loss[i] = torch.square(relu(self.ema_real[c] - logit))

            if self.effective_num_samples is not None:
                loss[i] *= self.effective_num_samples[c]

        return loss

    def loss_real(self, real_logits, real_classes):
        if not self.started:
            return 0.0

        loss = torch.empty(real_logits.size(), dtype=real_logits.dtype, device=real_logits.device)

        for i, (logit, c) in enumerate(zip(real_logits, real_classes)):
            c = c.argmax()
            loss[i] = torch.square(relu(logit - self.ema_fake[c]))

            if self.effective_num_samples is not None:
                loss[i] *= self.effective_num_samples[c]

        return loss
