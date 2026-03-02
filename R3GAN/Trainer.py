import torch
import torch.nn as nn

class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator
        
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])
        
    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1, Preprocessor=lambda x: x, do_gap_stuff=False):
        FakeSamples = self.Generator(Noise, Conditions)
        RealSamples = RealSamples.detach()
        
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)

        if self.Discriminator.gap and do_gap_stuff:
            with torch.no_grad():
                fake_logits = self.Discriminator(FakeSamples, Conditions)
                real_logits = self.Discriminator(RealSamples, Conditions)
                self.Discriminator.gap.update_fake_ema(fake_logits.detach().cpu().numpy(), Conditions.detach().cpu().numpy())
                self.Discriminator.gap.update_real_ema(real_logits.detach().cpu().numpy(), Conditions.detach().cpu().numpy())
        
        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        (Scale * AdversarialLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
    
    def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Conditions, Gamma, Scale=1, Preprocessor=lambda x: x, do_reg=True, reg_interval=1, do_gap_stuff=False):
        RealSamples = RealSamples.detach().requires_grad_(True)
        FakeSamples = self.Generator(Noise, Conditions).detach().requires_grad_(True)

        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)

        if self.Discriminator.gap and do_gap_stuff:
            with torch.no_grad():
                fake_logits = self.Discriminator(FakeSamples, Conditions)
                real_logits = self.Discriminator(RealSamples, Conditions)
                self.Discriminator.gap.update_fake_ema(fake_logits.detach().cpu().numpy(), Conditions.detach().cpu().numpy())
                self.Discriminator.gap.update_real_ema(real_logits.detach().cpu().numpy(), Conditions.detach().cpu().numpy())

        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        (Scale * AdversarialLoss.mean()).backward(retain_graph=do_reg)

        R1Penalty = torch.zeros(1, device=RealSamples.device)
        R2Penalty = torch.zeros(1, device=RealSamples.device)

        if do_reg:
            R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
            R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
            reg_loss = (Gamma / 2) * (R1Penalty + R2Penalty) * reg_interval  # scale by interval
            (Scale * reg_loss.mean()).backward()

        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]
