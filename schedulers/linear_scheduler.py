import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearNoiseScheduler:
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_product = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_product = torch.sqrt(self.alpha_product)
        self.sqrt_one_minus_alpha_prod = torch.sqrt(1. - self.alpha_product)

    def add_noise(self, original, noise, t):
        o_shape = original.shape
        batch_size = o_shape[0]

        sqrt_alpha_prod = self.sqrt_alpha_product[t].reshape(batch_size).to(device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alpha_prod[t].reshape(batch_size).to(device)

        for _ in range(len(o_shape) - 1):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_prod[t] * noise_pred)) / self.sqrt_alpha_product[t]
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t] * noise_pred) / self.sqrt_one_minus_alpha_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1. - self.alpha_product[t - 1]) / (1. - self.alpha_product[t])
            variance = variance * self.betas[t]
            sigma = variance ** .5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0

    def loss_coeff(self, t):
        return self.betas[t] / (2 * torch.sqrt(1. - self.betas[t]) * self.alphas[t] * (1. - self.alpha_product[t]))

