import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_ae_loss": r"$\mathcal{L}_{\mathrm{AE}} = \|x - \mathrm{decode}(\mathrm{encode}(x))\|_2^2$",
    "eq_reparam": r"$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$",
    "eq_kl": r"$D_{KL}\!\left(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\right) = \frac{1}{2}\sum_i \left(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1\right)$",
    "eq_elbo": r"$\mathcal{L}_{\mathrm{VAE}} = \|x-\hat x\|_2^2 \ +\ \beta\, D_{KL}\!\left(q(z|x)\,\|\,p(z)\right)$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=15)
        print("wrote", name)
