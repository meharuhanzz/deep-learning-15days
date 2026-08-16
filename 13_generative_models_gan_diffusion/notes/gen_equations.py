import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_gan_minimax": r"$\min_G \max_D\ \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_z[\log(1-D(G(z)))]$",
    "eq_gan_nonsat": r"$\mathcal{L}_G = -\log D(G(z)) \quad\text{(non-saturating, used instead of } \log(1-D(G(z)))\text{)}$",
    "eq_diffusion_forward": r"$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon\sim\mathcal{N}(0,I)$",
    "eq_diffusion_loss": r"$\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2\right]$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=15)
        print("wrote", name)
