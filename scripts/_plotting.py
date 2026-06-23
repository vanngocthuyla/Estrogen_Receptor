import os
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from _fitting import predict_response
from _fitting import param_each_receptor


def posterior_predict_curve(samples, experiment, ligand_dict):
    """
    Compute posterior predictive mean + 95% CI for a single ligand.
    Compatible with merged ER_alpha / ER_beta experiments.
    """

    receptor = experiment["name"]
    lname = ligand_dict["name"]

    # Flatten helper
    def flat(x):
        return np.asarray(x).reshape(-1)

    # Extract receptor-specific parameters
    params = param_each_receptor(samples, experiment)

    # Identify index of this ligand within experiment["ligands"]
    lig_idx = experiment["ligands"].index(ligand_dict)

    # Extract ligand-specific parameters
    logK_RE   = flat(params["logK_RE"][lig_idx])
    logK_REC  = flat(params["logK_REC"][lig_idx])
    log_scale = flat(params["log_scale"][lig_idx])
    log_sigma = flat(params["log_sigma"][lig_idx])

    sigma = np.exp(log_sigma)

    # Global receptor parameter
    logK_RC = flat(params["logK_RC"])

    # Input data
    logR = ligand_dict["logR"]
    logE = ligand_dict["logE"]
    logC = ligand_dict["logC"]

    # Predict for each posterior sample
    preds = []
    for i in range(len(logK_RE)):
        y = predict_response(
            logR, logE, logC,
            logK_RE[i],
            logK_RC[i],
            logK_REC[i],
            log_scale[i],
        )
        preds.append(np.array(y))

    preds = np.vstack(preds)

    # Posterior mean + 95% CI
    mean = preds.mean(axis=0)
    lo   = np.percentile(preds, 2.5, axis=0)
    hi   = np.percentile(preds, 97.5, axis=0)

    return mean, lo, hi


def plot_fit_for_ligand(samples, experiment, ligand_name, save=False, save_dir="."):
    """
    Plot posterior predictive fit for a ligand.
    Title format: Figure – Ligand – Receptor
    If save=True, saves PNG with sanitized filename.
    """

    receptor = experiment["name"]

    # Find ligand
    ligand = next(l for l in experiment["ligands"] if l["name"] == ligand_name)

    # Extract figure name
    fig_name = ligand.get("figure", "UnknownFigure")

    # Determine x-axis
    if "logC" in ligand and len(np.unique(ligand["logC"])) > 1:
        x = ligand["logC"]
        xlabel = "log[SRC3] (M)"
    else:
        x = ligand["logE"]
        xlabel = "log[Ligand] (M)"

    y_obs = ligand["response"]

    # Posterior predictive
    mean, lo, hi = posterior_predict_curve(samples, experiment, ligand)

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(x, y_obs, color="black", label="Data")
    plt.plot(x, mean, color="blue", label="Posterior mean")
    plt.fill_between(x, lo, hi, color="blue", alpha=0.2, label="95% CI")

    plt.xlabel(xlabel)
    plt.ylabel("Response")

    # Title
    title = f"{receptor} - {fig_name} – {ligand_name}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save if requested
    if save:
        # Sanitize filename
        fname = title.replace(" ", "_").replace("–", "-").replace("/", "_")
        path = os.path.join(save_dir, f"{fname}.png")
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")

    plt.show()


def plot_fit_for_ligands(samples, experiment, save=False, save_dir="."):
    """
    Plot all ligands in an experiment (ER_alpha or ER_beta).
    """

    for lig in experiment["ligands"]:
        ligand_name = lig["name"]
        plot_fit_for_ligand(
            samples,
            experiment,
            ligand_name,
            save=save,
            save_dir=save_dir
        )


def autocorr(x, max_lag=200):
    x = np.asarray(x)
    x = x - x.mean()
    result = np.correlate(x, x, mode='full')
    ac = result[result.size//2:]
    ac = ac / ac[0]
    return ac[:max_lag]


def diagnostic_panel(samples, param_name, max_lag=200):
    vals = np.array(samples[param_name])  # (chains, draws)

    fig, axs = plt.subplots(1, 3, figsize=(15,4))

    # Trace
    for c in range(vals.shape[0]):
        axs[0].plot(vals[c], alpha=0.7)
    axs[0].set_title("Trace")
    axs[0].set_xlabel("Iteration")

    # Autocorrelation
    for c in range(vals.shape[0]):
        ac = autocorr(vals[c], max_lag=max_lag)
        axs[1].plot(ac)
    axs[1].set_title("Autocorrelation")
    axs[1].set_xlabel("Lag")

    # Histogram (all chains pooled)
    axs[2].hist(vals.reshape(-1), bins=40, density=True, alpha=0.7)
    axs[2].set_title("Posterior density")

    fig.suptitle(param_name)
    plt.tight_layout()
    plt.show()