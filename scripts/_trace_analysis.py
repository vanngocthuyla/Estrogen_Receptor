import arviz as az
import pandas as pd

def summarize_parameters(samples):
    """
    Summarize MCMC parameters when samples come from:
        mcmc.get_samples(group_by_chain=False)

    samples: dict of arrays, each shaped (n_samples_total, ...)
    Returns: pandas DataFrame with mean, sd, r_hat, ess_bulk, ess_tail
    """

    # Convert flat samples → ArviZ InferenceData
    idata = az.from_dict(posterior=samples)

    # Compute summary
    summary_df = az.summary(
        idata,
        round_to=6,
        stat_focus="mean",
        hdi_prob=0.95,
    )

    return summary_df