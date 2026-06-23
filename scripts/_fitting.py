import os
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import arviz as az
import pickle

from _data import build_experiments_filtered, group_ER_experiments
from _prior import uniform_prior, normal_prior, logsigma_guesses
from _model import predict_response
from _trace_analysis import summarize_parameters

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------
# Prior
# ------------------------------------------------------------

def all_priors(experiments):
    """
    Build priors for:
      - receptor-specific logK_RC
      - ligand-specific logK_RE, logK_REC, log_scale, log_sigma
    Returns dict keyed by experiment index.
    """

    ER_alpha, ER_beta = group_ER_experiments(experiments)

    params = {}
    for exp in [ER_alpha, ER_beta]:
        if len(exp["ligands"])>0:
            if exp["name"] == "ER_alpha":
                receptor = "α"
            if exp["name"] == "ER_beta":
                receptor = "ß"

            # Gather ligand arrays
            lig_names = []
            for lig in exp['ligands']:
                lig_names.append(lig['name'])
            lig_names = np.unique(np.array(lig_names))

            # Global parameter for each receptor
            params[f"logK_RC_{receptor}"] = uniform_prior(f"logK_RC_{receptor}", -20.0, 0.0)

            # Local and share parameters between ligands
            for lig in lig_names:
                params[f"logK_RE_{receptor}_{lig}"]  = uniform_prior(f"logK_RE_{receptor}_{lig}", -20.0, 0.0)
                params[f"logK_REC_{receptor}_{lig}"] = uniform_prior(f"logK_REC_{receptor}_{lig}", -20.0, 0.0)

            # Local parameters for each dataset
            for lig_idx, lig in enumerate(exp['ligands']):
                lname = lig["name"]

                response = jnp.asarray(lig["response"], dtype=jnp.float64)
                lo, hi = logsigma_guesses(response)

                if not f"log_sigma_{receptor}_{lname}" in params.keys():
                    params[f"log_sigma_{receptor}_{lname}"] = uniform_prior(f"log_sigma_{receptor}_{lname}", lo, hi)
                else:
                    params[f"log_sigma_{receptor}_{lname}_{lig_idx}"] = uniform_prior(f"log_sigma_{receptor}_{lname}_{lig_idx}", lo, hi)

                if not f"log_scale_{receptor}_{lname}" in params.keys():
                    params[f"log_scale_{receptor}_{lname}"] = uniform_prior(f"log_scale_{receptor}_{lname}", 0.0, 50.0)
                else:
                    params[f"log_scale_{receptor}_{lname}_{lig_idx}"] = uniform_prior(f"log_scale_{receptor}_{lname}_{lig_idx}", 0.0, 50.0)

    return params


def param_each_receptor(params, experiment):
    """
    Return prior for each experiment:
      - receptor-specific logK_RC
      - ligand-specific logK_RE, logK_REC, log_scale, log_sigma
    """
    if experiment["name"] == "ER_alpha":
        receptor = "α"
    if experiment["name"] == "ER_beta":
        receptor = "ß"

    logK_RC = params[f"logK_RC_{receptor}"]
    logK_RE   = []
    logK_REC  = []
    log_scale = []
    log_sigma = []
    
    for lig_idx, lig in enumerate(experiment["ligands"]):
        
        lname = lig["name"]
        logK_RE.append(params[f"logK_RE_{receptor}_{lname}"])
        logK_REC.append(params[f"logK_REC_{receptor}_{lname}"])
        
        if f"log_scale_{receptor}_{lname}_{lig_idx}" in params.keys():
            log_scale.append(params[f"log_scale_{receptor}_{lname}_{lig_idx}"])
        else:
            log_scale.append(params[f"log_scale_{receptor}_{lname}"])
        if f"log_sigma_{receptor}_{lname}_{lig_idx}" in params.keys():
            log_sigma.append(params[f"log_sigma_{receptor}_{lname}_{lig_idx}"])
        else:
            log_sigma.append(params[f"log_sigma_{receptor}_{lname}"])

    return {
        "logK_RC": logK_RC,
        "logK_RE": jnp.array(logK_RE),
        "logK_REC": jnp.array(logK_REC),
        "log_scale": jnp.array(log_scale),
        "log_sigma": jnp.array(log_sigma),
    }

# ------------------------------------------------------------
# Binding model
# ------------------------------------------------------------

def global_binding_model(experiments):
    """
    Global model:
      - receptor-specific logK_RC
      - ligand-specific logK_RE, logK_REC, log_scale, log_sigma
    experiments: list of dicts as returned by build_experiments_filtered
    """
    
    # Receptor names: "ER_alpha", "ER_beta"
    receptors = sorted({exp["name"] for exp in experiments})

    # Parameters for all experiments
    all_params = all_priors(experiments)

    # Likelihood
    for exp in experiments:
        receptor = exp["name"]
        fig_name = exp["figure"]

        # Extract parameters for each experiment
        params = param_each_receptor(params=all_params, experiment=exp)

        # Gather ligand arrays
        lig_list = exp["ligands"]
        L = len(lig_list)

        logR  = jnp.stack([jnp.asarray(l["logR"])  for l in lig_list])   # (L, D)
        logE  = jnp.stack([jnp.asarray(l["logE"])  for l in lig_list])   # (L, D)
        logC  = jnp.stack([jnp.asarray(l["logC"])  for l in lig_list])   # (L, D)
        y_obs = jnp.stack([jnp.asarray(l["response"]) for l in lig_list]) # (L, D)

        # Vectorized predict_response over ligands
        def f(logR_i, logE_i, logC_i, logK_RE_i, logK_REC_i, log_scale_i):
            return predict_response(
                logR_i, logE_i, logC_i,
                logK_RE_i,
                params['logK_RC'],
                logK_REC_i,
                log_scale_i,
            )

        y_pred = jax.vmap(f)(logR, logE, logC, params['logK_RE'], params['logK_REC'], params['log_scale'])  # (L, D)
        sigma  = jnp.exp(params['log_sigma'])[:, None]                                                                # (L, 1)

        # NumPyro sampling
        for i, lig in enumerate(lig_list):
            numpyro.sample(
                f"y_{fig_name}_{receptor}_{lig['name']}",
                dist.Normal(y_pred[i], sigma[i]),
                obs=y_obs[i],
            )

# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

def run_inference(receptors=("ER_alpha", "ER_beta"),
                  figures=("Figure 4", "Figure 5"),
                  num_warmup=2000,
                  num_samples=5000,
                  num_chains=4,
                  save_dir=''):

    experiments = build_experiments_filtered(
        receptors=receptors,
        figures=figures,
    )

    kernel = NUTS(global_binding_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc.run(jax.random.PRNGKey(0), experiments)

    traces = mcmc.get_samples(group_by_chain=True)
    pickle.dump(traces, open(os.path.join(save_dir,'traces.pickle'), "wb"))

    summary = summarize_parameters(traces)
    summary.to_csv(os.path.join(save_dir,"Summary.csv"))

    return mcmc.get_samples(group_by_chain=False)