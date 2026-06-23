import jax
import jax.numpy as jnp

from _chemical_reactions import ChemicalReactions

# -------------------------------------------------------------------
# Static binding model definition
# -------------------------------------------------------------------

species = ['R', 'E', 'RE', 'C', 'RC', 'REC']

reactions = [
    {'R': 1,  'E': 1, 'RE': -1},   # logK_RE
    {'R': 1,  'C': 1, 'RC': -1},   # logK_RC
    {'RE': 1, 'C': 1, 'REC': -1},  # logK_REC
    {'RC': 1, 'E': 1, 'REC': -1},  # logK_RCE
]

conservation_equations = [
    {'R': +1, 'RE': +1, 'RC': +1, 'REC': +1},  # total R
    {'E': +1, 'RE': +1, 'REC': +1},            # total E
    {'C': +1, 'RC': +1, 'REC': +1},            # total C
]

binding_model = ChemicalReactions(reactions, conservation_equations)
REC_INDEX = binding_model.index_of_species['REC']

# -------------------------------------------------------------------
# Core single-dose binding computation
# -------------------------------------------------------------------

def _binding_single(logR, logE, logC,
                    logK_RE, logK_RC, logK_REC,
                    iters: int = 5):
    """
    Compute log-concentrations for a single (R,E,C) triple.
    """
    logK_RCE = logK_RE + logK_REC - logK_RC
    logKd = jnp.array([logK_RE, logK_RC, logK_REC, logK_RCE])
    logctot = jnp.array([logR, logE, logC])
    logc = binding_model.logceq(logKd, logctot)
    return logc


# -------------------------------------------------------------------
# Batched + jitted response
# -------------------------------------------------------------------

def _binding_response_batched(logR, logE, logC,
                              logK_RE, logK_RC, logK_REC,
                              log_scale,
                              iters: int = 5):
    """
    Vectorized over dose points (logR, logE, logC).
    logK_* and log_scale are scalars.
    """
    logR = jnp.atleast_1d(logR)
    logE = jnp.atleast_1d(logE)
    logC = jnp.atleast_1d(logC)

    def f(logR_i, logE_i, logC_i):
        logc = _binding_single(logR_i, logE_i, logC_i,
                               logK_RE, logK_RC, logK_REC,
                               iters)
        return logc[REC_INDEX]

    logREC = jax.vmap(f)(logR, logE, logC)
    return jnp.exp(logREC + log_scale)


# iters is static
_binding_response_batched_jit = jax.jit(
    _binding_response_batched,
    static_argnums=(7,)
)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def normalized_response(logRtot, logEtot, logCtot,
                        logK_RE, logK_RC, logK_REC,
                        scaling_factor: float = 1.0,
                        iters: int = 5):
    """
    Normalized response = [REC] * scaling_factor
    """
    log_scale = jnp.log(scaling_factor)
    return _binding_response_batched_jit(
        logRtot, logEtot, logCtot,
        logK_RE, logK_RC, logK_REC,
        log_scale,
        iters
    )


def predict_response(logR, logE, logC,
                     logK_RE, logK_RC, logK_REC,
                     log_scale,
                     iters: int = 5):
    """
    Direct response with log_scale parameter (for Bayesian regression).
    """
    return _binding_response_batched_jit(
        logR, logE, logC,
        logK_RE, logK_RC, logK_REC,
        log_scale,
        iters
    )