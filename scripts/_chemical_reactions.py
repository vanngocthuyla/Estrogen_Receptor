import numpy as np
import jax
import jax.numpy as jnp

class ChemicalReactions:
    """
    Computes equilibrium log-concentrations for a system of chemical equations
    using a Gauss–Newton solver, JAX-jitted for speed.
    Supports batched total concentrations over doses.
    """

    def __init__(self, chemical_equations, conservation_equations, iters=3):
        self.iters = iters

        # Collect species
        all_species = []
        for ceq in chemical_equations:
            for s in ceq.keys():
                if s not in all_species:
                    all_species.append(s)
        all_species = sorted(all_species)

        self.all_species = all_species
        self.index_of_species = {s: i for i, s in enumerate(all_species)}
        self.nspecies = len(all_species)
        self.nchemical_equations = len(chemical_equations)
        self.nconservation_equations = len(conservation_equations)

        # Stoichiometry matrix
        S = np.zeros((self.nspecies, self.nchemical_equations))
        for r, ceq in enumerate(chemical_equations):
            for s, nu in ceq.items():
                S[self.index_of_species[s], r] = nu

        # Conservation matrix
        C = np.zeros((self.nspecies, self.nconservation_equations))
        for r, ceq in enumerate(conservation_equations):
            for s, nu in ceq.items():
                C[self.index_of_species[s], r] += nu

        self.S = jnp.array(S)
        self.C = jnp.array(C)

        # Build solvers
        self._logceq_single_jit = self._build_logceq_single_jit()
        self._logceq_batched_jit = self._build_logceq_batched_jit()

    # ----------------------------------------------------------
    # Single-system Gauss–Newton solver
    # ----------------------------------------------------------
    def _build_logceq_single_jit(self):
        S = self.S
        C = self.C
        nspecies = self.nspecies
        ncons = self.nconservation_equations
        iters = self.iters

        def step(logc, carry):
            logKd, logctot, logci = carry

            # Stable log-sum-exp
            shift = jnp.max(logc)
            logsum = shift + jnp.log(jnp.exp(logc - shift) @ C)

            eps = jnp.concatenate((logc @ S - logKd, logsum - logctot))

            exp_term = jnp.exp(
                jnp.tile(logc, (ncons, 1)) -
                jnp.tile(logsum, (nspecies, 1)).T
            )
            J = jnp.vstack([S.T, C.T * exp_term])

            JTJ = J.T @ J
            JTeps = J.T @ eps
            delta = jnp.linalg.solve(JTJ, JTeps)

            new_logc = jnp.minimum(logc - delta, logci)
            return new_logc, jnp.sum(eps**2)

        def solve(logKd, logctot):
            maxlogc = jnp.max(logctot)
            logci = jnp.full((nspecies,), maxlogc)
            carry = (logKd, logctot, logci)
            logc0 = logci

            def body(logc, _):
                return step(logc, carry)

            logc_final, _ = jax.lax.scan(body, logc0, None, length=iters)
            return logc_final

        return jax.jit(solve)

    # ----------------------------------------------------------
    # Batched solver over doses
    # ----------------------------------------------------------
    def _build_logceq_batched_jit(self):
        solve_single = self._logceq_single_jit

        def solve_batched(logKd, logctot_batch):
            # logctot_batch: (ncons, N)
            N = logctot_batch.shape[1]

            def solve_i(i):
                return solve_single(logKd, logctot_batch[:, i])

            return jax.vmap(solve_i)(jnp.arange(N))

        return jax.jit(solve_batched)

    # Public API
    def logceq(self, logKd, logctot):
        return self._logceq_single_jit(logKd, logctot)

    def logceq_batched(self, logKd, logctot_batch):
        return self._logceq_batched_jit(logKd, logctot_batch)