from _data import build_experiments_filtered, group_ER_experiments
from _fitting import run_inference
from _plotting import plot_fit_for_ligands

def _main(dir, receptors=("ER_alpha", "ER_beta"), 
		  figures=("Figure 4", "Figure 5"),
		  num_warmup=100, num_samples=200, num_chains=4):

	experiments = build_experiments_filtered(
    	receptors=receptors,
    	figures=figures
	)

	if ()

	samples = run_inference(
	    receptors=receptors, figures=figures,
	    num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, 
	    save_dir=dir,
	)

	ER_alpha, _ = group_ER_experiments(experiments)
	plot_fit_for_ligands(samples, ER_alpha, save=True, save_dir=dir)

	return samples