import jax.numpy as jnp

def make_experiment_Fig_4(Rtot = 1E-9, Etot=25E-6):

    logC = jnp.log(10**(jnp.array([-11.0, -10.5, -10.0, -9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0])))
    logR = jnp.array([jnp.log(Rtot)]*len(logC)) # 1 nM
    logE = jnp.array([jnp.log(Etot)]*len(logC)) # 25 μM
    
    ER_alpha = {
        "name": "ER_alpha",
        "ligands": [
            {
                "name": "17b_E2",
                "response": jnp.array([20, 120, 420, 850, 1250, 1650, 1900, 2050, 2100, 2120, 2100])
            },
            {
                "name": "17a_E2",
                "response": jnp.array([0, 80, 300, 600, 1050, 1300, 1500, 1850, 2000, 2050, 2050])
            },
            {
                "name": "GEN",
                "response": jnp.array([0, 0, 20, 60, 180, 600, 1200, 1600, 1800, 1900, 1900])
            },
            {
                "name": "Equ",
                "response": jnp.array([0, 0, 10, 20, 50, 250,  700, 1200, 1450, 1550, 1600])
            },
            {
                "name": "Eqn",
                "response": jnp.array([0, 0, 0, 10, 20, 80, 300, 900, 1350, 1500, 1550])
            }
        ]
    }

    ER_beta = {
        "name": "ER_beta",
        "ligands": [
            {
                "name": "17b_E2",
                "response": jnp.array([0, 20, 120, 250, 380, 500, 650, 780, 900, 980, 1000])
            },
            {
                "name": "17a_E2",
                "response": jnp.array([0, 30, 150, 280, 430, 620, 800, 900, 960, 990, 1020])
            },
            {
                "name": "GEN",
                "response": jnp.array([0,  20, 100, 200, 330, 470, 600, 760, 900, 980, 1020])
            },
            {
                "name": "Equ",
                "response": jnp.array([0,  10,  80, 160, 300, 450, 620, 780, 920, 980, 1000])
            },
            {
                "name": "Eqn",
                "response": jnp.array([0,  10,  60, 140, 280, 420, 600, 740, 880, 980, 1020])
            }
        ]
    }

    for ligands in ER_alpha["ligands"]:
        ligands['logR'] = logR
        ligands['logE'] = logE
        ligands['logC'] = logC

    for ligands in ER_beta["ligands"]:
        ligands['logR'] = logR
        ligands['logE'] = logE
        ligands['logC'] = logC

    return ER_alpha, ER_beta


def make_experiment_Fig_5(Rtot=1E-9, Ctot=100E-9):

    ER_alpha = {
        "name": "ER_alpha",
        "ligands": [
            # {
            # "name": "17b_E2", 
            # "logE" : jnp.log(10**jnp.array([-9.4926, -9.1915, -8.7902, -8.4483, -8.1398, -7.8232, -7.5375, -7.2264, -6.9972, -6.7250, -6.4634])), 
            # "response": jnp.array([3.1746, 5.2910, 9.5238, 22.2222, 38.6243, 74.0741, 90.4762, 96.2963, 99.4709, 93.6508, 91.5344]),
            # },
            {
            "name": "EE2",
            "logE": jnp.log(10**jnp.array([-9.4926, -9.1460, -8.7902, -8.4693, -8.1197, -7.8426, -7.5375, -7.2264, -6.9798, -6.7417, -6.4794])),
            "response": jnp.array([3.1746, 5.8201, 12.1693, 24.3386, 49.7354, 79.3651, 96.8254, 97.3545, 98.4127, 98.9418, 100.5291]),
            },
            {
                "name": "DMS",
                "logE": jnp.log(10**jnp.array([-9.4926, -9.0557, -8.7684, -8.4483, -8.1398, -7.8232, -7.5375, -7.2085, -7.0145, -6.7417, -6.4634])),
                "response": jnp.array([2.1164, 3.1746, 4.7619, 9.5238, 21.6931, 37.5661, 66.6667, 88.3598, 94.1799, 98.9418, 95.7672]),
            },
            {
                "name": "DES",
                "logE": jnp.log(10**jnp.array([-9.0782, -8.7684, -8.4274, -8.0595, -7.8038, -7.5938, -7.5002, -7.3165, -7.2085, -7.0145, -6.7417])),
                "response": jnp.array([0.5291, 1.0582, 1.0582, 6.3492, 12.6984, 22.2222, 29.1005, 41.7989, 59.7884, 86.7725, 95.2381]),
            },
            {
                "name": "meso_Hex",
                "logE": jnp.log(10**jnp.array([-9.4926, -9.1007, -8.7251, -8.3857, -7.7845, -7.5938, -7.5002, -7.3165, -7.0319, -6.7584, -6.4634])),
                "response": jnp.array([1.0582, 0.0000, 5.2910, 13.2275, 34.9206, 49.7354, 69.3122, 83.5979, 94.1799, 98.4127, 99.4709]),
            },
            # {
            #     "name": "17b_E2",
            #     "logE": jnp.log(10**jnp.array([-10.0000, -9.5927, -9.3120, -9.0217, -8.7403, -8.4678, -8.2200, -7.9795, -7.7460, -7.5193, -7.0577])),
            #     "response": jnp.array([-0.5348, -0.5348, 3.2086, 9.6257, 19.7861, 41.1765, 73.2620, 82.3529, 92.5134, 94.6524, 100.0000]),
            # },
            {
                "name": "E1",
                "logE": jnp.log(10**jnp.array([-9.5927, -9.0038, -8.1390, -7.9165, -7.6697, -7.4305, -7.2274, -6.9881, -6.7837, -6.5851, -6.3925])),
                "response": jnp.array([0.5348, 0.0000, 8.5561, 17.1123, 31.0160, 51.3369, 66.8449, 80.2139, 88.7701, 97.3262, 97.8610]),
            },
            {
                "name": "E3",
                "logE": jnp.log(10**jnp.array([-10.0000, -9.5927, -9.0038, -8.7577, -8.5014, -8.2363, -7.9953, -7.7306, -7.5193, -7.2848, -6.7837])),
                "response": jnp.array([-0.5348, 1.0695, 3.2086, 7.4866, 16.5775, 36.8984, 55.0802, 76.4706, 88.7701, 95.7219, 101.6043]),
            },
            {
                "name": "Equ",
                "logE": jnp.log(10**jnp.array([-9.0217, -8.1713, -7.9165, -7.6697, -7.4452, -7.1988, -6.9743, -6.7971, -6.5851, -6.3925, -6.2054])),
                "response": jnp.array([-1.0695, 6.9519, 16.0428, 29.9465, 44.3850, 63.6364, 72.1925, 82.3529, 86.6310, 90.9091, 94.6524]),
            },
            {
                "name": "Eqn",
                "logE": jnp.log(10**jnp.array([-9.0038, -8.1552, -7.9009, -7.6849, -7.4452, -7.2131, -7.0020, -6.7702, -6.5851, -6.3925, -5.7329])),
                "response": jnp.array([1.0695, 1.0695, 2.6738, 4.2781, 12.8342, 27.2727, 43.8503, 63.1016, 77.0053, 86.6310, 100.0000]),
            },
            {
                "name": "GEN",
                "logE": jnp.log(10**jnp.array([-9.0038, -8.1552, -7.9009, -7.6849, -7.4452, -7.2274, -7.0020, -6.7971, -6.5851, -6.3925, -5.7216])),
                "response": jnp.array([-1.0695, 0.0000, 0.5348, 2.1390, 3.2086, 6.9519, 16.5775, 32.6203, 53.4759, 70.0535, 100.0000]),
            },
            {
                "name": "17b_E2",
                "logE": jnp.log(10**jnp.array([-9.9733, -9.4282, -8.6776, -8.3361, -7.9654, -7.6725, -7.3509, -7.0805, -6.7837, -6.4994, -6.2437])),
                "response": jnp.array([-0.5348, 3.2086, 14.4385, 26.7380, 47.0588, 77.5401, 84.4920, 87.7005, 99.4652, 97.8610, 99.4652]),
            },
            {
                "name": "17a_E2",
                "logE": jnp.log(10**jnp.array([-8.6776, -8.1378, -7.7967, -7.4899, -7.1951, -6.8936, -6.6223, -6.3447, -5.8240, -5.6098, -5.3891])),
                "response": jnp.array([-2.1390, 5.3476, 12.8342, 27.8075, 55.0802, 65.2406, 80.7487, 88.7701, 98.3957, 99.4652, 98.9305]),
            },
            {
                "name": "dl_Hex",
                "logE": jnp.log(10**jnp.array([-8.6776, -8.1160, -7.5908, -7.2921, -6.9864, -6.7295, -6.3959, -6.1772, -5.9024, -5.7159, -5.4035])),
                "response": jnp.array([0.5348, 2.6738, 8.5561, 17.6471, 28.8770, 44.3850, 62.5668, 73.7968, 81.2834, 91.4439, 97.8610]),
            },
        ]
    }
    

    ER_beta = {
        "name": "ER_beta",
        "ligands": [
            {
                "name": "17b_E2",
                "logE": jnp.log(10**jnp.array([-9.5,-9.0,-8.5,-8.0,-7.8,-7.6,-7.4,-7.2])),
                "response": jnp.array([0,2,10,40,75,95,100,100]),
            },
            {
                "name": "EE2",
                "logE": jnp.log(10**jnp.array([-8.0,-7.8,-7.6,-7.4,-7.2,-7.0,-6.8,-6.6])),
                "response": jnp.array([2,5,10,25,50,80,95,100]),
            },
            {
                "name": "DMS",
                "logE": jnp.log(10**jnp.array([-9.5,-9.0,-8.5,-8.0,-7.8,-7.6,-7.4,-7.2])),
                "response": jnp.array([0,1,8,35,70,90,98,100]),
            },
            {
                "name": "DES",
                "logE": jnp.log(10**jnp.array([-8.5,-8.0,-7.5,-7.2,-7.0,-6.8,-6.6,-6.4])),
                "response": jnp.array([0,3,8,20,30,50,80,98]),
            },
            {
                "name": "meso_Hex",
                "logE": jnp.log(10**jnp.array([-8.5,-8.0,-7.5,-7.2,-7.0,-6.8,-6.6,-6.4])),
                "response": jnp.array([0,4,10,18,30,52,85,100]),
            },
            {
                "name": "E1",
                "logE": jnp.log(10**jnp.array([-8.5,-8.0,-7.5,-7.0,-6.5,-6.0])),
                "response": jnp.array([2,10,30,65,90,100]),
            },
            {
                "name": "E3",
                "logE": jnp.log(10**jnp.array([-8.8,-8.3,-7.8,-7.3,-6.8,-6.3])),
                "response": jnp.array([2,12,35,70,92,100]),
            },
            {
                "name": "Equ",
                "logE": jnp.log(10**jnp.array([-8.5,-8.0,-7.5,-7.0,-6.5])),
                "response": jnp.array([3,15,45,80,100]),
            },
            {
                "name": "Eqn",
                "logE": jnp.log(10**jnp.array([-8.0,-7.5,-7.0,-6.5,-6.0])),
                "response": jnp.array([2,12,40,78,100]),
            },
            {
                "name": "GEN",
                "logE": jnp.log(10**jnp.array([-8.0,-7.5,-7.0,-6.5,-6.0])),
                "response": jnp.array([1,10,35,75,100]),
            },
            {
                "name": "17a_E2",
                "logE": jnp.log(10**jnp.array([-8.8,-8.3,-7.8,-7.3,-6.8,-6.3])),
                "response": jnp.array([0,4,18,50,82,98]),
            },
            {
                "name": "dl_Hex",
                "logE": jnp.log(10**jnp.array([-8.5,-8.0,-7.5,-7.0,-6.5,-6.0])),
                "response": jnp.array([0,2,8,35,85,100]),
            },
        ],
    }

    for ligands in ER_alpha["ligands"]:
        ligands['logR'] = jnp.array([jnp.log(Rtot)]*len(ligands['logE']))
        ligands['logC'] = jnp.array([jnp.log(Ctot)]*len(ligands['logE']))

    for ligands in ER_beta["ligands"]:
        ligands['logR'] = jnp.array([jnp.log(Rtot)]*len(ligands['logE']))
        ligands['logC'] = jnp.array([jnp.log(Ctot)]*len(ligands['logE']))

    return ER_alpha, ER_beta


# ------------------------------------------------------------
# Build experiment list
# ------------------------------------------------------------

def build_all_experiments():
    fig4_alpha, fig4_beta = make_experiment_Fig_4()
    fig5_alpha, fig5_beta = make_experiment_Fig_5()
    return [fig4_alpha, fig4_beta, fig5_alpha, fig5_beta]


def build_experiments_filtered(receptors=("ER_alpha", "ER_beta"),
                               figures=("Figure 4", "Figure 5")):
    """
    Returns a list of experiments, each of the form:
        {
            "name":   receptor name, e.g. "ER_alpha" or "ER_beta",
            "figure": "Figure 4" or "Figure 5",
            "ligands": [
                {
                    "name": "17b_E2",
                    "response": ...,
                    "logR": ...,
                    "logE": ...,
                    "logC": ...,
                },
                ...
            ],
        }
    """
    fig4_alpha, fig4_beta = make_experiment_Fig_4()
    fig5_alpha, fig5_beta = make_experiment_Fig_5()

    all_exps = [
        {
            "name": fig4_alpha["name"],   # "ER_alpha"
            "figure": "Figure 4",
            "ligands": fig4_alpha["ligands"],
        },
        {
            "name": fig4_beta["name"],    # "ER_beta"
            "figure": "Figure 4",
            "ligands": fig4_beta["ligands"],
        },
        {
            "name": fig5_alpha["name"],
            "figure": "Figure 5",
            "ligands": fig5_alpha["ligands"],
        },
        {
            "name": fig5_beta["name"],
            "figure": "Figure 5",
            "ligands": fig5_beta["ligands"],
        },
    ]

    filtered = []
    for exp in all_exps:
        receptor = exp["name"]
        fig_name = exp["figure"]
        if receptor in receptors and fig_name in figures:
            filtered.append(exp)

    return filtered


def build_experiments_combined(receptors=("ER_alpha", "ER_beta"),
                               figures=("Figure 4", "Figure 5")):
    """
    Returns a list of experiments, one per receptor, each of the form:
        {
            "name":   receptor name, e.g. "ER_alpha",
            "ligands": [
                {
                    "name": "17b_E2_4",   # renamed if needed
                    "response": ...,
                    "logR": ...,
                    "logE": ...,
                    "logC": ...,
                },
                ...
            ],
        }

    Ligands from Figure 4 and Figure 5 are merged.
    If a ligand name appears in both figures, suffixes _4 and _5 are added.
    """

    # Load your existing experiments
    fig4_alpha, fig4_beta = make_experiment_Fig_4()
    fig5_alpha, fig5_beta = make_experiment_Fig_5()

    # Organize by receptor
    per_receptor = {
        "ER_alpha": {"fig4": fig4_alpha["ligands"], "fig5": fig5_alpha["ligands"]},
        "ER_beta":  {"fig4": fig4_beta["ligands"],  "fig5": fig5_beta["ligands"]},
    }

    combined = []

    for receptor in receptors:
        if receptor not in per_receptor:
            continue

        ligs4 = per_receptor[receptor]["fig4"] if "Figure 4" in figures else []
        ligs5 = per_receptor[receptor]["fig5"] if "Figure 5" in figures else []

        # Collect ligand names to detect duplicates
        names4 = {lig["name"] for lig in ligs4}
        names5 = {lig["name"] for lig in ligs5}

        # If a ligand appears in both figures → rename
        duplicates = names4.intersection(names5)

        merged_ligands = []

        # Add Figure 4 ligands
        for lig in ligs4:
            new_name = lig["name"] + ("_4" if lig["name"] in duplicates else "")
            merged_ligands.append({
                "name": new_name,
                "response": lig["response"],
                "logR": lig["logR"],
                "logE": lig["logE"],
                "logC": lig["logC"],
            })

        # Add Figure 5 ligands
        for lig in ligs5:
            new_name = lig["name"] + ("_5" if lig["name"] in duplicates else "")
            merged_ligands.append({
                "name": new_name,
                "response": lig["response"],
                "logR": lig["logR"],
                "logE": lig["logE"],
                "logC": lig["logC"],
            })

        combined.append({
            "name": receptor,
            "ligands": merged_ligands,
        })

    return combined


def group_ER_experiments(experiments):
    """
    Build two dictionaries:
      - ER_alpha[ligand] = list of experiment objects
      - ER_beta[ligand]  = list of experiment objects

    Ligands included = union of Fig.4 and Fig.5 ligands.
    """

    ER_alpha = {"name": "ER_alpha"}
    ER_beta  = {"name": "ER_beta"}

    ER_alpha["ligands"] = []
    ER_beta["ligands"]  = []

    for exp in experiments:
        if exp["name"] == "ER_alpha":
            for lig in exp["ligands"]:
                lig['figure'] = exp['figure']
                ER_alpha["ligands"].append(lig)
        if exp["name"] == "ER_beta":
            for lig in exp["ligands"]:
                lig['figure'] = exp['figure']
                ER_beta["ligands"].append(lig)

    return ER_alpha, ER_beta