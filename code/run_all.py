"""Run the book's scripts and save their outputs.

    python code/run_all.py --quick            # every script that takes under about ten minutes on a 4-core CPU
    python code/run_all.py --only ch32 ch35   # selected chapters (prefix match on the file name)
    python code/run_all.py --list             # show the table of scripts, estimated runtimes, and data needs

Outputs go to outputs/<script>.txt (stdout and stderr), with a one-line summary per script at the end.
Scripts that need data download it on first use into data/ (BACE, ESOL, a chloroplast genome, PDB entries); the sandbox
that produced the book's numbers had an internet connection only for those downloads.  Runtimes are approximate
(minutes on 4 CPU cores, one script at a time); several scripts use random seeds fixed in the file, so outputs should match the book
up to library-version differences (and, for the torch scripts, up to CPU-thread-dependent floating-point differences).
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
# (file, approximate minutes, needs a download on first run, note)
SCRIPTS = [
    ("m01_language.py", 0.1, False, "Part 0"), ("m02_functions.py", 0.1, False, "Part 0"), ("m03_calculus.py", 0.1, False, "Part 0"), ("m04_multivariable.py", 0.2, False, "Part 0"),
    ("m05_linalg1.py", 0.1, False, "Part 0"), ("m06_linalg2.py", 0.1, False, "Part 0"), ("m07_probability.py", 0.5, False, "Part 0"), ("m08_statistics.py", 0.5, False, "Part 0"),
    ("m09_discrete.py", 0.2, False, "Part 0"), ("m10_numerics.py", 0.3, False, "Part 0; torch for the autograd example"),
    ("ch01_noise_ceiling.py", 0.2, False, ""), ("ch02_linear_algebra.py", 0.1, False, ""), ("ch03_optimization.py", 0.1, False, ""), ("ch04_statistics.py", 0.2, False, ""),
    ("ch05_information.py", 0.1, False, ""), ("ch06_dynamic_programming.py", 0.1, False, ""), ("ch06_training_skeleton.py", 1, False, "torch"), ("ch07_generalization.py", 1, False, ""),
    ("ch08_latent_variables.py", 0.5, False, ""), ("ch09_autograd.py", 0.2, False, ""), ("ch10_cnn_grammar.py", 3, False, "torch"), ("ch11_sequence_models.py", 0.2, False, ""),
    ("ch12_attention.py", 0.2, False, "torch"), ("ch13_representation.py", 3, False, "torch"), ("ch14_generative.py", 3, False, "torch"), ("ch15_diffusion.py", 4, False, "torch"),
    ("ch16_geometric.py", 4, False, "torch"), ("ch17_scaling.py", 0.5, False, ""), ("ch18_interpretability.py", 3, False, "torch"), ("ch19_cell_dynamics.py", 0.5, False, ""),
    ("ch20_mutation_bias.py", 0.5, False, ""), ("ch21_popgen.py", 1, False, ""), ("ch22_thermodynamic_model.py", 0.2, False, ""), ("ch23_global_epistasis.py", 1, False, ""),
    ("ch24_chemistry.py", 4, True, "rdkit; BACE and ESOL from MoleculeNet"), ("ch25_measurement.py", 2, False, ""), ("ch26_statgen.py", 3, False, ""), ("ch27_alignment.py", 2, False, ""),
    ("ch28_representation.py", 3, True, "chloroplast genome (GenBank NC_000932); tokenizers"), ("ch29_classical_models.py", 6, True, "pyjaspar; chloroplast genome"),
    ("ch30_single_cell_methods.py", 12, False, "torch"), ("ch30_false_correlation.py", 3, False, "torch"), ("ch31_counterfactual.py", 8, False, "torch"),
    ("ch32_dna_lm.py", 25, True, "torch; chloroplast genome; about 20 min of CPU training"), ("ch32b_composition_check.py", 0.2, True, "needs the genome"),
    ("ch33_rna_structure.py", 3, True, "ViennaRNA; chloroplast genome"), ("ch34_plm_toy.py", 12, False, "torch"), ("ch35_structure.py", 6, True, "PDB 1A8O"),
    ("ch36_design_surrogate.py", 20, False, "torch (small)"), ("ch36b_local_design.py", 10, False, "imports ch36_design_surrogate"), ("ch37_molecular_ml.py", 15, True, "rdkit, torch; BACE"),
    ("ch38_sc_fm.py", 12, False, "torch; imports ch30_single_cell_methods"), ("ch38b_contrastive.py", 8, False, "torch; imports ch38_sc_fm"), ("ch39_perturbation.py", 1, False, ""),
    ("ch40_multimodal.py", 15, False, "torch; set SKIP12=1 to run only the unpaired-alignment part"), ("ch41_functional_priors.py", 15, False, "set SKIP1=1 to run only the polygenic-prediction part"),
    ("ch42_ancestral.py", 0.5, False, ""), ("ch43_benchmarks.py", 2, False, ""), ("ch44_causal.py", 1, False, ""), ("ch45_shift.py", 2, False, ""), ("ch46_design.py", 10, False, ""),
    ("ch47_scaling_data.py", 5, True, "rdkit; BACE"), ("ch48_interp_circuit.py", 1, False, "torch"), ("ch53_neural_models.py", 8, False, "torch"), ("ch54_agent_verification.py", 0.3, False, ""),
    ("ch56_information_gain.py", 0.2, False, ""),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true", help="skip scripts estimated above 10 minutes"); ap.add_argument("--only", nargs="*", help="file-name prefixes to run")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs")); a = ap.parse_args()
    if a.list:
        print(f"{'script':34s} {'~min':>5s}  download  note")
        for f, m, d, n in SCRIPTS: print(f"{f:34s} {m:5.1f}  {'yes' if d else 'no ':8s}  {n}")
        return
    os.makedirs(a.outdir, exist_ok=True); summary = []
    for f, m, d, n in SCRIPTS:
        if a.only and not any(f.startswith(p) for p in a.only): continue
        if a.quick and m > 10 and not a.only: print(f"skip {f} (about {m:g} min)"); continue
        t0 = time.time(); print(f"running {f} ...", flush=True)
        with open(os.path.join(a.outdir, f.replace(".py", ".txt")), "w") as out:
            rc = subprocess.run([sys.executable, "-u", os.path.join(HERE, f)], stdout=out, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        summary.append((f, rc, time.time() - t0)); print(f"  exit {rc}, {summary[-1][2]:.0f} s", flush=True)
    print("\nsummary"); [print(f"{f:34s} {'ok' if rc == 0 else 'FAILED (' + str(rc) + ')':12s} {t:7.0f} s") for f, rc, t in summary]
    sys.exit(0 if all(rc == 0 for _, rc, _ in summary) else 1)

if __name__ == "__main__": main()
