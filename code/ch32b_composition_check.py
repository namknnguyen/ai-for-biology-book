"""Chapter 32 (follow-up check): why do stop-gain variants get a SMALLER likelihood drop than synonymous ones in ch32_dna_lm.py?
Hypothesis: the likelihood is dominated by base composition (the chloroplast genome is about 63% A+T), and stop codons (TAA, TAG, TGA) are made by changes toward A/T.
Test: an order-0 model (base frequencies only) scored the same way, plus the fraction of each variant class whose alternative allele is A or T."""
import os
import numpy as np
from Bio import SeqIO
from Bio.Data import CodonTable
from sklearn.metrics import roc_auc_score
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "NC_000932.gb")
rec = SeqIO.read(path, "genbank"); seq = str(rec.seq).upper(); L = len(seq)
IRb, IRa = (84170, 110434), (128214, 154478); TEST_U = (60000, 84170)
tab = CodonTable.unambiguous_dna_by_id[11].forward_table; stops = set(CodonTable.unambiguous_dna_by_id[11].stop_codons); comp = str.maketrans("ACGT", "TGCA")
train = seq[:TEST_U[0]] + seq[IRb[0]:IRa[0]]; freq = {b: train.count(b) / len(train) for b in "ACGT"}
print("training base composition: " + ", ".join(f"{b} {freq[b]:.3f}" for b in "ACGT") + f"  (A+T = {freq['A'] + freq['T']:.3f})")
rows = []                                                                    # (class, ref base, alt base) on the coding strand, as in ch32_dna_lm.py
for f in rec.features:
    if f.type != "CDS" or len(f.location.parts) != 1: continue
    s0, e0, strand = int(f.location.start), int(f.location.end), f.location.strand
    if s0 < TEST_U[0] + 60 or e0 > TEST_U[1] - 60 or (e0 - s0) % 3: continue
    cds = seq[s0:e0] if strand == 1 else seq[s0:e0][::-1].translate(comp)
    for j in range(len(cds) - 3):
        cod = cds[3 * (j // 3):3 * (j // 3) + 3]; p = j % 3
        for b in "ACGT":
            if b == cds[j]: continue
            new = cod[:p] + b + cod[p + 1:]; cls = "stop-gain" if new in stops else ("synonymous" if tab.get(new) == tab.get(cod) else "missense")
            rows.append((cls, cds[j], b))
cls = np.array([r[0] for r in rows]); ref = np.array([r[1] for r in rows]); alt = np.array([r[2] for r in rows])
drop0 = np.array([np.log2(freq[a]) - np.log2(freq[b]) for a, b in zip(ref, alt)])
at_gain = np.isin(alt, ["A", "T"]) & np.isin(ref, ["C", "G"]); at_loss = np.isin(alt, ["C", "G"]) & np.isin(ref, ["A", "T"])
print("\nclass         variants   alt is A/T   change toward A/T (G/C -> A/T)   change away from A/T (A/T -> G/C)   mean order-0 drop in log2-likelihood")
for c in ("synonymous", "missense", "stop-gain"):
    m = cls == c; print(f"{c:12s} {m.sum():8d}     {np.isin(alt[m], ['A', 'T']).mean():6.3f}            {at_gain[m].mean():6.3f}                            {at_loss[m].mean():6.3f}                       {drop0[m].mean():8.3f}")
print("\nAUROC of the order-0 (composition-only) drop in log-likelihood for ranking a class above synonymous variants:")
for c in ("missense", "stop-gain"):
    m = np.isin(cls, [c, "synonymous"]); print(f"  {c:10s} {roc_auc_score(cls[m] == c, drop0[m]):.3f}")
