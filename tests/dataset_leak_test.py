import csv
from dataset import make_splits
seq = {r['sample_id']: r['sequence_id'] for r in csv.DictReader(open('configs/cvc_clinicdb_sequences.csv'))}
for name in ['cvc_clinicdb', 'cvc_clinicdb_leaky']:
    tr, va = make_splits(name)
    ds = tr.dataset
    S = lambda sub: {seq[ds.samples[i][2]] for i in sub.indices}
    print(f'{name:22s} train={len(tr)} val={len(va)} train_seqs={len(S(tr))} val_seqs={len(S(va))} overlap={len(S(tr) & S(va))}')