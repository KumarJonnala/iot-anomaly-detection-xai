"""Data audit for AI4I 2020.

Run from the project root:
    python data_audit.py

Or with a custom path:
    python data_audit.py path/to/ai4i2020.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

def hr(c='=', w=70): print(c * w)
def section(t): print(); hr(); print(t); hr()

def load(path):
    if not path.exists():
        raise FileNotFoundError(f'Not found: {path}')
    return pd.read_csv(path)

def main(argv):
    # Try common locations automatically
    candidates = [
        Path(argv[1]) if len(argv) > 1 else None,
        Path('ai4i2020.csv'),
        Path('../data/ai4i2020.csv'),
        Path('data/ai4i2020.csv'),
    ]
    df = None
    for p in candidates:
        if p and p.exists():
            df = load(p)
            print(f'Loaded: {p}')
            break
    if df is None:
        raise FileNotFoundError(
            'Cannot find ai4i2020.csv. Pass the path as an argument:\n'
            '  python data_audit.py path/to/ai4i2020.csv')

    section('1. Dataset overview')
    print(f'Rows: {len(df):,}   Columns: {df.shape[1]}')
    print(f'machine_failure==1 : {int(df["Machine failure"].sum())}')
    for ft in FAILURE_COLS:
        print(f'  {ft}: {int(df[ft].sum())}')

    section('2. Label inconsistencies (339 vs 348)')
    n_flags = df[FAILURE_COLS].sum(axis=1)
    mf      = df['Machine failure'] == 1
    flag1   = n_flags >= 1
    print(f'A. machine_failure==1          : {int(mf.sum())}')
    print(f'B. any per-type flag           : {int(flag1.sum())}')
    print(f'A union B                      : {int((mf|flag1).sum())}')
    print(f'A intersect B                  : {int((mf&flag1).sum())}')
    print(f'In A only (noise rows)         : {int((mf&~flag1).sum())}')
    print(f'In B only (near-miss rows)     : {int((flag1&~mf).sum())}')
    print()
    print('Recommendation: use 348 (ground truth B) because each per-type flag')
    print('maps to a documented physics rule, matching how our detector works.')

    section('3. Physics-rule cross-verification')
    raw = df.copy()
    raw['temp_diff']   = raw['Process temperature [K]'] - raw['Air temperature [K]']
    raw['power_w']     = raw['Torque [Nm]'] * raw['Rotational speed [rpm]'] * (2*np.pi/60)
    raw['wear_torque'] = raw['Tool wear [min]'] * raw['Torque [Nm]']
    r_hdf = (raw['temp_diff'] < 8.6) & (raw['Rotational speed [rpm]'] < 1380)
    r_pwf = (raw['power_w'] < 3500) | (raw['power_w'] > 9000)
    r_osf = raw['wear_torque'] > raw['Type'].map({'L':11000,'M':12000,'H':13000})
    r_twf = (raw['Tool wear [min]'] >= 200) & (raw['Tool wear [min]'] <= 240)
    print(f'{"Rule":<10} {"flags":>6} {"target":>8} {"caught":>7} {"recall":>8} {"precision":>10}')
    for name, rule, target in [
        ('HDF', r_hdf,'HDF'), ('PWF', r_pwf,'PWF'),
        ('OSF', r_osf,'OSF'), ('TWF', r_twf,'TWF')
    ]:
        fl  = int(rule.sum())
        tn  = int(df[target].sum())
        ct  = int((rule & (df[target]==1)).sum())
        rec = ct/tn if tn else 0
        pre = ct/fl if fl else 0
        print(f'  rule_{name.lower():<5} {fl:>6d} {tn:>8d} {ct:>7d} {rec:>7.1%} {pre:>9.1%}')

    hr(); print('Audit complete.'); hr()

if __name__ == '__main__':
    main(sys.argv)
