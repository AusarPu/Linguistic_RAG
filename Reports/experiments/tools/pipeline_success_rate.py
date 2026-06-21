import os
import json
import re
from pathlib import Path
from typing import Dict, List
import csv
import math
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

FIG_SIZE = (6.0, 3.75)

ROOT = Path('/home/pushihao/RAG')
RUNS_ROOT = ROOT / 'Reports' / 'experiments' / 'datasets' / 'runs_GPT_oss'
OUT_CSV = RUNS_ROOT / 'aggregated_pipeline_success_summary_structured.csv'
PICS_DIR = ROOT / 'Reports' / 'docs' / 'pics'

def _classify(item: Dict) -> str:
    chunks = item.get('retrieved_chunk_ids') or []
    if len(chunks) == 0:
        return 'retrieval_failure'
    ans = item.get('system_answer') or ''
    t = ans.lower()
    if ('信息不足' in ans) or ('insufficient information' in t) or ('无法回答' in ans) or ('cannot answer' in t) or ('not enough information' in t):
        return 'faithful_refusal'
    return 'valid_response'

def _parse_cfg(path: Path) -> Dict:
    s = path.as_posix()
    m_tau = re.search(r"thr_chunk_([0-9.]+)", s)
    m_k = re.search(r"topk_final_(\d+)", s)
    m_seed = re.search(r"/(r\d+)/", s)
    m_dataset = re.search(r"/rag_evaluation_results/([^/]+)/", s)
    tau = float(m_tau.group(1)) if m_tau else None
    k_final = int(m_k.group(1)) if m_k else None
    seed = m_seed.group(1) if m_seed else None
    dataset = m_dataset.group(1) if m_dataset else None
    return {'tau': tau, 'k_final': k_final, 'seed': seed, 'dataset': dataset}

def _summarize_file(json_path: Path) -> Dict:
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    total = len(data)
    fail = 0
    refuse = 0
    for it in data:
        c = _classify(it)
        if c == 'retrieval_failure':
            fail += 1
        elif c == 'faithful_refusal':
            refuse += 1
    valid = total - fail - refuse
    p_fail = (fail / total) if total else 0.0
    p_refuse = (refuse / total) if total else 0.0
    p_valid = (valid / total) if total else 0.0
    return {
        'n_total': total,
        'n_fail': fail,
        'n_refuse': refuse,
        'n_valid': valid,
        'p_fail': p_fail,
        'p_refuse': p_refuse,
        'p_valid': p_valid,
    }

def _walk_files(root: Path) -> List[Path]:
    return list(root.glob('**/rag_evaluation_results/*/evaluation_results.json'))

def _make_id(tau: float, k_final: int) -> str:
    return f"threshold_{tau}_topk_final_{k_final}"

def _write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        'id','threshold','final_topk','run','dataset',
        'pipeline_n_total','pipeline_n_fail','pipeline_n_refuse','pipeline_n_valid',
        'pipeline_p_fail','pipeline_p_refuse','pipeline_p_valid','file'
    ]
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _aggregate_rows(files: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for p in files:
        cfg = _parse_cfg(p)
        agg = _summarize_file(p)
        rid = _make_id(cfg['tau'], cfg['k_final'])
        row = {
            'id': rid,
            'threshold': cfg['tau'],
            'final_topk': cfg['k_final'],
            'run': cfg['seed'],
            'dataset': cfg['dataset'],
            'pipeline_n_total': agg['n_total'],
            'pipeline_n_fail': agg['n_fail'],
            'pipeline_n_refuse': agg['n_refuse'],
            'pipeline_n_valid': agg['n_valid'],
            'pipeline_p_fail': agg['p_fail'],
            'pipeline_p_refuse': agg['p_refuse'],
            'pipeline_p_valid': agg['p_valid'],
            'file': p.as_posix(),
        }
        rows.append(row)
    return rows

def _avg_by_tau(rows: List[Dict], dataset: str, k_final: int) -> Dict[float, Dict[str, float]]:
    m: Dict[float, List[Dict[str, float]]] = {}
    for r in rows:
        if r['dataset'] != dataset:
            continue
        if r['final_topk'] != k_final:
            continue
        tau = float(r['threshold'])
        if tau not in m:
            m[tau] = []
        m[tau].append({'p_fail': r['pipeline_p_fail'], 'p_refuse': r['pipeline_p_refuse'], 'p_valid': r['pipeline_p_valid']})
    out: Dict[float, Dict[str, float]] = {}
    for tau, arr in sorted(m.items(), key=lambda x: x[0]):
        n = len(arr)
        pf = sum(x['p_fail'] for x in arr) / n if n else 0.0
        pr = sum(x['p_refuse'] for x in arr) / n if n else 0.0
        pv = sum(x['p_valid'] for x in arr) / n if n else 0.0
        out[tau] = {'p_fail': pf, 'p_refuse': pr, 'p_valid': pv}
    return out

def _plot_dataset(rows: List[Dict], dataset: str, k_final: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _avg_by_tau(rows, dataset, k_final)
    xs = list(stats.keys())
    pf = [stats[t]['p_fail'] for t in xs]
    pr = [stats[t]['p_refuse'] for t in xs]
    pv = [stats[t]['p_valid'] for t in xs]
    plt.figure(figsize=FIG_SIZE)
    plt.bar(xs, pf, label='Retrieval Failure')
    plt.bar(xs, pr, bottom=pf, label='Faithful Refusal')
    btm = [pf[i] + pr[i] for i in range(len(xs))]
    plt.bar(xs, pv, bottom=btm, label='Valid Response')
    plt.ylim(0,1)
    plt.xlabel('Threshold')
    plt.ylabel('Proportion')
    plt.title(f'Pipeline Success Rate – {dataset} (K={k_final})')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='lower right')
    out_path = out_dir / f'pipeline_success_{dataset}_k{k_final}.png'
    plt.tight_layout()
    plt.savefig(out_path.as_posix())
    plt.close()
    return out_path

def _micro_by_tau(rows: List[Dict], k_final: int, taus: List[float]) -> Dict[float, Dict[str, float]]:
    out: Dict[float, Dict[str, float]] = {}
    for tau in taus:
        for sys in ['Baseline','MARS']:
            total = 0
            fail = 0
            refuse = 0
            valid = 0
            for r in rows:
                if float(r['threshold']) != float(tau):
                    continue
                if r['final_topk'] != k_final:
                    continue
                run = r['run']
                if sys == 'Baseline' and run != 'r1':
                    continue
                if sys == 'MARS' and run != 'r4':
                    continue
                total += int(r['pipeline_n_total'])
                fail += int(r['pipeline_n_fail'])
                refuse += int(r['pipeline_n_refuse'])
                valid += int(r['pipeline_n_valid'])
            if total == 0:
                pf = pr = pv = 0.0
            else:
                pf = fail / total
                pr = refuse / total
                pv = valid / total
            key = f"{sys}:{tau}"
            out[tau] = out.get(tau, {})
            out[tau][key] = {'p_fail': pf, 'p_refuse': pr, 'p_valid': pv}
    return out

def _plot_micro_tau(rows: List[Dict], k_final: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    taus = [0.5, 0.6, 0.65, 0.7]
    stats = _micro_by_tau(rows, k_final, taus)
    x = list(taus)
    width = 0.35
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # Baseline bars
    b_pf = [stats[t][f"Baseline:{t}"]['p_fail'] for t in x]
    b_pr = [stats[t][f"Baseline:{t}"]['p_refuse'] for t in x]
    b_pv = [stats[t][f"Baseline:{t}"]['p_valid'] for t in x]
    # MARS bars
    m_pf = [stats[t][f"MARS:{t}"]['p_fail'] for t in x]
    m_pr = [stats[t][f"MARS:{t}"]['p_refuse'] for t in x]
    m_pv = [stats[t][f"MARS:{t}"]['p_valid'] for t in x]
    # positions
    import numpy as np
    idx = np.arange(len(x))
    # Baseline stacked
    ax.bar(idx - width/2, b_pf, width=width, label='Baseline – Retrieval Failure', color='#d62728')
    ax.bar(idx - width/2, b_pr, width=width, bottom=b_pf, label='Baseline – Faithful Refusal', color='#ff7f0e')
    ax.bar(idx - width/2, b_pv, width=width, bottom=[b_pf[i]+b_pr[i] for i in range(len(b_pf))], label='Baseline – Valid Response', color='#2ca02c')
    # MARS stacked
    ax.bar(idx + width/2, m_pf, width=width, label='MARS – Retrieval Failure', color='#8c564b')
    ax.bar(idx + width/2, m_pr, width=width, bottom=m_pf, label='MARS – Faithful Refusal', color='#bcbd22')
    ax.bar(idx + width/2, m_pv, width=width, bottom=[m_pf[i]+m_pr[i] for i in range(len(m_pf))], label='MARS – Valid Response', color='#1f77b4')
    ax.set_xticks(idx)
    ax.set_xticklabels([str(t) for t in x])
    ax.set_ylim(0,1)
    ax.set_xlabel('Threshold (τ)')
    ax.set_ylabel('Proportion')
    ax.set_title(f'Micro-Average Pipeline Success (K={k_final})')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', ncol=2)
    out_path = out_dir / f'pipeline_success_micro_tau_k{k_final}.png'
    fig.tight_layout()
    fig.savefig(out_path.as_posix())
    plt.close(fig)
    return out_path

def _micro_valid_vs_k(rows: List[Dict], tau: float, k_vals: List[int]) -> Dict[str, List[float]]:
    b_valid: List[float] = []
    m_valid: List[float] = []
    for k in k_vals:
        total_b = fail_b = refuse_b = valid_b = 0
        total_m = fail_m = refuse_m = valid_m = 0
        for r in rows:
            if float(r['threshold']) != float(tau):
                continue
            if r['final_topk'] != k:
                continue
            run = r['run']
            if run == 'r1':
                total_b += int(r['pipeline_n_total'])
                fail_b += int(r['pipeline_n_fail'])
                refuse_b += int(r['pipeline_n_refuse'])
                valid_b += int(r['pipeline_n_valid'])
            elif run == 'r4':
                total_m += int(r['pipeline_n_total'])
                fail_m += int(r['pipeline_n_fail'])
                refuse_m += int(r['pipeline_n_refuse'])
                valid_m += int(r['pipeline_n_valid'])
        b_valid.append((valid_b / total_b) if total_b else 0.0)
        m_valid.append((valid_m / total_m) if total_m else 0.0)
    return {'Baseline': b_valid, 'MARS': m_valid}

def _plot_micro_valid_vs_k(rows: List[Dict], tau: float, k_vals: List[int], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _micro_valid_vs_k(rows, tau, k_vals)
    import numpy as np
    x = np.array(k_vals)
    plt.figure(figsize=(4.5, 3.0))
    plt.plot(x, stats['Baseline'], marker='o', label='Baseline')
    plt.plot(x, stats['MARS'], marker='o', label='MARS')
    plt.ylim(0.20, 1.0)
    plt.xlabel('K_final')
    plt.ylabel('Valid Response Rate')
    plt.title(f'Micro-Average Valid Response vs K (τ={tau})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = out_dir / f'pipeline_success_micro_valid_vs_k_tau{tau}.png'
    plt.tight_layout()
    plt.savefig(out_path.as_posix())
    plt.close()
    return out_path

def main() -> None:
    files = _walk_files(RUNS_ROOT)
    rows = _aggregate_rows(files)
    _write_csv(rows, OUT_CSV)
    datasets = ['hotpotqa','natural_questions','triviaqa','ms_marco']
    for ds in datasets:
        _plot_dataset(rows, ds, 10, PICS_DIR)
    _plot_micro_tau(rows, 10, PICS_DIR)
    _plot_micro_valid_vs_k(rows, 0.65, [3,5,7,10], PICS_DIR)

if __name__ == '__main__':
    main()
