#!/usr/bin/env python3
"""Validate four matched test phases and report measured model throughput."""
import argparse,csv,json,math,pathlib,statistics
CASES=[((512,0),'pp512'),((2048,0),'pp2048'),((0,128),'tg128')]
SETS={'stock300':('stock.jsonl','stock',300),
      'unlock300':('unlocked.jsonl','unlocked',300),
      'stock420':('stock-420.jsonl','stock',420),
      'unlock420':('unlocked-420.jsonl','unlocked',420)}
COMPARISONS={
    'unlock_at_300':('stock300','unlock300'),
    'unlock_at_420':('stock420','unlock420'),
    'stock_420_vs_300':('stock300','stock420'),
    'unlock_420_vs_300':('unlock300','unlock420')}
def read(path,phase,power):
    ls=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(ls)==15 and len({z['id'] for z in ls})==15,(str(path),'model roster')
    assert all(x['phase']==phase and x['power_w']==power for x in ls),path
    return {x['id']:x for x in ls}
def tests(x):
    assert x['status']=='ok',(x['id'],'not successful')
    r={(int(y['n_prompt']),int(y['n_gen'])):y for y in x['tests']}
    assert set(r)=={x for x,_ in CASES},(x['id'],'missing test case')
    assert all(len(t['samples_ts'])==3 and t.get('backends')=='CUDA' and t.get('n_gpu_layers')==99 for t in r.values()),(x['id'],'backend/repeats/offload mismatch')
    return r
def match(a,b):
    for key in ('model_sha256','model_size_bytes','binary_sha256','parameters','quantization'):
        assert a[key]==b[key],(a['id'],key)
    A=tests(a);B=tests(b)
    for test,_ in CASES:
        for field in ('model_n_params','n_gpu_layers','n_threads','n_batch','n_ubatch','gpu_info'):
            assert A[test][field]==B[test][field],(a['id'],test,field)
def agg(xs):
    return {'geomean_gain_pct':100*(math.exp(statistics.mean(math.log(1+v/100) for v in xs))-1),
            'median_gain_pct':statistics.median(xs),'min_gain_pct':min(xs),
            'max_gain_pct':max(xs),'model_count':len(xs)}
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--results',default='results')
    opts=a.parse_args()
    out=pathlib.Path(opts.results).resolve()
    datasets={k:read(out/f,p,w) for k,(f,p,w) in SETS.items()}
    roster=set(datasets['stock300'])
    assert all(set(x)==roster for x in datasets.values()),'different model rosters'
    rows=[]; by_comparison={k:{label:[] for _,label in CASES} for k in COMPARISONS}
    noisy=[]
    for ident in sorted(roster):
        ref=datasets['stock300'][ident]
        for phase in datasets: match(ref,datasets[phase][ident])
        for key,label in CASES:
            values={p:float(tests(datasets[p][ident])[key]['avg_ts']) for p in SETS}
            cv={p:100*float(tests(datasets[p][ident])[key]['stddev_ts'])/v for p,v in values.items()}
            assert all(v>0 for v in values.values()),ident
            gains={}
            for cmp,(left,right) in COMPARISONS.items():
                gains[cmp]=100*(values[right]/values[left]-1)
                by_comparison[cmp][label].append(gains[cmp])
            high_var=max(cv.values())>5
            if high_var:noisy.append({'model':ident,'case':label,'max_cv_pct':round(max(cv.values()),2)})
            data=tests(ref)[key]
            rows.append({'model':ident,'test':label,'parameters_b':round(data['model_n_params']/1e9,3),
                'quantization':ref['quantization'],
                **{p+'_tok_s':round(v,4) for p,v in values.items()},
                **{p+'_cv_pct':round(v,2) for p,v in cv.items()},
                **{cmp+'_gain_pct':round(v,3) for cmp,v in gains.items()},
                'high_variability':high_var,
                'model_sha256':ref['model_sha256'],'binary_sha256':ref['binary_sha256']})
    assert len(rows)==45
    with (out/'power-matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)
    means={k:{case:agg(vals) for case,vals in cases.items()} for k,cases in by_comparison.items()}
    (out/'power-matrix-aggregates.json').write_text(json.dumps({
        'definitions':{
          'unlock_at_300':'unlocked / stock at 300 W',
          'unlock_at_420':'unlocked / stock at 420 W',
          'stock_420_vs_300':'stock 420 W / stock 300 W',
          'unlock_420_vs_300':'unlocked 420 W / unlocked 300 W'},
        'models':15,'cases':45,'repeats_per_case_per_phase':3,
        'all_model_and_binary_hashes_match':True,
        'high_variability_gt_5pct':noisy,'aggregates':means},indent=2)+'\n')
    fmt=lambda x:f'{x:+.2f}%'
    md=['# RTX 3090 four-way power and firmware comparison\n\n',
        'Same RTX 3090, identical CUDA llama-bench binary, 15 identically hashed GGUF models, three timing samples per case. ',
        'This is one sequential comparison, not a randomized repeated study. A power limit is a ceiling, not measured power consumption.\n\n',
        '## Geometric-mean throughput changes across 15 models\n\n',
        '| Workload | Unlock effect 300 W | Unlock effect 420 W | Stock: 420 vs 300 W | Unlocked: 420 vs 300 W |\n',
        '|---|---:|---:|---:|---:|\n']
    for label,heading in [('pp512','512-token prefill'),('pp2048','2048-token prefill'),('tg128','128-token generation')]:
        md.append('| '+heading+' | '+' | '.join(fmt(means[k][label]['geomean_gain_pct']) for k in COMPARISONS)+' |\n')
    md += ['\nPositive values mean the second condition is faster. The model-weighted geometric mean weights each model equally. ',
           'High variability (>5% within-phase coefficient of variation) appears in '+str(len(noisy))+' of 45 cases across one or more conditions.\n\n',
           '## Per-model throughput and unlock effect at 420 W\n\n',
           '| Model | Workload | Stock 300 tok/s | Unlock 300 tok/s | Stock 420 tok/s | Unlock 420 tok/s | 420 W unlock gain | Variability |\n',
           '|---|---|---:|---:|---:|---:|---:|---|\n']
    for r in rows:
        md.append('| '+r['model']+' | '+r['test']+' | '+
          ' | '.join(f'{r[p+"_tok_s"]:.2f}' for p in SETS)+' | '+
          fmt(r['unlock_at_420_gain_pct'])+' | '+('>5% CV' if r['high_variability'] else '')+' |\n')
    md += ['\n## Scope and limitations\n\n',
           '- These are quantized CUDA GGUF inference throughput tests, not full-precision GEMM, training throughput, answer accuracy, or hybrid multi-GPU Maya inference.\n',
           '- GPU offload, CUDA backend, model SHA-256, binary SHA-256, execution flags and GPU power cap were validated for each paired condition.\n',
           '- Three samples cannot establish reliable confidence intervals for small differences, especially pp512.\n',
           '- The run order was unlocked 300 W, stock 300 W, stock 420 W, unlocked 420 W. Environmental drift and benchmark order are confounders.\n',
           '- See stock-420.jsonl and unlocked-420.jsonl and the previous 300 W files for all individual timing samples. Thermal summaries distinguish sampled power draw from configured caps.\n']
    (out/'power-matrix.md').write_text(''.join(md))
    print('VALIDATED',len(rows),'CASES',len(roster),'MODELS; HIGH_VARIABILITY',len(noisy),flush=True)
    for cmp,results in means.items():
        print(cmp,{k:round(x['geomean_gain_pct'],3) for k,x in results.items()},flush=True)
if __name__=='__main__':main()
