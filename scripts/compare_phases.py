#!/usr/bin/env python3
"""Build public comparison CSV/Markdown from sanitized stock and unlock JSONL."""
import argparse,csv,json,pathlib,statistics
def load(path):
    models={}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():continue
        entry=json.loads(line);models[entry['id']]=entry
    return models
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stock',required=True);ap.add_argument('--unlocked',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    s=load(a.stock);u=load(a.unlocked)
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ident in sorted(set(s)|set(u)):
        x=s.get(ident);y=u.get(ident)
        if not x or not y or x['status']!='ok' or y['status']!='ok':
            rows.append({'model':ident,'test':'unmatched','status':f"stock={x['status'] if x else 'missing'},unlock={y['status'] if y else 'missing'}"})
            continue
        if x['model_sha256']!=y['model_sha256'] or x['binary_sha256']!=y['binary_sha256']:
            rows.append({'model':ident,'test':'invalid','status':'model or binary checksum mismatch'})
            continue
        if x['power_w']!=y['power_w']:
            rows.append({'model':ident,'test':'invalid','status':'power limit mismatch'})
            continue
        def by_test(d):return {(z['n_prompt'],z['n_gen']):z for z in d['tests']}
        sx=by_test(x);uy=by_test(y)
        for key in [(512,0),(2048,0),(0,128)]:
            if key not in sx or key not in uy:
                rows.append({'model':ident,'test':str(key),'status':'test missing'});continue
            old=float(sx[key]['avg_ts']);new=float(uy[key]['avg_ts'])
            cv_stock=float(sx[key].get('stddev_ts',0))/old*100 if old else 0
            cv_unlocked=float(uy[key].get('stddev_ts',0))/new*100 if new else 0
            rows.append({'model':ident,'test':'pp512' if key[0]==512 else 'pp2048' if key[0]==2048 else 'tg128',
                'status':'matched','stock_tokens_s':round(old,3),
                'unlocked_tokens_s':round(new,3),'gain_pct':round(100*(new/old-1),2),
                'cv_stock_pct':round(cv_stock,2),'cv_unlocked_pct':round(cv_unlocked,2),
                'sha256':x['model_sha256'],'quantization':x.get('quantization','')})
    keys=['model','test','status','stock_tokens_s','unlocked_tokens_s',
          'gain_pct','cv_stock_pct','cv_unlocked_pct','sha256','quantization']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    header='| Model | Test | Stock tok/s | Unlocked tok/s | Gain | Status |\n|---|---|---:|---:|---:|---|\n'
    lines=[header]
    for row in rows:
        lines.append('| '+str(row['model'])+' | '+str(row['test'])+' | '+
          str(row.get('stock_tokens_s','—'))+' | '+str(row.get('unlocked_tokens_s','—'))+
          ' | '+(str(row['gain_pct'])+'%' if 'gain_pct' in row else '—')+
          ' | '+str(row['status'])+' |\n')
    (out/'comparison.md').write_text(''.join(lines))
    print('ROWS',len(rows),'PAIRED',sum(x['status']=='matched' for x in rows))
if __name__=='__main__':main()
