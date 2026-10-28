#!/usr/bin/env python3
"""Render a public benchmark summary; input contains only sanitized model metadata."""
import argparse,json,pathlib
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--input',required=True)
 ap.add_argument('--output',required=True)
 ap.add_argument('--title',default='Model benchmark results')
 a=ap.parse_args()
 data=[json.loads(line) for line in pathlib.Path(a.input).read_text().splitlines() if line.strip()]
 data.sort(key=lambda x:(x.get('tests',[{}])[0].get('model_n_params') or 0,x['id']))
 lines=['# '+a.title+'\n\n',
        'Measured with the same pinned CUDA benchmark; tokens/sec are **not** a model-quality score.\n\n',
        '| Model | Quant | Parameters | pp512 tok/s | pp2048 tok/s | tg128 tok/s | Status |\n',
        '|---|---|---:|---:|---:|---:|---|\n']
 for row in data:
    tests={(z['n_prompt'],z['n_gen']):z for z in row.get('tests',[])}
    def v(key):
        x=tests.get(key)
        return f"{x['avg_ts']:.2f}" if x else '—'
    t=row.get('tests',[])
    par=round((t[0].get('model_n_params') or 0)/1e9,2) if t else None
    lines.append(f"| {row['id']} | {row.get('quantization') or '—'} | {par if par is not None else '—'} B | "
                 f"{v((512,0))} | {v((2048,0))} | {v((0,128))} | {row['status']} |\n")
 lines.append('\nSee the corresponding JSONL file for per-run samples, standard deviations, SHA256 hashes, and failure details.\n')
 pathlib.Path(a.output).write_text(''.join(lines))
 print('RENDERED',len(data),'MODELS',a.output)
if __name__=='__main__':main()
