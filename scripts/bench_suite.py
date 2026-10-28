#!/usr/bin/env python3
"""Reproducible single-3090 llama-bench harness; does NOT change GPU power/services.
Use in a separately supervised thermal and service-restoration controller.
Only sanitized numerical results may be published.
"""
import argparse,datetime,hashlib,json,os,pathlib,subprocess,sys,time

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def check_gpu(required_limit):
    q=subprocess.run(['nvidia-smi','--query-gpu=index,name,pci.bus_id,power.limit',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=12).stdout
    lines=q.strip().splitlines()
    if not lines or '3090' not in lines[0] or '01:00.0' not in lines[0]:
        raise RuntimeError('GPU0 must be RTX3090 at PCI 01:00.0')
    cap=float(lines[0].rsplit(',',1)[-1].strip())
    if abs(cap-required_limit)>1:raise RuntimeError(f'GPU0 power cap {cap} not {required_limit}')
    return q

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--binary',required=True)
    ap.add_argument('--phase',choices=['unlocked','stock'],required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--only',action='append',default=[])
    ap.add_argument('--power',type=int,default=300)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--skip-hash',action='store_true',help='use only when hashes were checked separately')
    args=ap.parse_args()
    out=pathlib.Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(pathlib.Path(args.manifest).read_text())
    models=manifest['models']
    if args.only:models=[x for x in models if x['id'] in args.only]
    if not models:raise SystemExit('No models selected')
    gpu=check_gpu(args.power)
    binary=pathlib.Path(args.binary).resolve()
    if not binary.is_file():raise RuntimeError('Benchmark binary missing')
    build_sha=sha256(binary)
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']='0'
    env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    env['GGML_CUDA_NO_PEER_COPY']='1'
    inventory={'phase':args.phase,'gpu_inventory':gpu,'binary_sha256':build_sha,
        'power_w':args.power,'binary_name':binary.name,'repeats':args.repeats,
        'source_git_commit':'6a1a922','cuda_toolkit':'12.8',
        'parameters':{'n_prompt':[512,2048],'n_gen':128,'n_gpu_layers':99,
            'threads':16,'batch':512,'ubatch':512,'device':'CUDA0',
            'flash_attn':'on','KV':'f16','split_mode':'none'}}
    (out/'environment.json').write_text(json.dumps(inventory,indent=2)+'\n')
    for model in models:
        ident=model['id'];path=pathlib.Path(model['path'])
        clean=''.join(c if c.isalnum() or c in '-_.' else '_' for c in ident)
        done=out/(clean+'.status.json')
        if done.exists():
            print('CHECKPOINT_SKIP',ident,flush=True);continue
        started=time.time()
        result={'phase':args.phase,'id':ident,'source':model.get('source'),
           'source_revision':model.get('source_revision'),'quantization':model.get('quantization'),
           'model_sha256':model.get('sha256'),'binary_sha256':build_sha,
           'power_w':args.power,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'status':'failed','tests':[]}
        try:
            check_gpu(args.power)
            if not path.is_file():raise FileNotFoundError('Missing model weight file')
            if path.stat().st_size<10_000_000:raise ValueError('Model too small to be valid GGUF')
            result['model_size_bytes']=path.stat().st_size
            if not args.skip_hash:
                digest=sha256(path)
                result['model_sha256']=digest
                if model.get('sha256') and digest.lower()!=model['sha256'].lower():
                    raise ValueError('Model SHA256 mismatch')
            command=[str(binary),'-m',str(path),'-dev','CUDA0','-sm','none',
                     '-mg','0','-ngl','99','-fa','on','-t','16',
                     '-b','512','-ub','512','-ctk','f16','-ctv','f16',
                     '-p','512,2048','-n','128','-r',str(args.repeats),'-o','jsonl']
            result['parameters']=inventory['parameters']
            print('BEGIN',args.phase,ident,'bytes',result['model_size_bytes'],flush=True)
            with (out/(clean+'.stdout.private.txt')).open('w') as stdout:
                with (out/(clean+'.stderr.private.txt')).open('w') as stderr:
                    proc=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,
                                        timeout=args.timeout,check=False)
            if proc.returncode:raise RuntimeError('llama-bench returned '+str(proc.returncode))
            for line in (out/(clean+'.stdout.private.txt')).read_text().splitlines():
                try:raw=json.loads(line)
                except json.JSONDecodeError:continue
                if not isinstance(raw,dict) or 'avg_ts' not in raw:continue
                record={k:raw.get(k) for k in ('model_type','model_n_params','model_size',
                    'n_prompt','n_gen','n_gpu_layers','n_threads','n_batch','n_ubatch',
                    'backends','gpu_info','avg_ts','stddev_ts','samples_ts','avg_ns','stddev_ns')}
                result['tests'].append(record)
            if len(result['tests'])!=3:
                raise RuntimeError('Expected pp512/pp2048/tg128 test rows, got '+str(len(result['tests'])))
            expected={(512,0),(2048,0),(0,128)}
            actual={(int(x['n_prompt']),int(x['n_gen'])) for x in result['tests']}
            if actual!=expected:raise RuntimeError('Unexpected benchmark cases '+str(actual))
            if any('CUDA' not in str(x.get('backends','')) for x in result['tests']):
                raise RuntimeError('CUDA backend was not selected')
            result['status']='ok'
            print('PASS',args.phase,ident,[(x['n_prompt'],x['n_gen'],round(x['avg_ts'],2)) for x in result['tests']],flush=True)
        except Exception as e:
            result['error']=str(e)[:190]
            print('FAIL',args.phase,ident,type(e).__name__,str(e)[:160],flush=True)
        finally:
            result['elapsed_s']=round(time.time()-started,2)
            done.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
            with (out/'results.jsonl').open('a') as f:f.write(json.dumps(result,sort_keys=True)+'\n')
    print('PHASE_FINISHED',args.phase,flush=True)
if __name__=='__main__':main()
