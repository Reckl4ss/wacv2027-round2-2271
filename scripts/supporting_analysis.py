"""Run supporting cached analyses or regenerate selected figures in isolation."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
from verify_bundle import verify

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['early-windows','width-partials','ntk-analysis','figures'])
    args=parser.parse_args()
    verify()
    output=ROOT/'runs'/args.task
    stage=output/'work'
    for name in ['code','results','figures']:(stage/name).mkdir(parents=True,exist_ok=True)
    for file in (ROOT/'data/frozen').glob('*.pkl'):shutil.copy2(file,stage/'results'/file.name)
    for file in (ROOT/'src/analysis').glob('*.py'):shutil.copy2(file,stage/'code'/file.name)
    if args.task=='figures':
        for file in (ROOT/'results/reference/core').glob('*_out.txt'):shutil.copy2(file,stage/'results'/file.name)
        for file in (ROOT/'src/figures').glob('*.py'):shutil.copy2(file,stage/'code'/file.name)
        commands=[['portability.py'],['pipeline.py']]
    else:
        commands=[{'early-windows':['t_window.py'],'width-partials':['width_partial.py'],'ntk-analysis':['ntk_experiment.py','--phase','analyze']}[args.task]]
    env=dict(os.environ,PYTHONUTF8='1',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    reports=[]
    for command in commands:
        with (output/(command[0]+'.stdout.log')).open('w',encoding='utf-8') as stdout,(output/(command[0]+'.stderr.log')).open('w',encoding='utf-8') as stderr:
            result=subprocess.run([sys.executable,str(stage/'code'/command[0]),*command[1:]],cwd=stage,env=env,stdout=stdout,stderr=stderr)
        reports.append({'script':command[0],'arguments':command[1:],'exit_code':result.returncode})
    report={'task':args.task,'status':'PASS' if all(x['exit_code']==0 for x in reports) else 'FAIL','commands':reports,'scope':'Execution check; numerical/visual comparison is a separate validation step.'}
    (output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
