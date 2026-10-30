"""Build both documents in isolation and refresh their cross-reference labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tectonic',default='tectonic')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/paper-build-renamed')
    args=parser.parse_args()
    executable=shutil.which(args.tectonic)
    if executable is None:
        parser.error('Tectonic was not found. Install Tectonic or supply --tectonic /path/to/tectonic.')
    output=args.output.resolve()
    if output==PAPER or PAPER in output.parents or output in PAPER.parents:
        parser.error('Choose an output directory outside the paper source and its parent directories.')
    stage=output/'source'
    shutil.copytree(PAPER,stage,dirs_exist_ok=True,ignore=shutil.ignore_patterns('build','font-cache'))
    (stage/'crossrefs').mkdir(exist_ok=True)
    for name in ['main','appendix']:(stage/'build'/name).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,FONTCONFIG_FILE=str(stage/'fonts.conf'))
    steps=[]
    stable=False
    for cycle in range(1,5):
        before={name:digest(stage/'crossrefs'/f'{name}.aux') for name in ['main','appendix']}
        for name in ['main','appendix']:
            command=[executable,f'{name}.tex','--outdir',f'build/{name}','--keep-intermediates','--keep-logs']
            logfile=output/f'cycle{cycle}_{name}.log'
            print(f'Building {name}.tex, cross-reference cycle {cycle}',flush=True)
            with logfile.open('w',encoding='utf-8') as log:
                process=subprocess.run(command,cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT)
            steps.append({'cycle':cycle,'document':name,'command':['tectonic',*command[1:]],'exit_code':process.returncode,'log':logfile.name})
            if process.returncode:
                raise RuntimeError(f'{name}.tex failed; inspect {logfile.name}')
            shutil.copy2(stage/'build'/name/f'{name}.aux',stage/'crossrefs'/f'{name}.aux')
        after={name:digest(stage/'crossrefs'/f'{name}.aux') for name in ['main','appendix']}
        if before==after:
            stable=True
            break
    if not stable:
        raise RuntimeError('Cross-reference labels did not stabilize within four cycles')
    issues=[]
    for name in ['main','appendix']:
        log=(stage/'build'/name/f'{name}.log').read_text(encoding='utf-8',errors='replace')
        patterns=[r'LaTeX Warning:.*undefined',r'There were undefined (?:references|citations)',
                  r'There were multiply-defined labels',r'Overfull \\[hv]box',r'(?m)^! ']
        for pattern in patterns:
            issues.extend({'document':name,'message':match.group(0)} for match in re.finditer(pattern,log,re.I))
        shutil.copy2(stage/'build'/name/f'{name}.pdf',output/f'{name}.pdf')
    report={'status':'FAIL' if issues else 'PASS','cross_reference_cycles':cycle,'cross_reference_snapshots_stable':stable,
            'steps':steps,'issues':issues,'outputs':{name:{'file':f'{name}.pdf','sha256':digest(output/f'{name}.pdf')} for name in ['main','appendix']},
            'scope':'Compilation and final log checks; visual/text comparison with reference PDFs is separate.'}
    (output/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
