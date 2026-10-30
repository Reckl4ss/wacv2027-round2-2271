"""Check every distributed file against MANIFEST.sha256 (stdlib only)."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def main():
    failures=[];count=0
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        expected,name=line.split('  ',1)
        target=(ROOT/name).resolve()
        if ROOT not in target.parents or not target.is_file():
            failures.append({'file':name,'reason':'missing/unsafe'})
        elif hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            failures.append({'file':name,'reason':'hash'})
        count+=1
    report={'status':'FAIL' if failures else 'PASS','checked':count,'failures':failures}
    print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
