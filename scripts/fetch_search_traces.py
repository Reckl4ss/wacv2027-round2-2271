"""Download, hash-check and optionally extract the fixed public trace artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

URL='https://github.com/Reckl4ss/wacv2027-round2-2271/releases/download/e4-traces-v1/e4-traces-v1.tar'
SHA256='8325148fe975c9eaf24fa6fca912c8ef5a2a8df96e9d3e7176742e2417ce344a'
SIZE=1744138240
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/search_traces')
    parser.add_argument('--extract',action='store_true')
    args=parser.parse_args()
    directory=args.directory.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'e4-traces-v1.tar'
    if not archive.exists():
        temp=directory/'e4-traces-v1.tar.partial'
        req=urllib.request.Request(URL,headers={'User-Agent':'nas-proxy-transport-reproduction'})
        count=0
        with urllib.request.urlopen(req,timeout=60) as response,temp.open('wb') as out:
            for chunk in iter(lambda:response.read(8*1024*1024),b''):
                out.write(chunk)
                count+=len(chunk)
                if count//(256*1024*1024)!=(count-len(chunk))//(256*1024*1024):
                    print(f'Downloaded {count}/{SIZE} bytes',flush=True)
        if temp.stat().st_size!=SIZE or digest(temp)!=SHA256:
            raise RuntimeError('Downloaded artifact does not match the pinned size/hash')
        temp.replace(archive)
    if archive.stat().st_size!=SIZE or digest(archive)!=SHA256:
        raise RuntimeError('Artifact does not match the pinned size/hash')
    report={'status':'PASS','archive_bytes':SIZE,'archive_sha256':SHA256,'url':URL}
    if args.extract:
        payload=directory/'payload'
        payload.mkdir(exist_ok=True)
        with tarfile.open(archive,'r:') as tar:
            members=tar.getmembers()
            for item in members:
                target=(payload/item.name).resolve()
                if payload not in target.parents or not (item.isfile() or item.isdir()):
                    raise RuntimeError(f'Unexpected archive member: {item.name}')
            tar.extractall(payload,filter='data')
        manifest=ROOT/'data/provenance/search_transfer/output_checksums_nested.sha256'
        checked=0
        for line in manifest.read_text().splitlines():
            if not line.strip():continue
            expected,name=line.split(maxsplit=1)
            path=(payload/name).resolve()
            if payload not in path.parents or not path.is_file() or digest(path)!=expected:
                raise RuntimeError('Manifest verification failed: '+name)
            checked+=1
        file_count=sum(item.isfile() for item in members)
        if checked!=179 or file_count!=179:
            raise RuntimeError(f'Unexpected manifest/archive count: {checked}/{file_count}')
        report.update(manifest_entries=checked,verified=checked,extracted_files=file_count)
    (directory/'download_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
