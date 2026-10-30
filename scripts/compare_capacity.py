"""Compare all nested capacity results, including the standard-proxy population."""
import pickle
import numpy as np

def compare(expected, actual, tolerance=1e-10):
    with expected.open('rb') as f:
        left = pickle.load(f)
    with actual.open('rb') as f:
        right = pickle.load(f)
    errors, numeric, maximum = [], 0, 0.0
    def visit(a, b, path):
        nonlocal numeric, maximum
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                errors.append({'field':path, 'reason':'keys differ'})
                return
            for key in a:
                visit(a[key], b[key], path+'/'+str(key))
        elif isinstance(a, (np.ndarray, list, tuple)):
            if not isinstance(b, (np.ndarray, list, tuple)) or len(a) != len(b):
                errors.append({'field':path, 'reason':'shape differs'})
                return
            for i,(x,y) in enumerate(zip(a,b)):
                visit(x,y,path+'/'+str(i))
        elif isinstance(a, (float,int,np.number)) and not isinstance(a,bool):
            numeric += 1
            if not np.isclose(a,b,rtol=0,atol=tolerance,equal_nan=True):
                errors.append({'field':path,'expected':str(a),'actual':str(b)})
            elif np.isfinite(a) and np.isfinite(b):
                maximum=max(maximum,abs(float(a)-float(b)))
        elif a != b:
            errors.append({'field':path,'expected':str(a),'actual':str(b)})
    visit(left,right,'')
    return {'status':'FAIL' if errors else 'PASS','numeric_values':numeric,
            'absolute_tolerance':tolerance,'max_abs_error':maximum,'differences':errors}
