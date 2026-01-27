import subprocess
import json
import tempfile
from pathlib import Path
def compute_self_consistent_kappa_eff(eigenvalues, P, quiet=True):
    """
    Call the Julia self_consistent_kappa_solver.jl script to compute kappa_eff.
    eigenvalues: list or np.ndarray of eigenvalues
    P: int, number of samples
    Returns: float kappa_eff (or None if not found)
    """

    julia_script = Path(__file__).parent.parent / 'julia_lib' / 'self_consistent_kappa_solver.jl'
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp:
        eig_json_path = Path(tmp.name)
        json.dump({"eigenvalues": list(np.asarray(eigenvalues).flatten()), "kappa_bare": 1.0}, tmp)
    cmd = [
        'julia', str(julia_script),
        str(eig_json_path), str(P)
    ]
    if quiet:
        cmd.append('--quiet')
    import subprocess, re
    try:
        sc_out = subprocess.check_output(cmd, text=True)
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            return float(match.group(1))
        else:
            print("Warning: could not parse kappa_eff from output.")
            return None
    finally:
        try:
            eig_json_path.unlink()
        except Exception:
            pass
def call_julia_theory(script_name, args_dict, quiet=True):

    """
    Call a Julia theory script with arguments from args_dict.
    script_name: e.g. 'compute_fcn2_erf_cubic_eigs.jl'
    args_dict: dict of argument name (str) to value (str, int, float)
    Returns: parsed JSON result from Julia script
    """
    julia_script = Path(__file__).parent.parent / 'julia_lib' / script_name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            'julia', str(julia_script)
        ]
        for k, v in args_dict.items():
            cmd.append(f'--{k}')
            cmd.append(str(v))
        cmd += ['--to', str(tmp_path)]
        if quiet:
            cmd.append('--quiet')
        print("Running Julia command:", ' '.join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        with open(tmp_path, 'r') as f:
            result = json.load(f)
            print( f"Julia output from {script_name}: {result}" )
        return result
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

def compute_fcn2_erf_cubic_eigs(d, n1, P, chi, kappa, epsilon=0.0, quiet=True):
    args = dict(d=str(d), n1=str(n1), P=str(P), chi=str(chi), kappa=str(kappa), epsilon=str(epsilon))
    return call_julia_theory('compute_fcn2_erf_cubic_eigs.jl', args, quiet=quiet)

def compute_fcn2_erf_eigs(d, n1, P, chi, kappa, epsilon=0.0, quiet=True):
    args = dict(d=str(d), n1=str(n1), P=str(P), chi=str(chi), kappa=str(kappa), epsilon=str(epsilon))
    return call_julia_theory('compute_fcn2_erf_eigs.jl', args, quiet=quiet)
