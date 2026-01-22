

import os
import sys
import json
import tempfile
import subprocess
import pytest
from pathlib import Path

# Standalone import logic: allow running as script or with pytest
try:
    from lib.julia_theory import call_julia_theory
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lib.julia_theory import call_julia_theory

def test_call_julia_theory_mock(monkeypatch):
    # Patch subprocess.run to simulate Julia output
    def fake_run(cmd, check, capture_output):
        # Find the --to argument to get the output path
        out_idx = cmd.index('--to') + 1
        out_path = cmd[out_idx]
        # Write a fake JSON result
        with open(out_path, 'w') as f:
            json.dump({'result': 42, 'args': dict(zip(cmd[2::2], cmd[3::2]))}, f)
        class Result:
            pass
        return Result()
    monkeypatch.setattr(subprocess, 'run', fake_run)
    # Call with dummy script and args
    script = 'dummy_script.jl'
    args = {'foo': 1, 'bar': 'baz'}
    result = call_julia_theory(script, args, quiet=True)
    assert result['result'] == 42
    assert result['args']['--foo'] == '1'
    assert result['args']['--bar'] == 'baz'

def test_call_julia_theory_real_script(tmp_path):
    # This test assumes a minimal Julia script exists and Julia is installed.
    # It will be skipped if not present.
    script_path = Path(__file__).parent.parent / 'julia_lib' / 'echo_args.jl'
    if not script_path.exists():
        pytest.skip('No echo_args.jl script available')
    args = {'x': 5, 'y': 'test'}
    result = call_julia_theory('echo_args.jl', args, quiet=True)
    assert 'x' in result and 'y' in result


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
