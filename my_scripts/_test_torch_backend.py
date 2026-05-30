"""Quick smoke-test for NeuralNetworkTorch and _make_nn."""
import sys, ast, inspect
import numpy as np

# ── 1. Syntax check ────────────────────────────────────────────────────────
with open('ClayPMFNeural.py') as f:
    src = f.read()
ast.parse(src)
print('AST OK')

# ── 2. Import ──────────────────────────────────────────────────────────────
from ClayPMFNeural import NeuralNetwork, NeuralNetworkTorch, _make_nn, ClayPMFNeuralEnsemble

# ── 3. numpy backend ───────────────────────────────────────────────────────
nn_np = _make_nn(3, (32, 16), backend='numpy', lr=1e-3, l2=1e-4, seed=0)
X = np.random.randn(200, 3).astype(np.float64)
y = np.random.randn(200).astype(np.float64)
losses_np = nn_np.train(X, y, epochs=5, batch_size=32, verbose=False)
out_np = nn_np.predict(X[:5])
print(f'numpy  losses[-1]={losses_np[-1]:.4e}  predict shape={out_np.shape}  dtype={out_np.dtype}')

# ── 4. torch backend ───────────────────────────────────────────────────────
try:
    nn_t = _make_nn(3, (32, 16), backend='torch', lr=1e-3, l2=1e-4, seed=0)
    losses_t = nn_t.train(X, y, epochs=5, batch_size=32, verbose=False)
    out_t = nn_t.predict(X[:5])
    print(f'torch  losses[-1]={losses_t[-1]:.4e}  predict shape={out_t.shape}  dtype={out_t.dtype}')

    # weight transfer round-trip
    W, b = nn_t.W, nn_t.b
    nn_t2 = _make_nn(3, (32, 16), backend='torch', seed=99)
    nn_t2.W = W
    nn_t2.b = b
    diff = float(np.max(np.abs(nn_t2.predict(X) - nn_t.predict(X))))
    print(f'weight transfer max diff: {diff:.2e}  (expect ~0)')

    # early stopping with val_split
    nn_es = _make_nn(3, (16,), backend='torch', lr=1e-3, l2=0, seed=7)
    losses_es = nn_es.train(X, y, epochs=100, batch_size=64,
                            val_split=0.1, patience=5, verbose=False)
    print(f'early-stop: stopped at epoch {len(losses_es)} / 100')
except ImportError as e:
    print(f'torch not installed, skipping torch tests: {e}')

# ── 5. Signature check (backend= kwarg present in all fit_* methods) ───────
for cls_name, cls in [('ClayPMFNeuralEnsemble', ClayPMFNeuralEnsemble)]:
    for mname in ('fit_smooth', 'fit_reweighted', 'fit_smooth_per_replicate'):
        sig = inspect.signature(getattr(cls, mname))
        assert 'backend' in sig.parameters, f'MISSING backend in {cls_name}.{mname}'
        default = sig.parameters['backend'].default
        print(f'  {cls_name}.{mname}: backend default={default!r}  OK')

# Also check ClayPMFNeural
from ClayPMFNeural import ClayPMFNeural
for mname in ('fit_smooth', 'fit_reweighted'):
    sig = inspect.signature(getattr(ClayPMFNeural, mname))
    assert 'backend' in sig.parameters, f'MISSING backend in ClayPMFNeural.{mname}'
    default = sig.parameters['backend'].default
    print(f'  ClayPMFNeural.{mname}: backend default={default!r}  OK')

print('\nAll checks passed.')
