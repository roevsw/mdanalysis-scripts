#!/usr/bin/env python3
"""
ClayPMFNeural.py
================
Neural-network PMF representations for ClayPMF3D objects.

Two approaches
--------------
A  (grid smoother)
    Train a NN directly on the existing WHAM PMF grid (pmf3d.pmf_3d).
    One training pass; gives a smooth, differentiable W(r, θ, n_cat).

B  (frame-reweighted)
    Compute per-frame WHAM weights from the raw trajectory arrays
    (z_data, theta_data, ncat_data) using the converged WHAM free energies
    pmf3d.f.  Build a fine-resolution unbiased P(r, θ, n) via a weighted
    histogram, then train a NN on that finer PMF.  Avoids coarse-binning
    artefacts of the original WHAM grid.

Usage
-----
    from ClayPMFNeural import ClayPMFNeural

    # Approach A ─ smooth the existing grid
    nn = ClayPMFNeural(pmf3d)
    nn.fit_smooth(epochs=1000)

    # Approach B ─ per-frame reweighting on finer grid
    nn.fit_reweighted(epochs=1000, n_r_bins=80, n_theta_bins=36)

    # Evaluate at arbitrary (r, theta, n_cat) arrays
    W_a = nn.predict(r_arr, theta_arr, n_arr)       # kJ/mol, Approach A
    W_b = nn.predict_b(r_arr, theta_arr, n_arr)     # kJ/mol, Approach B

    # Diagnostics
    nn.plot_losses()
    nn.plot_comparison_slice(n_cat_val=1)
    nn.plot_1d_marginals()

    # Save / reload
    nn.save('nn_pmf.npz')
    nn2 = ClayPMFNeural.load('nn_pmf.npz', pmf3d)
"""

import random
import warnings
import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import torch as _torch_probe
    torch_available = True
    del _torch_probe
except ImportError:
    torch_available = False

__all__ = ['NeuralNetwork', 'NeuralNetworkTorch', 'ClayPMFNeural', 'ClayPMFNeuralEnsemble',
           'set_global_seed', 'torch_available']

_CLAYPMFNEURAL_VERSION = '1.1.0'


def set_global_seed(seed):
    """Set random seeds for NumPy, Python's random module, and PyTorch (if available).

    Call this once before constructing or training any model to get fully
    reproducible results across runs.

    Parameters
    ----------
    seed : int
    """
    np.random.seed(seed)
    random.seed(seed)
    if torch_available:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _git_hash():
    """Return the current HEAD git hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'

# ---------------------------------------------------------------------------
# Numerically stable helpers
# ---------------------------------------------------------------------------

def _logsumexp(a, axis=None, keepdims=False):
    """Numerically stable log-sum-exp; NaN entries are ignored (treated as -inf)."""
    a_max = np.nanmax(a, axis=axis, keepdims=True)
    out = np.log(np.nansum(np.exp(a - a_max), axis=axis, keepdims=True))
    out += a_max
    if not keepdims:
        out = out.squeeze(axis=axis)
    return out


# ---------------------------------------------------------------------------
# Marginal target correction helper
# ---------------------------------------------------------------------------

def _apply_marginal_correction(X, y, W3d, r_centers, kT, beta,
                                w1d_target, w1d_r=None, lambda_marginal=1.0,
                                max_correction_kJmol=15.0):
    """
    Shift 3-D training targets so their r-marginal matches *w1d_target*.

    For each r-bin, adds ``lambda_marginal * (w1d_target(r) - W_1d_ref(r))``
    to every training point in that bin, where ``W_1d_ref(r)`` is the
    r-marginal of the reference 3-D PMF *W3d* computed via log-sum-exp.

    Because adding a constant δ(r) to every point in an r-slice shifts the
    log-sum-exp marginal by exactly δ(r):

        -kT log Σ_{θ,n} exp(−β (y + δ)) = W_1d_ref(r) + δ

    the corrected targets satisfy W_1d(r) = w1d_target(r) exactly when
    the NN perfectly reproduces them (λ = 1) and approximately for λ < 1.

    Parameters
    ----------
    X               : (N, 3)          training inputs  [r, θ, n_cat]
    y               : (N,)            training targets  [kJ/mol]  (copied)
    W3d             : (n_r, n_θ, n_cat) reference 3-D PMF  [kJ/mol]
    r_centers       : (n_r,)          r bin-centres  [nm]
    kT, beta        : float           thermal energy (kJ/mol) and 1/kT
    w1d_target      : (M,)            target 1-D PMF W(r)  [kJ/mol]
    w1d_r           : (M,) or None    r-coordinates for *w1d_target*  [nm];
                                      if None, assumed aligned to *r_centers*
    lambda_marginal : float           blending factor (default 1.0 = full
                                      correction; 0.0 = no correction)
    max_correction_kJmol : float
        Hard cap on |Δ(r)| before applying to training targets (default
        15 kJ/mol ≈ 6 kT at 298 K).  Guards against artifact deep wells in
        W3d or mismatched bulk-reference levels producing very large shifts
        that push y_corr outside the sampled W-range and spike MSE gradients.

    Returns
    -------
    y_corr : (N,)  corrected training targets  [kJ/mol]
    """
    # ── Guard 1: clamp extreme W3d values before Boltzmann marginal sum ──
    # Very negative W3d entries (artifact deep wells, e.g. barely-sampled bins)
    # produce huge exp(−β·W) terms that dominate W_1d_ref, making delta blow up.
    # Treat values more than 10 kT below the per-r-slice minimum as NaN so
    # they are ignored by _logsumexp (which skips NaN).
    _W3d_safe = W3d.astype(float, copy=True)
    _W_floor  = np.nanmin(_W3d_safe) - 10.0 * kT   # 10 kT below global min
    _W3d_safe[_W3d_safe < _W_floor] = np.nan

    # r-marginal of the reference 3-D PMF
    W_1d_ref = -kT * _logsumexp(-beta * _W3d_safe, axis=(1, 2))   # (n_r,)

    # Interpolate w1d_target onto the r_centers grid
    _r_src   = r_centers if w1d_r is None else np.asarray(w1d_r,      dtype=float)
    _W_src   = np.asarray(w1d_target, dtype=float)
    w1d_grid = np.interp(r_centers, _r_src, _W_src, left=np.nan, right=np.nan)

    # Per-bin correction: Δ(r) = w1d_target(r) − W_1d_ref(r)
    delta = w1d_grid - W_1d_ref   # (n_r,)  [kJ/mol]

    # ── Guard 2: clamp delta so y_corr stays in the sampled W-range ──────
    d_raw_max = float(np.nanmax(np.abs(delta)))
    if d_raw_max > max_correction_kJmol:
        print(f"  [marginal correction]  WARNING: |Δ|_max = {d_raw_max:.2f} kJ/mol "
              f"exceeds cap ({max_correction_kJmol:.1f} kJ/mol) — clamping.")
    delta = np.clip(delta, -max_correction_kJmol, max_correction_kJmol)

    # Locate each training point in the nearest r-bin
    r_idx = np.searchsorted(r_centers, X[:, 0], side='right') - 1
    r_idx = np.clip(r_idx, 0, len(r_centers) - 1)

    # Apply correction where Δ is finite
    delta_pts = delta[r_idx]
    valid     = np.isfinite(delta_pts)
    y_corr    = y.copy()
    y_corr[valid] += lambda_marginal * delta_pts[valid]

    n_v   = int(valid.sum())
    d_max = float(np.nanmax(np.abs(delta)))
    print(f"  [marginal correction]  {n_v:,}/{len(y):,} points adjusted, "
          f"|Δ|_max = {d_max:.3f} kJ/mol, λ = {lambda_marginal:.3f}")
    return y_corr


# ---------------------------------------------------------------------------
# NeuralNetwork — pure numpy, Adam optimizer, backprop from scratch
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Fully-connected feedforward NN with tanh hidden layers and linear output.

    Architecture
    ------------
    Input (d) → [hidden_sizes[0], tanh] → … → [hidden_sizes[-1], tanh] → Output (1, linear)

    Optimiser
    ---------
    Adam (β₁=0.9, β₂=0.999, ε=1e-8) with optional L2 weight decay.

    Parameters
    ----------
    input_dim    : int
    hidden_sizes : sequence of int
    lr           : float  Adam learning rate (default 1e-3)
    l2           : float  L2 regularisation coefficient (default 1e-4)
    seed         : int or None
    """

    def __init__(self, input_dim, hidden_sizes, lr=1e-3, l2=1e-4, seed=None,
                 verbose=True):
        if seed is not None:
            np.random.seed(seed)

        self.lr  = float(lr)
        self.l2  = float(l2)

        # Build weight / bias arrays (Xavier init)
        dims = [input_dim] + list(hidden_sizes) + [1]
        self.W = []
        self.b = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W.append(np.random.randn(fan_out, fan_in) * scale)
            self.b.append(np.zeros(fan_out))

        self.n_hidden = len(hidden_sizes)   # number of hidden layers

        # Adam moment estimates
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(bi) for bi in self.b]
        self.vb = [np.zeros_like(bi) for bi in self.b]
        self.t  = 0   # global step counter

    # --- forward pass -------------------------------------------------------

    def _forward(self, X):
        """
        X : (N, input_dim)
        Returns (out, zs, acts)
            out  : (N,)        linear output flattened
            zs   : list of (N, fan_out)  pre-activations per layer
            acts : list of (N, *)         input + post-activations per layer
        """
        acts = [X]
        zs   = []
        h    = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b          # (N, fan_out)
            zs.append(z)
            h = np.tanh(z) if i < self.n_hidden else z   # linear output
            acts.append(h)
        return acts[-1].ravel(), zs, acts

    def predict(self, X):
        """X : (N, input_dim) or (input_dim,). Returns W values (N,)."""
        out, _, _ = self._forward(np.atleast_2d(X))
        return out

    # --- training step ------------------------------------------------------

    def train_step(self, X, y, sample_weight=None):
        """
        One Adam gradient step.

        Parameters
        ----------
        X             : (N, d)
        y             : (N,)   target PMF values
        sample_weight : (N,) or None  per-sample multiplicative weights

        Returns
        -------
        loss : float   weighted MSE
        """
        N = X.shape[0]
        out, zs, acts = self._forward(X)

        err = out - y                                # (N,)
        sqrt_w = None
        if sample_weight is not None:
            sqrt_w = np.sqrt(sample_weight)          # compute once; √w·√w = w
            err    = err * sqrt_w                    # weighted residual: √w·(out−y)

        loss = float(np.mean(err ** 2))              # = (1/N)Σ w·(out−y)²

        # Seed backprop gradient: d_loss / d_out  (N, 1)
        # d/d_out[(1/N)Σ w·(out−y)²] = (2/N)·w·(out−y) = (2/N)·err·√w
        grad = (2.0 / N) * err.reshape(-1, 1)
        if sqrt_w is not None:
            grad = grad * sqrt_w.reshape(-1, 1)

        grads_W = []
        grads_b = []

        for i in range(len(self.W) - 1, -1, -1):
            # Apply activation derivative for hidden layers
            if i < self.n_hidden:
                grad = grad * (1.0 - acts[i + 1] ** 2)   # tanh'(z) = 1 − tanh²(z)

            gw = grad.T @ acts[i]          # (fan_out, fan_in)
            gb = np.sum(grad, axis=0)      # (fan_out,)

            if self.l2 > 0:
                gw += 2.0 * self.l2 * self.W[i]

            grads_W.insert(0, gw)
            grads_b.insert(0, gb)

            if i > 0:
                grad = grad @ self.W[i]    # propagate to previous layer

        # Adam update
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t

        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * grads_W[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * grads_W[i] ** 2
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * grads_b[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * grads_b[i] ** 2

            self.W[i] -= self.lr * (self.mW[i] / bc1) / (np.sqrt(self.vW[i] / bc2) + eps)
            self.b[i] -= self.lr * (self.mb[i] / bc1) / (np.sqrt(self.vb[i] / bc2) + eps)

        return loss

    # --- training loop -------------------------------------------------------

    def _val_loss(self, X_val, y_val):
        """Unweighted MSE on the validation set (no gradient, no side-effects)."""
        out = self._forward(X_val)[0].ravel()
        return float(np.mean((out - y_val) ** 2))

    def _snapshot_weights(self):
        """Return a deep copy of all trainable parameters + Adam state."""
        return (
            [w.copy() for w in self.W],
            [b.copy() for b in self.b],
            [m.copy() for m in self.mW],
            [v.copy() for v in self.vW],
            [m.copy() for m in self.mb],
            [v.copy() for v in self.vb],
            self.t,
        )

    def _restore_weights(self, snapshot):
        W, b, mW, vW, mb, vb, t = snapshot
        self.W  = [w.copy() for w in W]
        self.b  = [b_.copy() for b_ in b]
        self.mW = [m.copy() for m in mW]
        self.vW = [v.copy() for v in vW]
        self.mb = [m.copy() for m in mb]
        self.vb = [v.copy() for v in vb]
        self.t  = t

    def train(self, X, y, sample_weight=None, epochs=500, batch_size=512,
              verbose=True, print_every=50, lr_schedule=None, lr_min=1e-6,
              val_split=0.0, patience=200):
        """
        Full training loop.

        Parameters
        ----------
        lr_schedule : None or 'cosine'
            Cosine annealing decays self.lr → lr_min over the full epoch
            budget.  None (default) keeps a fixed learning rate.
        lr_min      : float
            Floor learning rate for cosine annealing (default 1e-6).
        val_split   : float
            Fraction of data held out as a validation set for early stopping
            (default 0.0 = disabled).  When > 0 the training set is reduced
            accordingly.
        patience    : int
            Number of consecutive epochs with no improvement in val_loss
            before stopping (default 200).  Set high when using cosine
            annealing — the loss can plateau for long stretches mid-schedule.

        Returns
        -------
        losses : list of float   mean *training* loss per epoch
        """
        N = X.shape[0]
        losses = []
        _lr_base = self.lr   # save for cosine annealing / restore

        # ── Optional validation split ──────────────────────────────────────
        use_early_stop = val_split > 0.0
        if use_early_stop:
            n_val   = max(1, int(N * val_split))
            _perm   = np.random.permutation(N)
            _vi     = _perm[:n_val]
            _ti     = _perm[n_val:]
            X_val,  y_val  = X[_vi],  y[_vi]
            X_tr,   y_tr   = X[_ti],  y[_ti]
            sw_tr = sample_weight[_ti] if sample_weight is not None else None
            N_tr  = len(X_tr)
            best_val       = np.inf
            patience_count = 0
            best_snap      = self._snapshot_weights()
            best_epoch     = 0
            if verbose:
                print(f"  Early stopping: val={n_val} / train={N_tr} "
                      f"(val_split={val_split:.0%}, patience={patience})")
        else:
            X_tr, y_tr, sw_tr, N_tr = X, y, sample_weight, N
        # ──────────────────────────────────────────────────────────────────

        for epoch in range(epochs):
            # --- optional cosine LR annealing ---
            if lr_schedule == 'cosine':
                self.lr = float(
                    lr_min + 0.5 * (_lr_base - lr_min)
                    * (1.0 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
                )

            idx  = np.random.permutation(N_tr)
            Xs   = X_tr[idx];  ys = y_tr[idx]
            sws  = sw_tr[idx] if sw_tr is not None else None

            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, N_tr, batch_size):
                end = min(start + batch_size, N_tr)
                swb = sws[start:end] if sws is not None else None
                epoch_loss += self.train_step(Xs[start:end], ys[start:end], swb)
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)

            # ── Early stopping check ───────────────────────────────────────
            if use_early_stop:
                val_loss = self._val_loss(X_val, y_val)
                if val_loss < best_val:
                    best_val       = val_loss
                    patience_count = 0
                    best_snap      = self._snapshot_weights()
                    best_epoch     = epoch
                else:
                    patience_count += 1

                if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                    lr_str = f"  lr={self.lr:.2e}" if lr_schedule else ""
                    print(f"  epoch {epoch:5d}/{epochs}  "
                          f"train={avg:.5e}  val={val_loss:.5e}"
                          f"  patience={patience_count}/{patience}{lr_str}")

                if patience_count >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best val={best_val:.5e} at epoch {best_epoch})")
                    self._restore_weights(best_snap)
                    break
            # ──────────────────────────────────────────────────────────────
            elif verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                lr_str = f"  lr={self.lr:.2e}" if lr_schedule else ""
                print(f"  epoch {epoch:5d}/{epochs}  loss={avg:.5e}{lr_str}")

        # Always restore best weights when validation was used (handles the
        # case where we exhaust epochs without triggering patience).
        if use_early_stop:
            self._restore_weights(best_snap)
            if verbose:
                print(f"  Best weights restored from epoch {best_epoch} "
                      f"(val={best_val:.5e})")

        self.lr = _lr_base   # restore original LR
        return losses


# ---------------------------------------------------------------------------
# NeuralNetworkTorch — PyTorch drop-in (MPS / CUDA / CPU auto-select)
# ---------------------------------------------------------------------------

class NeuralNetworkTorch:
    """
    PyTorch drop-in replacement for :class:`NeuralNetwork`.

    Runs on MPS (Apple M1/M2 GPU), CUDA, or CPU — auto-detected at
    construction.  Uses float32 internally (MPS has no float64 support).

    External API is **identical** to :class:`NeuralNetwork`:

    * ``predict(X)`` — numpy in → numpy float64 out
    * ``train(X, y, ...)`` — same full signature, returns list of float losses
    * ``.W`` / ``.b`` — properties that return ``list[numpy.ndarray]``
      (so :meth:`~ClayPMFNeuralEnsemble.save` works transparently)
    * ``.W`` / ``.b`` setters accept ``list[numpy.ndarray]``
      (so :meth:`~ClayPMFNeuralEnsemble.load` works transparently)
    * ``.n_hidden`` int attribute

    Parameters
    ----------
    input_dim    : int
    hidden_sizes : sequence of int
    lr           : float   Adam learning rate  (default 1e-3)
    l2           : float   L2 regularisation   (default 1e-4)
    seed         : int or None
    """

    def __init__(self, input_dim, hidden_sizes, lr=1e-3, l2=1e-4, seed=None,
                 verbose=True):
        try:
            import torch
            import torch.optim as _optim
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for backend='torch'.  "
                "Install via:  pip install torch"
            ) from exc

        self._torch = torch
        self._optim = _optim

        if seed is not None:
            torch.manual_seed(seed)

        self.lr       = float(lr)
        self.l2       = float(l2)
        self.n_hidden = len(hidden_sizes)
        self._dims    = [input_dim] + list(hidden_sizes) + [1]

        # Device: MPS (M1/M2) > CUDA > CPU
        if torch.backends.mps.is_available():
            self._dev = torch.device('mps')
        elif torch.cuda.is_available():
            self._dev = torch.device('cuda')
        else:
            self._dev = torch.device('cpu')
        self._dtype = torch.float32   # MPS has no float64

        # Xavier-initialised weight / bias leaf tensors (requires_grad=True)
        self._W_t = []
        self._b_t = []
        for fan_in, fan_out in zip(self._dims[:-1], self._dims[1:]):
            scale = float((2.0 / (fan_in + fan_out)) ** 0.5)
            w = (torch.randn(fan_out, fan_in,
                             dtype=self._dtype, device=self._dev) * scale)
            b = torch.zeros(fan_out, dtype=self._dtype, device=self._dev)
            w.requires_grad_(True)
            b.requires_grad_(True)
            self._W_t.append(w)
            self._b_t.append(b)

        self._rebuild_optimizer()
        if verbose:
            print(f"  [NeuralNetworkTorch]  device={self._dev}  "
                  f"float32  arch={self._dims}")

    # ── Optimizer management ──────────────────────────────────────────────

    def _rebuild_optimizer(self):
        """Create / reset the Adam optimizer (called after weight assignment)."""
        self._optimizer = self._optim.Adam(
            self._W_t + self._b_t, lr=self.lr, weight_decay=0.0
        )

    # ── Properties: .W and .b as numpy lists (save / load compatibility) ──

    @property
    def W(self):
        """List of weight matrices as float64 numpy arrays."""
        return [np.array(w.detach().cpu().tolist(), dtype=np.float64)
                for w in self._W_t]

    @W.setter
    def W(self, W_list):
        torch = self._torch
        with torch.no_grad():
            for i, w_np in enumerate(W_list):
                self._W_t[i].copy_(
                    torch.tensor(w_np, dtype=self._dtype, device=self._dev)
                )
        self._rebuild_optimizer()

    @property
    def b(self):
        """List of bias vectors as float64 numpy arrays."""
        return [np.array(b.detach().cpu().tolist(), dtype=np.float64)
                for b in self._b_t]

    @b.setter
    def b(self, b_list):
        torch = self._torch
        with torch.no_grad():
            for i, b_np in enumerate(b_list):
                self._b_t[i].copy_(
                    torch.tensor(b_np, dtype=self._dtype, device=self._dev)
                )
        self._rebuild_optimizer()

    # ── Forward pass ─────────────────────────────────────────────────────

    def _forward_t(self, X_t):
        """Torch forward pass.  X_t : (N, d) float32 tensor on device."""
        h = X_t
        for i, (W, b) in enumerate(zip(self._W_t, self._b_t)):
            z = h @ W.t() + b
            h = self._torch.tanh(z) if i < self.n_hidden else z
        return h.squeeze(-1)   # (N,)

    # ── Public API (identical to NeuralNetwork) ───────────────────────────

    def predict(self, X):
        """X : numpy (N, d) or (d,).  Returns numpy float64 (N,)."""
        torch = self._torch
        X_t = torch.tensor(
            np.atleast_2d(X).astype(np.float32),
            dtype=self._dtype, device=self._dev
        )
        with torch.no_grad():
            out = self._forward_t(X_t)
        return np.array(out.detach().cpu().tolist(), dtype=np.float64)

    def train(self, X, y, sample_weight=None, epochs=500, batch_size=512,
              verbose=True, print_every=50, lr_schedule=None, lr_min=1e-6,
              val_split=0.0, patience=200):
        """
        Full training loop — identical signature to NeuralNetwork.train().

        Reported losses are pure MSE (no L2 term), matching the numpy version.

        Returns
        -------
        losses : list of float   per-epoch training MSE (kJ/mol)²
        """
        torch    = self._torch
        _lr_base = self.lr

        N = X.shape[0]
        losses = []

        # Convert to float32 tensors on device (once)
        X_t  = torch.tensor(X.astype(np.float32),  dtype=self._dtype, device=self._dev)
        y_t  = torch.tensor(y.astype(np.float32),  dtype=self._dtype, device=self._dev)
        sw_t = None
        if sample_weight is not None:
            sw_t = torch.tensor(
                sample_weight.astype(np.float32), dtype=self._dtype, device=self._dev
            )

        # ── Optional validation split (early stopping) ────────────────────
        use_early_stop = val_split > 0.0
        if use_early_stop:
            n_val   = max(1, int(N * val_split))
            perm    = torch.randperm(N, device=self._dev)
            vi, ti  = perm[:n_val], perm[n_val:]
            X_val_t, y_val_t = X_t[vi], y_t[vi]
            X_tr_t,  y_tr_t  = X_t[ti], y_t[ti]
            sw_tr_t = sw_t[ti] if sw_t is not None else None
            N_tr    = int(ti.shape[0])
            best_val       = float('inf')
            patience_count = 0
            best_W = [w.data.clone() for w in self._W_t]
            best_b = [b.data.clone() for b in self._b_t]
            best_epoch = 0
            if verbose:
                print(f"  Early stopping: val={n_val} / train={N_tr} "
                      f"(val_split={val_split:.0%}, patience={patience})")
        else:
            X_tr_t, y_tr_t, sw_tr_t, N_tr = X_t, y_t, sw_t, N
        # ─────────────────────────────────────────────────────────────────

        for epoch in range(epochs):
            # Cosine LR annealing
            if lr_schedule == 'cosine':
                new_lr = float(
                    lr_min + 0.5 * (_lr_base - lr_min)
                    * (1.0 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
                )
                for pg in self._optimizer.param_groups:
                    pg['lr'] = new_lr

            # Shuffle indices and run batch loop
            idx  = torch.randperm(N_tr, device=self._dev)
            Xs, ys = X_tr_t[idx], y_tr_t[idx]
            sws  = sw_tr_t[idx] if sw_tr_t is not None else None

            epoch_mse = 0.0
            n_batches = 0

            for start in range(0, N_tr, batch_size):
                end  = min(start + batch_size, N_tr)
                Xb, yb = Xs[start:end], ys[start:end]
                swb  = sws[start:end] if sws is not None else None

                self._optimizer.zero_grad()
                out = self._forward_t(Xb)
                res = out - yb

                if swb is not None:
                    mse = (swb * res * res).mean()
                else:
                    mse = (res * res).mean()

                # L2 penalty: gradient contribution = 2*l2*W — matches numpy
                if self.l2 > 0:
                    l2_reg = sum((w * w).sum() for w in self._W_t)
                    total  = mse + self.l2 * l2_reg
                else:
                    total  = mse

                total.backward()
                self._optimizer.step()

                epoch_mse += float(mse.item())   # pure MSE (no L2 term)
                n_batches  += 1

            avg = epoch_mse / max(n_batches, 1)
            losses.append(avg)

            # ── Early stopping check ──────────────────────────────────────
            if use_early_stop:
                with torch.no_grad():
                    val_out  = self._forward_t(X_val_t)
                    val_loss = float(((val_out - y_val_t) ** 2).mean().item())

                if val_loss < best_val:
                    best_val       = val_loss
                    patience_count = 0
                    best_W = [w.data.clone() for w in self._W_t]
                    best_b = [b.data.clone() for b in self._b_t]
                    best_epoch = epoch
                else:
                    patience_count += 1

                if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                    _lr_now = self._optimizer.param_groups[0]['lr']
                    lr_str  = f"  lr={_lr_now:.2e}" if lr_schedule else ""
                    print(f"  epoch {epoch:5d}/{epochs}  "
                          f"train={avg:.5e}  val={val_loss:.5e}"
                          f"  patience={patience_count}/{patience}{lr_str}")

                if patience_count >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best val={best_val:.5e} at epoch {best_epoch})")
                    with torch.no_grad():
                        for w, bw in zip(self._W_t, best_W):
                            w.copy_(bw)
                        for bv, bb in zip(self._b_t, best_b):
                            bv.copy_(bb)
                    break
            # ─────────────────────────────────────────────────────────────
            elif verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                _lr_now = self._optimizer.param_groups[0]['lr']
                lr_str  = f"  lr={_lr_now:.2e}" if lr_schedule else ""
                print(f"  epoch {epoch:5d}/{epochs}  loss={avg:.5e}{lr_str}")

        # Restore best weights when early stopping was used
        if use_early_stop:
            with torch.no_grad():
                for w, bw in zip(self._W_t, best_W):
                    w.copy_(bw)
                for bv, bb in zip(self._b_t, best_b):
                    bv.copy_(bb)
            if verbose:
                print(f"  Best weights restored from epoch {best_epoch} "
                      f"(val={best_val:.5e})")

        # Restore base LR in optimizer and attribute
        for pg in self._optimizer.param_groups:
            pg['lr'] = _lr_base
        self.lr = _lr_base
        return losses


# ---------------------------------------------------------------------------
# Factory: select numpy or torch backend
# ---------------------------------------------------------------------------

def _make_nn(input_dim, hidden_sizes, backend='numpy', **kwargs):
    """
    Return a :class:`NeuralNetwork` (``backend='numpy'``) or
    :class:`NeuralNetworkTorch` (``backend='torch'``).

    Parameters
    ----------
    input_dim    : int
    hidden_sizes : tuple of int
    backend      : ``'numpy'`` (default) or ``'torch'``
    **kwargs     : ``lr``, ``l2``, ``seed`` — forwarded to the constructor

    Returns
    -------
    NeuralNetwork or NeuralNetworkTorch
    """
    if backend == 'torch':
        return NeuralNetworkTorch(input_dim, hidden_sizes, **kwargs)
    if backend != 'numpy':
        raise ValueError(
            f"backend must be 'numpy' or 'torch', got {backend!r}"
        )
    return NeuralNetwork(input_dim, hidden_sizes, **kwargs)


# ---------------------------------------------------------------------------
# ClayPMFNeural — high-level wrapper for ClayPMF3D
# ---------------------------------------------------------------------------

class ClayPMFNeural:
    """
    Neural-network PMF for ClayPMF3D objects.

    Parameters
    ----------
    pmf3d        : ClayPMF3D
        Must have run_wham_3d() (and optionally reference_to_bulk()) called.
    hidden_sizes : tuple of int
        Default architecture (64, 64, 32).
    temperature  : float or None
        If None, uses pmf3d.T (attribute) or 298.15 K.
    seed         : int or None
    """

    def __init__(self, pmf3d, hidden_sizes=(64, 64, 32),
                 temperature=None, seed=42):
        self.pmf3d        = pmf3d
        self.hidden_sizes = tuple(hidden_sizes)
        self.seed         = seed

        T = temperature or getattr(pmf3d, 'T', None) or getattr(pmf3d, 'temperature', 298.15)
        self.T    = float(T)
        self.kT   = 8.314462618e-3 * self.T   # kJ/mol
        self.beta = 1.0 / self.kT

        # --- Approach A ---
        self.nn_a      = None
        self.norm_a    = None   # (mean_3, std_3) input normalisation
        self.losses_a  = None

        # --- Approach B ---
        self.nn_b      = None
        self.norm_b    = None
        self.losses_b  = None
        # Training r_min recorded here so the plotter knows which bins were
        # excluded from each NN fit (see ClayPMFPlotter.plot_3d_ensemble_marginals).
        self.r_min_a   = None
        self.r_min_b   = None
        # Fine-grid PMF built by fit_reweighted (stored for inspection)
        self.W_fine     = None
        self.r_fine     = None
        self.theta_fine = None
        self.n_fine     = None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _grid_inputs_a(self):
        """
        Build (X, y) training data for Approach A from the WHAM PMF grid.

        Meshes all (r, θ, n_cat) combinations from the pmf3d grid centres,
        then removes rows where the PMF is NaN (unvisited or unsampled cells).

        Returns
        -------
        X : ndarray, shape (N, 3)
            Input features [r (nm), θ (deg), n_cat] for each finite grid cell.
        y : ndarray, shape (N,)
            Corresponding PMF values (kJ/mol).
        """
        p = self.pmf3d
        R, TH, N = np.meshgrid(
            p.r_centers, p.theta_centers, p.cation_centers, indexing='ij'
        )
        X = np.column_stack([R.ravel(), TH.ravel(), N.ravel()])
        y = p.pmf_3d.ravel()
        mask = np.isfinite(y)
        return X[mask], y[mask]

    @staticmethod
    def _normalise(X, norm=None):
        """
        Zero-mean, unit-variance normalisation.

        Parameters
        ----------
        X    : (N, d)
        norm : (mean, std) or None (fit from X)

        Returns
        -------
        X_norm : (N, d)
        norm   : (mean (d,), std (d,))
        """
        if norm is None:
            mean = X.mean(axis=0)
            std  = X.std(axis=0)
            std[std < 1e-12] = 1.0
            norm = (mean, std)
        return (X - norm[0]) / norm[1], norm

    def _compute_frame_weights(self, stride=1):
        """
        Compute per-frame WHAM weights for all raw trajectory frames.

        For frame i from sub-window k (harmonic center r₀_k), the
        MBAR/WHAM unnormalized weight is:

            w_i ∝ 1 / Σ_m  N_m · exp( f_m − β·½k·(r_i − r₀_m)² )

        where the sum is over all 2·n_windows sub-windows, using the
        already-converged WHAM free energies pmf3d.f.

        Parameters
        ----------
        stride : int   subsample every stride-th frame (default 1 = all)

        Returns
        -------
        r_all    : (Nf,)   |z| values
        theta_all: (Nf,)   tilt angle (degrees)
        ncat_all : (Nf,)   cation coordination (float)
        w_all    : (Nf,)   normalised WHAM weights (sum = 1)
        """
        p = self.pmf3d
        if p.f is None:
            raise RuntimeError(
                "pmf3d.f is None — call pmf3d.run_wham_3d() first."
            )
        if any(x is None for x in (p.z_data, p.theta_data, p.ncat_data)):
            raise RuntimeError(
                "Raw trajectory data not available. "
                "Call pmf3d.load_trajectory_data() before fit_reweighted()."
            )

        # Sub-window harmonic centres and WHAM parameters
        r0_list = []
        for c1, c2 in p.window_centers:
            r0_list.append(abs(float(c1)))
            r0_list.append(abs(float(c2)))
        r0   = np.array(r0_list)          # (M,)
        f    = p.f.copy()                 # (M,)
        N_sw = p.n_snapshots.astype(float)  # (M,)  frames per sub-window
        k    = float(p.k)
        beta = self.beta

        # Collect raw frames
        r_list, th_list, nc_list = [], [], []
        for i, ((z1, z2), (th1, th2), (nc1, nc2)) in enumerate(
            zip(p.z_data, p.theta_data, p.ncat_data)
        ):
            for z, th, nc in ((z1, th1, nc1), (z2, th2, nc2)):
                if stride > 1:
                    z  = z[::stride]
                    th = th[::stride]
                    nc = nc[::stride]
                r_list.append(np.abs(z).astype(float))
                th_list.append(th.astype(float))
                nc_list.append(nc.astype(float))

        r_all     = np.concatenate(r_list)
        theta_all = np.concatenate(th_list)
        ncat_all  = np.concatenate(nc_list)
        Nf        = len(r_all)

        print(f"  Total frames: {Nf:,}  (stride={stride})")

        # Per-frame denominator (numerically stable via log-sum-exp)
        # log_term[i, m] = log(N_m) + f_m − β·½k·(r_i − r₀_m)²
        # shape (Nf, M)
        V_mat     = 0.5 * k * (r_all[:, None] - r0[None, :]) ** 2   # (Nf, M)
        log_terms = np.log(N_sw[None, :] + 1e-300) + f[None, :] - beta * V_mat

        log_denom = _logsumexp(log_terms, axis=1)   # (Nf,)
        log_w     = -log_denom                       # unnormalised log weights

        # Normalise
        log_w -= _logsumexp(log_w)
        w_all  = np.exp(log_w)

        ess = 1.0 / float(np.sum(w_all ** 2))
        print(f"  Effective sample size (ESS): {ess:.0f}  "
              f"({100*ess/Nf:.1f}% of {Nf:,})")

        return r_all, theta_all, ncat_all, w_all

    def _eval_on_wham_grid(self, nn, norm):
        """
        Evaluate *nn* at every point on the pmf3d WHAM grid.

        Parameters
        ----------
        nn   : NeuralNetwork or NeuralNetworkTorch
        norm : tuple (mean, std)  normalisation from :meth:`_normalise`

        Returns
        -------
        W : ndarray, shape (n_r, n_theta, n_cat)
            NN-predicted PMF values (kJ/mol) on the full WHAM grid.
        """
        p = self.pmf3d
        R, TH, N = np.meshgrid(
            p.r_centers, p.theta_centers, p.cation_centers, indexing='ij'
        )
        X = np.column_stack([R.ravel(), TH.ravel(), N.ravel()])
        Xn, _ = self._normalise(X, norm)
        return nn.predict(Xn).reshape(R.shape)

    def _marginals_from_nn(self, approach, mask_unvisited=True):
        """
        Compute 1-D marginals W(r), W(θ), W(n_cat) from trained NN.

        Parameters
        ----------
        approach        : 'a' or 'b'
        mask_unvisited  : bool   if True (default), bins with no WHAM data are
                                 excluded from the logsumexp integration.  Set
                                 False to integrate over the full NN surface.

        Returns
        -------
        pmf_r     : ndarray, shape (n_r,)
            Radial marginal W(r) (kJ/mol), min-shifted to 0.
        pmf_theta : ndarray, shape (n_theta,)
            Angular marginal W(θ) (kJ/mol), min-shifted to 0.
        pmf_ncat  : ndarray, shape (n_cat,)
            Cation-count marginal W(n) (kJ/mol), min-shifted to 0.
        """
        p  = self.pmf3d
        nn = self.nn_a if approach == 'a' else self.nn_b
        nm = self.norm_a if approach == 'a' else self.norm_b

        W3d = self._eval_on_wham_grid(nn, nm)   # (n_r, n_theta, n_cat)

        if mask_unvisited:
            occupied = np.isfinite(p.pmf_3d)
            W3d_masked = np.where(occupied, W3d, np.nan)
        else:
            W3d_masked = W3d

        # W_marginal(x) = -kT ln[ Σ_{other} exp(-β W(x, other)) · Δ·Δ ]
        log_sum_r   = _logsumexp(-self.beta * W3d_masked, axis=(1, 2))  # (n_r,)
        log_sum_th  = _logsumexp(-self.beta * W3d_masked, axis=(0, 2))  # (n_th,)
        log_sum_n   = _logsumexp(-self.beta * W3d_masked, axis=(0, 1))  # (n_cat,)

        pmf_r   = -self.kT * log_sum_r;   pmf_r   -= pmf_r.min()
        pmf_th  = -self.kT * log_sum_th;  pmf_th  -= pmf_th.min()
        pmf_n   = -self.kT * log_sum_n;   pmf_n   -= pmf_n.min()

        return pmf_r, pmf_th, pmf_n

    # -----------------------------------------------------------------------
    # Approach A — grid smoother
    # -----------------------------------------------------------------------

    def fit_smooth(self, hidden_sizes=None, epochs=1000, lr=1e-3, l2=1e-4,
                   batch_size=512, r_min=None, pmf_max=None, verbose=True,
                   lr_schedule=None, lr_min=1e-6,
                   w1d_target=None, w1d_r=None, lambda_marginal=1.0,
                   val_split=0.0, patience=200, backend='numpy'):
        """
        Approach A: fit a NN to the existing WHAM PMF grid (pmf3d.pmf_3d).

        The NN learns a smooth, continuous W(r, θ, n_cat).  It can then be
        queried at arbitrary coordinates for interpolation or Kd integrals.

        Parameters
        ----------
        hidden_sizes : tuple or None  (default: self.hidden_sizes)
        epochs       : int
        lr           : float  Adam learning rate
        l2           : float  L2 weight decay
        batch_size   : int
        r_min        : float or None  exclude training points with r < r_min (nm)
        pmf_max      : float or None  exclude training points with W > pmf_max (kJ/mol)
        verbose      : bool
        w1d_target   : array-like or None
            Optional 1-D PMF target W(r) [kJ/mol].  When provided, training
            targets are shifted so the NN's r-marginal matches this curve.
            Must be on the same energy scale as pmf3d.pmf_3d.
        w1d_r        : array-like or None
            r-coordinates [nm] for *w1d_target*.  If None, *w1d_target* must
            be aligned to pmf3d.r_centers.
        lambda_marginal : float
            Blending factor for the marginal correction (default 1.0 = full
            correction).  Only used when *w1d_target* is not None.

        Returns
        -------
        losses : list of float  per-epoch training losses
        """
        if hidden_sizes is None:
            hidden_sizes = self.hidden_sizes

        print("=== Approach A: Grid Smoother ===")
        X, y = self._grid_inputs_a()
        n_raw = len(y)
        if r_min is not None:
            mask = X[:, 0] >= r_min
            X, y = X[mask], y[mask]
        if pmf_max is not None:
            mask = y <= pmf_max
            X, y = X[mask], y[mask]
        print(f"  Non-NaN grid cells: {len(y):,}  "
              f"({100*len(y)/n_raw:.0f}% of {n_raw:,} finite points"
              + (f", r_min={r_min}" if r_min else "")
              + (f", pmf_max={pmf_max}" if pmf_max else "") + ")")

        # --- Optional marginal target correction ---
        if w1d_target is not None:
            y = _apply_marginal_correction(
                X, y, W3d=self.pmf3d.pmf_3d, r_centers=self.pmf3d.r_centers,
                kT=self.kT, beta=self.beta,
                w1d_target=w1d_target, w1d_r=w1d_r,
                lambda_marginal=lambda_marginal,
            )

        X_norm, self.norm_a = self._normalise(X)
        self.nn_a   = _make_nn(3, hidden_sizes, lr=lr, l2=l2,
                               seed=self.seed, backend=backend)
        self.losses_a = self.nn_a.train(
            X_norm, y, epochs=epochs, batch_size=batch_size,
            verbose=verbose, print_every=max(1, epochs // 10),
            lr_schedule=lr_schedule, lr_min=lr_min,
            val_split=val_split, patience=patience,
        )
        print(f"  Final loss: {self.losses_a[-1]:.4e} (kJ/mol)²")
        self.r_min_a = r_min
        return self.losses_a

    # -----------------------------------------------------------------------
    # Approach B — per-frame reweighted PMF
    # -----------------------------------------------------------------------

    def fit_reweighted(self, hidden_sizes=None, epochs=1000, lr=1e-3, l2=1e-4,
                       batch_size=1024, n_r_bins=80, n_theta_bins=36,
                       stride=1, r_min=None, pmf_max=None, verbose=True,
                       lr_schedule=None, lr_min=1e-6,
                       w1d_target=None, w1d_r=None, lambda_marginal=1.0,
                       val_split=0.0, patience=200, backend='numpy'):
        """
        Approach B: per-frame WHAM weights → fine-grid PMF → NN.

        Steps
        -----
        1. Collect all raw frames from pmf3d.z_data / theta_data / ncat_data.
        2. Compute per-frame WHAM weights using converged pmf3d.f.
        3. Build fine-resolution unbiased PMF via weighted histogram.
        4. Train NN on that fine PMF grid.

        Parameters
        ----------
        hidden_sizes  : tuple or None
        epochs        : int
        lr, l2        : float
        batch_size    : int
        n_r_bins      : int   fine r-grid resolution (default 80)
        n_theta_bins  : int   fine θ-grid resolution (default 36)
        stride        : int   subsample raw frames (1 = use all)
        r_min         : float or None  exclude training points with r < r_min (nm)
        pmf_max       : float or None  exclude training points with W > pmf_max (kJ/mol)
        verbose       : bool
        w1d_target    : array-like or None
            Optional 1-D PMF target W(r) [kJ/mol].  Training targets are
            shifted so the NN's r-marginal matches this curve.  Must be on the
            same energy scale as the fine-grid PMF (W_fine).
        w1d_r         : array-like or None
            r-coordinates [nm] for *w1d_target*.  If None, *w1d_target* must
            be aligned to the fine-grid r bin-centres (r_ctrs).
        lambda_marginal : float
            Blending factor for the marginal correction (default 1.0).

        Returns
        -------
        losses : list of float
        """
        if hidden_sizes is None:
            hidden_sizes = self.hidden_sizes

        print("=== Approach B: Frame-Reweighted PMF ===")

        # --- Step 1: per-frame weights ---
        print("Step 1: Computing per-frame WHAM weights…")
        r_all, theta_all, ncat_all, w_all = self._compute_frame_weights(stride)

        # --- Step 2: weighted fine-grid histogram ---
        print("Step 2: Building fine-grid P_unbiased via weighted histogram…")
        p = self.pmf3d

        r_edges  = np.linspace(p.r_bins[0],     p.r_bins[-1],     n_r_bins   + 1)
        th_edges = np.linspace(p.theta_bins[0], p.theta_bins[-1], n_theta_bins + 1)
        n_edges  = p.cation_bins.copy()   # keep original integer bins

        r_ctrs  = 0.5 * (r_edges[:-1]  + r_edges[1:])
        th_ctrs = 0.5 * (th_edges[:-1] + th_edges[1:])
        n_ctrs  = p.cation_centers.copy()

        dr  = float(r_edges[1]  - r_edges[0])
        dth = float(th_edges[1] - th_edges[0])
        dn  = 1.0

        # Clip n_cat to declared range before histogramming
        ncat_clip = np.clip(ncat_all, p.cation_range[0], p.cation_range[1])

        # Weighted 3D histogram — bins sum to total weight (≈1)
        H_w, _ = np.histogramdd(
            np.column_stack([r_all, theta_all, ncat_clip]),
            bins=[r_edges, th_edges, n_edges],
            weights=w_all,
            density=False,
        )

        # Convert to probability density and PMF
        P_fine = H_w / (dr * dth * dn)   # P_fine * dr * dth * dn sums to 1
        with np.errstate(divide='ignore', invalid='ignore'):
            W_fine = np.where(P_fine > 0, -self.kT * np.log(P_fine), np.nan)
        # ── align W_fine zero-level to match pmf_3d ────────────────────────
        # When pmf_3d has been bulk-referenced (W_bulk≈0, W_well<0), apply the
        # same convention so NN-B colours map onto the same scale as WHAM.
        # Bulk region = first n_bulk r-bins (small r = far from clay).
        # Fall back to min=0 when reference_to_bulk() has not been called.
        if getattr(p, 'bulk_correction_enabled', False):
            _bulk_frac = getattr(p, 'bulk_fraction', 0.2)
            _n_bulk    = max(1, int(_bulk_frac * n_r_bins))
            _W_r       = np.nanmedian(W_fine, axis=(1, 2))   # (n_r_bins,)
            _shift     = float(np.nanmedian(_W_r[:_n_bulk]))
            W_fine    -= _shift if np.isfinite(_shift) else np.nanmin(W_fine)
        else:
            W_fine -= np.nanmin(W_fine)
        # ───────────────────────────────────────────────────────────────────

        self.W_fine     = W_fine
        self.r_fine     = r_ctrs
        self.theta_fine = th_ctrs
        self.n_fine     = n_ctrs

        finite_frac = 100.0 * np.sum(np.isfinite(W_fine)) / W_fine.size
        print(f"  Fine grid: r({n_r_bins}) × θ({n_theta_bins}) × "
              f"n_cat({p.n_cation_bins})  "
              f"({finite_frac:.0f}% populated)")
        print(f"  W range: [{np.nanmin(W_fine):.2f}, "
              f"{np.nanmax(W_fine):.2f}] kJ/mol")

        # --- Step 3: train NN on fine-grid PMF ---
        print("Step 3: Training NN on fine-grid PMF…")
        Rf, THF, NF = np.meshgrid(r_ctrs, th_ctrs, n_ctrs, indexing='ij')
        X_b = np.column_stack([Rf.ravel(), THF.ravel(), NF.ravel()])
        y_b = W_fine.ravel()
        mask = np.isfinite(y_b)
        X_b, y_b = X_b[mask], y_b[mask]
        if r_min is not None:
            keep = X_b[:, 0] >= r_min
            X_b, y_b = X_b[keep], y_b[keep]
        if pmf_max is not None:
            keep = y_b <= pmf_max
            X_b, y_b = X_b[keep], y_b[keep]
        print(f"  Training points: {len(y_b):,}"
              + (f"  (r≥{r_min})" if r_min else "")
              + (f"  (W≤{pmf_max})" if pmf_max else ""))

        # --- Optional marginal target correction ---
        if w1d_target is not None:
            y_b = _apply_marginal_correction(
                X_b, y_b, W3d=W_fine, r_centers=r_ctrs,
                kT=self.kT, beta=self.beta,
                w1d_target=w1d_target, w1d_r=w1d_r,
                lambda_marginal=lambda_marginal,
            )

        X_bn, self.norm_b = self._normalise(X_b)
        self.nn_b = _make_nn(3, hidden_sizes, lr=lr, l2=l2,
                             seed=self.seed, backend=backend)
        self.losses_b = self.nn_b.train(
            X_bn, y_b, epochs=epochs, batch_size=batch_size,
            verbose=verbose, print_every=max(1, epochs // 10),
            lr_schedule=lr_schedule, lr_min=lr_min,
            val_split=val_split, patience=patience,
        )
        print(f"  Final loss: {self.losses_b[-1]:.4e} (kJ/mol)²")
        self.r_min_b = r_min
        return self.losses_b

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def _predict_from(self, nn, norm, r, theta, n_cat):
        """Shared prediction backend — evaluates nn at given coordinates."""
        r     = np.atleast_1d(np.asarray(r,     dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        n_cat = np.atleast_1d(np.asarray(n_cat, dtype=float))
        if np.any(r < 0):
            warnings.warn(
                f"Negative r values detected (min={r.min():.3f}). "
                "r is distance from surface and should be >= 0."
            )
        if np.any((theta < 0) | (theta > np.pi)):
            warnings.warn(
                f"theta values outside [0, pi] detected "
                f"(min={theta.min():.3f}, max={theta.max():.3f}). "
                "theta is a polar angle in radians."
            )
        if np.any(n_cat < 0):
            warnings.warn(
                f"Negative n_cat values detected (min={n_cat.min():.3f}). "
                "n_cat is a coordination number and should be >= 0."
            )
        # Broadcast scalars to match the longest array
        n = max(len(r), len(theta), len(n_cat))
        if len(r)     == 1: r     = np.broadcast_to(r,     (n,))
        if len(theta) == 1: theta = np.broadcast_to(theta, (n,))
        if len(n_cat) == 1: n_cat = np.broadcast_to(n_cat, (n,))
        X = np.column_stack([r, theta, n_cat])
        Xn, _ = self._normalise(X, norm)
        return nn.predict(Xn)

    def predict(self, r, theta, n_cat):
        """
        Approach A: evaluate W(r, θ, n_cat) in kJ/mol.

        Parameters
        ----------
        r, theta, n_cat : array-like (broadcast-compatible)

        Returns
        -------
        W : ndarray  (kJ/mol)
        """
        if self.nn_a is None:
            raise RuntimeError("Call fit_smooth() first.")
        return self._predict_from(self.nn_a, self.norm_a, r, theta, n_cat)

    def predict_b(self, r, theta, n_cat):
        """Approach B: evaluate W(r, θ, n_cat) in kJ/mol."""
        if self.nn_b is None:
            raise RuntimeError("Call fit_reweighted() first.")
        return self._predict_from(self.nn_b, self.norm_b, r, theta, n_cat)

    def predict_both(self, r, theta, n_cat):
        """
        Evaluate both NN-A and NN-B at the same coordinates.

        Returns
        -------
        W_a, W_b : ndarray  (kJ/mol)
        """
        W_a = self.predict(r, theta, n_cat)   if self.nn_a else None
        W_b = self.predict_b(r, theta, n_cat) if self.nn_b else None
        return W_a, W_b

    # -----------------------------------------------------------------------
    # Hyperparameter tuning
    # -----------------------------------------------------------------------

    def tune_hyperparameters(self, approach='a', param_grid=None, cv=5,
                             cv_epochs=300, batch_size=512,
                             cv_patience=0,
                             r_min=None, pmf_max=None,
                             backend='numpy', verbose=True):
        """
        k-fold cross-validation grid search over hidden_sizes, lr, l2.

        Only Approach A (grid smoother) is supported.  Approach B requires
        expensive per-frame weight recomputation on each fold; use
        ``fit_reweighted(val_split=..., patience=...)`` to tune that instead.

        Parameters
        ----------
        approach     : str   only 'a' is currently supported
        param_grid   : dict or None
            Keys: 'hidden_sizes', 'lr', 'l2'.  If None a default 36-combo
            grid is used (4 architectures × 3 lr × 3 l2).
        cv           : int   number of CV folds (default 5)
        cv_epochs    : int   max epochs per fold (default 300)
        batch_size   : int
        cv_patience  : int   early-stop patience on inner val loss
            (0 = disabled; >0 uses val_split=0.1 inside each fold's train)
        r_min        : float or None   same filtering as fit_smooth
        pmf_max      : float or None   same filtering as fit_smooth
        backend      : 'numpy' or 'torch'
        verbose      : bool   print per-combo progress (one line per combo)

        Returns
        -------
        results : list of dict, sorted by mean val MSE ascending (best first)
            Each dict has keys: 'hidden_sizes', 'lr', 'l2',
            'val_mse' (mean over folds), 'fold_mse' (list per fold).
        """
        if approach != 'a':
            raise NotImplementedError(
                "tune_hyperparameters only supports approach='a'. "
                "For Approach B use fit_reweighted(val_split=..., patience=...)."
            )
        if param_grid is None:
            param_grid = {
                'hidden_sizes': [(32,), (64,), (64, 32), (64, 64, 32)],
                'lr':           [1e-4, 3e-4, 1e-3],
                'l2':           [0.0,  1e-5,  1e-4],
            }

        # Build training data once (same pipeline as fit_smooth)
        X, y = self._grid_inputs_a()
        if r_min   is not None:
            mask = X[:, 0] >= r_min;  X, y = X[mask], y[mask]
        if pmf_max is not None:
            mask = y <= pmf_max;      X, y = X[mask], y[mask]
        N = len(y)
        if N < cv:
            raise ValueError(
                f"Only {N} training points — cannot split into {cv} folds."
            )

        # Normalise on full data so all folds share the same input scale
        X_norm, _ = self._normalise(X)

        # k-fold index split
        rng     = np.random.default_rng(self.seed if self.seed is not None else 0)
        indices = rng.permutation(N)
        folds   = [idx.tolist() for idx in np.array_split(indices, cv)]

        keys   = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        print(f"Grid search: {len(combos)} combos × {cv} folds = "
              f"{len(combos) * cv} fits  ({N:,} points, cv_epochs={cv_epochs})")

        results = []
        for i, values in enumerate(combos):
            params = dict(zip(keys, values))
            hs = params.get('hidden_sizes', self.hidden_sizes)
            lr = params.get('lr',           1e-3)
            l2 = params.get('l2',           1e-4)

            fold_mse = []
            for fi, val_idx in enumerate(folds):
                val_mask = np.zeros(N, dtype=bool)
                val_mask[val_idx] = True
                X_tr, y_tr = X_norm[~val_mask], y[~val_mask]
                X_va, y_va = X_norm[val_mask],  y[val_mask]

                nn = _make_nn(3, hs, lr=lr, l2=l2,
                              seed=self.seed, backend=backend)
                nn.train(X_tr, y_tr, epochs=cv_epochs, batch_size=batch_size,
                         verbose=False)
                mse = float(np.mean((nn.predict(X_va) - y_va) ** 2))
                fold_mse.append(mse)
                if verbose:
                    print(f"  [{i+1:>2}/{len(combos)}] "
                          f"hidden={hs} lr={lr:.0e} l2={l2:.0e}  "
                          f"fold {fi+1}/{cv}  val_MSE={mse:.4e}")

            results.append({
                'hidden_sizes': hs,
                'lr':           lr,
                'l2':           l2,
                'val_mse':      float(np.mean(fold_mse)),
                'fold_mse':     fold_mse,
            })

        results.sort(key=lambda d: d['val_mse'])
        best = results[0]
        print(f"\nBest: hidden={best['hidden_sizes']}  "
              f"lr={best['lr']:.0e}  l2={best['l2']:.0e}  "
              f"val_MSE={best['val_mse']:.4e} (kJ/mol)²")
        return results

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, path, include_metadata=True):
        """
        Save trained NN weights and normalisations to a .npz file.

        Parameters
        ----------
        path             : str    e.g. 'nn_pmf.npz'
        include_metadata : bool   if True (default), store version, timestamp
                                  and git hash in the file for provenance.
        """
        data = {
            'hidden_sizes': np.array(self.hidden_sizes),
            'seed':         np.array(self.seed if self.seed is not None else -1),
            'T':            np.array(self.T),
            'r_min_a':      np.array(float('nan') if self.r_min_a is None else self.r_min_a),
            'r_min_b':      np.array(float('nan') if self.r_min_b is None else self.r_min_b),
        }
        if include_metadata:
            data['_version']   = np.array([_CLAYPMFNEURAL_VERSION], dtype='U')
            data['_timestamp'] = np.array([datetime.now().isoformat()], dtype='U')
            data['_git_hash']  = np.array([_git_hash()], dtype='U')

        for tag, nn, norm in (('a', self.nn_a, self.norm_a),
                               ('b', self.nn_b, self.norm_b)):
            if nn is not None:
                for i, (W, b) in enumerate(zip(nn.W, nn.b)):
                    data[f'{tag}_W{i}'] = W
                    data[f'{tag}_b{i}'] = b
                data[f'{tag}_norm_mean'] = norm[0]
                data[f'{tag}_norm_std']  = norm[1]
                data[f'{tag}_n_layers']  = np.array(len(nn.W))
                if tag == 'b' and self.W_fine is not None:
                    data['W_fine']     = self.W_fine
                    data['r_fine']     = self.r_fine
                    data['theta_fine'] = self.theta_fine
                    data['n_fine']     = self.n_fine

        np.savez_compressed(path, **data)
        print(f"Saved NN weights to {path}")

    @classmethod
    def load(cls, path, pmf3d):
        """
        Load trained NN weights from a .npz file.

        Parameters
        ----------
        path  : str
        pmf3d : ClayPMF3D   the associated pmf3d object

        Returns
        -------
        ClayPMFNeural instance with nn_a and/or nn_b populated
        """
        d = np.load(path, allow_pickle=False)
        for key, label in (('_version', 'version'), ('_timestamp', 'saved'),
                           ('_git_hash', 'git')):
            if key in d:
                print(f"  [{label}] {str(d[key][0])}")
        hidden_sizes = tuple(int(x) for x in d['hidden_sizes'])
        seed  = int(d['seed'])
        if seed < 0:
            seed = None
        T = float(d['T'])

        obj = cls(pmf3d, hidden_sizes=hidden_sizes, temperature=T, seed=seed)
        # Restore training r_min (NaN sentinel means None / not set)
        for attr, key in (('r_min_a', 'r_min_a'), ('r_min_b', 'r_min_b')):
            if key in d:
                val = float(d[key])
                setattr(obj, attr, None if np.isnan(val) else val)

        for tag in ('a', 'b'):
            if f'{tag}_n_layers' not in d:
                continue
            n_layers = int(d[f'{tag}_n_layers'])
            W_list = [d[f'{tag}_W{i}'] for i in range(n_layers)]
            b_list = [d[f'{tag}_b{i}'] for i in range(n_layers)]
            norm   = (d[f'{tag}_norm_mean'], d[f'{tag}_norm_std'])
            input_dim = W_list[0].shape[1]
            hs = [W_list[i].shape[0] for i in range(n_layers - 1)]
            nn = NeuralNetwork(input_dim, hs, seed=seed)
            nn.W = W_list
            nn.b = b_list
            if tag == 'a':
                obj.nn_a, obj.norm_a = nn, norm
            else:
                obj.nn_b, obj.norm_b = nn, norm
                if 'W_fine' in d:
                    obj.W_fine     = d['W_fine']
                    obj.r_fine     = d['r_fine']
                    obj.theta_fine = d['theta_fine']
                    obj.n_fine     = d['n_fine']

        print(f"Loaded NN weights from {path}")
        return obj

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------

    def plot_losses(self, figsize=(11, 4)):
        """
        Side-by-side training loss curves for Approach A and B.

        Returns
        -------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, losses, label, color in zip(
            axes,
            [self.losses_a, self.losses_b],
            ['Approach A: Grid Smoother', 'Approach B: Reweighted'],
            ['steelblue', 'tomato'],
        ):
            if losses is None:
                ax.text(0.5, 0.5, 'Not fitted', ha='center', va='center',
                        fontsize=12, transform=ax.transAxes, color='grey')
                ax.set_title(label)
            else:
                ax.semilogy(losses, color=color, lw=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE loss  (kJ/mol)²')
                ax.set_title(label)
                ax.grid(True, alpha=0.3)
                ax.text(0.98, 0.95, f'final={losses[-1]:.3e}',
                        ha='right', va='top', transform=ax.transAxes,
                        fontsize=9, color=color)
        plt.tight_layout()
        return fig

    def plot_comparison_slice(self, n_cat_val=1, figsize=(16, 5),
                               vmax=None, r_min=None, cmap='viridis',
                               mask_unvisited=True, n_levels=40,
                               zero_at='bulk'):
        """
        W(r, θ) at fixed n_cat: WHAM grid vs NN-A vs NN-B.

        Parameters
        ----------
        n_cat_val       : int     cation coordination value to slice at
        figsize         : tuple
        vmax            : float or None   colorbar ceiling (kJ/mol)
        r_min           : float or None   clip colorbar at bins with
                                          surface-distance ≥ r_min (nm from surface)
        cmap            : str
        mask_unvisited  : bool   if True (default), NN predictions at grid cells
                                 with no WHAM data are hidden (NaN).  Set False to
                                 see the NN extrapolation into unvisited regions.
        zero_at         : {'bulk', 'min'}  reference convention.  'bulk' subtracts
                                 the median of the first 20 %% of r-bins (pore centre)
                                 so adsorption wells appear negative.

        Returns
        -------
        fig : matplotlib Figure
        """
        p     = self.pmf3d
        n_idx = int(np.argmin(np.abs(p.cation_centers - n_cat_val)))
        n_val = p.cation_centers[n_idx]

        # Surface-relative x-axis
        x_surf = float(getattr(p, 'z_clay_surface', None) or 0.0)
        r_surf = x_surf - p.r_centers   # 0 at surface, negative toward bulk

        # Panels to draw
        panels = [('WHAM grid', None, None)]
        if self.nn_a is not None:
            panels.append(('NN-A  (smooth)', self.nn_a, self.norm_a))
        if self.nn_b is not None:
            panels.append(('NN-B  (reweighted)', self.nn_b, self.norm_b))

        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=True)
        if ncols == 1:
            axes = [axes]

        # Determine colour scale
        W_wham = p.pmf_3d[:, :, n_idx].copy()

        # Guard: if this cation coordination state was never sampled, skip it
        if not np.any(np.isfinite(W_wham)):
            plt.close(fig)
            print(f"  n_cat={n_val:.0f}: no WHAM data — skipping panel.")
            return None

        # ── Bulk reference (same convention as plot_3d_slices / plot_3d_conditional) ──
        if zero_at == 'bulk':
            n_b = max(1, int(0.2 * p.n_r_bins))
            _bulk_shift = float(np.nanmedian(p.pmf_3d[:n_b, :, :]))
        else:
            _bulk_shift = float(np.nanmin(p.pmf_3d))
        W_wham -= _bulk_shift

        # r_min mask (applied per-panel for vmax auto-detection)
        r_mask = None
        if r_min is not None:
            r_mask = (x_surf - p.r_centers) >= r_min

        # Occupancy mask from WHAM: True where data existed
        wham_occupied = np.isfinite(W_wham)

        # ── Pre-pass: collect each panel's data so each gets its own colour scale ──
        _panel_data = []
        for title, nn, nm in panels:
            if nn is None:
                Z = W_wham.copy()
            else:
                R_q, TH_q = np.meshgrid(p.r_centers, p.theta_centers, indexing='ij')
                N_q = np.full(R_q.shape, n_val)
                X_q = np.column_stack([R_q.ravel(), TH_q.ravel(), N_q.ravel()])
                Xn, _ = self._normalise(X_q, nm)
                Z = nn.predict(Xn).reshape(R_q.shape) - _bulk_shift
                if mask_unvisited:
                    Z = np.where(wham_occupied, Z, np.nan)
            _panel_data.append((title, nn, nm, Z))

        for ax, (title, nn, nm, Z_plot) in zip(axes, _panel_data):
            # Per-panel colour scale
            if r_mask is not None and r_mask.any():
                _vmax = vmax if vmax is not None else float(np.nanpercentile(Z_plot[r_mask, :], 98))
            else:
                _vmax = vmax if vmax is not None else float(np.nanpercentile(Z_plot, 98))
            _vmin = float(np.nanmin(Z_plot))
            if _vmax <= _vmin:
                _vmax = _vmin + 1e-6  # avoid degenerate level range
            _cnorm  = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
            _levels = np.linspace(_vmin, _vmax, n_levels + 1)

            Z_plot = np.clip(Z_plot, _vmin, _vmax)

            cf = ax.contourf(r_surf, p.theta_centers, Z_plot.T,
                             levels=_levels, cmap=cmap, norm=_cnorm, extend='neither')
            ax.axvline(0.0, color='white', lw=1.5, ls='--', alpha=0.9,
                       label='surface')
            ax.set_xlabel('Distance from surface (nm)')
            ax.set_title(f'{title}\n$n_{{\\rm cat}}={n_val:.0f}$')
            plt.colorbar(cf, ax=ax, label='W  (kJ/mol)', pad=0.02)

        axes[0].set_ylabel('θ  (°)')
        plt.suptitle(
            f'W(r, θ) slice at $n_{{\\rm cat}} = {n_val:.0f}$',
            y=1.01, fontsize=12
        )
        plt.tight_layout()
        return fig

    def plot_1d_marginals(self, figsize=(15, 4), mask_unvisited=True):
        """
        Three-panel comparison of W(r), W(θ), W(n_cat) marginals.

        Shows WHAM result and NN-A / NN-B where available.

        Parameters
        ----------
        figsize         : tuple
        mask_unvisited  : bool   passed to _marginals_from_nn.  True (default)
                                 restricts integration to bins with WHAM data.
                                 Set False to integrate over the full NN surface.

        Returns
        -------
        fig : matplotlib Figure
        """
        p   = self.pmf3d
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        x_surf = float(getattr(p, 'z_clay_surface', None) or 0.0)
        r_surf = x_surf - p.r_centers

        # Pre-compute NN marginals (vectorised over entire grid)
        nn_margs = {}
        for tag in ('a', 'b'):
            nn = self.nn_a if tag == 'a' else self.nn_b
            if nn is not None:
                nn_margs[tag] = self._marginals_from_nn(tag, mask_unvisited=mask_unvisited)

        specs = [
            (0, r_surf,          p.pmf_r,      'Distance from surface (nm)', 'W(r)'),
            (1, p.theta_centers, p.pmf_theta,  'θ  (°)',                     'W(θ)'),
            (2, p.cation_centers,p.pmf_cation, '$n_{\\rm cat}$',             'W($n_{\\rm cat}$)'),
        ]

        for panel_idx, x, wham_y, xlabel, title in specs:
            ax = axes[panel_idx]
            ax.plot(x, wham_y, 'k-', lw=2.5, label='WHAM', zorder=3)
            for tag, color, style, label in (
                ('a', 'steelblue', '--', 'NN-A'),
                ('b', 'tomato',   ':',  'NN-B'),
            ):
                if tag in nn_margs:
                    y_nn = nn_margs[tag][panel_idx]
                    ax.plot(x, y_nn, color=color, ls=style, lw=2, label=label)

            if panel_idx == 2:
                ax.set_xticks(p.cation_centers)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('W  (kJ/mol)')
            ax.set_title(title)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            if panel_idx == 0:
                ax.axvline(0.0, color='grey', lw=1, ls='--', alpha=0.7)

        plt.tight_layout()
        return fig

    def plot_fine_pmf_slice(self, n_cat_val=1, figsize=(14, 5),
                             vmax=None, cmap='viridis', n_levels=40,
                             zero_at='bulk'):
        """
        Compare the original WHAM grid PMF and the fine-grid B PMF
        (W_fine) for the same n_cat slice.

        Only available after fit_reweighted().

        Each panel uses its own colour scale so the fine-grid NN-B range
        does not compress the WHAM colour contrast (or vice versa).

        Parameters
        ----------
        n_cat_val : int
        figsize   : tuple
        vmax      : float or None  shared colorbar ceiling; if None each
                    panel auto-scales to its own 98th percentile.
        cmap      : str
        n_levels  : int
        zero_at   : {'bulk', 'min'}  bulk-reference convention.

        Returns
        -------
        fig : matplotlib Figure
        """
        if self.W_fine is None:
            raise RuntimeError("Call fit_reweighted() first.")

        p = self.pmf3d
        x_surf = float(getattr(p, 'z_clay_surface', None) or 0.0)

        # n_cat index in original grid and fine grid
        nidx_wham = int(np.argmin(np.abs(p.cation_centers - n_cat_val)))
        nidx_fine = int(np.argmin(np.abs(self.n_fine - n_cat_val)))
        n_val = p.cation_centers[nidx_wham]

        # ── Bulk reference (same convention as plot_3d_slices) ──
        if zero_at == 'bulk':
            n_b = max(1, int(0.2 * p.n_r_bins))
            _bulk_shift = float(np.nanmedian(p.pmf_3d[:n_b, :, :]))
        else:
            _bulk_shift = float(np.nanmin(p.pmf_3d))

        W_wham = p.pmf_3d[:, :, nidx_wham].copy() - _bulk_shift
        W_b    = self.W_fine[:, :, nidx_fine].copy() - _bulk_shift

        # Guard: if the WHAM slice has no data this n_cat was never sampled
        if not np.any(np.isfinite(W_wham)):
            print(f"  n_cat={n_val:.0f}: no WHAM data — skipping.")
            return None

        r_surf_wham = x_surf - p.r_centers
        r_surf_fine = x_surf - self.r_fine

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        for ax, r_s, th_s, W_s, title in [
            (axes[0], r_surf_wham, p.theta_centers, W_wham, 'WHAM grid'),
            (axes[1], r_surf_fine, self.theta_fine, W_b,    'B fine grid'),
        ]:
            # Per-panel colour scale
            _vmax = vmax if vmax is not None else float(np.nanpercentile(W_s, 98))
            _vmin = float(np.nanmin(W_s))
            # Guard against all-NaN panels (e.g. fine grid has no coverage here)
            if not np.isfinite(_vmin) or not np.isfinite(_vmax) or _vmin >= _vmax:
                ax.set_title(f'{title}\n(no data)')
                continue
            _cnorm  = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
            _levels = np.linspace(_vmin, _vmax, n_levels + 1)

            Wc = np.clip(W_s, _vmin, _vmax)
            cf = ax.contourf(r_s, th_s, Wc.T, levels=_levels,
                             cmap=cmap, norm=_cnorm, extend='neither')
            ax.axvline(0.0, color='white', lw=1.5, ls='--', alpha=0.9)
            ax.set_xlabel('Distance from surface (nm)')
            ax.set_title(f'{title}\n$n_{{\\rm cat}} = {n_val:.0f}$')
            plt.colorbar(cf, ax=ax, label='W  (kJ/mol)', pad=0.02)

        axes[0].set_ylabel('θ  (°)')
        plt.suptitle(
            f'WHAM vs fine-grid B  at $n_{{\\rm cat}} = {n_val:.0f}$',
            y=1.01, fontsize=12
        )
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# ClayPMFNeuralEnsemble — multi-replicate pooled NN PMF
# ---------------------------------------------------------------------------

class ClayPMFNeuralEnsemble:
    """
    Neural-network PMF trained on *pooled* data from multiple independent
    replicate :class:`ClayPMF3D` objects.

    Each replicate is an independent umbrella-sampling run sharing the same
    box and pulling protocol.  Pooling their WHAM grid cells (Approach A) or
    raw trajectory frames (Approach B) into a single NN training session
    yields a smoother, better-constrained free-energy surface than any
    single run could provide.

    Parameters
    ----------
    pmf3d_list   : list of ClayPMF3D
        At least two elements.  Each must have ``run_wham_3d()`` (Approach A)
        or additionally ``load_trajectory_data()`` (Approach B) called first.
        The **first** element is used as the *reference replicate* — its
        ``r_centers``, ``theta_centers``, and ``cation_centers`` define all
        grid axes used for predictions and plots.
    hidden_sizes : tuple of int
        Hidden layer widths.  Default ``(64, 64, 32)``.
    temperature  : float or None
        Simulation temperature (K).  Falls back to ``pmf3d_list[0].T``,
        then 298.15 K.
    seed         : int or None
        Random seed for reproducible NN initialisation.

    Notes
    -----
    All replicates should share the same simulation box so their grid axes
    are (nearly) identical.  No inter-grid interpolation is performed.
    Before pooling, each replicate's PMF is independently bulk-referenced
    so they all share the same energy zero.

    Examples
    --------
    ::

        from ClayPMFNeural import ClayPMFNeuralEnsemble

        ensemble = ClayPMFNeuralEnsemble([pmf1, pmf2, pmf3, pmf4, pmf5])
        ensemble.fit_smooth(epochs=500)

        fig = ensemble.plot_comparison_slice(n_cat_val=1)
        fig = ensemble.plot_1d_marginals()
        fig = ensemble.plot_uncertainty_map(n_cat_val=1)

        ensemble.save('ensemble_nn.npz')
        ens2 = ClayPMFNeuralEnsemble.load('ensemble_nn.npz', [pmf1, ...])
    """

    def __init__(self, pmf3d_list, hidden_sizes=(64, 64, 32),
                 temperature=None, seed=42):
        if len(pmf3d_list) < 2:
            raise ValueError(
                "ClayPMFNeuralEnsemble requires at least 2 replicates."
            )
        self.pmf3d_list   = list(pmf3d_list)
        self.pmf3d        = pmf3d_list[0]        # reference replicate
        self.n_replicates = len(pmf3d_list)
        self.hidden_sizes = tuple(hidden_sizes)
        self.seed         = seed

        T = (temperature
             or getattr(self.pmf3d, 'T', None)
             or getattr(self.pmf3d, 'temperature', None)
             or 298.15)
        self.T    = float(T)
        self.kT   = 8.314462618e-3 * self.T
        self.beta = 1.0 / self.kT

        # Approach-A NN
        self.nn_a     = None
        self.norm_a   = None
        self.losses_a = None
        # Approach-B NN
        self.nn_b     = None
        self.norm_b   = None
        self.losses_b = None
        # Training r_min recorded here so the plotter knows which bins were
        # excluded from each NN fit (see ClayPMFPlotter.plot_3d_ensemble_marginals).
        self.r_min_a  = None
        self.r_min_b  = None
        # Fine-grid PMF (Approach B)
        self.W_fine     = None
        self.r_fine     = None
        self.theta_fine = None
        self.n_fine     = None
        # Per-replicate NNs (deep ensemble, trained via fit_smooth_per_replicate)
        self._nn_per_replicate   = []
        self._norm_per_replicate = []
        self._losses_per_rep     = []

    # -----------------------------------------------------------------------
    # Static helpers (identical to ClayPMFNeural)
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalise(X, norm=None):
        """Zero-mean, unit-variance normalisation.  Fits from X if norm is None."""
        if norm is None:
            mean = X.mean(axis=0)
            std  = X.std(axis=0)
            std[std < 1e-12] = 1.0
            norm = (mean, std)
        return (X - norm[0]) / norm[1], norm

    def _predict_from(self, nn, norm, r, theta, n_cat):
        """Shared prediction back-end."""
        r     = np.atleast_1d(np.asarray(r,     dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        n_cat = np.atleast_1d(np.asarray(n_cat, dtype=float))
        if np.any(r < 0):
            warnings.warn(
                f"Negative r values detected (min={r.min():.3f}). "
                "r is distance from surface and should be >= 0."
            )
        if np.any((theta < 0) | (theta > np.pi)):
            warnings.warn(
                f"theta values outside [0, pi] detected "
                f"(min={theta.min():.3f}, max={theta.max():.3f}). "
                "theta is a polar angle in radians."
            )
        if np.any(n_cat < 0):
            warnings.warn(
                f"Negative n_cat values detected (min={n_cat.min():.3f}). "
                "n_cat is a coordination number and should be >= 0."
            )
        n = max(len(r), len(theta), len(n_cat))
        if len(r)     == 1: r     = np.broadcast_to(r,     (n,))
        if len(theta) == 1: theta = np.broadcast_to(theta, (n,))
        if len(n_cat) == 1: n_cat = np.broadcast_to(n_cat, (n,))
        X  = np.column_stack([r, theta, n_cat])
        Xn, _ = self._normalise(X, norm)
        return nn.predict(Xn)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _bulk_shift_single(self, pmf3d, zero_at='bulk'):
        """Return the bulk-reference energy shift for one pmf3d."""
        if zero_at == 'bulk':
            n_b   = max(1, int(0.2 * pmf3d.n_r_bins))
            shift = float(np.nanmedian(pmf3d.pmf_3d[:n_b, :, :]))
        else:
            shift = float(np.nanmin(pmf3d.pmf_3d))
        return shift if np.isfinite(shift) else 0.0

    def _pool_grid_inputs_a(self, r_min=None, pmf_max=None, zero_at='bulk'):
        """
        Collect and concatenate (X, y) training data from all replicates.

        Each replicate's PMF is independently bulk-referenced before pooling
        so all replicates share the same energy zero.

        Parameters
        ----------
        r_min   : float or None   exclude points with r < r_min (nm)
        pmf_max : float or None   exclude points with W > pmf_max (kJ/mol)
        zero_at : 'bulk' or 'min'

        Returns
        -------
        X_all : ndarray  (N_total, 3)
        y_all : ndarray  (N_total,)
        """
        X_list, y_list = [], []
        for k, pmf3d in enumerate(self.pmf3d_list):
            p  = pmf3d
            R, TH, N = np.meshgrid(
                p.r_centers, p.theta_centers, p.cation_centers, indexing='ij'
            )
            X  = np.column_stack([R.ravel(), TH.ravel(), N.ravel()])
            sh = self._bulk_shift_single(p, zero_at)
            y  = (p.pmf_3d - sh).ravel()
            ok = np.isfinite(y)
            X_list.append(X[ok])
            y_list.append(y[ok])
            print(f"  Replicate {k + 1}: {ok.sum():,} finite cells  "
                  f"(bulk shift = {sh:.3f} kJ/mol)")

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)

        if r_min is not None:
            keep  = X_all[:, 0] >= r_min
            X_all, y_all = X_all[keep], y_all[keep]
        if pmf_max is not None:
            keep  = y_all <= pmf_max
            X_all, y_all = X_all[keep], y_all[keep]

        return X_all, y_all

    def mean_wham_pmf(self, zero_at='bulk'):
        """
        Pointwise mean and inter-replicate standard deviation of the 3-D PMF.

        All replicates are independently bulk-referenced before stacking.

        Parameters
        ----------
        zero_at : 'bulk' or 'min'

        Returns
        -------
        mean_pmf : ndarray  (n_r, n_theta, n_cat)
        std_pmf  : ndarray  (n_r, n_theta, n_cat)
        """
        grids = []
        for pmf3d in self.pmf3d_list:
            sh = self._bulk_shift_single(pmf3d, zero_at)
            grids.append(pmf3d.pmf_3d - sh)
        stack    = np.stack(grids, axis=0)        # (n_rep, n_r, n_theta, n_cat)
        mean_pmf = np.nanmean(stack, axis=0)
        std_pmf  = np.nanstd(stack,  axis=0)
        return mean_pmf, std_pmf

    def boltzmann_ensemble_marginals(self, zero_at='bulk'):
        """
        Compute 1-D marginals W(r), W(θ), W(n_cat) by Boltzmann-factor
        averaging across replicates before marginalising.

        Unlike pointwise PMF averaging (Jensen-biased), this computes:

            <BF>(r,θ,n) = (1/N_rep) Σ_k  exp(-β W_k(r,θ,n))
            W_ens(r)    = -kT ln [ Σ_{θ,n} <BF>(r,θ,n) ]

        Grid cells that are NaN in a replicate contribute zero Boltzmann
        weight (correct: they have no sampled probability mass).

        Parameters
        ----------
        zero_at : 'bulk' or 'min'
            Bulk-reference convention applied per replicate before averaging.

        Returns
        -------
        pmf_r  : ndarray  (n_r,)      kJ/mol, min-shifted to 0
        pmf_th : ndarray  (n_theta,)  kJ/mol, min-shifted to 0
        pmf_n  : ndarray  (n_cat,)    kJ/mol, min-shifted to 0
        """
        # Bulk-referenced stack: (n_rep, n_r, n_theta, n_cat)
        grids = []
        for pmf3d in self.pmf3d_list:
            sh = self._bulk_shift_single(pmf3d, zero_at)
            grids.append(pmf3d.pmf_3d - sh)
        stack = np.stack(grids, axis=0)

        # Boltzmann factors; NaN cells → 0 (no probability mass)
        bf = np.exp(-self.beta * stack)
        bf = np.where(np.isfinite(bf), bf, 0.0)

        # Average Boltzmann factors across replicates
        avg_bf = np.mean(bf, axis=0)              # (n_r, n_theta, n_cat)
        avg_bf = np.where(avg_bf > 0.0, avg_bf, np.nan)

        # Marginalise: nansum treats fully-unvisited cells as zero
        sum_r  = np.nansum(avg_bf, axis=(1, 2))   # (n_r,)
        sum_th = np.nansum(avg_bf, axis=(0, 2))   # (n_theta,)
        sum_n  = np.nansum(avg_bf, axis=(0, 1))   # (n_cat,)

        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_r  = np.where(sum_r  > 0, -self.kT * np.log(sum_r),  np.nan)
            pmf_th = np.where(sum_th > 0, -self.kT * np.log(sum_th), np.nan)
            pmf_n  = np.where(sum_n  > 0, -self.kT * np.log(sum_n),  np.nan)

        pmf_r  -= np.nanmin(pmf_r)
        pmf_th -= np.nanmin(pmf_th)
        pmf_n  -= np.nanmin(pmf_n)

        return pmf_r, pmf_th, pmf_n

    def _eval_on_wham_grid(self, nn, norm):
        """
        Evaluate *nn* at every point on the reference replicate's WHAM grid.

        Parameters
        ----------
        nn   : NeuralNetwork or NeuralNetworkTorch
        norm : tuple (mean, std)  normalisation from :meth:`_normalise`

        Returns
        -------
        W : ndarray, shape (n_r, n_theta, n_cat)
            NN-predicted PMF values (kJ/mol) on the reference replicate's grid.
        """
        p = self.pmf3d
        R, TH, N = np.meshgrid(
            p.r_centers, p.theta_centers, p.cation_centers, indexing='ij'
        )
        X  = np.column_stack([R.ravel(), TH.ravel(), N.ravel()])
        Xn, _ = self._normalise(X, norm)
        return nn.predict(Xn).reshape(R.shape)

    def _marginals_from_nn(self, approach, mask_unvisited=True):
        """
        Compute 1-D marginals W(r), W(θ), W(n_cat) from the trained NN.

        The occupancy mask uses the mean WHAM PMF (finite wherever *any*
        replicate has data).

        Parameters
        ----------
        approach       : 'a' or 'b'
        mask_unvisited : bool

        Returns
        -------
        pmf_r  : ndarray, shape (n_r,)
            Radial marginal W(r) (kJ/mol), min-shifted to 0.
        pmf_th : ndarray, shape (n_theta,)
            Angular marginal W(θ) (kJ/mol), min-shifted to 0.
        pmf_n  : ndarray, shape (n_cat,)
            Cation-count marginal W(n) (kJ/mol), min-shifted to 0.
        """
        nn = self.nn_a  if approach == 'a' else self.nn_b
        nm = self.norm_a if approach == 'a' else self.norm_b
        W3d = self._eval_on_wham_grid(nn, nm)

        if mask_unvisited:
            mean_pmf, _ = self.mean_wham_pmf()
            W3d = np.where(np.isfinite(mean_pmf), W3d, np.nan)

        log_sum_r  = _logsumexp(-self.beta * W3d, axis=(1, 2))
        log_sum_th = _logsumexp(-self.beta * W3d, axis=(0, 2))
        log_sum_n  = _logsumexp(-self.beta * W3d, axis=(0, 1))

        pmf_r  = -self.kT * log_sum_r;  pmf_r  -= np.nanmin(pmf_r)
        pmf_th = -self.kT * log_sum_th; pmf_th -= np.nanmin(pmf_th)
        pmf_n  = -self.kT * log_sum_n;  pmf_n  -= np.nanmin(pmf_n)
        return pmf_r, pmf_th, pmf_n

    def _compute_frame_weights_single(self, pmf3d, stride=1):
        """
        Per-frame WHAM weights for one replicate's trajectory.

        Analogous to :meth:`ClayPMFNeural._compute_frame_weights` but
        takes the pmf3d object explicitly.

        Returns
        -------
        r_all, theta_all, ncat_all : ndarray  (N_frames,)
        w_all                      : ndarray  (N_frames,)   sums to 1
        """
        p = pmf3d
        if p.f is None:
            raise RuntimeError(
                "pmf3d.f is None — call run_wham_3d() first."
            )
        if any(x is None for x in (p.z_data, p.theta_data, p.ncat_data)):
            raise RuntimeError(
                "Raw trajectory data not available — "
                "call pmf3d.load_trajectory_data() (or equivalent) first."
            )

        r0_list = []
        for c1, c2 in p.window_centers:
            r0_list.append(abs(float(c1)))
            r0_list.append(abs(float(c2)))
        r0    = np.array(r0_list)
        f     = p.f.copy()
        N_sw  = p.n_snapshots.astype(float)
        k_spr = float(p.k)
        beta  = self.beta

        r_list, th_list, nc_list = [], [], []
        for (z1, z2), (th1, th2), (nc1, nc2) in zip(
            p.z_data, p.theta_data, p.ncat_data
        ):
            for z, th, nc in ((z1, th1, nc1), (z2, th2, nc2)):
                if stride > 1:
                    z = z[::stride]; th = th[::stride]; nc = nc[::stride]
                r_list.append(np.abs(z).astype(float))
                th_list.append(th.astype(float))
                nc_list.append(nc.astype(float))

        r_all     = np.concatenate(r_list)
        theta_all = np.concatenate(th_list)
        ncat_all  = np.concatenate(nc_list)
        Nf        = len(r_all)

        # Sanity-check consistency; these must all equal 2 * n_windows.
        n_windows_r0 = len(r0)
        n_windows_f  = len(f)
        if n_windows_r0 != n_windows_f:
            raise RuntimeError(
                f"_compute_frame_weights_single: inconsistent window counts — "
                f"len(r0)={n_windows_r0} (from window_centers, {n_windows_r0 // 2} windows × 2) != "
                f"len(f)={n_windows_f} (from WHAM, {n_windows_f // 2} windows × 2). "
                "This usually means the WHAM cache was computed with a different "
                "number of windows than are currently loaded. "
                "Fix: delete the cache file and re-run with force_recompute=True "
                "so ClayPMF3D rebuilds histograms and WHAM from scratch."
            )

        V_mat     = 0.5 * k_spr * (r_all[:, None] - r0[None, :]) ** 2
        log_terms = np.log(N_sw[None, :] + 1e-300) + f[None, :] - beta * V_mat
        log_denom = _logsumexp(log_terms, axis=1)
        log_w     = -log_denom
        log_w    -= _logsumexp(log_w)
        w_all     = np.exp(log_w)

        ess = 1.0 / float(np.sum(w_all ** 2))
        print(f"    {Nf:,} frames  —  ESS = {ess:.0f} ({100*ess/Nf:.1f}%)")
        return r_all, theta_all, ncat_all, w_all

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit_smooth(self, hidden_sizes=None, epochs=1000, lr=1e-3, l2=1e-4,
                   batch_size=512, r_min=None, pmf_max=None,
                   zero_at='bulk', verbose=True, lr_schedule=None, lr_min=1e-6,
                   w1d_target=None, w1d_r=None, lambda_marginal=1.0,
                   val_split=0.0, patience=200, backend='numpy'):
        """
        Approach A: train one NN on pooled WHAM PMF grid data.

        Finite grid cells from all replicates are collected, independently
        bulk-referenced, then concatenated into a single training set.
        This typically gives ~N_replicates × more training points than a
        single-replicate fit.

        Parameters
        ----------
        hidden_sizes : tuple or None   defaults to ``self.hidden_sizes``
        epochs       : int
        lr           : float   Adam learning rate
        l2           : float   L2 weight decay
        batch_size   : int
        r_min        : float or None   exclude training points with r < r_min (nm)
        pmf_max      : float or None   exclude training points with W > pmf_max (kJ/mol)
        zero_at      : 'bulk' or 'min'
        verbose      : bool
        w1d_target   : array-like or None
            Optional 1-D PMF target W(r) [kJ/mol].  Training targets from
            all replicates are shifted so the NN's r-marginal matches this
            curve.  The correction is computed from the Boltzmann-averaged
            mean 3-D PMF across replicates.
        w1d_r        : array-like or None
            r-coordinates [nm] for *w1d_target*.  If None, *w1d_target* must
            be aligned to the reference replicate's r_centers.
        lambda_marginal : float
            Blending factor for the marginal correction (default 1.0).

        Returns
        -------
        losses : list of float   per-epoch MSE (kJ/mol)²
        """
        if hidden_sizes is None:
            hidden_sizes = self.hidden_sizes

        print(f"=== Ensemble Approach A  ({self.n_replicates} replicates) ===")
        X, y = self._pool_grid_inputs_a(
            r_min=r_min, pmf_max=pmf_max, zero_at=zero_at
        )
        print(f"  Total training points: {len(y):,}"
              + (f"  (r≥{r_min})"   if r_min   else "")
              + (f"  (W≤{pmf_max})" if pmf_max else ""))

        # --- Optional marginal target correction ---
        if w1d_target is not None:
            W3d_ref, _ = self.mean_wham_pmf(zero_at=zero_at)
            y = _apply_marginal_correction(
                X, y, W3d=W3d_ref, r_centers=self.pmf3d.r_centers,
                kT=self.kT, beta=self.beta,
                w1d_target=w1d_target, w1d_r=w1d_r,
                lambda_marginal=lambda_marginal,
            )

        X_norm, self.norm_a = self._normalise(X)
        self.nn_a   = _make_nn(3, hidden_sizes, lr=lr, l2=l2,
                               seed=self.seed, backend=backend)
        self.losses_a = self.nn_a.train(
            X_norm, y, epochs=epochs, batch_size=batch_size,
            verbose=verbose, print_every=max(1, epochs // 10),
            lr_schedule=lr_schedule, lr_min=lr_min,
            val_split=val_split, patience=patience,
        )
        print(f"  Final loss: {self.losses_a[-1]:.4e} (kJ/mol)²")
        self.r_min_a = r_min
        return self.losses_a

    def fit_reweighted(self, hidden_sizes=None, epochs=1000, lr=1e-3, l2=1e-4,
                       batch_size=1024, n_r_bins=80, n_theta_bins=36,
                       stride=1, r_min=None, pmf_max=None,
                       zero_at='bulk', verbose=True, lr_schedule=None, lr_min=1e-6,
                       w1d_target=None, w1d_r=None, lambda_marginal=1.0,
                       val_split=0.0, patience=200, backend='numpy',
                       n_workers=1):
        """
        Approach B: pool per-frame WHAM weights from all replicates →
        fine-grid histogram → NN.

        For each replicate the WHAM weights are computed independently
        (using that replicate's converged free energies ``pmf3d.f``).
        Frames from all replicates are then pooled with equal per-replicate
        weight before building one fine-resolution unbiased histogram.

        Requires ``load_trajectory_data()`` (or equivalent) to have been
        called on each replicate so that ``z_data``, ``theta_data``, and
        ``ncat_data`` are populated.

        Parameters
        ----------
        hidden_sizes  : tuple or None
        epochs, lr, l2, batch_size : as in :meth:`fit_smooth`
        n_r_bins      : int   fine r-grid resolution (default 80)
        n_theta_bins  : int   fine θ-grid resolution (default 36)
        stride        : int   subsample raw frames (1 = use every frame)
        r_min         : float or None
        pmf_max       : float or None
        zero_at       : 'bulk' or 'min'
        verbose       : bool
        w1d_target    : array-like or None
            Optional 1-D PMF target W(r) [kJ/mol].  Training targets are
            shifted so the NN's r-marginal matches this curve.  Must be on
            the same energy scale as the pooled fine-grid PMF (W_fine).
        w1d_r         : array-like or None
            r-coordinates [nm] for *w1d_target*.  If None, *w1d_target* must
            be aligned to the fine-grid r bin-centres (r_ctrs).
        lambda_marginal : float
            Blending factor for the marginal correction (default 1.0).
        n_workers : int
            Number of threads used to compute per-replicate WHAM weights
            in parallel (default 1 = sequential).  Set to -1 to use
            ``min(8, n_replicates)`` threads automatically.
            Uses :class:`~concurrent.futures.ThreadPoolExecutor` — no
            pickling; numpy releases the GIL for heavy array operations.

        Returns
        -------
        losses : list of float
        """
        if hidden_sizes is None:
            hidden_sizes = self.hidden_sizes

        print(f"=== Ensemble Approach B  ({self.n_replicates} replicates) ===")

        # Step 1: per-frame WHAM weights per replicate
        _n_workers = (
            min(8, self.n_replicates) if n_workers == -1
            else max(1, int(n_workers))
        )
        print(f"Step 1: Computing per-frame WHAM weights…"
              f"  (n_workers={_n_workers})")
        with ThreadPoolExecutor(max_workers=_n_workers) as ex:
            futures = [
                ex.submit(self._compute_frame_weights_single, p3d, stride)
                for p3d in self.pmf3d_list
            ]
            _results = [f.result() for f in futures]

        r_pool, th_pool, nc_pool, w_pool = [], [], [], []
        for r_k, th_k, nc_k, w_k in _results:
            # Give each replicate equal weight: scale by 1/n_replicates
            w_pool.append(w_k / self.n_replicates)
            r_pool.append(r_k); th_pool.append(th_k); nc_pool.append(nc_k)

        r_all     = np.concatenate(r_pool)
        theta_all = np.concatenate(th_pool)
        ncat_all  = np.concatenate(nc_pool)
        w_all     = np.concatenate(w_pool)   # sums to 1
        print(f"  Pooled frames: {len(r_all):,}")

        # Step 2: weighted fine-grid histogram on reference bin edges
        print("Step 2: Building pooled fine-grid PMF via weighted histogram…")
        p = self.pmf3d   # reference replicate

        r_edges  = np.linspace(p.r_bins[0],     p.r_bins[-1],     n_r_bins    + 1)
        th_edges = np.linspace(p.theta_bins[0], p.theta_bins[-1], n_theta_bins + 1)
        n_edges  = p.cation_bins.copy()

        r_ctrs  = 0.5 * (r_edges[:-1] + r_edges[1:])
        th_ctrs = 0.5 * (th_edges[:-1] + th_edges[1:])
        n_ctrs  = p.cation_centers.copy()

        dr  = float(r_edges[1]  - r_edges[0])
        dth = float(th_edges[1] - th_edges[0])
        dn  = 1.0

        ncat_clip = np.clip(ncat_all, p.cation_range[0], p.cation_range[1])

        H_w, _ = np.histogramdd(
            np.column_stack([r_all, theta_all, ncat_clip]),
            bins=[r_edges, th_edges, n_edges],
            weights=w_all,
            density=False,
        )

        P_fine = H_w / (dr * dth * dn)
        with np.errstate(divide='ignore', invalid='ignore'):
            W_fine = np.where(P_fine > 0,
                              -self.kT * np.log(P_fine),
                              np.nan)

        # Bulk reference on fine grid
        if getattr(p, 'bulk_correction_enabled', False):
            _bulk_frac = getattr(p, 'bulk_fraction', 0.2)
            _n_bulk    = max(1, int(_bulk_frac * n_r_bins))
            _W_r       = np.nanmedian(W_fine, axis=(1, 2))
            _shift     = float(np.nanmedian(_W_r[:_n_bulk]))
            W_fine    -= _shift if np.isfinite(_shift) else np.nanmin(W_fine)
        else:
            W_fine -= np.nanmin(W_fine)

        self.W_fine     = W_fine
        self.r_fine     = r_ctrs
        self.theta_fine = th_ctrs
        self.n_fine     = n_ctrs

        finite_frac = 100.0 * np.sum(np.isfinite(W_fine)) / W_fine.size
        print(f"  Fine grid: r({n_r_bins}) × θ({n_theta_bins}) × "
              f"n_cat({p.n_cation_bins})  ({finite_frac:.0f}% populated)")
        print(f"  W range: [{np.nanmin(W_fine):.2f}, "
              f"{np.nanmax(W_fine):.2f}] kJ/mol")

        # Step 3: train NN on fine-grid PMF
        print("Step 3: Training NN on fine-grid PMF…")
        Rf, THF, NF = np.meshgrid(r_ctrs, th_ctrs, n_ctrs, indexing='ij')
        X_b = np.column_stack([Rf.ravel(), THF.ravel(), NF.ravel()])
        y_b = W_fine.ravel()
        mask = np.isfinite(y_b)
        X_b, y_b = X_b[mask], y_b[mask]
        if r_min is not None:
            keep = X_b[:, 0] >= r_min;  X_b, y_b = X_b[keep], y_b[keep]
        if pmf_max is not None:
            keep = y_b <= pmf_max;       X_b, y_b = X_b[keep], y_b[keep]
        print(f"  Training points: {len(y_b):,}"
              + (f"  (r≥{r_min})"   if r_min   else "")
              + (f"  (W≤{pmf_max})" if pmf_max else ""))

        # --- Optional marginal target correction ---
        if w1d_target is not None:
            y_b = _apply_marginal_correction(
                X_b, y_b, W3d=W_fine, r_centers=r_ctrs,
                kT=self.kT, beta=self.beta,
                w1d_target=w1d_target, w1d_r=w1d_r,
                lambda_marginal=lambda_marginal,
            )

        X_bn, self.norm_b = self._normalise(X_b)
        self.nn_b   = _make_nn(3, hidden_sizes, lr=lr, l2=l2,
                               seed=self.seed, backend=backend)
        self.losses_b = self.nn_b.train(
            X_bn, y_b, epochs=epochs, batch_size=batch_size,
            verbose=verbose, print_every=max(1, epochs // 10),
            lr_schedule=lr_schedule, lr_min=lr_min,
            val_split=val_split, patience=patience,
        )
        print(f"  Final loss: {self.losses_b[-1]:.4e} (kJ/mol)²")
        self.r_min_b = r_min
        return self.losses_b

    def fit_smooth_per_replicate(self, hidden_sizes=None, epochs=1000, lr=1e-3,
                                  l2=1e-4, batch_size=256, r_min=None, pmf_max=None,
                                  zero_at='bulk', verbose=True, lr_schedule=None,
                                  lr_min=1e-6, val_split=0.0, patience=200,
                                  backend='numpy'):
        """
        Deep-ensemble approach: train one independent NN *per replicate*.

        Unlike :meth:`fit_smooth` (which pools all replicates into a single
        NN), this method trains ``n_replicates`` separate NNs, each on its
        own WHAM grid.  Different trajectory snapshots → genuine inter-
        replicate physical variance, which :meth:`predict_with_uncertainty`
        exposes as a principled uncertainty estimate.

        This is the *deep ensembles* approach (Lakshminarayanan 2017) and
        empirically outperforms MC Dropout for uncertainty quality,
        especially in the sparse high-n_cat bins where dropout is
        overconfident.

        Results are stored in ``self._nn_per_replicate`` and
        ``self._norm_per_replicate``.  The pooled ``self.nn_a`` is **not**
        affected — both approaches can coexist.

        Parameters
        ----------
        hidden_sizes : tuple or None   defaults to ``self.hidden_sizes``
        epochs       : int
        lr           : float   Adam learning rate
        l2           : float   L2 weight decay
        batch_size   : int
        r_min        : float or None
        pmf_max      : float or None
        zero_at      : 'bulk' or 'min'
        verbose      : bool
        lr_schedule  : str or None   e.g. 'cosine'
        lr_min       : float
        val_split    : float   fraction held out for early-stopping
        patience     : int     early-stopping patience (epochs)

        Returns
        -------
        losses_list : list of lists   per-epoch MSE for each replicate
        """
        if hidden_sizes is None:
            hidden_sizes = self.hidden_sizes

        print(f"=== Per-Replicate Deep Ensemble  ({self.n_replicates} NNs) ===")
        self._nn_per_replicate   = []
        self._norm_per_replicate = []
        self._losses_per_rep     = []

        for k, pmf3d in enumerate(self.pmf3d_list):
            print(f"\n--- Replicate {k + 1}/{self.n_replicates} ---")
            p   = pmf3d
            R, TH, N = np.meshgrid(
                p.r_centers, p.theta_centers, p.cation_centers, indexing='ij'
            )
            X  = np.column_stack([R.ravel(), TH.ravel(), N.ravel()])
            sh = self._bulk_shift_single(p, zero_at)
            y  = (p.pmf_3d - sh).ravel()
            ok = np.isfinite(y)
            X, y = X[ok], y[ok]
            if r_min is not None:
                keep = X[:, 0] >= r_min;  X, y = X[keep], y[keep]
            if pmf_max is not None:
                keep = y <= pmf_max;       X, y = X[keep], y[keep]
            print(f"  {len(y):,} training points  (bulk shift = {sh:.3f} kJ/mol)")

            # Different seed per replicate → diverse initialisation
            seed_k = None if self.seed is None else self.seed + k
            nn     = _make_nn(3, hidden_sizes, lr=lr, l2=l2,
                              seed=seed_k, backend=backend)
            Xn, norm = self._normalise(X)
            losses = nn.train(
                Xn, y, epochs=epochs, batch_size=batch_size,
                verbose=verbose, print_every=max(1, epochs // 10),
                lr_schedule=lr_schedule, lr_min=lr_min,
                val_split=val_split, patience=patience,
            )
            print(f"  Final loss: {losses[-1]:.4e} (kJ/mol)²")
            self._nn_per_replicate.append(nn)
            self._norm_per_replicate.append(norm)
            self._losses_per_rep.append(losses)

        rms_vals = [float(ls[-1]) ** 0.5 for ls in self._losses_per_rep]
        print(f"\nPer-replicate training complete.")
        print(f"  RMS per replicate: "
              + ", ".join(f"{v:.2f}" for v in rms_vals)
              + " kJ/mol")
        print(f"  Mean RMS: {sum(rms_vals)/len(rms_vals):.2f} kJ/mol")
        print("  Use predict_with_uncertainty() for mean ± std.")
        return self._losses_per_rep

    # -----------------------------------------------------------------------
    # Prediction  (identical interface to ClayPMFNeural)
    # -----------------------------------------------------------------------

    def predict(self, r, theta, n_cat):
        """
        Approach A: evaluate W(r, θ, n_cat) from the ensemble NN (kJ/mol).

        Parameters
        ----------
        r, theta, n_cat : array-like  (broadcast-compatible)

        Returns
        -------
        W : ndarray  (kJ/mol)
        """
        if self.nn_a is None:
            raise RuntimeError("Call fit_smooth() first.")
        return self._predict_from(self.nn_a, self.norm_a, r, theta, n_cat)

    def predict_b(self, r, theta, n_cat):
        """Approach B: evaluate W(r, θ, n_cat) from the ensemble NN (kJ/mol)."""
        if self.nn_b is None:
            raise RuntimeError("Call fit_reweighted() first.")
        return self._predict_from(self.nn_b, self.norm_b, r, theta, n_cat)

    def predict_both(self, r, theta, n_cat):
        """
        Evaluate both NN-A and NN-B at the same coordinates.

        Returns
        -------
        W_a, W_b : ndarray or None  (kJ/mol)
        """
        W_a = self.predict(r,   theta, n_cat) if self.nn_a else None
        W_b = self.predict_b(r, theta, n_cat) if self.nn_b else None
        return W_a, W_b

    def predict_with_uncertainty(self, r, theta, n_cat):
        """
        Deep-ensemble uncertainty estimate for W(r, θ, n_cat).

        Requires :meth:`fit_smooth_per_replicate` to have been called first.
        Each of the ``n_replicates`` NNs was trained on a different
        trajectory → the spread across predictions captures genuine
        physical (inter-replicate) variance rather than model variance.

        Parameters
        ----------
        r, theta, n_cat : array-like  (broadcast-compatible scalars or arrays)

        Returns
        -------
        mean : ndarray  (kJ/mol)   pointwise mean across replicates
        std  : ndarray  (kJ/mol)   pointwise std  across replicates

        Example
        -------
        ::

            r     = np.linspace(0.2, 0.9, 200)
            theta = np.full(200, np.pi / 4)
            n_cat = np.zeros(200)
            mean, std = ensemble.predict_with_uncertainty(r, theta, n_cat)
            plt.fill_between(r, mean - std, mean + std, alpha=0.3)
            plt.plot(r, mean)
        """
        if not self._nn_per_replicate:
            raise RuntimeError(
                "Per-replicate NNs not found.  "
                "Call fit_smooth_per_replicate() first."
            )
        preds = np.array([
            self._predict_from(nn, norm, r, theta, n_cat)
            for nn, norm in zip(self._nn_per_replicate,
                                self._norm_per_replicate)
        ])  # shape: (n_replicates, N)
        return preds.mean(axis=0), preds.std(axis=0)

    # -----------------------------------------------------------------------
    # Uncertainty map
    # -----------------------------------------------------------------------

    def uncertainty_map(self, zero_at='bulk'):
        """
        Inter-replicate standard deviation of the PMF at every grid point.

        Wraps :meth:`mean_wham_pmf` for convenience.

        Parameters
        ----------
        zero_at : 'bulk' or 'min'

        Returns
        -------
        mean_pmf : ndarray  (n_r, n_theta, n_cat)
        std_pmf  : ndarray  (n_r, n_theta, n_cat)
        """
        return self.mean_wham_pmf(zero_at=zero_at)

    # -----------------------------------------------------------------------
    # Hyperparameter tuning
    # -----------------------------------------------------------------------

    def tune_hyperparameters(self, approach='a', param_grid=None, cv=5,
                             cv_epochs=300, batch_size=512,
                             cv_patience=0,
                             r_min=None, pmf_max=None, zero_at='bulk',
                             backend='numpy', verbose=True):
        """
        k-fold cross-validation grid search over hidden_sizes, lr, l2.

        Only Approach A (pooled grid smoother) is supported.  Approach B
        requires expensive per-frame weight recomputation on each fold; use
        ``fit_reweighted(val_split=..., patience=...)`` to tune that instead.

        Parameters
        ----------
        approach     : str   only 'a' is currently supported
        param_grid   : dict or None
            Keys: 'hidden_sizes', 'lr', 'l2'.  If None a default 36-combo
            grid is used (4 architectures × 3 lr × 3 l2).
        cv           : int   number of CV folds (default 5)
        cv_epochs    : int   max epochs per fold (default 300)
        batch_size   : int
        cv_patience  : int   early-stop patience on inner val loss
            (0 = disabled; >0 uses val_split=0.1 inside each fold's train)
        r_min        : float or None   same filtering as fit_smooth
        pmf_max      : float or None   same filtering as fit_smooth
        zero_at      : 'bulk' or 'min'
        backend      : 'numpy' or 'torch'
        verbose      : bool   print per-combo progress (one line per combo)

        Returns
        -------
        results : list of dict, sorted by mean val MSE ascending (best first)
            Each dict has keys: 'hidden_sizes', 'lr', 'l2',
            'val_mse' (mean over folds), 'fold_mse' (list per fold).
        """
        if approach != 'a':
            raise NotImplementedError(
                "tune_hyperparameters only supports approach='a'. "
                "For Approach B use fit_reweighted(val_split=..., patience=...)."
            )
        if param_grid is None:
            param_grid = {
                'hidden_sizes': [(32,), (64,), (64, 32), (64, 64, 32)],
                'lr':           [1e-4, 3e-4, 1e-3],
                'l2':           [0.0,  1e-5,  1e-4],
            }

        # Build training data once (same pipeline as fit_smooth)
        X, y = self._pool_grid_inputs_a(
            r_min=r_min, pmf_max=pmf_max, zero_at=zero_at
        )
        N = len(y)
        if N < cv:
            raise ValueError(
                f"Only {N} training points — cannot split into {cv} folds."
            )

        # Normalise on full data so all folds share the same input scale
        X_norm, _ = self._normalise(X)

        # k-fold index split
        rng     = np.random.default_rng(self.seed if self.seed is not None else 0)
        indices = rng.permutation(N)
        folds   = [idx.tolist() for idx in np.array_split(indices, cv)]

        keys   = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        print(f"Grid search: {len(combos)} combos × {cv} folds = "
              f"{len(combos) * cv} fits  ({N:,} points, cv_epochs={cv_epochs})")

        _inner_val = 0.1 if cv_patience > 0 else 0.0

        results = []
        for i, values in enumerate(combos):
            params = dict(zip(keys, values))
            hs = params.get('hidden_sizes', self.hidden_sizes)
            lr = params.get('lr',           1e-3)
            l2 = params.get('l2',           1e-4)

            fold_mse = []
            for val_idx in folds:
                val_mask = np.zeros(N, dtype=bool)
                val_mask[val_idx] = True
                X_tr, y_tr = X_norm[~val_mask], y[~val_mask]
                X_va, y_va = X_norm[val_mask],  y[val_mask]

                nn = _make_nn(3, hs, lr=lr, l2=l2,
                              seed=self.seed, backend=backend, verbose=False)
                nn.train(X_tr, y_tr, epochs=cv_epochs, batch_size=batch_size,
                         val_split=_inner_val, patience=cv_patience,
                         verbose=False)
                mse = float(np.mean((nn.predict(X_va) - y_va) ** 2))
                fold_mse.append(mse)

            mean_mse = float(np.mean(fold_mse))
            std_mse  = float(np.std(fold_mse))
            if verbose:
                print(f"  [{i+1:>2}/{len(combos)}] "
                      f"hidden={hs} lr={lr:.0e} l2={l2:.0e}  "
                      f"val_MSE={mean_mse:.4e} ± {std_mse:.2e}")

            results.append({
                'hidden_sizes': hs,
                'lr':           lr,
                'l2':           l2,
                'val_mse':      mean_mse,
                'fold_mse':     fold_mse,
            })

        results.sort(key=lambda d: d['val_mse'])
        best = results[0]
        print(f"\nBest: hidden={best['hidden_sizes']}  "
              f"lr={best['lr']:.0e}  l2={best['l2']:.0e}  "
              f"val_MSE={best['val_mse']:.4e} (kJ/mol)²")
        return results

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, path, include_metadata=True):
        """
        Save trained NN weights and normalisations to a .npz file.

        The file is compatible with :meth:`load` (not with
        ``ClayPMFNeural.load``).

        Parameters
        ----------
        path             : str    e.g. ``'ensemble_nn.npz'``
        include_metadata : bool   if True (default), store version, timestamp
                                  and git hash in the file for provenance.
        """
        data = {
            'hidden_sizes': np.array(self.hidden_sizes),
            'seed':         np.array(self.seed if self.seed is not None else -1),
            'T':            np.array(self.T),
            'n_replicates': np.array(self.n_replicates),
            'r_min_a':      np.array(float('nan') if self.r_min_a is None else self.r_min_a),
            'r_min_b':      np.array(float('nan') if self.r_min_b is None else self.r_min_b),
        }
        if include_metadata:
            data['_version']   = np.array([_CLAYPMFNEURAL_VERSION], dtype='U')
            data['_timestamp'] = np.array([datetime.now().isoformat()], dtype='U')
            data['_git_hash']  = np.array([_git_hash()], dtype='U')
        for tag, nn, norm in (('a', self.nn_a, self.norm_a),
                               ('b', self.nn_b, self.norm_b)):
            if nn is not None:
                for i, (W, b) in enumerate(zip(nn.W, nn.b)):
                    data[f'{tag}_W{i}'] = W
                    data[f'{tag}_b{i}'] = b
                data[f'{tag}_norm_mean'] = norm[0]
                data[f'{tag}_norm_std']  = norm[1]
                data[f'{tag}_n_layers']  = np.array(len(nn.W))
                if tag == 'b' and self.W_fine is not None:
                    data['W_fine']     = self.W_fine
                    data['r_fine']     = self.r_fine
                    data['theta_fine'] = self.theta_fine
                    data['n_fine']     = self.n_fine
        # Per-replicate NNs (deep ensemble)
        data['n_per_rep'] = np.array(len(self._nn_per_replicate))
        for k, (nn, norm) in enumerate(
            zip(self._nn_per_replicate, self._norm_per_replicate)
        ):
            for i, (W, b) in enumerate(zip(nn.W, nn.b)):
                data[f'rep{k}_W{i}'] = W
                data[f'rep{k}_b{i}'] = b
            data[f'rep{k}_norm_mean'] = norm[0]
            data[f'rep{k}_norm_std']  = norm[1]
            data[f'rep{k}_n_layers']  = np.array(len(nn.W))
        # Training loss histories (empty array = not fitted)
        data['losses_a'] = np.array(self.losses_a) if self.losses_a is not None else np.array([])
        data['losses_b'] = np.array(self.losses_b) if self.losses_b is not None else np.array([])
        np.savez_compressed(path, **data)
        print(f"Saved ensemble NN weights to {path}")

    @classmethod
    def load(cls, path, pmf3d_list):
        """
        Load saved ensemble NN weights from a .npz file.

        Parameters
        ----------
        path        : str   .npz file written by :meth:`save`
        pmf3d_list  : list of ClayPMF3D

        Returns
        -------
        ClayPMFNeuralEnsemble with ``nn_a`` and/or ``nn_b`` populated
        """
        d = np.load(path, allow_pickle=False)
        for key, label in (('_version', 'version'), ('_timestamp', 'saved'),
                           ('_git_hash', 'git')):
            if key in d:
                print(f"  [{label}] {str(d[key][0])}")
        hidden_sizes = tuple(int(x) for x in d['hidden_sizes'])
        seed = int(d['seed']); seed = None if seed < 0 else seed
        T    = float(d['T'])

        obj = cls(pmf3d_list, hidden_sizes=hidden_sizes,
                  temperature=T, seed=seed)
        # Restore training r_min (NaN sentinel means None / not set)
        for attr, key in (('r_min_a', 'r_min_a'), ('r_min_b', 'r_min_b')):
            if key in d:
                val = float(d[key])
                setattr(obj, attr, None if np.isnan(val) else val)

        for tag in ('a', 'b'):
            if f'{tag}_n_layers' not in d:
                continue
            n_layers = int(d[f'{tag}_n_layers'])
            W_list   = [d[f'{tag}_W{i}'] for i in range(n_layers)]
            b_list   = [d[f'{tag}_b{i}'] for i in range(n_layers)]
            norm     = (d[f'{tag}_norm_mean'], d[f'{tag}_norm_std'])
            input_dim = W_list[0].shape[1]
            hs = [W_list[i].shape[0] for i in range(n_layers - 1)]
            nn = NeuralNetwork(input_dim, hs, seed=seed)
            nn.W = W_list; nn.b = b_list
            if tag == 'a':
                obj.nn_a, obj.norm_a = nn, norm
            else:
                obj.nn_b, obj.norm_b = nn, norm
                if 'W_fine' in d:
                    obj.W_fine     = d['W_fine']
                    obj.r_fine     = d['r_fine']
                    obj.theta_fine = d['theta_fine']
                    obj.n_fine     = d['n_fine']

        # Per-replicate NNs (deep ensemble)
        n_per_rep = int(d['n_per_rep']) if 'n_per_rep' in d else 0
        for k in range(n_per_rep):
            key = f'rep{k}_n_layers'
            if key not in d:
                continue
            n_layers  = int(d[key])
            W_list    = [d[f'rep{k}_W{i}'] for i in range(n_layers)]
            b_list    = [d[f'rep{k}_b{i}'] for i in range(n_layers)]
            norm      = (d[f'rep{k}_norm_mean'], d[f'rep{k}_norm_std'])
            input_dim = W_list[0].shape[1]
            hs        = [W_list[i].shape[0] for i in range(n_layers - 1)]
            seed_k    = None if seed is None else seed + k
            nn        = NeuralNetwork(input_dim, hs, seed=seed_k)
            nn.W = W_list; nn.b = b_list
            obj._nn_per_replicate.append(nn)
            obj._norm_per_replicate.append(norm)

        # Restore training loss histories if present
        for attr, key in (('losses_a', 'losses_a'), ('losses_b', 'losses_b')):
            if key in d and len(d[key]) > 0:
                setattr(obj, attr, d[key].tolist())
        print(f"Loaded ensemble NN weights from {path}")
        return obj

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------

    def plot_losses(self, figsize=(11, 4)):
        """
        Side-by-side training loss curves for Approach A and B.

        Delegates to ``ClayPMFPlotter.plot_losses``.

        Returns
        -------
        fig : matplotlib Figure
        """
        from ClayPMFPlotter import ClayPMFPlotter
        plotter = ClayPMFPlotter(ensemble=self)
        return plotter.plot_losses(figsize=figsize)

    def plot_comparison_slice(self, n_cat_val=1, figsize=None,
                               vmax=None, r_min=None, cmap='viridis',
                               mask_unvisited=True, n_levels=40,
                               zero_at='bulk', show_uncertainty=True):
        """
        W(r, θ) at fixed n_cat: mean WHAM, NN-A, NN-B, and (optionally) σ.

        Delegates to ``ClayPMFPlotter.plot_comparison_slice``.

        Returns
        -------
        fig : matplotlib Figure or None
        """
        from ClayPMFPlotter import ClayPMFPlotter
        plotter = ClayPMFPlotter(ensemble=self)
        return plotter.plot_comparison_slice(
            n_cat_val=n_cat_val, figsize=figsize, vmax=vmax, r_min=r_min,
            cmap=cmap, mask_unvisited=mask_unvisited, n_levels=n_levels,
            zero_at=zero_at, show_uncertainty=show_uncertainty,
        )

    def plot_1d_marginals(self, figsize=(15, 4), mask_unvisited=True,
                           zero_at='bulk'):
        """
        W(r), W(θ), W(n_cat): mean WHAM ± 1 σ versus NN-A and NN-B.

        Delegates to ``ClayPMFPlotter.plot_3d_ensemble_marginals``.

        Parameters
        ----------
        figsize        : tuple
        mask_unvisited : bool   passed to :meth:`_marginals_from_nn`
        zero_at        : 'bulk' or 'min'

        Returns
        -------
        fig : matplotlib Figure
        axes : ndarray of Axes
        """
        from ClayPMFPlotter import ClayPMFPlotter
        plotter = ClayPMFPlotter(ensemble=self)
        fig = plotter.plot_3d_ensemble_marginals(
            figsize=figsize,
            zero_at=zero_at,
            mask_unvisited=mask_unvisited,
        )
        return fig

    def plot_uncertainty_map(self, n_cat_val=1, figsize=(7, 5),
                              cmap='Reds', zero_at='bulk', n_levels=30):
        """Delegates to ``ClayPMFPlotter.plot_uncertainty_map``."""
        from ClayPMFPlotter import ClayPMFPlotter
        plotter = ClayPMFPlotter(ensemble=self)
        return plotter.plot_uncertainty_map(
            n_cat_val=n_cat_val, figsize=figsize,
            cmap=cmap, zero_at=zero_at, n_levels=n_levels,
        )
