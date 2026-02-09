# -*- coding: utf-8 -*-
"""
libtorch.py — SymPy-to-Torch conversion and torchquad integration.

Part of libphysics.

Two-function design for performance:
    1. torchify()            — expensive SymPy work done ONCE (change-of-vars + lambdify)
    2. torchquad_integrate() — cheap torch-only integration, called MANY times

Usage
=====
        from libphysics.libtorch import torchify, torchquad_integrate

        # Step 1: build once (slow: SymPy subs + lambdify, ~200-500 ms)
        wigner = torchify(
                integrand,
                variables=[y_A, y_B],
                limits=[(y_A, -oo, oo), (y_B, -oo, oo)],
                params=[x_A, p_A, x_B, p_B],
        )

        # Step 2: integrate many times (fast: pure torch, ~3-8 ms each)
        re, im = torchquad_integrate(wigner, params_values=[1, 1, 1, 1], N=121**2)
"""
from dataclasses import dataclass
from typing import List, Callable

import numpy as np
import torch
from sympy import lambdify, Symbol, tan, cos, pi, oo, Integer


# ---------------------------------------------------------------------------
# Default torch module mapping for lambdify
# ---------------------------------------------------------------------------
def _default_torch_modules():
    return {
        # Trigonometric
        "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
        "asin": torch.asin, "acos": torch.acos, "atan": torch.atan,
        "atan2": torch.atan2,
        # Hyperbolic
        "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
        "asinh": torch.asinh, "acosh": torch.acosh, "atanh": torch.atanh,
        # Exponentials / logs
        "exp": torch.exp, "log": torch.log, "ln": torch.log,
        "log10": torch.log10, "log2": torch.log2,
        # Roots / powers
        "sqrt": torch.sqrt, "Pow": torch.pow,
        # Misc
        "Abs": torch.abs, "sign": torch.sign,
        "floor": torch.floor, "ceiling": torch.ceil,
        "Min": torch.minimum, "Max": torch.maximum,
        # Piecewise / heaviside
        "Heaviside": lambda x: torch.heaviside(x, torch.zeros_like(x)),
        # Constants
        "pi": getattr(torch, "pi", float(np.pi)),
        "E": float(np.e),
    }


# ---------------------------------------------------------------------------
# TorchExpr — container returned by torchify when limits are provided
# ---------------------------------------------------------------------------
@dataclass
class TorchExpr:
    """Pre-built torch function + finite integration domain."""
    func: Callable                # f(new_var_0, …, new_var_n, param_0, …, param_m)
    domain: List[List[float]]     # finite box for torchquad
    dim: int                      # number of integration variables
    n_params: int                 # number of extra parameters


# ---------------------------------------------------------------------------
# torchify  —  expensive work done ONCE
# ---------------------------------------------------------------------------
def torchify(expr, variables, limits=None, params=None, eps=1e-8, modules=None):
    """
    Convert a SymPy expression into a torch-compatible function via lambdify.

    If *limits* are provided, performs change-of-variables for infinite /
    semi-infinite limits at the **SymPy level** (symbolic, done once) and
    multiplies by the analytic Jacobian.  Returns a ``TorchExpr`` ready
    for ``torchquad_integrate``.

    If *limits* are **not** provided, returns a plain callable (simple
    lambdify with torch modules).

    Parameters
    ----------
    expr : sympy.Expr
        Symbolic expression (may be complex-valued).
    variables : list[sympy.Symbol]
        Integration variables (order matters).
    limits : list[tuple] or None
        ``(var, lower, upper)`` for each variable.  ``sympy.oo`` /
        ``-sympy.oo`` trigger automatic change of variables.
    params : list[sympy.Symbol] or None
        Extra symbolic parameters that appear in *expr* but are **not**
        integrated over.  If ``None``, inferred automatically.
    eps : float
        Small cutoff for mapping infinite limits (``±π/2 ∓ eps``).
    modules : list[dict] or None
        Custom lambdify module list.  ``None`` → default torch mapping.

    Returns
    -------
    TorchExpr   if *limits* were given  (use with ``torchquad_integrate``)
    callable    if *limits* were ``None`` (plain torch function)
    """
    if modules is None:
        modules = [_default_torch_modules()]

    variables = list(variables)

    # ------------------------------------------------------------------
    # Simple case: no limits → plain lambdify
    # ------------------------------------------------------------------
    if limits is None:
        return lambdify(tuple(variables), expr, modules=modules)

    # ------------------------------------------------------------------
    # With limits: symbolic change of variables (done once, fast forever)
    # ------------------------------------------------------------------
    limit_map = {lim[0]: (lim[1], lim[2]) for lim in limits}

    new_vars = []
    domain = []
    subs_map = {}
    jacobian = Integer(1)

    for v in variables:
        lower, upper = limit_map[v]

        if lower == -oo and upper == oo:
            # (-∞, ∞) → v = tan(t),  dv = sec²(t) dt
            t_v = Symbol(f"t_{v.name}", real=True)
            subs_map[v] = tan(t_v)
            jacobian *= 1 / cos(t_v) ** 2
            new_vars.append(t_v)
            domain.append([float(-pi / 2 + eps), float(pi / 2 - eps)])

        elif lower != -oo and upper == oo:
            # (a, ∞) → v = a + tan²(t),  dv = 2 tan(t) sec²(t) dt
            t_v = Symbol(f"t_{v.name}", real=True)
            subs_map[v] = lower + tan(t_v) ** 2
            jacobian *= 2 * tan(t_v) / cos(t_v) ** 2
            new_vars.append(t_v)
            domain.append([0.0, float(pi / 2 - eps)])

        elif lower == -oo and upper != oo:
            # (-∞, b) → v = b - tan²(t),  dv = -2 tan(t) sec²(t) dt
            t_v = Symbol(f"t_{v.name}", real=True)
            subs_map[v] = upper - tan(t_v) ** 2
            jacobian *= -2 * tan(t_v) / cos(t_v) ** 2
            new_vars.append(t_v)
            domain.append([0.0, float(pi / 2 - eps)])

        else:
            # finite [a, b]
            new_vars.append(v)
            domain.append([float(lower), float(upper)])

    # Substitute + bake in the Jacobian — all symbolic, done ONCE
    expr_work = expr.subs(subs_map) * jacobian

    # Infer params from remaining free symbols
    if params is None:
        params = sorted(
            list(expr_work.free_symbols - set(new_vars)),
            key=lambda s: s.name,
        )
    else:
        params = list(params)

    # Lambdify: new_vars first, then params
    arglist = tuple(new_vars + params)
    func = lambdify(arglist, expr_work, modules=modules)

    return TorchExpr(func=func, domain=domain, dim=len(new_vars), n_params=len(params))


# ---------------------------------------------------------------------------
# torchquad_integrate  —  cheap work, called MANY times
# ---------------------------------------------------------------------------
def torchquad_integrate(texpr, params_values=None, method=None, N=21):
    """
    Numerically integrate a ``TorchExpr`` using torchquad.

    Parameters
    ----------
    texpr : TorchExpr
        Object returned by ``torchify(..., limits=...)``.
    params_values : list or None
        Numerical values for the parameters (same order as *params* in torchify).
    method : torchquad integrator or None
        Defaults to ``Simpson()``.
    N : int
        Integrator resolution (torchquad's *N* parameter).

    Returns
    -------
    re, im : torch.Tensor
        Real and imaginary parts of the integral.
    """
    from torchquad import Simpson

    if method is None:
        method = Simpson()

    param_vals = list(params_values) if params_values else []

    def f_re(d):
        vals = texpr.func(*[d[:, i] for i in range(d.shape[1])], *param_vals)
        vals = torch.as_tensor(vals, device=d.device)
        return vals.real if torch.is_complex(vals) else vals

    def f_im(d):
        vals = texpr.func(*[d[:, i] for i in range(d.shape[1])], *param_vals)
        vals = torch.as_tensor(vals, device=d.device)
        return vals.imag if torch.is_complex(vals) else torch.zeros_like(vals)

    re = method.integrate(f_re, dim=texpr.dim, N=N, integration_domain=texpr.domain)
    im = method.integrate(f_im, dim=texpr.dim, N=N, integration_domain=texpr.domain)
    return re, im