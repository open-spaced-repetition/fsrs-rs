#!/usr/bin/env python3
"""
Generate analytical gradient Rust code for the FSRS model.

This script uses SymPy to symbolically derive the partial derivatives of the
FSRS loss function with respect to all 21 model parameters, then generates
optimized Rust code using Common Subexpression Elimination (CSE).

The generated code is mathematically equivalent to Burn's dynamic autodiff
backward pass (BPTT), enabling a zero-dependency analytical gradient backend.

Usage:
    python scripts/generate_gradients.py

Output:
    src/analytical_gradients.rs

Mathematical reference:
    All equations are transcribed from src/model.rs and src/training.rs
    in the fsrs-rs repository (FSRS v6, 21 parameters).
"""

import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import sympy as sp
from sympy import (
    Integer,
    Rational,
    Symbol,
    diff,
    exp,
    log,
    symbols,
)
from sympy import cse as sympy_cse
from sympy.printing.rust import RustCodePrinter

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_FILE = PROJECT_ROOT / "src" / "analytical_gradients.rs"

N_PARAMS = 21


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Rust Code Printer (f32 instead of f64)
# ═══════════════════════════════════════════════════════════════════════════════


class F32RustPrinter(RustCodePrinter):
    """Rust code printer targeting f32 with correct integer/float handling.

    Key design decision: In Rust, method calls like `.exp()`, `.ln()`,
    `.powf()` bind tighter than arithmetic operators. So `a*b.exp()` means
    `a * b.exp()`, NOT `(a*b).exp()`. We must parenthesize compound
    expressions when they are used as method call receivers.
    """

    def _needs_parens(self, expr):
        """Check if an expression needs parentheses when used as a method receiver."""
        return not (expr.is_Symbol or expr.is_Number or expr.is_Atom)

    def _wrap(self, expr):
        """Print an expression, wrapping in parens if it's compound."""
        code = self._print(expr)
        if self._needs_parens(expr):
            return f"({code})"
        return code

    def _print_Float(self, expr):
        return f"{float(expr)}_f32"

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        if q == 1:
            return f"{p}.0_f32"
        return f"({p}.0_f32 / {q}.0_f32)"

    def _print_Integer(self, expr):
        return f"{int(expr)}.0_f32"

    def _print_NegativeOne(self, expr):
        return "-1.0_f32"

    def _print_One(self, expr):
        return "1.0_f32"

    def _print_Zero(self, expr):
        return "0.0_f32"

    def _print_Pow(self, expr):
        base_str = self._wrap(expr.base)
        if expr.exp == -1:
            return f"(1.0_f32 / {base_str})"
        if expr.exp == Rational(1, 2):
            return f"{base_str}.sqrt()"
        if expr.exp == Rational(-1, 2):
            return f"(1.0_f32 / {base_str}.sqrt())"
        if expr.exp.is_integer:
            exp_val = int(expr.exp)
            return f"{base_str}.powi({exp_val})"
        exp_str = self._print(expr.exp)
        return f"{base_str}.powf({exp_str})"

    def _print_log(self, expr):
        return f"{self._wrap(expr.args[0])}.ln()"

    def _print_exp(self, expr):
        return f"{self._wrap(expr.args[0])}.exp()"


_printer = F32RustPrinter()


def to_rust(expr):
    """Convert SymPy expression to Rust f32 expression string."""
    if expr == 0:
        return "0.0_f32"
    code = _printer.doprint(expr)
    # Replace parameter symbols w0..w20 → w[0]..w[20] (longest first)
    for i in range(20, -1, -1):
        code = re.sub(rf"\bw{i}\b", f"w[{i}]", code)
    return code


# ═══════════════════════════════════════════════════════════════════════════════
# CSE Helper
# ═══════════════════════════════════════════════════════════════════════════════


def cse_and_rust(expr_dict, indent=4):
    """
    Apply Common Subexpression Elimination to a dict of {name: sympy_expr}.
    Returns (cse_lines, result_dict) where:
      - cse_lines: list of Rust 'let xN = ...;' lines
      - result_dict: {name: rust_expr_string} for each input name
    """
    names = list(expr_dict.keys())
    exprs = [expr_dict[n] for n in names]
    ind = " " * indent

    replacements, reduced = sympy_cse(exprs)

    cse_lines = []
    for sym, expr in replacements:
        cse_lines.append(f"{ind}let {sym}: f32 = {to_rust(expr)};")

    result = {}
    for name, expr in zip(names, reduced):
        result[name] = to_rust(expr)

    return cse_lines, result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Symbolic Model Definition
#
# All equations transcribed from src/model.rs (FSRS v6, 21 parameters).
# Variable names match the Rust source where possible.
# ═══════════════════════════════════════════════════════════════════════════════

print("Defining symbolic model...")

# 21 FSRS parameters
w = [Symbol(f"w{i}", real=True) for i in range(N_PARAMS)]

# State variables (clamped values from previous step)
s = Symbol("s", positive=True, real=True)  # S_prev
d = Symbol("d", positive=True, real=True)  # D_prev
t = Symbol("t", nonneg=True, real=True)  # delta_t at current step
r = Symbol("r", real=True)  # rating as float (for parameterized fns)

# ─── Forgetting Curve ─────────────────────────────────────────────────────
# model.rs:55-58  power_forgetting_curve()
# decay = -w[20]
# factor = (0.9^(1/decay)) - 1 = exp(ln(0.9)/decay) - 1
# R(t, S) = (t/S * factor + 1)^decay

decay = -w[20]
ln_09 = log(Rational(9, 10))  # ln(0.9) exact
factor = exp(ln_09 / decay) - 1
R_expr = (t / s * factor + 1) ** decay

# ─── Init Difficulty ──────────────────────────────────────────────────────
# model.rs:131-132  init_difficulty()
# D_init(rating) = w[4] - exp(w[5] * (rating - 1)) + 1


def D_init(rating_val):
    return w[4] - exp(w[5] * (rating_val - 1)) + 1


# ─── Difficulty Update ────────────────────────────────────────────────────
# model.rs:135-141  linear_damping() + next_difficulty() + mean_reversion()
# delta_d = -w[6] * (rating - 3)
# D_raw = D_prev + (10 - D_prev) / 9 * delta_d
# D_mr = w[7] * D_init(4) + (1 - w[7]) * D_raw


def D_next(d_prev, rating_val):
    """Difficulty after mean reversion, before clamp."""
    delta_d = -w[6] * (rating_val - 3)
    d_raw = d_prev + (10 - d_prev) / 9 * delta_d
    d_init_4 = D_init(4)
    return w[7] * d_init_4 + (1 - w[7]) * d_raw


# ─── Stability After Success ─────────────────────────────────────────────
# model.rs:71-93  stability_after_success()
# S_success = S * (exp(w[8]) * (11 - D) * S^(-w[9]) * (exp((1-R)*w[10]) - 1)
#              * hard_penalty * easy_bonus + 1)
# hard_penalty = w[15] if rating==2 else 1
# easy_bonus   = w[16] if rating==4 else 1


def S_success(r_int):
    """Stability after success for integer rating r_int (2, 3, or 4)."""
    hp = w[15] if r_int == 2 else Integer(1)
    eb = w[16] if r_int == 4 else Integer(1)
    inner = (
        exp(w[8])
        * (11 - d)
        * s ** (-w[9])
        * (exp((1 - R_expr) * w[10]) - 1)
        * hp
        * eb
    )
    return s * (inner + 1)


# ─── Stability After Failure ─────────────────────────────────────────────
# model.rs:95-109  stability_after_failure()
# S_fail_raw = w[11] * D^(-w[12]) * ((S+1)^w[13] - 1) * exp((1-R)*w[14])
# S_fail_min = S / exp(w[17] * w[18])
# S_fail = min(S_fail_raw, S_fail_min)   [via mask_where]

S_fail_raw_expr = (
    w[11] * d ** (-w[12]) * ((s + 1) ** w[13] - 1) * exp((1 - R_expr) * w[14])
)
S_fail_min_expr = s / exp(w[17] * w[18])

# ─── Stability Short-Term ────────────────────────────────────────────────
# model.rs:111-119  stability_short_term()
# sinc = exp(w[17] * (rating - 3 + w[18])) * S^(-w[19])
# if rating >= 2: sinc_eff = max(sinc, 1.0)  [clamp_min]
# else:           sinc_eff = sinc
# S_short = S * sinc_eff

sinc_expr = exp(w[17] * (r - 3 + w[18])) * s ** (-w[19])
S_short_expr = s * sinc_expr  # Unclamped form


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Derivative Computation
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing derivatives (this may take a moment)...")


def nonzero_param_grads(expr, wrt_symbols=None):
    """Compute ∂expr/∂w[i] for i=0..20, return dict of {i: deriv} for non-zero."""
    if wrt_symbols is None:
        wrt_symbols = w
    result = {}
    for i, wi in enumerate(wrt_symbols):
        deriv = diff(expr, wi)
        if deriv != 0:
            result[i] = deriv
    return result


# ─── A. Forgetting Curve Derivatives ──────────────────────────────────────
dR_ds = diff(R_expr, s)
dR_dw20 = diff(R_expr, w[20])

# ─── B. Difficulty Update Derivatives ─────────────────────────────────────
D_new_sym = D_next(d, r)
dD_dd = diff(D_new_sym, d)
dD_dw_map = nonzero_param_grads(D_new_sym)

# ─── C. Init Difficulty Derivatives ───────────────────────────────────────
D_init_sym = D_init(r)
dDi_dw_map = nonzero_param_grads(D_init_sym)

# ─── D. Stability Success Derivatives (r=2,3,4) ──────────────────────────
success_data = {}
for r_int in [2, 3, 4]:
    expr = S_success(r_int)
    data = {
        "fwd": expr,
        "ds_ds": diff(expr, s),
        "ds_dd": diff(expr, d),
        "ds_dw": nonzero_param_grads(expr),
    }
    success_data[r_int] = data
    print(f"  Success r={r_int}: {len(data['ds_dw'])} non-zero param grads")

# ─── E. Stability Failure Raw Derivatives ─────────────────────────────────
fail_raw_data = {
    "fwd": S_fail_raw_expr,
    "ds_ds": diff(S_fail_raw_expr, s),
    "ds_dd": diff(S_fail_raw_expr, d),
    "ds_dw": nonzero_param_grads(S_fail_raw_expr),
}
print(f"  Failure raw: {len(fail_raw_data['ds_dw'])} non-zero param grads")

# ─── F. Stability Failure Min Derivatives ─────────────────────────────────
fail_min_data = {
    "fwd": S_fail_min_expr,
    "ds_ds": diff(S_fail_min_expr, s),
    "ds_dw": nonzero_param_grads(S_fail_min_expr),
}
print(f"  Failure min: {len(fail_min_data['ds_dw'])} non-zero param grads")

# ─── G. Short-Term Stability Derivatives ──────────────────────────────────
short_data = {
    "fwd": S_short_expr,
    "sinc": sinc_expr,
    "ds_ds": diff(S_short_expr, s),
    "ds_dw": nonzero_param_grads(S_short_expr),
}
print(f"  Short-term: {len(short_data['ds_dw'])} non-zero param grads")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Rust Code Generation
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Rust code...")


def build_stability_function(name, doc, data, has_d_dep=True, extra_returns=""):
    """Build a Rust function for a stability branch.

    data must have: fwd, ds_ds, ds_dw (dict)
    Optionally: ds_dd (if has_d_dep)
    """
    expr_dict = OrderedDict()
    expr_dict["s_new"] = data["fwd"]
    expr_dict["ds_ds"] = data["ds_ds"]
    if has_d_dep and "ds_dd" in data:
        expr_dict["ds_dd"] = data["ds_dd"]
    if "sinc" in data:
        expr_dict["sinc_val"] = data["sinc"]
    for i, expr in sorted(data["ds_dw"].items()):
        expr_dict[f"g{i}"] = expr

    cse_lines, results = cse_and_rust(expr_dict)

    params = "w: &[f32; 21], s: f32, d: f32, t: f32"
    if not has_d_dep:
        params = "w: &[f32; 21], s: f32, r: f32"

    lines = []
    lines.append(f"/// {doc}")
    if has_d_dep:
        lines.append(
            f"fn {name}(w: &[f32; 21], s: f32, d: f32, t: f32)"
            " -> (f32, f32, f32, [f32; 21]) {"
        )
    elif "sinc" in data:
        lines.append(
            f"fn {name}(w: &[f32; 21], s: f32, r: f32)"
            " -> (f32, f32, f32, [f32; 21]) {"
        )
    else:
        lines.append(
            f"fn {name}(w: &[f32; 21], s: f32)"
            " -> (f32, f32, [f32; 21]) {"
        )

    lines.extend(cse_lines)
    if cse_lines:
        lines.append("")
    lines.append(f"    let s_new: f32 = {results['s_new']};")
    lines.append(f"    let ds_ds: f32 = {results['ds_ds']};")
    if has_d_dep and "ds_dd" in data:
        lines.append(f"    let ds_dd: f32 = {results['ds_dd']};")

    lines.append("    let mut ds_dw = [0.0_f32; 21];")
    for i in sorted(data["ds_dw"].keys()):
        lines.append(f"    ds_dw[{i}] = {results[f'g{i}']};")

    if has_d_dep:
        lines.append("    (s_new, ds_ds, ds_dd, ds_dw)")
    elif "sinc" in data:
        lines.append(f"    let sinc_val: f32 = {results['sinc_val']};")
        lines.append("    (s_new, sinc_val, ds_ds, ds_dw)")
    else:
        lines.append("    (s_new, ds_ds, ds_dw)")
    lines.append("}")
    return "\n".join(lines)


# ─── Generate forgetting curve function ───────────────────────────────────
def gen_forgetting_curve():
    expr_dict = OrderedDict()
    expr_dict["r_val"] = R_expr
    expr_dict["dr_ds"] = dR_ds
    expr_dict["dr_dw20"] = dR_dw20
    cse_lines, results = cse_and_rust(expr_dict)

    lines = []
    lines.append("/// Power forgetting curve R(t, S) and its gradients.")
    lines.append("/// Returns (R, dR/dS, dR/dw20)")
    lines.append(
        "fn forgetting_curve_grad(w: &[f32; 21], t: f32, s: f32)"
        " -> (f32, f32, f32) {"
    )
    lines.extend(cse_lines)
    if cse_lines:
        lines.append("")
    lines.append(f"    let r_val: f32 = {results['r_val']};")
    lines.append(f"    let dr_ds: f32 = {results['dr_ds']};")
    lines.append(f"    let dr_dw20: f32 = {results['dr_dw20']};")
    lines.append("    (r_val, dr_ds, dr_dw20)")
    lines.append("}")
    return "\n".join(lines)


# ─── Generate difficulty update function ──────────────────────────────────
def gen_difficulty_update():
    expr_dict = OrderedDict()
    expr_dict["d_new"] = D_new_sym
    expr_dict["dd_dd"] = dD_dd
    for i, expr in sorted(dD_dw_map.items()):
        expr_dict[f"g{i}"] = expr
    cse_lines, results = cse_and_rust(expr_dict)

    lines = []
    lines.append("/// Difficulty update with mean reversion (before clamp).")
    lines.append("/// Returns (D_new_unclamped, dD/dD_prev, dD/dw[0..21])")
    lines.append(
        "fn difficulty_update_grad(w: &[f32; 21], d: f32, r: f32)"
        " -> (f32, f32, [f32; 21]) {"
    )
    lines.extend(cse_lines)
    if cse_lines:
        lines.append("")
    lines.append(f"    let d_new: f32 = {results['d_new']};")
    lines.append(f"    let dd_dd: f32 = {results['dd_dd']};")
    lines.append("    let mut dd_dw = [0.0_f32; 21];")
    for i in sorted(dD_dw_map.keys()):
        lines.append(f"    dd_dw[{i}] = {results[f'g{i}']};")
    lines.append("    (d_new, dd_dd, dd_dw)")
    lines.append("}")
    return "\n".join(lines)


# ─── Generate init difficulty function ────────────────────────────────────
def gen_init_difficulty():
    expr_dict = OrderedDict()
    expr_dict["d_init"] = D_init_sym
    for i, expr in sorted(dDi_dw_map.items()):
        expr_dict[f"g{i}"] = expr
    cse_lines, results = cse_and_rust(expr_dict)

    lines = []
    lines.append("/// Init difficulty D_init(rating) and its gradients.")
    lines.append("/// Returns (D_init, dD/dw[0..21])")
    lines.append(
        "fn init_difficulty_grad(w: &[f32; 21], r: f32)"
        " -> (f32, [f32; 21]) {"
    )
    lines.extend(cse_lines)
    if cse_lines:
        lines.append("")
    lines.append(f"    let d_init: f32 = {results['d_init']};")
    lines.append("    let mut dd_dw = [0.0_f32; 21];")
    for i in sorted(dDi_dw_map.keys()):
        lines.append(f"    dd_dw[{i}] = {results[f'g{i}']};")
    lines.append("    (d_init, dd_dw)")
    lines.append("}")
    return "\n".join(lines)


# ─── Generate success functions ───────────────────────────────────────────
def gen_success(r_int):
    data = success_data[r_int]
    return build_stability_function(
        f"stability_success_r{r_int}_grad",
        f"Stability after success (rating {r_int}) and its gradients.",
        data,
        has_d_dep=True,
    )


# ─── Generate failure functions ───────────────────────────────────────────
def gen_failure_raw():
    return build_stability_function(
        "stability_failure_raw_grad",
        "Stability after failure (raw path) and its gradients.",
        fail_raw_data,
        has_d_dep=True,
    )


def gen_failure_min():
    return build_stability_function(
        "stability_failure_min_grad",
        "Stability after failure (min path: S / exp(w17*w18)).",
        fail_min_data,
        has_d_dep=False,
    )


# ─── Generate short-term function ─────────────────────────────────────────
def gen_short_term():
    """Generate the unclamped short-term stability function."""
    expr_dict = OrderedDict()
    expr_dict["s_new"] = S_short_expr
    expr_dict["sinc_val"] = sinc_expr
    expr_dict["ds_ds"] = short_data["ds_ds"]
    for i, expr in sorted(short_data["ds_dw"].items()):
        expr_dict[f"g{i}"] = expr
    cse_lines, results = cse_and_rust(expr_dict)

    lines = []
    lines.append("/// Short-term stability (unclamped sinc) and its gradients.")
    lines.append("/// Returns (S_new, sinc, dS/dS_prev, dS/dw[0..21])")
    lines.append(
        "fn stability_short_term_grad(w: &[f32; 21], s: f32, r: f32)"
        " -> (f32, f32, f32, [f32; 21]) {"
    )
    lines.extend(cse_lines)
    if cse_lines:
        lines.append("")
    lines.append(f"    let s_new: f32 = {results['s_new']};")
    lines.append(f"    let sinc_val: f32 = {results['sinc_val']};")
    lines.append(f"    let ds_ds: f32 = {results['ds_ds']};")
    lines.append("    let mut ds_dw = [0.0_f32; 21];")
    for i in sorted(short_data["ds_dw"].keys()):
        lines.append(f"    ds_dw[{i}] = {results[f'g{i}']};")
    lines.append("    (s_new, sinc_val, ds_ds, ds_dw)")
    lines.append("}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Assemble the Rust File
# ═══════════════════════════════════════════════════════════════════════════════


def build_rust_file():
    """Assemble the complete generated Rust source file."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Header ────────────────────────────────────────────────────────────
    header = f"""\
//! AUTO-GENERATED by scripts/generate_gradients.py — DO NOT EDIT MANUALLY
//!
//! Generated: {timestamp}
//!
//! Analytical gradients for the FSRS model (21 parameters).
//! Mathematically equivalent to Burn's dynamic autodiff backward pass.
//!
//! The functions below implement per-step Jacobians for the FSRS state
//! transition equations and a full BPTT (backpropagation through time) loop.

#![allow(
    clippy::excessive_precision,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    unused_variables,
    unused_assignments,
    unused_parens,
    non_snake_case
)]

/// Constants matching simulation.rs
const S_MIN: f32 = 0.001;
const S_MAX: f32 = 36500.0;
const D_MIN: f32 = 1.0;
const D_MAX: f32 = 10.0;

/// Result of a single BPTT step: forward values + Jacobians + parameter grads.
#[derive(Debug, Clone)]
pub struct StepGrads {{
    pub s_new: f32,
    pub d_new: f32,
    pub ds_ds: f32,       // ∂S_new/∂S_prev
    pub ds_dd: f32,       // ∂S_new/∂D_prev
    pub dd_dd: f32,       // ∂D_new/∂D_prev (note: ∂D/∂S = 0 always)
    pub ds_dw: [f32; 21], // ∂S_new/∂w[i]
    pub dd_dw: [f32; 21], // ∂D_new/∂w[i]
}}

impl StepGrads {{
    /// Passthrough step (padding, rating == 0): no state change.
    fn passthrough(s: f32, d: f32) -> Self {{
        Self {{
            s_new: s,
            d_new: d,
            ds_ds: 1.0,
            ds_dd: 0.0,
            dd_dd: 1.0,
            ds_dw: [0.0; 21],
            dd_dw: [0.0; 21],
        }}
    }}
}}
"""

    # ── Generated math functions ──────────────────────────────────────────
    print("  Generating forgetting_curve_grad...")
    fc_code = gen_forgetting_curve()
    print("  Generating difficulty_update_grad...")
    du_code = gen_difficulty_update()
    print("  Generating init_difficulty_grad...")
    di_code = gen_init_difficulty()
    print("  Generating stability_success_r2_grad...")
    ss2_code = gen_success(2)
    print("  Generating stability_success_r3_grad...")
    ss3_code = gen_success(3)
    print("  Generating stability_success_r4_grad...")
    ss4_code = gen_success(4)
    print("  Generating stability_failure_raw_grad...")
    fr_code = gen_failure_raw()
    print("  Generating stability_failure_min_grad...")
    fm_code = gen_failure_min()
    print("  Generating stability_short_term_grad...")
    st_code = gen_short_term()

    # ── Template functions (branching + BPTT) ─────────────────────────────
    template = r"""
/// BCE loss and its gradient w.r.t. retrievability R.
/// Returns (loss, dL/dR)
fn bce_loss_grad(r: f32, label: f32, weight: f32) -> (f32, f32) {
    // Clamp r to avoid log(0) / division by zero
    let r_safe = r.clamp(1e-10, 1.0 - 1e-10);
    let loss = -(label * r_safe.ln() + (1.0 - label) * (1.0 - r_safe).ln()) * weight;
    let dl_dr = -(label / r_safe - (1.0 - label) / (1.0 - r_safe)) * weight;
    (loss, dl_dr)
}

/// Compute a single step of the FSRS model with analytical gradients.
///
/// Handles all branching: init, padding, short-term, failure, success.
/// Applies clamp masks to zero out gradients at boundaries (matching Burn).
///
/// # Arguments
/// * `w` - 21 FSRS parameters
/// * `s_prev` - Previous stability (output of last step, already clamped)
/// * `d_prev` - Previous difficulty (output of last step, already clamped)
/// * `delta_t` - Days since last review at this step
/// * `rating` - Rating at this step (0=padding, 1-4=actual)
/// * `is_first` - Whether this is the first step in the sequence (nth == 0)
fn step_grad(
    w: &[f32; 21],
    s_prev: f32,
    d_prev: f32,
    delta_t: f32,
    rating: u32,
    is_first: bool,
) -> StepGrads {
    // ── Padding: passthrough ──────────────────────────────────────────
    if rating == 0 {
        return StepGrads::passthrough(s_prev, d_prev);
    }

    // ── Input clamps (model.rs:151-152) ──────────────────────────────
    // For non-init steps, s_prev/d_prev are from previous step's output
    // clamp, so they're always within bounds → input clamp grad = 1.
    let s_c = s_prev.clamp(S_MIN, S_MAX);
    let d_c = d_prev.clamp(D_MIN, D_MAX);

    // ── Init case (first step, zero state) ───────────────────────────
    if is_first && s_prev == 0.0 {
        // S_init = w[rating - 1]  (table lookup, model.rs:127-129)
        let s_init = w[rating as usize - 1];

        // D_init(rating) = w[4] - exp(w[5]*(r-1)) + 1  (model.rs:131-132)
        let r_f = rating as f32;
        let (d_init, dd_dw_init) = init_difficulty_grad(w, r_f);

        // Output clamps
        let s_new = s_init.clamp(S_MIN, S_MAX);
        let d_new = d_init.clamp(D_MIN, D_MAX);
        let s_active = s_init > S_MIN && s_init < S_MAX;
        let d_active = d_init > D_MIN && d_init < D_MAX;

        let mut ds_dw = [0.0_f32; 21];
        if s_active {
            ds_dw[rating as usize - 1] = 1.0;
        }

        let mut dd_dw = [0.0_f32; 21];
        if d_active {
            for i in 0..21 {
                dd_dw[i] = dd_dw_init[i];
            }
        }

        return StepGrads {
            s_new,
            d_new,
            ds_ds: 0.0,
            ds_dd: 0.0,
            dd_dd: 0.0,
            ds_dw,
            dd_dw,
        };
    }

    // ── Difficulty update (shared for all non-init branches) ─────────
    // model.rs:168-169  next_difficulty + mean_reversion + clamp
    let r_f = rating as f32;
    let (d_unclamped, dd_dd_raw, mut dd_dw) = difficulty_update_grad(w, d_c, r_f);
    let d_new = d_unclamped.clamp(D_MIN, D_MAX);

    // Zero D grads at clamp boundaries (matching Burn)
    let d_active = d_unclamped > D_MIN && d_unclamped < D_MAX;
    let dd_dd = if d_active { dd_dd_raw } else { 0.0 };
    if !d_active {
        dd_dw = [0.0; 21];
    }

    // ── Stability branch selection ───────────────────────────────────
    let (s_unclamped, ds_ds_raw, ds_dd_raw, mut ds_dw) = if delta_t == 0.0 {
        // Short-term path (model.rs:111-119, 166)
        let (s_st, sinc_val, ds_ds_st, ds_dw_st) =
            stability_short_term_grad(w, s_c, r_f);

        if rating >= 2 && sinc_val < 1.0 {
            // clamp_min(sinc, 1.0) is active → S_new = S_prev, grad = 0
            // (Burn zeros the gradient when clamped, model.rs:118)
            (s_c, 1.0_f32, 0.0_f32, [0.0_f32; 21])
        } else {
            (s_st, ds_ds_st, 0.0_f32, ds_dw_st)
        }
    } else if rating == 1 {
        // Failure path (model.rs:95-109, 165)
        let (s_raw, ds_ds_raw, ds_dd_raw, ds_dw_raw) =
            stability_failure_raw_grad(w, s_c, d_c, delta_t);
        let (s_min, ds_ds_min, ds_dw_min) =
            stability_failure_min_grad(w, s_c);

        if s_min < s_raw {
            // min path active: use S_fail_min
            (s_min, ds_ds_min, 0.0_f32, ds_dw_min)
        } else {
            // raw path active: use S_fail_raw
            (s_raw, ds_ds_raw, ds_dd_raw, ds_dw_raw)
        }
    } else {
        // Success path (model.rs:71-93, 164-165)
        match rating {
            2 => stability_success_r2_grad(w, s_c, d_c, delta_t),
            3 => stability_success_r3_grad(w, s_c, d_c, delta_t),
            4 => stability_success_r4_grad(w, s_c, d_c, delta_t),
            _ => unreachable!(),
        }
    };

    // ── Output S clamp (model.rs:188) ────────────────────────────────
    let s_new = s_unclamped.clamp(S_MIN, S_MAX);
    let s_active = s_unclamped > S_MIN && s_unclamped < S_MAX;
    let ds_ds = if s_active { ds_ds_raw } else { 0.0 };
    let ds_dd = if s_active { ds_dd_raw } else { 0.0 };
    if !s_active {
        ds_dw = [0.0; 21];
    }

    StepGrads {
        s_new,
        d_new,
        ds_ds,
        ds_dd,
        dd_dd,
        ds_dw,
        dd_dw,
    }
}

/// Full forward-backward pass (BPTT) for a single training example.
///
/// Processes a sequence of reviews and computes the analytical gradient
/// of the BCE loss with respect to all 21 FSRS parameters.
///
/// This is mathematically equivalent to:
///   model.forward_classification() + loss.backward()
/// from src/training.rs (with Reduction::Sum, single item).
///
/// # Arguments
/// * `w` - 21 FSRS parameters
/// * `delta_ts` - Review history: days since last review at each step
/// * `ratings` - Review history: rating (1-4) at each step, 0 for padding
/// * `final_delta_t` - Days elapsed for the current (evaluated) review
/// * `label` - Ground truth: 1.0 if recalled, 0.0 if forgotten
/// * `weight` - Sample weight for the loss
///
/// # Returns
/// `(loss, grad_w)` where `grad_w[i]` = ∂loss/∂w[i]
pub fn forward_backward(
    w: &[f32; 21],
    delta_ts: &[f32],
    ratings: &[u32],
    final_delta_t: f32,
    label: f32,
    weight: f32,
) -> (f32, [f32; 21]) {
    let n = delta_ts.len();
    assert_eq!(n, ratings.len());

    // ── Forward pass: compute intermediate states + store Jacobians ───
    let mut s_states = Vec::with_capacity(n + 1);
    let mut d_states = Vec::with_capacity(n + 1);
    s_states.push(0.0_f32);
    d_states.push(0.0_f32);

    let mut step_results: Vec<StepGrads> = Vec::with_capacity(n);

    for k in 0..n {
        let sg = step_grad(
            w,
            s_states[k],
            d_states[k],
            delta_ts[k],
            ratings[k],
            k == 0,
        );
        s_states.push(sg.s_new);
        d_states.push(sg.d_new);
        step_results.push(sg);
    }

    // ── Final retrievability and loss ────────────────────────────────
    let s_final = s_states[n];
    let (r_final, dr_ds, dr_dw20) = forgetting_curve_grad(w, final_delta_t, s_final);
    let (loss, dl_dr) = bce_loss_grad(r_final, label, weight);

    // ── Backward pass: accumulate gradients via BPTT ─────────────────
    let mut grad_w = [0.0_f32; 21];

    // Adjoints: dL/dS_N and dL/dD_N
    let mut adj_s = dl_dr * dr_ds;
    let mut adj_d = 0.0_f32; // Loss doesn't depend directly on D_final

    // Direct gradient of w[20] through the final forgetting curve
    grad_w[20] += dl_dr * dr_dw20;

    // Backpropagate through the sequence
    for k in (0..n).rev() {
        let sg = &step_results[k];

        // Accumulate parameter gradients at this step
        for i in 0..21 {
            grad_w[i] += adj_s * sg.ds_dw[i] + adj_d * sg.dd_dw[i];
        }

        // Propagate adjoints to previous step
        // Note: dD/dS = 0 always (difficulty doesn't depend on stability)
        let new_adj_s = adj_s * sg.ds_ds; // + adj_d * 0
        let new_adj_d = adj_s * sg.ds_dd + adj_d * sg.dd_dd;
        adj_s = new_adj_s;
        adj_d = new_adj_d;
    }

    (loss, grad_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference test data from src/training.rs test_loss_and_grad.
    /// Uses DEFAULT_PARAMETERS and specific batch data.
    #[test]
    fn test_analytical_gradient_matches_burn() {
        // DEFAULT_PARAMETERS from inference.rs
        let w: [f32; 21] = [
            0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
            1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
            1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
        ];

        // Test batch from test_loss_and_grad (4 items, 6 timesteps each)
        // t_historys[step][item]:
        let t_historys: [[f32; 4]; 6] = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 3.0],
            [1.0, 3.0, 3.0, 5.0],
            [3.0, 6.0, 6.0, 12.0],
        ];
        // r_historys[step][item]:
        let r_historys: [[u32; 4]; 6] = [
            [1, 2, 3, 4],
            [3, 4, 2, 4],
            [1, 4, 4, 3],
            [4, 3, 3, 3],
            [3, 1, 3, 3],
            [2, 3, 3, 4],
        ];
        let delta_ts: [f32; 4] = [4.0, 11.0, 12.0, 23.0];
        let labels: [f32; 4] = [1.0, 1.0, 1.0, 0.0];
        let weights: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

        // Expected gradient from Burn (from test_loss_and_grad)
        let expected_grad: [f32; 21] = [
            -0.095688485, -0.0051607806, -0.0012249565, 0.007462064,
            0.03650761, -0.082112335, 0.0593964, -2.1474836,
            0.57626534, -2.8751316, 0.7154875, -0.028993709,
            0.0099172965, -0.2189217, -0.0017800558, -0.089381434,
            0.299141, 0.068104014, -0.011605468, -0.25398168,
            0.27700496,
        ];

        // Compute analytical gradients for all 4 items and sum
        let mut total_loss = 0.0_f32;
        let mut total_grad = [0.0_f32; 21];

        for item in 0..4 {
            let item_delta_ts: Vec<f32> =
                (0..6).map(|step| t_historys[step][item]).collect();
            let item_ratings: Vec<u32> =
                (0..6).map(|step| r_historys[step][item]).collect();

            let (loss, grad) = forward_backward(
                &w,
                &item_delta_ts,
                &item_ratings,
                delta_ts[item],
                labels[item],
                weights[item],
            );
            total_loss += loss;
            for i in 0..21 {
                total_grad[i] += grad[i];
            }
        }

        // Expected total loss
        let expected_loss = 4.0466027_f32;
        let loss_err = (total_loss - expected_loss).abs();
        assert!(
            loss_err < 1e-4,
            "Loss mismatch: got {total_loss}, expected {expected_loss}, err={loss_err}"
        );

        // Check gradient match
        for i in 0..21 {
            let err = (total_grad[i] - expected_grad[i]).abs();
            let rel = if expected_grad[i].abs() > 1e-8 {
                err / expected_grad[i].abs()
            } else {
                err
            };
            assert!(
                err < 1e-3 || rel < 1e-2,
                "Gradient mismatch at w[{i}]: got {}, expected {}, abs_err={err}, rel_err={rel}",
                total_grad[i], expected_grad[i]
            );
        }
    }
}
"""

    # ── Assemble complete file ────────────────────────────────────────
    sections = [
        header,
        "// ═══════════════════════════════════════════════════════════",
        "// Generated mathematical functions (SymPy + CSE optimized)",
        "// ═══════════════════════════════════════════════════════════\n",
        fc_code,
        "",
        du_code,
        "",
        di_code,
        "",
        ss2_code,
        "",
        ss3_code,
        "",
        ss4_code,
        "",
        fr_code,
        "",
        fm_code,
        "",
        st_code,
        "",
        "// ═══════════════════════════════════════════════════════════",
        "// Template functions (branching logic + BPTT)",
        "// ═══════════════════════════════════════════════════════════",
        template,
    ]

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    rust_code = build_rust_file()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rust_code)

    line_count = rust_code.count("\n") + 1
    print(f"\nWritten {line_count} lines to {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
