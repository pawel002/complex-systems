GREEK = {
    "alpha": r"\alpha",
    "beta":  r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "zeta": r"\zeta",
    "eta": r"\eta",
    "theta": r"\theta",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "xi": r"\xi",
    "pi": r"\pi",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "phi": r"\phi",
    "chi": r"\chi",
    "psi": r"\psi",
    "omega": r"\omega",
    "n": "n",  # leave as is (latin)
}

def key_to_tex(k: str) -> str:
    """Convert a parameter key to a LaTeX/mathtext token."""
    if k in GREEK:
        return GREEK[k]
    # handle simple subscript like tau_d, x_0, etc.
    if "_" in k:
        main, sub = k.split("_", 1)
        main_tex = GREEK.get(main, main)
        return rf"{main_tex}_{{{sub}}}"
    return k  # fallback

def fmt_val(v, precision=2):
    if isinstance(v, float):
        return f"{v:.{precision}f}".rstrip("0").rstrip(".")
    return str(v)

def params_to_mathtext(params: dict, order=None, precision=2, wrap=False):
    """Return a mathtext string like r'$\sigma=0.88, \rho=0.62, \ldots$'."""
    if order is None:
        order = list(params.keys())
    chunks = [rf"{key_to_tex(k)}={fmt_val(params[k], precision)}" for k in order]
    if wrap:
        # split into two lines roughly in half (simple and robust)
        mid = len(chunks) // 2
        line1 = r",\ ".join(chunks[:mid])
        line2 = r",\ ".join(chunks[mid:])
        return rf"$ {line1} $"+ "\n" + rf"$ {line2} $"
    else:
        return rf"$ " + r",\ ".join(chunks) + r" $"