import torch
import torch.nn as nn


def _open_uniform_knots(n_ctrl_pts: int, degree: int, *, dtype=None, device=None):
    """
    Create the usual open‑uniform knot vector
        [0 … 0,  evenly‑spaced internal knots , 1 … 1]
    with multiplicity degree+1 at each end.
    """
    n_knots = n_ctrl_pts + degree + 1
    kvec = torch.zeros(n_knots, dtype=dtype, device=device)

    # internal knots (if any) evenly spaced in (0, 1)
    n_internal = n_knots - 2 * (degree + 1)
    if n_internal > 0:
        kvec[degree + 1 : -degree - 1] = torch.linspace(
            0.0, 1.0, n_internal + 2, dtype=dtype, device=device
        )[1:-1]

    # right boundary
    kvec[-degree - 1 :] = 1.0
    return kvec


def _bspline_basis(t: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Vectorised Cox–de Boor evaluation of all B‑spline basis functions N_{i,degree}(t)
    for every time point in `t`.

    Args
    ----
    t      : (B,) or (B,1) tensor of query times in [0, 1].
    knots  : (n_knots,) non‑decreasing knot vector (monotone, 0 … 1 Inclusive).
    degree : spline degree (0 = piecewise constant, 3 = cubic, …).

    Returns
    -------
    basis  : (B, n_ctrl_pts) tensor, where each row sums to 1.
    """
    if t.dim() == 2:
        t = t.squeeze(-1)
    B = t.shape[0]
    device, dtype = t.device, t.dtype
    n_ctrl_pts = knots.numel() - degree - 1

    # k = 0 -------------------------------------------------------------------
    N = torch.zeros(B, n_ctrl_pts, device=device, dtype=dtype)
    for i in range(n_ctrl_pts):
        left, right = knots[i], knots[i + 1]
        mask = (t >= left) & (t < right)
        # make the last knot inclusive so that t = 1 maps to the final span
        if i == n_ctrl_pts - 1:
            mask = mask | (t == knots[-1])
        N[:, i] = mask.to(dtype)

    # k = 1 … degree ----------------------------------------------------------
    for k in range(1, degree + 1):
        N_new = torch.zeros_like(N)
        for i in range(n_ctrl_pts):
            # left term -------------------------------------------------------
            denom1 = knots[i + k] - knots[i]
            if denom1 > 0:
                N_new[:, i] += ((t - knots[i]) / denom1) * N[:, i]

            # right term ------------------------------------------------------
            if i + 1 < n_ctrl_pts:
                denom2 = knots[i + k + 1] - knots[i + 1]
                if denom2 > 0:
                    N_new[:, i] += ((knots[i + k + 1] - t) / denom2) * N[:, i + 1]

        N = N_new
    return N


class SplineWeightEmbedding(nn.Module):
    """
    B‑spline time‑series embedding layer:

        (n_outputs, n_ctrl_pts) learnable control‑point matrix  ——►  R^{B × n_outputs}
    """

    def __init__(self, n_outputs: int, n_ctrl_pts: int, degree: int = 3, prior=0, dropout=0):
        """
        Parameters
        ----------
        n_outputs : number of independent output channels (O).
        n_ctrl_pts: control points per output (C).
        degree    : spline degree (default cubic, degree = 3).
        """
        super().__init__()
        if degree >= n_ctrl_pts:
            raise ValueError("`degree` must be < `n_ctrl_pts`.")

        self.degree = degree
        self.register_buffer(
            "knots", _open_uniform_knots(n_ctrl_pts, degree, dtype=torch.float32)
        )

        # Learnable control‑point grid — shape (O, C)
        self.ctrl_pts = nn.Parameter(prior*torch.ones(n_outputs, n_ctrl_pts))

        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the spline at times `t` ∈ [0, 1].

        Parameters
        ----------
        t : (B,) or (B,1) tensor of query times.

        Returns
        -------
        (B, n_outputs) tensor of spline values.
        """
        basis = _bspline_basis(t, self.knots, self.degree)          # (B, C)
        out = basis @ self.ctrl_pts.t()                            # (B, O)

        return self.dropout(out)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke‑test / usage example
if __name__ == "__main__":
    B, O, C = 8, 4, 7                       # batch, outputs, control points
    spline = SplineWeightEmbedding(O, C)    # cubic by default (degree = 3)

    T = torch.rand(B, 1).requires_grad_()   # (B,1) times in [0,1]
    Y = spline(T)                           # (B,O) values

    assert Y.shape == (B, O)
    # Differentiability check --------------------------------------------------
    loss = Y.sum()
    loss.backward()

    print("ctrl_pts.grad shape:", spline.ctrl_pts.grad.shape)  # → (O, C)
    print("t.grad shape       :", T.grad.shape)                # → (B, 1)
