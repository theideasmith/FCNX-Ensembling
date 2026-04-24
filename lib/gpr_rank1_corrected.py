
# ---------------------------------------------------------------------------
# GPR with rank-1 kernel corrections via Sherman-Morrison
# ---------------------------------------------------------------------------
import torch
def gpr_rank1_corrected(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    ridge: float,
    theory_eigs: dict,
    P: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform GPR using the bare arcsin kernel corrected by rank-1 updates for
    the He1 and He3 target modes, exploiting Sherman-Morrison to avoid a
    second O(n³) factorisation.
 
    Kernel surgery (operator picture)
    ----------------------------------
    The GPR integral operator has Mercer expansion
 
        K_GPR(x,x') = Σ_k  λ_k^GPR · φ_k(x) φ_k(x')
 
    The adapted operator replaces λ_k^GPR with λ_k^adapted for the modes k
    we care about (He1-target, He3-target).  The kernel difference is
 
        ΔK = Σ_{k ∈ S}  δλ_k · φ_k(x) φ_k(x'),   δλ_k = λ_k^adapted − λ_k^GPR
 
    In the finite-sample Gram-matrix picture each term becomes a rank-1 matrix:
 
        ΔK_train = δλ_k · φ_k(X_train) φ_k(X_train)ᵀ
 
    We therefore never build the corrected kernel explicitly.  Instead we:
 
      1.  Factorise  A = K_train + σ²I  once with Cholesky  (O(n³)).
      2.  Apply Sherman-Morrison for each mode correction  (O(n²) each).
      3.  Use the updated inverse to compute the GPR dual weights α and then
          the predictive mean on the test set.
 
    Cross-kernel correction
    -----------------------
    The test predictions require K_cross = K(X_test, X_train), which also
    receives a rank-1 correction:
 
        ΔK_cross[i, j] = δλ_k · φ_k(x_test_i) · φ_k(x_train_j)
 
    This is computed cheaply as an outer product and added to K_cross before
    the final matrix-vector product.
    """
    n_train = X_train.shape[0]
    X_all = torch.cat([X_train, X_test], dim=0)
 
    # ------------------------------------------------------------------
    # Step 0: bare arcsin kernel
    # ------------------------------------------------------------------
    K_all = arcsin_kernel(X_all)
    K_train = K_all[:n_train, :n_train]       # (n_train, n_train)
    K_cross = K_all[n_train:, :n_train]       # (n_test,  n_train)
 
    # ------------------------------------------------------------------
    # Step 1: Cholesky factorisation of the bare regularised kernel
    #   A = K_train + (ridge/P) · I
    # ------------------------------------------------------------------
    reg = ridge / P
    eye = torch.eye(n_train, device=K_train.device, dtype=K_train.dtype)
    A = K_train + reg * eye
    chol = torch.linalg.cholesky(A)
    # Initial inverse via Cholesky solve against the identity — O(n³) but
    # only done once.  For very large n one could keep the Cholesky factor and
    # apply solves lazily; for our n this is fine.
    A_inv = torch.cholesky_solve(eye, chol)   # (n_train, n_train)
 
    # ------------------------------------------------------------------
    # Step 2: theory eigenvalue gaps δλ_k = λ_adapted − λ_GPR
    #
    # The Julia solver returns λ^adapted (the field-theory corrected value).
    # We need the GPR eigenvalue for the same mode.  For the arcsin kernel
    # under the Gaussian measure the He1 and He3 eigenvalues are known
    # analytically; see e.g. Williams & Rasmussen §4.3.  However, the most
    # self-consistent approach here is to read off λ^GPR by projecting the
    # empirical Gram matrix onto the Hermite modes — that is,
    #
    #   λ_k^GPR ≈ φ_k(X_train)ᵀ K_train φ_k(X_train) / ‖φ_k‖⁴
    #
    # scaled by P so units match.  We compute this directly below.
    # ------------------------------------------------------------------
    modes = _hermite_mode_vectors(X_train)   # dict of (n_train,) tensors
    modes_test = _hermite_mode_vectors(X_test)
 
    # scale_factor converts field-theory units → empirical Gram-matrix units
    scale_factor = float(P)
 
    corrections: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for key in ("h1t", "h3t"):
        lam_adapted_ft = theory_eigs.get(
            "lH1T" if key == "h1t" else "lH3T"
        )
        if lam_adapted_ft is None:
            continue
 
        phi_train = modes[key]          # shape (n_train,)
        phi_test  = modes_test[key]     # shape (n_test,)
 
        # Empirical GPR eigenvalue for this mode:
        #   λ_k^GPR = (φᵀ K_train φ) / (φᵀφ)
        # This is the Rayleigh quotient of K_train w.r.t. φ, which equals the
        # eigenvalue when φ is a true eigenfunction.  Under finite-sample
        # fluctuations it is the best single-number summary of the kernel's
        # spectral weight on this mode.
        Kphi = K_train @ phi_train                              # (n_train,)
        phi_norm_sq = float(phi_train @ phi_train)
        lam_gpr_empirical = float(phi_train @ Kphi) / phi_norm_sq
 
        # Adapted eigenvalue in empirical units
        lam_adapted_empirical = float(lam_adapted_ft) * scale_factor
 
        # Gap: positive → adapted gives more weight to this mode than bare GPR
        delta_lambda = lam_adapted_empirical - lam_gpr_empirical
 
        corrections.append((phi_train, phi_test, delta_lambda))
 
    # ------------------------------------------------------------------
    # Step 3: Sherman-Morrison updates to A_inv, one per mode
    #
    # After k updates the effective inverse is
    #   (K_train + σ²I + Σ_k δλ_k φ_k φ_kᵀ)⁻¹
    # Each application costs O(n²) and is numerically exact (no approximation).
    # ------------------------------------------------------------------
    for phi_train, _phi_test, delta_lambda in corrections:
        # Normalize φ to match the convention φᵀφ/n ~ 1 used by the solver.
        # Sherman-Morrison is basis-independent, so the normalization cancels,
        # but we absorb it into delta_lambda to keep the formula clean.
        n = phi_train.shape[0]
        norm_sq = float(phi_train @ phi_train)
        # Renormalize: φ̂ = φ / ‖φ‖,  δλ̂ = δλ · ‖φ‖²
        # so that δλ · φ φᵀ = δλ̂ · φ̂ φ̂ᵀ unchanged.
        # (This keeps the outer product update identical; we just pass the
        # un-normalized φ and the original δλ — the math is the same.)
        A_inv = _sherman_morrison_update(A_inv, phi_train, delta_lambda)
 
    # ------------------------------------------------------------------
    # Step 4: corrected cross-kernel  K_cross_corrected
    #
    # K_cross(x_test, x_train) receives the same rank-1 additions:
    #   K_cross_corrected = K_cross + Σ_k δλ_k · φ_k(X_test) ⊗ φ_k(X_train)
    # ------------------------------------------------------------------
    K_cross_corrected = K_cross.clone()
    for phi_train, phi_test, delta_lambda in corrections:
        # outer product: (n_test, n_train)
        K_cross_corrected = K_cross_corrected + delta_lambda * torch.outer(phi_test, phi_train)
 
    # ------------------------------------------------------------------
    # Step 5: GPR prediction
    #   α = A_inv y_train          (dual weights, O(n²))
    #   ŷ = K_cross_corrected α   (prediction,   O(n_test · n_train))
    # ------------------------------------------------------------------
    alpha = A_inv @ y_train                  # (n_train,)
    y_pred = K_cross_corrected @ alpha       # (n_test,)
 
    # Return y_pred plus training and full kernel for API compatibility
    K_train_corrected = K_train.clone()
    for phi_train, _phi_test, delta_lambda in corrections:
        K_train_corrected = K_train_corrected + delta_lambda * torch.outer(phi_train, phi_train)
 
    return y_pred, K_train_corrected, K_all