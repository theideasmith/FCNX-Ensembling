# Notes

Discussion notes for the thesis advisor, using the MSc thesis notation.

| File | Description |
|------|-------------|
| `readin_action_mixed_J_GL.tex` / `.pdf` | Two-layer read-in action with **nondiagonal** He1–He3 block \(J\) (mixed space), then Ginzburg–Landau analysis of inner vs outer transitions. |
| `fcn3_matrix_eos.tex` / `.pdf` | Self-contained: three-layer erf field theory, mean-field decoupling, equations of state as full Hermite matrices with a non-Gaussian hidden field; diagonal Gaussian limit; why N1 → ∞ is not a two-layer net. Appendix: the exact posterior sampler, its fast GPU implementation and validation. |

Rebuild PDF:

```bash
cd notes && conda run -n tex tectonic <file>.tex
```
