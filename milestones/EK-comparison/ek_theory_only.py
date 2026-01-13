"""
Standalone EK Prediction Calculator

Computes EK theoretical predictions without network training.
Useful for:
- Verifying formulas independently
- Quick theoretical analysis
- Comparing with existing experimental results
- Testing different hyperparameter combinations
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class EKConfig:
    """Configuration for standalone EK analysis."""
    chi: float = 1.0
    kappa: float = 0.5
    P_factor: float = 1.5
    d_range: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.d_range is None:
            self.d_range = list(range(2, 11))


def compute_ek_prediction_factor(d: int, kappa: float) -> float:
    """
    Compute the EK prediction factor.
    
    This represents the normalized network output according to EK theory:
    factor = (1/d) / (1/d + kappa)
    
    Interpretation:
    - As d → ∞: factor → 0 (network output vanishes)
    - As kappa → 0: factor → 1 (maximum network contribution)
    - As kappa → ∞: factor → 0 (strong regularization dominates)
    """
    lH = 1.0 / d
    factor = lH / (lH + kappa)
    return factor


def compute_ek_loss_components(
    d: int,
    P: int,
    kappa: float = 0.5,
    chi: float = 1.0
) -> Dict[str, float]:
    """
    Compute EK loss components in detail.
    
    Returns:
    - lH: NNGP eigenvalue (1/d)
    - numerator_bias: κ (numerator of bias)
    - denominator: lH + κ (common denominator)
    - bias: (κ / (lH + κ))²
    - variance: (κ / (χ × P)) × (lH / (lH + κ))
    - total_term: (bias + variance) × χ / κ
    - loss: Final loss value
    """
    lH = 1.0 / d
    numerator_bias = kappa
    denominator = lH + kappa
    
    # Bias: (κ / (lH + κ))²
    bias = (numerator_bias / denominator) ** 2
    
    # Variance: (κ / (χ × P)) × (lH / (lH + κ))
    variance = (kappa / (chi * P)) * (lH / denominator)
    
    # Total before rescaling
    total_unscaled = bias + variance
    
    # Final loss: (bias + variance) × χ / κ
    loss = total_unscaled * chi / kappa
    
    return {
        'lH': lH,
        'kappa': kappa,
        'P': P,
        'chi': chi,
        'd': d,
        'numerator_bias': numerator_bias,
        'denominator': denominator,
        'bias': bias,
        'variance': variance,
        'total_unscaled': total_unscaled,
        'scaling_factor': chi / kappa,
        'loss': loss,
    }


def analyze_ek_predictions(config: EKConfig) -> Dict[int, Dict[str, Any]]:
    """Compute EK predictions for all d values."""
    results = {}
    
    for d in config.d_range:
        P = int(np.round(config.P_factor * d))
        
        factor = compute_ek_prediction_factor(d, config.kappa)
        components = compute_ek_loss_components(d, P, config.kappa, config.chi)
        
        results[d] = {
            'P': P,
            'prediction_factor': factor,
            'components': components,
        }
    
    return results


def print_ek_analysis(config: EKConfig, results: Dict) -> None:
    """Print detailed EK analysis table."""
    print("\n" + "="*110)
    print("EK THEORETICAL ANALYSIS (No Network Training)")
    print("="*110)
    print(f"Configuration: χ={config.chi}, κ={config.kappa}, P=1.5d (rounded)")
    print("="*110)
    
    # Header
    print(f"{'d':<4} {'P':<5} {'lH':<10} {'Factor':<12} {'Bias':<13} {'Variance':<13} {'Loss':<13}")
    print("-"*110)
    
    # Data rows
    for d in sorted(results.keys()):
        data = results[d]
        comp = data['components']
        
        print(f"{d:<4} {data['P']:<5} {comp['lH']:<10.6f} {data['prediction_factor']:<12.6f} "
              f"{comp['bias']:<13.6e} {comp['variance']:<13.6e} {comp['loss']:<13.6e}")
    
    print("="*110)


def print_detailed_analysis(config: EKConfig, results: Dict) -> None:
    """Print very detailed analysis for each d value."""
    print("\n" + "="*100)
    print("DETAILED EK COMPONENT ANALYSIS")
    print("="*100)
    
    for d in sorted(results.keys()):
        data = results[d]
        comp = data['components']
        
        print(f"\n{'─'*100}")
        print(f"Input Dimension: d = {d}")
        print(f"{'─'*100}")
        print(f"  NNGP Eigenvalue (lH = 1/d)           : {comp['lH']:.10f}")
        print(f"  Sample Size (P = 1.5d rounded)       : {data['P']}")
        print(f"  EK Constant (κ)                      : {comp['kappa']:.10f}")
        print(f"  Scaling Regime (χ)                   : {comp['chi']:.10f}")
        print(f"\n  Prediction Factor: (1/d) / (1/d + κ) : {data['prediction_factor']:.10f}")
        print(f"\n  Loss Components:")
        print(f"    Numerator (κ)                     : {comp['numerator_bias']:.10e}")
        print(f"    Denominator (lH + κ)              : {comp['denominator']:.10e}")
        print(f"    Bias = (κ / (lH + κ))²           : {comp['bias']:.10e}")
        print(f"    Variance = (κ / (χ×P)) × (lH / denom) : {comp['variance']:.10e}")
        print(f"    Bias + Variance (unscaled)        : {comp['total_unscaled']:.10e}")
        print(f"    Scaling Factor (χ / κ)            : {comp['scaling_factor']:.10f}")
        print(f"\n  🎯 FINAL LOSS = {comp['loss']:.10e}")
        
        # Ratio analysis
        if comp['total_unscaled'] > 0:
            bias_ratio = comp['bias'] / comp['total_unscaled'] * 100
            var_ratio = comp['variance'] / comp['total_unscaled'] * 100
            print(f"\n  Bias-Variance Ratio:")
            print(f"    Bias % of total:                 {bias_ratio:.2f}%")
            print(f"    Variance % of total:             {var_ratio:.2f}%")


def export_results_json(config: EKConfig, results: Dict, output_path: Path) -> None:
    """Export results to JSON."""
    export_data = {}
    
    for d, data in results.items():
        export_data[str(d)] = {
            'P': data['P'],
            'ek_prediction_factor': data['prediction_factor'],
            'ek_loss': data['components'],
        }
    
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / 'ek_theoretical_predictions.json'
    
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✓ Results exported to: {json_path}")


def analyze_scaling_behavior(config: EKConfig, results: Dict) -> None:
    """Analyze how loss scales with dimension."""
    print("\n" + "="*80)
    print("SCALING BEHAVIOR ANALYSIS")
    print("="*80)
    
    d_values = np.array(sorted(results.keys()), dtype=float)
    losses = np.array([results[int(d)]['components']['loss'] for d in d_values])
    biases = np.array([results[int(d)]['components']['bias'] for d in d_values])
    variances = np.array([results[int(d)]['components']['variance'] for d in d_values])
    
    # Fit power law: y = a * d^b
    log_d = np.log10(d_values)
    log_loss = np.log10(losses)
    
    # Linear fit in log-log space
    coeffs_loss = np.polyfit(log_d, log_loss, 1)
    power_loss = coeffs_loss[0]
    
    coeffs_bias = np.polyfit(log_d, np.log10(biases), 1)
    power_bias = coeffs_bias[0]
    
    coeffs_var = np.polyfit(log_d, np.log10(variances), 1)
    power_var = coeffs_var[0]
    
    print(f"\nPower-Law Fits (y ∝ d^b in log-log space):")
    print(f"  Loss:     ∝ d^{power_loss:.4f}")
    print(f"  Bias:     ∝ d^{power_bias:.4f}")
    print(f"  Variance: ∝ d^{power_var:.4f}")
    
    print(f"\nInterpretation:")
    if power_loss < 0:
        print(f"  → Loss DECREASES with dimension (d^{power_loss:.2f})")
    elif power_loss > 0:
        print(f"  → Loss INCREASES with dimension (d^{power_loss:.2f})")
    else:
        print(f"  → Loss is approximately CONSTANT with dimension")
    
    # Check which term dominates
    bias_mean = np.mean(biases)
    var_mean = np.mean(variances)
    
    print(f"\nDominant Term Analysis:")
    print(f"  Mean Bias:     {bias_mean:.6e}")
    print(f"  Mean Variance: {var_mean:.6e}")
    
    if bias_mean > var_mean:
        print(f"  → Bias dominates (ratio: {bias_mean/var_mean:.2f}:1)")
    else:
        print(f"  → Variance dominates (ratio: {var_mean/bias_mean:.2f}:1)")


def main():
    """Main execution function."""
    print("="*100)
    print("EK Theoretical Predictions - Standalone Analysis")
    print("="*100)
    
    # Setup
    config = EKConfig()
    print(f"\nConfiguration:")
    print(f"  χ (scaling):     {config.chi}")
    print(f"  κ (EK constant): {config.kappa}")
    print(f"  P factor:        {config.P_factor}")
    print(f"  d range:         {config.d_range}")
    
    # Compute predictions
    print(f"\nComputing EK predictions...")
    results = analyze_ek_predictions(config)
    
    # Display results
    print_ek_analysis(config, results)
    print_detailed_analysis(config, results)
    analyze_scaling_behavior(config, results)
    
    # Export results
    output_dir = Path('/home/akiva/FCNX-Ensembling/milestones/EK-comparison/data/theory')
    export_results_json(config, results, output_dir)
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)
    
    return config, results


if __name__ == '__main__':
    config, results = main()
