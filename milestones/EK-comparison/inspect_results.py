"""
Result Inspection Utility

Provides convenient functions for inspecting and analyzing results
from the EK-comparison milestone.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ResultInspector:
    """Utility class for inspecting EK-comparison results."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize inspector with results directory."""
        if results_dir is None:
            results_dir = Path('/home/akiva/FCNX-Ensembling/milestones/EK-comparison/data/results')
        
        self.results_dir = results_dir
        self.config = None
        self.comparison = None
        
        if results_dir.exists():
            self.load_results()
    
    def load_results(self):
        """Load configuration and comparison data."""
        config_path = self.results_dir / 'config.json'
        comparison_path = self.results_dir / 'ek_comparison.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        if comparison_path.exists():
            with open(comparison_path, 'r') as f:
                self.comparison = json.load(f)
    
    def print_config(self):
        """Print experiment configuration."""
        if self.config is None:
            print("❌ Configuration not loaded")
            return
        
        print("\n" + "="*70)
        print("EXPERIMENT CONFIGURATION")
        print("="*70)
        
        for key, value in sorted(self.config.items()):
            if isinstance(value, list):
                print(f"  {key:<25}: {value}")
            else:
                print(f"  {key:<25}: {value}")
        
        print("="*70)
    
    def print_summary_statistics(self):
        """Print summary statistics."""
        if self.comparison is None:
            print("❌ Comparison data not loaded")
            return
        
        d_values = sorted([int(d) for d in self.comparison.keys()])
        
        ek_factors = []
        ek_losses = []
        emp_losses = []
        biases = []
        variances = []
        P_values = []
        
        for d in d_values:
            data = self.comparison[str(d)]
            ek_factors.append(data['ek_prediction_factor'])
            ek_loss = data['ek_loss']
            ek_losses.append(ek_loss['loss'])
            emp_losses.append(data['empirical_loss_mean'])
            biases.append(ek_loss['bias'])
            variances.append(ek_loss['variance'])
            P_values.append(data['P'])
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\n📊 EK Prediction Factors (Feature Importance):")
        print(f"  Mean:     {np.mean(ek_factors):.6f}")
        print(f"  Min:      {np.min(ek_factors):.6f} (d={d_values[np.argmin(ek_factors)]})")
        print(f"  Max:      {np.max(ek_factors):.6f} (d={d_values[np.argmax(ek_factors)]})")
        print(f"  Std:      {np.std(ek_factors):.6f}")
        
        print(f"\n💥 EK Loss (Theory):")
        print(f"  Mean:     {np.mean(ek_losses):.6e}")
        print(f"  Min:      {np.min(ek_losses):.6e} (d={d_values[np.argmin(ek_losses)]})")
        print(f"  Max:      {np.max(ek_losses):.6e} (d={d_values[np.argmax(ek_losses)]})")
        print(f"  Reduction: {(1 - np.min(ek_losses)/np.max(ek_losses))*100:.1f}%")
        
        print(f"\n📈 Empirical Loss (Experiment):")
        print(f"  Mean:     {np.mean(emp_losses):.6e}")
        print(f"  Min:      {np.min(emp_losses):.6e} (d={d_values[np.argmin(emp_losses)]})")
        print(f"  Max:      {np.max(emp_losses):.6e} (d={d_values[np.argmax(emp_losses)]})")
        print(f"  Reduction: {(1 - np.min(emp_losses)/np.max(emp_losses))*100:.1f}%")
        
        print(f"\n⚖️ Bias-Variance:")
        print(f"  Mean Bias:     {np.mean(biases):.6e}")
        print(f"  Mean Variance: {np.mean(variances):.6e}")
        print(f"  Ratio (Bias/Var): {np.mean(biases)/np.mean(variances):.2f}:1")
        
        print(f"\n🎯 Theory vs Empirical Agreement:")
        relative_errors = [abs(e-t)/max(abs(t), 1e-10)*100 
                          for e, t in zip(emp_losses, ek_losses)]
        print(f"  Mean Relative Error: {np.mean(relative_errors):.2f}%")
        print(f"  Median Relative Error: {np.median(relative_errors):.2f}%")
        print(f"  Max Relative Error: {np.max(relative_errors):.2f}%")
        
        print("="*70)
    
    def print_detailed_for_d(self, d: int):
        """Print detailed information for a specific dimension."""
        if self.comparison is None:
            print("❌ Comparison data not loaded")
            return
        
        if str(d) not in self.comparison:
            print(f"❌ No data for d={d}")
            return
        
        data = self.comparison[str(d)]
        ek_loss = data['ek_loss']
        
        print("\n" + "="*70)
        print(f"DETAILED ANALYSIS FOR d={d}")
        print("="*70)
        
        print(f"\n📐 Network Configuration:")
        print(f"  Input Dimension (d):              {d}")
        print(f"  Sample Size (P):                  {data['P']}")
        print(f"  Hidden Width (N):                 512")
        print(f"  Total Networks:                   {data['num_samples']}")
        
        print(f"\n🧮 EK Theoretical Predictions:")
        print(f"  Prediction Factor (1/d)/(1/d+κ): {data['ek_prediction_factor']:.10f}")
        print(f"  NNGP Eigenvalue (lH):            {ek_loss['lH']:.10f}")
        print(f"  Bias Term:                       {ek_loss['bias']:.10e}")
        print(f"  Variance Term:                   {ek_loss['variance']:.10e}")
        print(f"  Total EK Loss:                   {ek_loss['loss']:.10e}")
        
        print(f"\n📊 Empirical Results:")
        print(f"  Mean Loss:                       {data['empirical_loss_mean']:.10e}")
        print(f"  Std Loss:                        {data['empirical_loss_std']:.10e}")
        print(f"  Standard Error:                  {data['empirical_loss_std']/np.sqrt(data['num_samples']):.10e}")
        
        print(f"\n📈 Bias-Variance Decomposition (%):")
        total = ek_loss['bias'] + ek_loss['variance']
        if total > 0:
            bias_pct = ek_loss['bias'] / total * 100
            var_pct = ek_loss['variance'] / total * 100
            print(f"  Bias:                            {bias_pct:.2f}%")
            print(f"  Variance:                        {var_pct:.2f}%")
        
        print(f"\n🎯 Theory-Empirical Agreement:")
        rel_error = abs(data['empirical_loss_mean'] - ek_loss['loss']) / \
                   max(abs(ek_loss['loss']), 1e-10) * 100
        print(f"  Relative Error:                  {rel_error:.2f}%")
        print(f"  Absolute Difference:             {abs(data['empirical_loss_mean'] - ek_loss['loss']):.10e}")
        
        print("="*70)
    
    def compare_dimensions(self, d_values: Optional[list] = None):
        """Compare results across multiple dimensions."""
        if self.comparison is None:
            print("❌ Comparison data not loaded")
            return
        
        if d_values is None:
            d_values = sorted([int(d) for d in self.comparison.keys()])
        
        print("\n" + "="*100)
        print("CROSS-DIMENSION COMPARISON")
        print("="*100)
        print(f"{'d':<4} {'P':<4} {'Factor':<10} {'EK Loss':<13} {'Emp Loss':<13} {'Rel Err %':<10} {'Bias %':<8}")
        print("-"*100)
        
        for d in d_values:
            if str(d) not in self.comparison:
                continue
            
            data = self.comparison[str(d)]
            ek_loss = data['ek_loss']
            
            factor = data['ek_prediction_factor']
            ek_l = ek_loss['loss']
            emp_l = data['empirical_loss_mean']
            rel_err = abs(emp_l - ek_l) / max(abs(ek_l), 1e-10) * 100
            
            total = ek_loss['bias'] + ek_loss['variance']
            bias_pct = ek_loss['bias'] / total * 100 if total > 0 else 0
            
            print(f"{d:<4} {data['P']:<4} {factor:<10.6f} {ek_l:<13.6e} {emp_l:<13.6e} {rel_err:<10.2f} {bias_pct:<8.1f}")
        
        print("="*100)
    
    def export_comparison_table(self, output_file: Optional[Path] = None):
        """Export comparison as formatted table."""
        if output_file is None:
            output_file = self.results_dir / 'comparison_table.txt'
        
        lines = []
        lines.append("="*110)
        lines.append("EK-COMPARISON RESULTS TABLE")
        lines.append("="*110)
        
        if self.config:
            lines.append(f"Configuration: N={self.config['N']}, χ={self.config['chi']}, " +
                        f"κ={self.config['kappa_ek']}, P=1.5d")
        
        lines.append("="*110)
        lines.append(f"{'d':<5} {'P':<5} {'Factor':<12} {'EK Loss':<15} {'Emp Loss':<15} {'Bias':<15} {'Var':<15} {'RelErr%':<10}")
        lines.append("-"*110)
        
        if self.comparison:
            for d in sorted([int(d) for d in self.comparison.keys()]):
                data = self.comparison[str(d)]
                ek_loss = data['ek_loss']
                
                rel_err = abs(data['empirical_loss_mean'] - ek_loss['loss']) / \
                         max(abs(ek_loss['loss']), 1e-10) * 100
                
                lines.append(
                    f"{d:<5} {data['P']:<5} {data['ek_prediction_factor']:<12.6f} " +
                    f"{ek_loss['loss']:<15.6e} {data['empirical_loss_mean']:<15.6e} " +
                    f"{ek_loss['bias']:<15.6e} {ek_loss['variance']:<15.6e} {rel_err:<10.2f}"
                )
        
        lines.append("="*110)
        
        output = "\n".join(lines)
        
        with open(output_file, 'w') as f:
            f.write(output)
        
        print(output)
        print(f"\n✓ Table exported to: {output_file}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("EK-Comparison Result Inspector")
    print("="*70)
    
    inspector = ResultInspector()
    
    if inspector.config is None or inspector.comparison is None:
        print("\n❌ Results not found. Please run ek_comparison.py first.")
        print(f"   Expected location: /home/akiva/FCNX-Ensembling/milestones/EK-comparison/data/results/")
        return
    
    print(f"\n✓ Successfully loaded results")
    print(f"  Config file: {inspector.results_dir / 'config.json'}")
    print(f"  Data file:   {inspector.results_dir / 'ek_comparison.json'}")
    
    # Print all information
    inspector.print_config()
    inspector.print_summary_statistics()
    inspector.compare_dimensions()
    
    # Print details for each dimension
    d_values = sorted([int(d) for d in inspector.comparison.keys()])
    for d in d_values:
        inspector.print_detailed_for_d(d)
    
    # Export table
    inspector.export_comparison_table()
    
    print("\n" + "="*70)
    print("Inspection complete!")
    print("="*70)


if __name__ == '__main__':
    main()
