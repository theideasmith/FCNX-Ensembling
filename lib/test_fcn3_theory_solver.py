import unittest
import os
from pathlib import Path
from fcn3_theory_solver import fcn3_theory_solver

class TestFCN3TheorySolver(unittest.TestCase):
    def test_basic_run(self):
        # Use small parameters for a fast test
        d = 10
        n1 = 10
        P = 20
        chi = 2.0
        kappa = 1.5
        epsilon = 0.01
        # Use a temporary output file and check result structure
        result = fcn3_theory_solver(
            d=d,
            n1=n1,
            P=P,
            chi=chi,
            kappa=kappa,
            epsilon=epsilon,
            quiet=True
        )
        self.assertIsInstance(result, dict)
        self.assertIn("target", result)
        self.assertIn("perpendicular", result)
        # Check some expected keys in the output
        self.assertTrue("lJ3T" in result["target"] or "lJ1T" in result["target"])

if __name__ == "__main__":
    unittest.main()
