#!/usr/bin/env python3
"""Compatibility entry point for the current N-dimensional PMF analyzer."""

try:
    from .analyze_ND import PMFAnalyzer, main
except ImportError:
    from analyze_ND import PMFAnalyzer, main

__all__ = ["PMFAnalyzer", "main"]


if __name__ == "__main__":
    main()
