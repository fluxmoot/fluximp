#!/usr/bin/env python3
"""ExperimentsService - Manages enabled experiments.

This service tracks which experiments are enabled via a frozen set
of experiment names. It provides a simple is_enabled method to check
if a specific experiment is active.
"""

from typing import FrozenSet


class ExperimentsService:
    """Service for managing and checking experiment status.
    
    Experiments are tracked via a frozen set of enabled experiment names.
    The set is frozen at initialization to prevent runtime modifications.
    """

    # insert_lines has not had any testing yet
    # move_lines has not had any testing yet
    # adjust_indent has not had any testing yet
    # adjust_indent_exact has not had any testing yet
    # search_list_view has not had any testing yet
    
    # Experiment pairs that are mutually exclusive
    MUTUALLY_EXCLUSIVE = [
        ("insert_lines", "insert_line_validation"),
        ("move_lines", "move_line_validation"),
        ("adjust_indent", "adjust_indent_validation"),
        ("adjust_indent_exact", "adjust_indent_exact_validation"),
    ]
    
    def __init__(self, enabled_experiments: FrozenSet[str] = None):
        """Initialize with a frozen set of enabled experiment names.
        
        Args:
            enabled_experiments: Frozen set of experiment names that are enabled.
                                  Defaults to an empty set if not provided.
        """
        self._enabled = enabled_experiments if enabled_experiments else frozenset()
        
        # Validate mutually exclusive experiments
        self._validate_mutually_exclusive()
    
    def _validate_mutually_exclusive(self) -> None:
        """Validate that mutually exclusive experiments are not both enabled.
        
        Raises:
            AssertionError: If any pair of mutually exclusive experiments are both enabled.
        """
        for exp1, exp2 in self.MUTUALLY_EXCLUSIVE:
            if exp1 in self._enabled and exp2 in self._enabled:
                raise AssertionError(
                    f"Mutually exclusive experiments '{exp1}' and '{exp2}' cannot both be enabled. "
                    f"Please disable one of them."
                )
    
    def is_enabled(self, experiment_name: str) -> bool:
        """Check if a specific experiment is enabled.
        
        Args:
            experiment_name: Name of the experiment to check.
            
        Returns:
            True if the experiment is enabled, False otherwise.
        """
        return experiment_name in self._enabled
