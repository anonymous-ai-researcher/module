"""
NFMR Core Module

This module provides the core functionality for noise-free module retrieval
from ALCI knowledge bases.
"""

from .ontology import (
    Concept,
    Top,
    Bottom,
    AtomicConcept,
    Negation,
    Conjunction,
    Disjunction,
    ExistentialRestriction,
    UniversalRestriction,
    Role,
    ConceptInclusion,
    Clause,
    OntologyKB,
    make_conjunction,
    make_disjunction,
)

from .normalizer import (
    DefinerManager,
    LinearNormalizer,
    normalize_kb_for_concept,
    normalize_kb_for_role,
)

from .eliminator import (
    ConceptEliminator,
    RoleEliminator,
    eliminate_concept,
    eliminate_role,
)

from .forgetting import (
    NoiseFreeForgetting,
    RetrievalStatus,
    RetrievalResult,
    compute_zero_noise_module,
    forget_symbols,
)

__all__ = [
    # Ontology constructs
    'Concept',
    'Top',
    'Bottom',
    'AtomicConcept',
    'Negation',
    'Conjunction',
    'Disjunction',
    'ExistentialRestriction',
    'UniversalRestriction',
    'Role',
    'ConceptInclusion',
    'Clause',
    'OntologyKB',
    'make_conjunction',
    'make_disjunction',
    
    # Normalization
    'DefinerManager',
    'LinearNormalizer',
    'normalize_kb_for_concept',
    'normalize_kb_for_role',
    
    # Elimination
    'ConceptEliminator',
    'RoleEliminator',
    'eliminate_concept',
    'eliminate_role',
    
    # Forgetting
    'NoiseFreeForgetting',
    'RetrievalStatus',
    'RetrievalResult',
    'compute_zero_noise_module',
    'forget_symbols',
]
