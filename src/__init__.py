"""
NFMR: Noise-Free Module Retrieval

A Python implementation of noise-free module retrieval from large-scale
knowledge bases using forgetting-based methods.

This package provides:
- Core forgetting algorithm for ALCI knowledge bases
- Linear-time normalization strategy
- RAG pipeline with zero-noise retrieval
- Evaluation framework with Bio-RAG-Bench

Example:
    >>> from nfmr import OntologyKB, NoiseFreeForgetting
    >>> kb = OntologyKB.from_owl("ontology.owl")
    >>> forgetting = NoiseFreeForgetting(kb)
    >>> module = forgetting.compute_module({"Disease", "hasSymptom"})
"""

__version__ = "0.1.0"
__author__ = "SIGIR 2026 Authors"

from .core import (
    OntologyKB,
    NoiseFreeForgetting,
    RetrievalResult,
    RetrievalStatus,
    compute_zero_noise_module,
    forget_symbols,
)

from .rag import (
    RAGPipeline,
    RAGResponse,
)

from .evaluation import (
    BioRAGBenchGenerator,
    BioRAGBenchEvaluator,
)

__all__ = [
    # Version
    '__version__',
    
    # Core
    'OntologyKB',
    'NoiseFreeForgetting',
    'RetrievalResult',
    'RetrievalStatus',
    'compute_zero_noise_module',
    'forget_symbols',
    
    # RAG
    'RAGPipeline',
    'RAGResponse',
    
    # Evaluation
    'BioRAGBenchGenerator',
    'BioRAGBenchEvaluator',
]
