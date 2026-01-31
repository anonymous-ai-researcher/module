"""
NFMR: Noise-Free Module Retrieval
Main Forgetting Algorithm

This module implements Algorithm 1 from the paper: the complete zero-noise
knowledge retrieval process that iteratively eliminates symbols while
handling definers and detecting cycles.
"""

from __future__ import annotations
from typing import Set, Optional, Tuple, List, Dict, FrozenSet
from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import time
import logging

from .ontology import OntologyKB, Clause, ConceptInclusion
from .normalizer import DefinerManager, LinearNormalizer
from .eliminator import (
    ConceptEliminator, RoleEliminator,
    eliminate_concept, eliminate_role
)

logger = logging.getLogger(__name__)


class RetrievalStatus(Enum):
    """Status of a retrieval operation."""
    SUCCESS = auto()
    CYCLE_DETECTED = auto()
    TIMEOUT = auto()
    MEMORY_ERROR = auto()
    INCOMPLETE = auto()


@dataclass
class RetrievalResult:
    """Result of a zero-noise retrieval operation."""
    status: RetrievalStatus
    module: Optional[OntologyKB]
    time_seconds: float
    memory_mb: float
    num_definers_introduced: int
    num_definers_remaining: int
    symbols_eliminated: Set[str]
    message: str = ""
    
    @property
    def is_success(self) -> bool:
        """Check if retrieval was successful."""
        return self.status == RetrievalStatus.SUCCESS
    
    @property
    def is_zero_noise(self) -> bool:
        """Check if the result is truly zero-noise (no definers remaining)."""
        return self.is_success and self.num_definers_remaining == 0


def compute_kb_hash(kb: OntologyKB) -> str:
    """Compute a hash of the KB for cycle detection."""
    # Sort clauses for deterministic hashing
    clause_strs = sorted(str(c) for c in kb.clauses)
    combined = "\n".join(clause_strs)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class NoiseFreeForgetting:
    """
    Main class for computing zero-noise modules via forgetting.
    
    Implements the retrieval algorithm from the paper:
    1. Iteratively eliminate each symbol in the forgetting vocabulary
    2. Handle introduced definers
    3. Detect and handle cycles via blocking
    
    Attributes:
        kb: The input knowledge base
        definer_manager: Manager for tracking definers
        reasoner: Optional DL reasoner for subsumption checking
        timeout: Maximum time in seconds
        max_memory_gb: Maximum memory in GB
    """
    
    def __init__(
        self,
        kb: OntologyKB,
        reasoner=None,
        timeout: int = 300,
        max_memory_gb: float = 9.0
    ):
        """
        Initialize the forgetting system.
        
        Args:
            kb: Input knowledge base
            reasoner: Optional DL reasoner
            timeout: Timeout in seconds
            max_memory_gb: Maximum memory in GB
        """
        self.kb = kb
        self.reasoner = reasoner
        self.timeout = timeout
        self.max_memory_gb = max_memory_gb
        self.definer_manager = DefinerManager()
        self._history: Set[str] = set()
    
    def compute_module(
        self, 
        target_vocab: Set[str]
    ) -> RetrievalResult:
        """
        Compute a zero-noise module for the given target vocabulary.
        
        The module will contain only symbols from target_vocab and
        preserve all entailments over target_vocab.
        
        Args:
            target_vocab: Set of concept and role names to keep
            
        Returns:
            RetrievalResult with the computed module
        """
        forgetting_vocab = self.kb.signature - target_vocab
        return self.forget(forgetting_vocab)
    
    def forget(
        self, 
        forgetting_vocab: Set[str]
    ) -> RetrievalResult:
        """
        Forget (eliminate) symbols from the knowledge base.
        
        Args:
            forgetting_vocab: Set of symbols to eliminate
            
        Returns:
            RetrievalResult with the result of forgetting
        """
        start_time = time.time()
        self._history.clear()
        self.definer_manager = DefinerManager()
        
        logger.info(f"Starting forgetting of {len(forgetting_vocab)} symbols")
        
        current_kb = self.kb.copy()
        eliminated: Set[str] = set()
        
        try:
            # Phase 1: Eliminate symbols in forgetting vocabulary
            for symbol in forgetting_vocab:
                if time.time() - start_time > self.timeout:
                    return self._make_result(
                        RetrievalStatus.TIMEOUT,
                        current_kb,
                        start_time,
                        eliminated,
                        message="Timeout during symbol elimination"
                    )
                
                if symbol in current_kb.concept_names:
                    current_kb, _ = eliminate_concept(
                        current_kb, symbol, self.definer_manager, self.reasoner
                    )
                elif symbol in current_kb.role_names:
                    current_kb, _ = eliminate_role(
                        current_kb, symbol, self.definer_manager, self.reasoner
                    )
                
                eliminated.add(symbol)
                logger.debug(f"Eliminated symbol: {symbol}")
            
            # Phase 2: Eliminate introduced definers
            definers_to_eliminate = self.definer_manager.all_definers.copy()
            
            while definers_to_eliminate:
                definer = definers_to_eliminate.pop()
                
                if time.time() - start_time > self.timeout:
                    return self._make_result(
                        RetrievalStatus.TIMEOUT,
                        current_kb,
                        start_time,
                        eliminated,
                        message="Timeout during definer elimination"
                    )
                
                # Check for cycle via blocking
                kb_hash = compute_kb_hash(current_kb)
                if kb_hash in self._history:
                    logger.warning(f"Cycle detected at definer {definer}")
                    return self._make_result(
                        RetrievalStatus.CYCLE_DETECTED,
                        current_kb,
                        start_time,
                        eliminated,
                        message=f"Cycle detected while eliminating definer {definer}"
                    )
                
                self._history.add(kb_hash)
                
                # Eliminate the definer
                if definer in current_kb.concept_names:
                    old_num_definers = self.definer_manager.num_definers
                    current_kb, _ = eliminate_concept(
                        current_kb, definer, self.definer_manager, self.reasoner
                    )
                    
                    # Check for new definers
                    new_definers = self.definer_manager.all_definers - eliminated - definers_to_eliminate
                    definers_to_eliminate.update(new_definers)
                
                eliminated.add(definer)
                logger.debug(f"Eliminated definer: {definer}")
            
            # Check final result
            remaining_definers = self.definer_manager.all_definers & current_kb.signature
            
            if remaining_definers:
                return self._make_result(
                    RetrievalStatus.INCOMPLETE,
                    current_kb,
                    start_time,
                    eliminated,
                    message=f"Definers remaining: {remaining_definers}"
                )
            
            # Simplify the final KB
            current_kb = self._simplify_kb(current_kb)
            
            return self._make_result(
                RetrievalStatus.SUCCESS,
                current_kb,
                start_time,
                eliminated,
                message="Successfully computed zero-noise module"
            )
            
        except MemoryError:
            return self._make_result(
                RetrievalStatus.MEMORY_ERROR,
                current_kb,
                start_time,
                eliminated,
                message="Memory limit exceeded"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def _make_result(
        self,
        status: RetrievalStatus,
        kb: OntologyKB,
        start_time: float,
        eliminated: Set[str],
        message: str = ""
    ) -> RetrievalResult:
        """Create a RetrievalResult."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        remaining_definers = self.definer_manager.all_definers & kb.signature
        
        return RetrievalResult(
            status=status,
            module=kb,
            time_seconds=time.time() - start_time,
            memory_mb=memory_mb,
            num_definers_introduced=self.definer_manager.num_definers,
            num_definers_remaining=len(remaining_definers),
            symbols_eliminated=eliminated,
            message=message
        )
    
    def _simplify_kb(self, kb: OntologyKB) -> OntologyKB:
        """Apply simplifications to the KB."""
        simplified_clauses = set()
        
        for clause in kb.clauses:
            s = clause.simplify()
            if s is not None and not s.is_tautology():
                # Check for subsumption (optional, requires reasoner)
                simplified_clauses.add(s)
        
        return OntologyKB.from_clauses(simplified_clauses)


def compute_zero_noise_module(
    kb: OntologyKB,
    target_vocab: Set[str],
    timeout: int = 300,
    reasoner=None
) -> RetrievalResult:
    """
    Convenience function to compute a zero-noise module.
    
    Args:
        kb: Input knowledge base
        target_vocab: Symbols to keep in the module
        timeout: Timeout in seconds
        reasoner: Optional DL reasoner
        
    Returns:
        RetrievalResult with the computed module
    """
    forgetting = NoiseFreeForgetting(kb, reasoner=reasoner, timeout=timeout)
    return forgetting.compute_module(target_vocab)


def forget_symbols(
    kb: OntologyKB,
    forgetting_vocab: Set[str],
    timeout: int = 300,
    reasoner=None
) -> RetrievalResult:
    """
    Convenience function to forget symbols from a KB.
    
    Args:
        kb: Input knowledge base
        forgetting_vocab: Symbols to eliminate
        timeout: Timeout in seconds
        reasoner: Optional DL reasoner
        
    Returns:
        RetrievalResult with the result
    """
    forgetting = NoiseFreeForgetting(kb, reasoner=reasoner, timeout=timeout)
    return forgetting.forget(forgetting_vocab)


# Module-level exports
__all__ = [
    'NoiseFreeForgetting',
    'RetrievalStatus',
    'RetrievalResult',
    'compute_zero_noise_module',
    'forget_symbols',
]
