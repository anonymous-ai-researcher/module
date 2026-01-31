"""
NFMR: Noise-Free Module Retrieval
Linear-Time Normalization Module

This module implements the novel linear-time normalization strategy that
confines definer introduction to a single preprocessing pass, avoiding
exponential blow-up in the forgetting process.
"""

from __future__ import annotations
from typing import Set, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid

from .ontology import (
    Concept, Top, Bottom, AtomicConcept, Negation, Conjunction, Disjunction,
    ExistentialRestriction, UniversalRestriction, Role, Clause, OntologyKB,
    ConceptInclusion, make_disjunction
)


class LiteralType(Enum):
    """Classification of literals for normalization."""
    POSITIVE_ATOMIC = auto()      # A
    NEGATIVE_ATOMIC = auto()      # ¬A
    EXISTENTIAL = auto()          # ∃r.C
    EXISTENTIAL_INV = auto()      # ∃r⁻.C
    UNIVERSAL = auto()            # ∀r.C
    UNIVERSAL_INV = auto()        # ∀r⁻.C
    OTHER = auto()


@dataclass
class DefinerManager:
    """
    Manages the creation and tracking of definers.
    
    Definers are fresh concept names introduced during normalization
    to represent complex subformulas.
    """
    prefix: str = "D_"
    _counter: int = field(default=0, init=False)
    _definers: Set[str] = field(default_factory=set, init=False)
    _definer_map: Dict[Concept, str] = field(default_factory=dict, init=False)
    
    def create_definer(self, concept: Optional[Concept] = None) -> str:
        """
        Create a new definer, optionally associated with a concept.
        
        Args:
            concept: Optional concept to associate with the definer
            
        Returns:
            Fresh definer name
        """
        # Check if we already have a definer for this concept
        if concept is not None and concept in self._definer_map:
            return self._definer_map[concept]
        
        definer = f"{self.prefix}{self._counter}"
        self._counter += 1
        self._definers.add(definer)
        
        if concept is not None:
            self._definer_map[concept] = definer
        
        return definer
    
    def is_definer(self, name: str) -> bool:
        """Check if a name is a definer."""
        return name in self._definers
    
    @property
    def all_definers(self) -> Set[str]:
        """Return all created definers."""
        return self._definers.copy()
    
    @property
    def num_definers(self) -> int:
        """Return the number of definers created."""
        return len(self._definers)
    
    def get_concept_for_definer(self, definer: str) -> Optional[Concept]:
        """Get the concept associated with a definer, if any."""
        for concept, d in self._definer_map.items():
            if d == definer:
                return concept
        return None


def classify_literal(literal: Concept, symbol: str) -> LiteralType:
    """
    Classify a literal with respect to a symbol.
    
    Args:
        literal: The literal to classify
        symbol: The symbol being eliminated
        
    Returns:
        The type of the literal
    """
    if isinstance(literal, AtomicConcept):
        if literal.name == symbol:
            return LiteralType.POSITIVE_ATOMIC
        return LiteralType.OTHER
    
    if isinstance(literal, Negation):
        inner = literal.operand
        if isinstance(inner, AtomicConcept) and inner.name == symbol:
            return LiteralType.NEGATIVE_ATOMIC
    
    if isinstance(literal, ExistentialRestriction):
        if literal.role.is_inverse:
            return LiteralType.EXISTENTIAL_INV
        return LiteralType.EXISTENTIAL
    
    if isinstance(literal, UniversalRestriction):
        if literal.role.is_inverse:
            return LiteralType.UNIVERSAL_INV
        return LiteralType.UNIVERSAL
    
    return LiteralType.OTHER


def count_symbol_occurrences(clause: Clause, symbol: str) -> int:
    """Count the number of occurrences of a symbol in a clause."""
    count = 0
    for literal in clause.literals:
        if literal.contains_symbol(symbol):
            count += 1
    return count


def get_filler_with_symbol(
    literal: Concept, 
    symbol: str
) -> Optional[Tuple[Concept, Role, bool]]:
    """
    Get the filler containing the symbol from a role restriction.
    
    Returns:
        Tuple of (filler, role, is_existential) or None
    """
    if isinstance(literal, ExistentialRestriction):
        if literal.filler.contains_symbol(symbol):
            return (literal.filler, literal.role, True)
    elif isinstance(literal, UniversalRestriction):
        if literal.filler.contains_symbol(symbol):
            return (literal.filler, literal.role, False)
    return None


class LinearNormalizer:
    """
    Implements linear-time normalization for ALCI knowledge bases.
    
    The normalizer transforms clauses into A-reduced or r-reduced form
    by introducing definers in a controlled manner, ensuring that the
    number of definers grows linearly with the KB size.
    """
    
    def __init__(self, definer_manager: Optional[DefinerManager] = None):
        """
        Initialize the normalizer.
        
        Args:
            definer_manager: Optional DefinerManager for definer tracking
        """
        self.definer_manager = definer_manager or DefinerManager()
    
    def normalize_for_concept(
        self, 
        kb: OntologyKB, 
        concept_name: str
    ) -> Tuple[OntologyKB, Set[Clause]]:
        """
        Normalize KB to A-reduced form for eliminating a concept name.
        
        A clause is in A-reduced form if it has the form C ⊔ L, where:
        - L ∈ {A, ¬A, ∃r.A, ∃r.¬A, ∀r.A, ∀r.¬A, ∃r⁻.A, ∃r⁻.¬A, ∀r⁻.A, ∀r⁻.¬A}
        - C is A-free
        
        Args:
            kb: The input knowledge base
            concept_name: The concept name to normalize for
            
        Returns:
            Tuple of (normalized KB, set of A-clauses)
        """
        clauses = kb.clauses
        normalized_clauses: Set[Clause] = set()
        a_clauses: Set[Clause] = set()
        new_definitions: List[Clause] = []
        
        for clause in clauses:
            if not clause.contains_symbol(concept_name):
                # Clause doesn't contain A, keep as is
                normalized_clauses.add(clause)
                continue
            
            # Count occurrences of A in the clause
            occurrences = count_symbol_occurrences(clause, concept_name)
            
            if occurrences == 1:
                # Check if A is in a valid position (A-reduced form)
                if self._is_a_reduced(clause, concept_name):
                    a_clauses.add(clause)
                    continue
            
            # Need to normalize this clause
            normalized, definitions = self._normalize_clause_for_concept(
                clause, concept_name
            )
            
            for nc in normalized:
                if nc.contains_symbol(concept_name):
                    a_clauses.add(nc)
                else:
                    normalized_clauses.add(nc)
            
            new_definitions.extend(definitions)
        
        # Add definition clauses
        for defn in new_definitions:
            if defn.contains_symbol(concept_name):
                a_clauses.add(defn)
            else:
                normalized_clauses.add(defn)
        
        # Create new KB from all clauses
        all_clauses = normalized_clauses | a_clauses
        return OntologyKB.from_clauses(all_clauses), a_clauses
    
    def normalize_for_role(
        self, 
        kb: OntologyKB, 
        role_name: str
    ) -> Tuple[OntologyKB, Set[Clause]]:
        """
        Normalize KB to r-reduced form for eliminating a role name.
        
        A clause is in r-reduced form if it has the form C ⊔ L, where:
        - L ∈ {∃r.D, ∃r⁻.D, ∀r.D, ∀r⁻.D}
        - C and D are r-free
        
        Args:
            kb: The input knowledge base
            role_name: The role name to normalize for
            
        Returns:
            Tuple of (normalized KB, set of r-clauses)
        """
        clauses = kb.clauses
        normalized_clauses: Set[Clause] = set()
        r_clauses: Set[Clause] = set()
        new_definitions: List[Clause] = []
        
        for clause in clauses:
            if not clause.contains_symbol(role_name):
                # Clause doesn't contain r, keep as is
                normalized_clauses.add(clause)
                continue
            
            if self._is_r_reduced(clause, role_name):
                r_clauses.add(clause)
                continue
            
            # Need to normalize this clause
            normalized, definitions = self._normalize_clause_for_role(
                clause, role_name
            )
            
            for nc in normalized:
                if nc.contains_symbol(role_name):
                    r_clauses.add(nc)
                else:
                    normalized_clauses.add(nc)
            
            new_definitions.extend(definitions)
        
        # Add definition clauses
        for defn in new_definitions:
            if defn.contains_symbol(role_name):
                r_clauses.add(defn)
            else:
                normalized_clauses.add(defn)
        
        all_clauses = normalized_clauses | r_clauses
        return OntologyKB.from_clauses(all_clauses), r_clauses
    
    def _is_a_reduced(self, clause: Clause, concept_name: str) -> bool:
        """Check if a clause is in A-reduced form."""
        a_literals = []
        
        for lit in clause.literals:
            if not lit.contains_symbol(concept_name):
                continue
            
            # Check valid forms for A-occurrence
            if isinstance(lit, AtomicConcept) and lit.name == concept_name:
                a_literals.append(lit)
            elif isinstance(lit, Negation):
                inner = lit.operand
                if isinstance(inner, AtomicConcept) and inner.name == concept_name:
                    a_literals.append(lit)
            elif isinstance(lit, (ExistentialRestriction, UniversalRestriction)):
                filler = lit.filler
                # Filler must be A or ¬A
                if isinstance(filler, AtomicConcept) and filler.name == concept_name:
                    a_literals.append(lit)
                elif isinstance(filler, Negation):
                    if isinstance(filler.operand, AtomicConcept):
                        if filler.operand.name == concept_name:
                            a_literals.append(lit)
                else:
                    return False
            else:
                return False
        
        return len(a_literals) == 1
    
    def _is_r_reduced(self, clause: Clause, role_name: str) -> bool:
        """Check if a clause is in r-reduced form."""
        r_count = 0
        
        for lit in clause.literals:
            if not lit.contains_symbol(role_name):
                continue
            
            r_count += 1
            
            # Must be a role restriction with r
            if not isinstance(lit, (ExistentialRestriction, UniversalRestriction)):
                return False
            
            if lit.role.name != role_name:
                return False
            
            # Filler must be r-free
            if lit.filler.contains_symbol(role_name):
                return False
        
        return r_count == 1
    
    def _normalize_clause_for_concept(
        self, 
        clause: Clause, 
        concept_name: str
    ) -> Tuple[List[Clause], List[Clause]]:
        """
        Normalize a single clause for concept elimination.
        
        Returns:
            Tuple of (normalized clauses, definition clauses)
        """
        result_clauses = []
        definitions = []
        
        # Separate A-containing and A-free literals
        a_literals = []
        other_literals = []
        
        for lit in clause.literals:
            if lit.contains_symbol(concept_name):
                a_literals.append(lit)
            else:
                other_literals.append(lit)
        
        # Handle each A-literal
        for a_lit in a_literals:
            new_lit, new_defs = self._extract_a_literal(a_lit, concept_name)
            
            # Create clause with this A-literal and all other literals
            new_clause_lits = frozenset(other_literals + [new_lit])
            result_clauses.append(Clause(new_clause_lits))
            definitions.extend(new_defs)
        
        return result_clauses, definitions
    
    def _normalize_clause_for_role(
        self, 
        clause: Clause, 
        role_name: str
    ) -> Tuple[List[Clause], List[Clause]]:
        """
        Normalize a single clause for role elimination.
        
        Returns:
            Tuple of (normalized clauses, definition clauses)
        """
        result_clauses = []
        definitions = []
        
        # Separate r-containing and r-free literals
        r_literals = []
        other_literals = []
        
        for lit in clause.literals:
            if lit.contains_symbol(role_name):
                r_literals.append(lit)
            else:
                other_literals.append(lit)
        
        # Handle each r-literal
        for r_lit in r_literals:
            new_lit, new_defs = self._extract_r_literal(r_lit, role_name)
            
            # Create clause with this r-literal and all other literals
            new_clause_lits = frozenset(other_literals + [new_lit])
            result_clauses.append(Clause(new_clause_lits))
            definitions.extend(new_defs)
        
        return result_clauses, definitions
    
    def _extract_a_literal(
        self, 
        literal: Concept, 
        concept_name: str
    ) -> Tuple[Concept, List[Clause]]:
        """
        Extract an A-reduced literal from a complex literal.
        
        Transformation rules:
        1. If L = ∃R.C where A ∈ sig(C) and C ≠ A:
           Replace C with fresh Z, add ¬Z ⊔ C
        2. If L = ∀R.C where A ∈ sig(C) and C ≠ A:
           Replace C with fresh Z, add ¬Z ⊔ C
           
        Returns:
            Tuple of (transformed literal, definition clauses)
        """
        definitions = []
        
        if isinstance(literal, AtomicConcept):
            if literal.name == concept_name:
                return literal, []
        
        if isinstance(literal, Negation):
            inner = literal.operand
            if isinstance(inner, AtomicConcept) and inner.name == concept_name:
                return literal, []
            # Recursively handle negation
            new_inner, defs = self._extract_a_literal(inner, concept_name)
            definitions.extend(defs)
            return Negation(new_inner), definitions
        
        if isinstance(literal, ExistentialRestriction):
            filler = literal.filler
            if filler.contains_symbol(concept_name):
                # Check if filler is already A or ¬A
                if isinstance(filler, AtomicConcept) and filler.name == concept_name:
                    return literal, []
                if isinstance(filler, Negation):
                    if isinstance(filler.operand, AtomicConcept):
                        if filler.operand.name == concept_name:
                            return literal, []
                
                # Need to introduce a definer
                definer_name = self.definer_manager.create_definer(filler)
                definer = AtomicConcept(definer_name)
                
                # Add definition: ¬Z ⊔ C
                def_clause = Clause(frozenset([Negation(definer), filler]))
                definitions.append(def_clause)
                
                return ExistentialRestriction(literal.role, definer), definitions
            return literal, []
        
        if isinstance(literal, UniversalRestriction):
            filler = literal.filler
            if filler.contains_symbol(concept_name):
                # Check if filler is already A or ¬A
                if isinstance(filler, AtomicConcept) and filler.name == concept_name:
                    return literal, []
                if isinstance(filler, Negation):
                    if isinstance(filler.operand, AtomicConcept):
                        if filler.operand.name == concept_name:
                            return literal, []
                
                # Need to introduce a definer
                definer_name = self.definer_manager.create_definer(filler)
                definer = AtomicConcept(definer_name)
                
                # Add definition: ¬Z ⊔ C
                def_clause = Clause(frozenset([Negation(definer), filler]))
                definitions.append(def_clause)
                
                return UniversalRestriction(literal.role, definer), definitions
            return literal, []
        
        return literal, []
    
    def _extract_r_literal(
        self, 
        literal: Concept, 
        role_name: str
    ) -> Tuple[Concept, List[Clause]]:
        """
        Extract an r-reduced literal from a complex literal.
        
        Transformation rules:
        1. If L contains r elsewhere, replace L with fresh Z, add ¬Z ⊔ L
        2. If L = ∃R.C or ∀R.C with r ∈ sig(C), replace C with Z, add ¬Z ⊔ C
        
        Returns:
            Tuple of (transformed literal, definition clauses)
        """
        definitions = []
        
        if isinstance(literal, (ExistentialRestriction, UniversalRestriction)):
            role = literal.role
            filler = literal.filler
            
            if role.name == role_name:
                # Role matches, check if filler is r-free
                if filler.contains_symbol(role_name):
                    # Need to introduce definer for filler
                    definer_name = self.definer_manager.create_definer(filler)
                    definer = AtomicConcept(definer_name)
                    
                    def_clause = Clause(frozenset([Negation(definer), filler]))
                    definitions.append(def_clause)
                    
                    if isinstance(literal, ExistentialRestriction):
                        return ExistentialRestriction(role, definer), definitions
                    else:
                        return UniversalRestriction(role, definer), definitions
                
                return literal, []
            else:
                # Different role, but might contain r in filler
                if filler.contains_symbol(role_name):
                    new_filler, defs = self._extract_r_literal(filler, role_name)
                    definitions.extend(defs)
                    
                    if isinstance(literal, ExistentialRestriction):
                        return ExistentialRestriction(role, new_filler), definitions
                    else:
                        return UniversalRestriction(role, new_filler), definitions
        
        # For other complex literals containing r, introduce a definer
        if literal.contains_symbol(role_name):
            definer_name = self.definer_manager.create_definer(literal)
            definer = AtomicConcept(definer_name)
            
            def_clause = Clause(frozenset([Negation(definer), literal]))
            definitions.append(def_clause)
            
            return definer, definitions
        
        return literal, []


def normalize_kb_for_concept(
    kb: OntologyKB, 
    concept_name: str,
    definer_manager: Optional[DefinerManager] = None
) -> Tuple[OntologyKB, Set[Clause], DefinerManager]:
    """
    Convenience function to normalize a KB for concept elimination.
    
    Args:
        kb: Input knowledge base
        concept_name: Concept to normalize for
        definer_manager: Optional DefinerManager
        
    Returns:
        Tuple of (normalized KB, A-clauses, DefinerManager)
    """
    dm = definer_manager or DefinerManager()
    normalizer = LinearNormalizer(dm)
    normalized_kb, a_clauses = normalizer.normalize_for_concept(kb, concept_name)
    return normalized_kb, a_clauses, dm


def normalize_kb_for_role(
    kb: OntologyKB, 
    role_name: str,
    definer_manager: Optional[DefinerManager] = None
) -> Tuple[OntologyKB, Set[Clause], DefinerManager]:
    """
    Convenience function to normalize a KB for role elimination.
    
    Args:
        kb: Input knowledge base
        role_name: Role to normalize for
        definer_manager: Optional DefinerManager
        
    Returns:
        Tuple of (normalized KB, r-clauses, DefinerManager)
    """
    dm = definer_manager or DefinerManager()
    normalizer = LinearNormalizer(dm)
    normalized_kb, r_clauses = normalizer.normalize_for_role(kb, role_name)
    return normalized_kb, r_clauses, dm
