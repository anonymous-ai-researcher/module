"""
NFMR: Noise-Free Module Retrieval
Core Ontology Module - ALCI Knowledge Base Representation

This module provides classes for representing ALCI Description Logic constructs
including concepts, roles, axioms, and knowledge bases.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, List, Optional, Dict, FrozenSet, Iterator, Union
from enum import Enum, auto
import hashlib
import copy


class ConceptType(Enum):
    """Enumeration of ALCI concept types."""
    TOP = auto()
    BOTTOM = auto()
    ATOMIC = auto()
    NEGATION = auto()
    CONJUNCTION = auto()
    DISJUNCTION = auto()
    EXISTENTIAL = auto()
    UNIVERSAL = auto()


class Concept(ABC):
    """
    Abstract base class for ALCI concepts.
    
    ALCI concepts include:
    - ⊤ (top), ⊥ (bottom)
    - A (atomic concept)
    - ¬C (negation)
    - C ⊓ D (conjunction)
    - C ⊔ D (disjunction)
    - ∃R.C (existential restriction)
    - ∀R.C (universal restriction)
    """
    
    @property
    @abstractmethod
    def concept_type(self) -> ConceptType:
        """Return the type of this concept."""
        pass
    
    @property
    @abstractmethod
    def signature(self) -> Set[str]:
        """Return the signature (concept and role names) of this concept."""
        pass
    
    @property
    def concept_names(self) -> Set[str]:
        """Return only concept names in the signature."""
        return {s for s in self.signature if not s.startswith('r_')}
    
    @property
    def role_names(self) -> Set[str]:
        """Return only role names in the signature."""
        return {s for s in self.signature if s.startswith('r_')}
    
    @abstractmethod
    def contains_symbol(self, symbol: str) -> bool:
        """Check if this concept contains the given symbol."""
        pass
    
    @abstractmethod
    def substitute(self, symbol: str, replacement: 'Concept') -> 'Concept':
        """Substitute a symbol with a concept."""
        pass
    
    @abstractmethod
    def nnf(self) -> 'Concept':
        """Convert to Negation Normal Form."""
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Top(Concept):
    """Represents the top concept ⊤ (universal concept)."""
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.TOP
    
    @property
    def signature(self) -> Set[str]:
        return set()
    
    def contains_symbol(self, symbol: str) -> bool:
        return False
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        return self
    
    def nnf(self) -> Concept:
        return self
    
    def __hash__(self) -> int:
        return hash("TOP")
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Top)
    
    def __str__(self) -> str:
        return "⊤"


@dataclass(frozen=True)
class Bottom(Concept):
    """Represents the bottom concept ⊥ (empty concept)."""
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.BOTTOM
    
    @property
    def signature(self) -> Set[str]:
        return set()
    
    def contains_symbol(self, symbol: str) -> bool:
        return False
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        return self
    
    def nnf(self) -> Concept:
        return self
    
    def __hash__(self) -> int:
        return hash("BOTTOM")
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Bottom)
    
    def __str__(self) -> str:
        return "⊥"


@dataclass(frozen=True)
class AtomicConcept(Concept):
    """Represents an atomic concept name A."""
    name: str
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.ATOMIC
    
    @property
    def signature(self) -> Set[str]:
        return {self.name}
    
    def contains_symbol(self, symbol: str) -> bool:
        return self.name == symbol
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        if self.name == symbol:
            return replacement
        return self
    
    def nnf(self) -> Concept:
        return self
    
    def __hash__(self) -> int:
        return hash(("ATOMIC", self.name))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, AtomicConcept) and self.name == other.name
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Role:
    """
    Represents a role in ALCI.
    
    Roles can be:
    - r (role name)
    - r⁻ (inverse role)
    """
    name: str
    is_inverse: bool = False
    
    @property
    def signature(self) -> Set[str]:
        return {self.name}
    
    def inverse(self) -> 'Role':
        """Return the inverse of this role."""
        return Role(self.name, not self.is_inverse)
    
    def __hash__(self) -> int:
        return hash(("ROLE", self.name, self.is_inverse))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Role):
            return False
        return self.name == other.name and self.is_inverse == other.is_inverse
    
    def __str__(self) -> str:
        if self.is_inverse:
            return f"{self.name}⁻"
        return self.name


@dataclass(frozen=True)
class Negation(Concept):
    """Represents negation ¬C."""
    operand: Concept
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.NEGATION
    
    @property
    def signature(self) -> Set[str]:
        return self.operand.signature
    
    def contains_symbol(self, symbol: str) -> bool:
        return self.operand.contains_symbol(symbol)
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        return Negation(self.operand.substitute(symbol, replacement))
    
    def nnf(self) -> Concept:
        """Push negation inward to achieve NNF."""
        inner = self.operand
        
        if isinstance(inner, Top):
            return Bottom()
        elif isinstance(inner, Bottom):
            return Top()
        elif isinstance(inner, AtomicConcept):
            return self
        elif isinstance(inner, Negation):
            return inner.operand.nnf()
        elif isinstance(inner, Conjunction):
            # ¬(C ⊓ D) ≡ ¬C ⊔ ¬D
            return Disjunction(
                frozenset(Negation(op).nnf() for op in inner.operands)
            )
        elif isinstance(inner, Disjunction):
            # ¬(C ⊔ D) ≡ ¬C ⊓ ¬D
            return Conjunction(
                frozenset(Negation(op).nnf() for op in inner.operands)
            )
        elif isinstance(inner, ExistentialRestriction):
            # ¬∃R.C ≡ ∀R.¬C
            return UniversalRestriction(
                inner.role,
                Negation(inner.filler).nnf()
            )
        elif isinstance(inner, UniversalRestriction):
            # ¬∀R.C ≡ ∃R.¬C
            return ExistentialRestriction(
                inner.role,
                Negation(inner.filler).nnf()
            )
        return self
    
    def __hash__(self) -> int:
        return hash(("NEG", self.operand))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Negation) and self.operand == other.operand
    
    def __str__(self) -> str:
        if isinstance(self.operand, AtomicConcept):
            return f"¬{self.operand}"
        return f"¬({self.operand})"


@dataclass(frozen=True)
class Conjunction(Concept):
    """Represents conjunction C ⊓ D (and multiple operands)."""
    operands: FrozenSet[Concept]
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.CONJUNCTION
    
    @property
    def signature(self) -> Set[str]:
        result = set()
        for op in self.operands:
            result.update(op.signature)
        return result
    
    def contains_symbol(self, symbol: str) -> bool:
        return any(op.contains_symbol(symbol) for op in self.operands)
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        return Conjunction(
            frozenset(op.substitute(symbol, replacement) for op in self.operands)
        )
    
    def nnf(self) -> Concept:
        return Conjunction(frozenset(op.nnf() for op in self.operands))
    
    def __hash__(self) -> int:
        return hash(("CONJ", self.operands))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Conjunction) and self.operands == other.operands
    
    def __str__(self) -> str:
        if len(self.operands) == 0:
            return "⊤"
        parts = sorted(str(op) for op in self.operands)
        return f"({' ⊓ '.join(parts)})"


@dataclass(frozen=True)
class Disjunction(Concept):
    """Represents disjunction C ⊔ D (and multiple operands)."""
    operands: FrozenSet[Concept]
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.DISJUNCTION
    
    @property
    def signature(self) -> Set[str]:
        result = set()
        for op in self.operands:
            result.update(op.signature)
        return result
    
    def contains_symbol(self, symbol: str) -> bool:
        return any(op.contains_symbol(symbol) for op in self.operands)
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        return Disjunction(
            frozenset(op.substitute(symbol, replacement) for op in self.operands)
        )
    
    def nnf(self) -> Concept:
        return Disjunction(frozenset(op.nnf() for op in self.operands))
    
    def __hash__(self) -> int:
        return hash(("DISJ", self.operands))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Disjunction) and self.operands == other.operands
    
    def __str__(self) -> str:
        if len(self.operands) == 0:
            return "⊥"
        parts = sorted(str(op) for op in self.operands)
        return f"({' ⊔ '.join(parts)})"


@dataclass(frozen=True)
class ExistentialRestriction(Concept):
    """Represents existential restriction ∃R.C."""
    role: Role
    filler: Concept
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.EXISTENTIAL
    
    @property
    def signature(self) -> Set[str]:
        return self.role.signature | self.filler.signature
    
    def contains_symbol(self, symbol: str) -> bool:
        return symbol in self.role.signature or self.filler.contains_symbol(symbol)
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        return ExistentialRestriction(
            self.role,
            self.filler.substitute(symbol, replacement)
        )
    
    def nnf(self) -> Concept:
        return ExistentialRestriction(self.role, self.filler.nnf())
    
    def __hash__(self) -> int:
        return hash(("EXISTS", self.role, self.filler))
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, ExistentialRestriction) and 
                self.role == other.role and self.filler == other.filler)
    
    def __str__(self) -> str:
        return f"∃{self.role}.{self.filler}"


@dataclass(frozen=True)
class UniversalRestriction(Concept):
    """Represents universal restriction ∀R.C."""
    role: Role
    filler: Concept
    
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.UNIVERSAL
    
    @property
    def signature(self) -> Set[str]:
        return self.role.signature | self.filler.signature
    
    def contains_symbol(self, symbol: str) -> bool:
        return symbol in self.role.signature or self.filler.contains_symbol(symbol)
    
    def substitute(self, symbol: str, replacement: Concept) -> Concept:
        return UniversalRestriction(
            self.role,
            self.filler.substitute(symbol, replacement)
        )
    
    def nnf(self) -> Concept:
        return UniversalRestriction(self.role, self.filler.nnf())
    
    def __hash__(self) -> int:
        return hash(("FORALL", self.role, self.filler))
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, UniversalRestriction) and 
                self.role == other.role and self.filler == other.filler)
    
    def __str__(self) -> str:
        return f"∀{self.role}.{self.filler}"


@dataclass(frozen=True)
class ConceptInclusion:
    """
    Represents a concept inclusion (CI) axiom: C ⊑ D
    
    This is the only axiom type in our normalized ALCI-KB.
    """
    lhs: Concept  # Left-hand side (subsumee)
    rhs: Concept  # Right-hand side (subsumer)
    
    @property
    def signature(self) -> Set[str]:
        """Return the signature of this axiom."""
        return self.lhs.signature | self.rhs.signature
    
    @property
    def concept_names(self) -> Set[str]:
        """Return concept names in this axiom."""
        return self.lhs.concept_names | self.rhs.concept_names
    
    @property
    def role_names(self) -> Set[str]:
        """Return role names in this axiom."""
        return self.lhs.role_names | self.rhs.role_names
    
    def contains_symbol(self, symbol: str) -> bool:
        """Check if this axiom contains the given symbol."""
        return self.lhs.contains_symbol(symbol) or self.rhs.contains_symbol(symbol)
    
    def to_clause(self) -> 'Clause':
        """
        Convert to clausal form: C ⊑ D becomes ⊤ ⊑ ¬C ⊔ D
        """
        neg_lhs = Negation(self.lhs).nnf()
        return Clause(frozenset([neg_lhs, self.rhs]))
    
    def __hash__(self) -> int:
        return hash(("CI", self.lhs, self.rhs))
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, ConceptInclusion) and 
                self.lhs == other.lhs and self.rhs == other.rhs)
    
    def __str__(self) -> str:
        return f"{self.lhs} ⊑ {self.rhs}"


@dataclass(frozen=True)
class Clause:
    """
    Represents a clause in clausal normal form: ⊤ ⊑ L₁ ⊔ ... ⊔ Lₙ
    
    We represent this as a set of literals (the disjuncts).
    """
    literals: FrozenSet[Concept]
    
    @property
    def signature(self) -> Set[str]:
        """Return the signature of this clause."""
        result = set()
        for lit in self.literals:
            result.update(lit.signature)
        return result
    
    def contains_symbol(self, symbol: str) -> bool:
        """Check if this clause contains the given symbol."""
        return any(lit.contains_symbol(symbol) for lit in self.literals)
    
    def is_tautology(self) -> bool:
        """Check if this clause is a tautology."""
        # Check for complementary literals
        for lit in self.literals:
            if isinstance(lit, Top):
                return True
            if isinstance(lit, Negation):
                if lit.operand in self.literals:
                    return True
            elif Negation(lit) in self.literals:
                return True
        return False
    
    def simplify(self) -> Optional['Clause']:
        """Simplify the clause, return None if tautology."""
        if self.is_tautology():
            return None
        
        # Remove bottom literals
        new_literals = frozenset(
            lit for lit in self.literals 
            if not isinstance(lit, Bottom)
        )
        
        if len(new_literals) == 0:
            return Clause(frozenset([Bottom()]))
        
        return Clause(new_literals)
    
    def to_concept_inclusion(self) -> ConceptInclusion:
        """Convert back to concept inclusion form."""
        if len(self.literals) == 0:
            return ConceptInclusion(Top(), Bottom())
        elif len(self.literals) == 1:
            return ConceptInclusion(Top(), list(self.literals)[0])
        else:
            return ConceptInclusion(Top(), Disjunction(self.literals))
    
    def __hash__(self) -> int:
        return hash(("CLAUSE", self.literals))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Clause) and self.literals == other.literals
    
    def __str__(self) -> str:
        if len(self.literals) == 0:
            return "⊤ ⊑ ⊥"
        parts = sorted(str(lit) for lit in self.literals)
        return f"⊤ ⊑ {' ⊔ '.join(parts)}"
    
    def __len__(self) -> int:
        return len(self.literals)


class OntologyKB:
    """
    Represents an ALCI ontology knowledge base.
    
    The KB is a finite set of concept inclusions (CIs).
    """
    
    def __init__(self, axioms: Optional[Set[ConceptInclusion]] = None):
        """Initialize the KB with a set of axioms."""
        self._axioms: Set[ConceptInclusion] = axioms if axioms else set()
        self._clauses: Optional[Set[Clause]] = None
        self._signature_cache: Optional[Set[str]] = None
    
    @classmethod
    def from_owl(cls, path: str) -> 'OntologyKB':
        """
        Load an ontology from an OWL file.
        
        Args:
            path: Path to the OWL file
            
        Returns:
            OntologyKB instance
        """
        from .utils.owl_parser import OWLParser
        parser = OWLParser()
        return parser.parse(path)
    
    @classmethod
    def from_clauses(cls, clauses: Set[Clause]) -> 'OntologyKB':
        """Create KB from a set of clauses."""
        kb = cls()
        kb._clauses = clauses
        kb._axioms = {c.to_concept_inclusion() for c in clauses}
        return kb
    
    def add_axiom(self, axiom: ConceptInclusion) -> None:
        """Add an axiom to the KB."""
        self._axioms.add(axiom)
        self._invalidate_cache()
    
    def remove_axiom(self, axiom: ConceptInclusion) -> None:
        """Remove an axiom from the KB."""
        self._axioms.discard(axiom)
        self._invalidate_cache()
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._clauses = None
        self._signature_cache = None
    
    @property
    def axioms(self) -> Set[ConceptInclusion]:
        """Return all axioms in the KB."""
        return self._axioms.copy()
    
    @property
    def clauses(self) -> Set[Clause]:
        """Return the KB in clausal form."""
        if self._clauses is None:
            self._clauses = set()
            for axiom in self._axioms:
                clause = axiom.to_clause()
                simplified = clause.simplify()
                if simplified is not None:
                    self._clauses.add(simplified)
        return self._clauses.copy()
    
    @property
    def signature(self) -> Set[str]:
        """Return the signature (all concept and role names)."""
        if self._signature_cache is None:
            self._signature_cache = set()
            for axiom in self._axioms:
                self._signature_cache.update(axiom.signature)
        return self._signature_cache.copy()
    
    @property
    def concept_names(self) -> Set[str]:
        """Return all concept names in the KB."""
        result = set()
        for axiom in self._axioms:
            result.update(axiom.concept_names)
        return result
    
    @property
    def role_names(self) -> Set[str]:
        """Return all role names in the KB."""
        result = set()
        for axiom in self._axioms:
            result.update(axiom.role_names)
        return result
    
    def get_axioms_with_symbol(self, symbol: str) -> Set[ConceptInclusion]:
        """Return all axioms containing the given symbol."""
        return {ax for ax in self._axioms if ax.contains_symbol(symbol)}
    
    def get_clauses_with_symbol(self, symbol: str) -> Set[Clause]:
        """Return all clauses containing the given symbol."""
        return {c for c in self.clauses if c.contains_symbol(symbol)}
    
    def copy(self) -> 'OntologyKB':
        """Create a deep copy of the KB."""
        return OntologyKB(self._axioms.copy())
    
    def __len__(self) -> int:
        """Return the number of axioms."""
        return len(self._axioms)
    
    def __iter__(self) -> Iterator[ConceptInclusion]:
        """Iterate over axioms."""
        return iter(self._axioms)
    
    def __contains__(self, axiom: ConceptInclusion) -> bool:
        """Check if an axiom is in the KB."""
        return axiom in self._axioms
    
    def __str__(self) -> str:
        lines = [f"OntologyKB with {len(self)} axioms:"]
        for ax in sorted(self._axioms, key=str):
            lines.append(f"  {ax}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"OntologyKB(|axioms|={len(self)}, |sig|={len(self.signature)})"


# Utility functions for concept construction
def make_conjunction(*concepts: Concept) -> Concept:
    """Create a conjunction from multiple concepts."""
    flat = set()
    for c in concepts:
        if isinstance(c, Conjunction):
            flat.update(c.operands)
        elif not isinstance(c, Top):
            flat.add(c)
    
    if len(flat) == 0:
        return Top()
    if len(flat) == 1:
        return list(flat)[0]
    if any(isinstance(c, Bottom) for c in flat):
        return Bottom()
    
    return Conjunction(frozenset(flat))


def make_disjunction(*concepts: Concept) -> Concept:
    """Create a disjunction from multiple concepts."""
    flat = set()
    for c in concepts:
        if isinstance(c, Disjunction):
            flat.update(c.operands)
        elif not isinstance(c, Bottom):
            flat.add(c)
    
    if len(flat) == 0:
        return Bottom()
    if len(flat) == 1:
        return list(flat)[0]
    if any(isinstance(c, Top) for c in flat):
        return Top()
    
    return Disjunction(frozenset(flat))
