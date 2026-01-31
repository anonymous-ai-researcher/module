"""
NFMR: Noise-Free Module Retrieval
Elimination Module

This module implements the combination rules for eliminating concept and role
names from normalized ALCI knowledge bases while preserving semantic equivalence.
"""

from __future__ import annotations
from typing import Set, List, Tuple, Optional, Dict, FrozenSet
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product

from .ontology import (
    Concept, Top, Bottom, AtomicConcept, Negation, Conjunction, Disjunction,
    ExistentialRestriction, UniversalRestriction, Role, Clause, OntologyKB,
    ConceptInclusion, make_disjunction, make_conjunction
)
from .normalizer import DefinerManager, LinearNormalizer


class PremiseType(Enum):
    """Classification of premises for the combination rules."""
    # Positive premises for concept elimination
    P_ATOMIC = auto()          # B ⊔ A
    P_EXISTS = auto()          # C ⊔ ∃r.A
    P_EXISTS_INV = auto()      # D ⊔ ∃r⁻.A
    P_FORALL = auto()          # φ ⊔ ∀r.A
    
    # Negative premises for concept elimination
    N_ATOMIC = auto()          # E ⊔ ¬A
    N_EXISTS = auto()          # F ⊔ ∃r.¬A
    N_EXISTS_INV = auto()      # G ⊔ ∃r⁻.¬A
    N_FORALL = auto()          # ψ ⊔ ∀r.¬A
    
    # Premises for role elimination
    R_EXISTS = auto()          # C ⊔ ∃r.D
    R_EXISTS_INV = auto()      # E ⊔ ∃r⁻.F
    R_FORALL = auto()          # V ⊔ ∀r.W
    R_FORALL_INV = auto()      # X ⊔ ∀r⁻.Y


@dataclass
class ClassifiedPremise:
    """A clause classified as a premise for combination rules."""
    clause: Clause
    premise_type: PremiseType
    residue: FrozenSet[Concept]  # C (the part not containing the symbol)
    target_literal: Concept       # The literal containing the symbol
    role: Optional[Role] = None   # Role if applicable
    filler: Optional[Concept] = None  # Filler if applicable


def classify_concept_premise(
    clause: Clause, 
    concept_name: str
) -> Optional[ClassifiedPremise]:
    """
    Classify an A-reduced clause as a positive or negative premise.
    
    Args:
        clause: An A-reduced clause
        concept_name: The concept name being eliminated
        
    Returns:
        ClassifiedPremise or None if not classifiable
    """
    a_literal = None
    residue = []
    
    for lit in clause.literals:
        if lit.contains_symbol(concept_name):
            if a_literal is not None:
                # Multiple A-literals - not properly reduced
                return None
            a_literal = lit
        else:
            residue.append(lit)
    
    if a_literal is None:
        return None
    
    residue_set = frozenset(residue)
    
    # Classify based on the A-literal form
    if isinstance(a_literal, AtomicConcept) and a_literal.name == concept_name:
        return ClassifiedPremise(
            clause=clause,
            premise_type=PremiseType.P_ATOMIC,
            residue=residue_set,
            target_literal=a_literal
        )
    
    if isinstance(a_literal, Negation):
        inner = a_literal.operand
        if isinstance(inner, AtomicConcept) and inner.name == concept_name:
            return ClassifiedPremise(
                clause=clause,
                premise_type=PremiseType.N_ATOMIC,
                residue=residue_set,
                target_literal=a_literal
            )
    
    if isinstance(a_literal, ExistentialRestriction):
        role = a_literal.role
        filler = a_literal.filler
        
        if isinstance(filler, AtomicConcept) and filler.name == concept_name:
            ptype = PremiseType.P_EXISTS_INV if role.is_inverse else PremiseType.P_EXISTS
            return ClassifiedPremise(
                clause=clause,
                premise_type=ptype,
                residue=residue_set,
                target_literal=a_literal,
                role=role,
                filler=filler
            )
        elif isinstance(filler, Negation):
            if isinstance(filler.operand, AtomicConcept):
                if filler.operand.name == concept_name:
                    ptype = PremiseType.N_EXISTS_INV if role.is_inverse else PremiseType.N_EXISTS
                    return ClassifiedPremise(
                        clause=clause,
                        premise_type=ptype,
                        residue=residue_set,
                        target_literal=a_literal,
                        role=role,
                        filler=filler
                    )
    
    if isinstance(a_literal, UniversalRestriction):
        role = a_literal.role
        filler = a_literal.filler
        
        if isinstance(filler, AtomicConcept) and filler.name == concept_name:
            ptype = PremiseType.P_FORALL
            return ClassifiedPremise(
                clause=clause,
                premise_type=ptype,
                residue=residue_set,
                target_literal=a_literal,
                role=role,
                filler=filler
            )
        elif isinstance(filler, Negation):
            if isinstance(filler.operand, AtomicConcept):
                if filler.operand.name == concept_name:
                    ptype = PremiseType.N_FORALL
                    return ClassifiedPremise(
                        clause=clause,
                        premise_type=ptype,
                        residue=residue_set,
                        target_literal=a_literal,
                        role=role,
                        filler=filler
                    )
    
    return None


def classify_role_premise(
    clause: Clause, 
    role_name: str
) -> Optional[ClassifiedPremise]:
    """
    Classify an r-reduced clause as a premise for role elimination.
    
    Args:
        clause: An r-reduced clause
        role_name: The role name being eliminated
        
    Returns:
        ClassifiedPremise or None if not classifiable
    """
    r_literal = None
    residue = []
    
    for lit in clause.literals:
        if lit.contains_symbol(role_name):
            if r_literal is not None:
                return None
            r_literal = lit
        else:
            residue.append(lit)
    
    if r_literal is None:
        return None
    
    residue_set = frozenset(residue)
    
    if isinstance(r_literal, ExistentialRestriction):
        role = r_literal.role
        if role.name == role_name:
            ptype = PremiseType.R_EXISTS_INV if role.is_inverse else PremiseType.R_EXISTS
            return ClassifiedPremise(
                clause=clause,
                premise_type=ptype,
                residue=residue_set,
                target_literal=r_literal,
                role=role,
                filler=r_literal.filler
            )
    
    if isinstance(r_literal, UniversalRestriction):
        role = r_literal.role
        if role.name == role_name:
            ptype = PremiseType.R_FORALL_INV if role.is_inverse else PremiseType.R_FORALL
            return ClassifiedPremise(
                clause=clause,
                premise_type=ptype,
                residue=residue_set,
                target_literal=r_literal,
                role=role,
                filler=r_literal.filler
            )
    
    return None


class ConceptEliminator:
    """
    Implements the combination rules for concept name elimination.
    
    Given an A-reduced KB, eliminates concept name A by applying all
    16 combinations of positive and negative premises.
    """
    
    def __init__(self, reasoner=None):
        """
        Initialize the eliminator.
        
        Args:
            reasoner: Optional DL reasoner for subsumption checking
        """
        self.reasoner = reasoner
    
    def eliminate(
        self, 
        a_clauses: Set[Clause], 
        concept_name: str
    ) -> Set[Clause]:
        """
        Eliminate a concept name from a set of A-reduced clauses.
        
        Args:
            a_clauses: Set of A-reduced clauses
            concept_name: The concept name to eliminate
            
        Returns:
            Set of resulting clauses (A-free)
        """
        # Classify premises
        positive_premises: Dict[PremiseType, List[ClassifiedPremise]] = {
            PremiseType.P_ATOMIC: [],
            PremiseType.P_EXISTS: [],
            PremiseType.P_EXISTS_INV: [],
            PremiseType.P_FORALL: [],
        }
        
        negative_premises: Dict[PremiseType, List[ClassifiedPremise]] = {
            PremiseType.N_ATOMIC: [],
            PremiseType.N_EXISTS: [],
            PremiseType.N_EXISTS_INV: [],
            PremiseType.N_FORALL: [],
        }
        
        for clause in a_clauses:
            classified = classify_concept_premise(clause, concept_name)
            if classified is None:
                continue
            
            if classified.premise_type in positive_premises:
                positive_premises[classified.premise_type].append(classified)
            elif classified.premise_type in negative_premises:
                negative_premises[classified.premise_type].append(classified)
        
        # Apply all 16 combinations
        result_clauses: Set[Clause] = set()
        
        for pos_type, pos_list in positive_premises.items():
            for neg_type, neg_list in negative_premises.items():
                combined = self._combine(pos_list, neg_list, pos_type, neg_type)
                result_clauses.update(combined)
        
        # Simplify and filter
        simplified = set()
        for clause in result_clauses:
            s = clause.simplify()
            if s is not None and not s.is_tautology():
                simplified.add(s)
        
        return simplified
    
    def _combine(
        self,
        pos_premises: List[ClassifiedPremise],
        neg_premises: List[ClassifiedPremise],
        pos_type: PremiseType,
        neg_type: PremiseType
    ) -> Set[Clause]:
        """
        Apply the appropriate combination rule based on premise types.
        
        This implements the 16 combination rules from Figure 1 of the paper.
        """
        results = set()
        
        for pos, neg in product(pos_premises, neg_premises):
            combined = self._apply_combination_rule(pos, neg, pos_type, neg_type)
            results.update(combined)
        
        return results
    
    def _apply_combination_rule(
        self,
        pos: ClassifiedPremise,
        neg: ClassifiedPremise,
        pos_type: PremiseType,
        neg_type: PremiseType
    ) -> List[Clause]:
        """Apply a specific combination rule."""
        B = pos.residue
        E = neg.residue
        
        # Rule 1: combine(P⁺_⊘(A), P⁻_⊘(A)) = B ⊔ E
        if pos_type == PremiseType.P_ATOMIC and neg_type == PremiseType.N_ATOMIC:
            return [Clause(B | E)]
        
        # Rule 2: combine(P⁺_⊘(A), P⁻_∃(A)) = F ⊔ ∃u.B
        if pos_type == PremiseType.P_ATOMIC and neg_type == PremiseType.N_EXISTS:
            r = neg.role
            new_lit = ExistentialRestriction(r, self._residue_to_concept(B))
            return [Clause(E | frozenset([new_lit]))]
        
        # Rule 3: combine(P⁺_⊘(A), P⁻_∃⁻(A)) = G ⊔ ∃v⁻.B
        if pos_type == PremiseType.P_ATOMIC and neg_type == PremiseType.N_EXISTS_INV:
            r = neg.role
            new_lit = ExistentialRestriction(r, self._residue_to_concept(B))
            return [Clause(E | frozenset([new_lit]))]
        
        # Rule 4: combine(P⁺_⊘(A), P⁻_∀(A)) = ψ ⊔ ∀w.B
        if pos_type == PremiseType.P_ATOMIC and neg_type == PremiseType.N_FORALL:
            r = neg.role
            new_lit = UniversalRestriction(r, self._residue_to_concept(B))
            return [Clause(E | frozenset([new_lit]))]
        
        # Rule 5: combine(P⁺_∃(A), P⁻_⊘(A)) = C ⊔ ∃r.E
        if pos_type == PremiseType.P_EXISTS and neg_type == PremiseType.N_ATOMIC:
            r = pos.role
            new_lit = ExistentialRestriction(r, self._residue_to_concept(E))
            return [Clause(B | frozenset([new_lit]))]
        
        # Rule 6: combine(P⁺_∃(A), P⁻_∃(A)) = {C ⊔ ∃r.⊤, F ⊔ ∃u.⊤}
        if pos_type == PremiseType.P_EXISTS and neg_type == PremiseType.N_EXISTS:
            r_pos = pos.role
            r_neg = neg.role
            lit1 = ExistentialRestriction(r_pos, Top())
            lit2 = ExistentialRestriction(r_neg, Top())
            return [Clause(B | frozenset([lit1])), Clause(E | frozenset([lit2]))]
        
        # Rule 7: combine(P⁺_∃(A), P⁻_∃⁻(A)) = {C ⊔ ∃r.⊤, G ⊔ ∃v⁻.⊤}
        if pos_type == PremiseType.P_EXISTS and neg_type == PremiseType.N_EXISTS_INV:
            r_pos = pos.role
            r_neg = neg.role
            lit1 = ExistentialRestriction(r_pos, Top())
            lit2 = ExistentialRestriction(r_neg, Top())
            return [Clause(B | frozenset([lit1])), Clause(E | frozenset([lit2]))]
        
        # Rule 8: combine(P⁺_∃(A), P⁻_∀(A)) = C ⊔ ∃r.∀w⁻.ψ
        if pos_type == PremiseType.P_EXISTS and neg_type == PremiseType.N_FORALL:
            r_pos = pos.role
            r_neg = neg.role.inverse()
            inner = UniversalRestriction(r_neg, self._residue_to_concept(E))
            new_lit = ExistentialRestriction(r_pos, inner)
            return [Clause(B | frozenset([new_lit]))]
        
        # Rule 9: combine(P⁺_∃⁻(A), P⁻_⊘(A)) = D ⊔ ∃s⁻.E
        if pos_type == PremiseType.P_EXISTS_INV and neg_type == PremiseType.N_ATOMIC:
            r = pos.role
            new_lit = ExistentialRestriction(r, self._residue_to_concept(E))
            return [Clause(B | frozenset([new_lit]))]
        
        # Rule 10: combine(P⁺_∃⁻(A), P⁻_∃(A)) = {D ⊔ ∃s⁻.⊤, F ⊔ ∃u.⊤}
        if pos_type == PremiseType.P_EXISTS_INV and neg_type == PremiseType.N_EXISTS:
            r_pos = pos.role
            r_neg = neg.role
            lit1 = ExistentialRestriction(r_pos, Top())
            lit2 = ExistentialRestriction(r_neg, Top())
            return [Clause(B | frozenset([lit1])), Clause(E | frozenset([lit2]))]
        
        # Rule 11: combine(P⁺_∃⁻(A), P⁻_∃⁻(A)) = {D ⊔ ∃s⁻.⊤, G ⊔ ∃v⁻.⊤}
        if pos_type == PremiseType.P_EXISTS_INV and neg_type == PremiseType.N_EXISTS_INV:
            r_pos = pos.role
            r_neg = neg.role
            lit1 = ExistentialRestriction(r_pos, Top())
            lit2 = ExistentialRestriction(r_neg, Top())
            return [Clause(B | frozenset([lit1])), Clause(E | frozenset([lit2]))]
        
        # Rule 12: combine(P⁺_∃⁻(A), P⁻_∀(A)) = D ⊔ ∃s⁻.∀w⁻.ψ
        if pos_type == PremiseType.P_EXISTS_INV and neg_type == PremiseType.N_FORALL:
            r_pos = pos.role
            r_neg = neg.role.inverse()
            inner = UniversalRestriction(r_neg, self._residue_to_concept(E))
            new_lit = ExistentialRestriction(r_pos, inner)
            return [Clause(B | frozenset([new_lit]))]
        
        # Rule 13: combine(P⁺_∀(A), P⁻_⊘(A)) = φ ⊔ ∀t.E
        if pos_type == PremiseType.P_FORALL and neg_type == PremiseType.N_ATOMIC:
            r = pos.role
            new_lit = UniversalRestriction(r, self._residue_to_concept(E))
            return [Clause(B | frozenset([new_lit]))]
        
        # Rule 14: combine(P⁺_∀(A), P⁻_∃(A)) = F ⊔ ∃u.∀t⁻.φ
        if pos_type == PremiseType.P_FORALL and neg_type == PremiseType.N_EXISTS:
            r_pos = pos.role.inverse()
            r_neg = neg.role
            inner = UniversalRestriction(r_pos, self._residue_to_concept(B))
            new_lit = ExistentialRestriction(r_neg, inner)
            return [Clause(E | frozenset([new_lit]))]
        
        # Rule 15: combine(P⁺_∀(A), P⁻_∃⁻(A)) = G ⊔ ∃v⁻.∀t⁻.φ
        if pos_type == PremiseType.P_FORALL and neg_type == PremiseType.N_EXISTS_INV:
            r_pos = pos.role.inverse()
            r_neg = neg.role
            inner = UniversalRestriction(r_pos, self._residue_to_concept(B))
            new_lit = ExistentialRestriction(r_neg, inner)
            return [Clause(E | frozenset([new_lit]))]
        
        # Rule 16: combine(P⁺_∀(A), P⁻_∀(A)) = φ ⊔ ψ ⊔ ∀t.⊥ (if t = w)
        if pos_type == PremiseType.P_FORALL and neg_type == PremiseType.N_FORALL:
            r_pos = pos.role
            r_neg = neg.role
            if r_pos == r_neg:
                new_lit = UniversalRestriction(r_pos, Bottom())
                return [Clause(B | E | frozenset([new_lit]))]
            return []
        
        return []
    
    def _residue_to_concept(self, residue: FrozenSet[Concept]) -> Concept:
        """Convert a residue (set of literals) to a single concept."""
        if len(residue) == 0:
            return Top()
        if len(residue) == 1:
            return list(residue)[0]
        return make_disjunction(*residue)


class RoleEliminator:
    """
    Implements the combination rules for role name elimination.
    
    Given an r-reduced KB, eliminates role name r by applying the
    4 combinations of existential and universal premises.
    """
    
    def __init__(self, reasoner=None):
        """
        Initialize the eliminator.
        
        Args:
            reasoner: Optional DL reasoner for subsumption checking
        """
        self.reasoner = reasoner
    
    def eliminate(
        self, 
        r_clauses: Set[Clause], 
        role_name: str
    ) -> Set[Clause]:
        """
        Eliminate a role name from a set of r-reduced clauses.
        
        Args:
            r_clauses: Set of r-reduced clauses
            role_name: The role name to eliminate
            
        Returns:
            Set of resulting clauses (r-free)
        """
        # Classify premises
        exists_premises: List[ClassifiedPremise] = []
        exists_inv_premises: List[ClassifiedPremise] = []
        forall_premises: List[ClassifiedPremise] = []
        forall_inv_premises: List[ClassifiedPremise] = []
        
        for clause in r_clauses:
            classified = classify_role_premise(clause, role_name)
            if classified is None:
                continue
            
            if classified.premise_type == PremiseType.R_EXISTS:
                exists_premises.append(classified)
            elif classified.premise_type == PremiseType.R_EXISTS_INV:
                exists_inv_premises.append(classified)
            elif classified.premise_type == PremiseType.R_FORALL:
                forall_premises.append(classified)
            elif classified.premise_type == PremiseType.R_FORALL_INV:
                forall_inv_premises.append(classified)
        
        # Apply 4 combinations
        result_clauses: Set[Clause] = set()
        
        # Rule 1: combine(P⁺_∃(r), P⁻_∀(r))
        for pos, neg in product(exists_premises, forall_premises):
            if self._check_disjointness(pos.filler, neg.filler):
                combined = self._combine_exists_forall(pos, neg)
                result_clauses.update(combined)
        
        # Rule 2: combine(P⁺_∃(r), P⁻_∀⁻(r))
        for pos, neg in product(exists_premises, forall_inv_premises):
            if self._check_disjointness(pos.filler, neg.residue_concept()):
                combined = self._combine_exists_forall_inv(pos, neg)
                result_clauses.update(combined)
        
        # Rule 3: combine(P⁺_∃⁻(r), P⁻_∀(r))
        for pos, neg in product(exists_inv_premises, forall_premises):
            if self._check_disjointness(pos.filler, neg.residue_concept()):
                combined = self._combine_exists_inv_forall(pos, neg)
                result_clauses.update(combined)
        
        # Rule 4: combine(P⁺_∃⁻(r), P⁻_∀⁻(r))
        for pos, neg in product(exists_inv_premises, forall_inv_premises):
            if self._check_disjointness(pos.filler, neg.filler):
                combined = self._combine_exists_inv_forall_inv(pos, neg)
                result_clauses.update(combined)
        
        # Simplify and filter
        simplified = set()
        for clause in result_clauses:
            s = clause.simplify()
            if s is not None and not s.is_tautology():
                simplified.add(s)
        
        return simplified
    
    def _check_disjointness(
        self, 
        concept1: Concept, 
        concept2: Concept
    ) -> bool:
        """
        Check if C ⊓ D ⊑ ⊥ (concepts are disjoint).
        
        This is a side condition for the role elimination rules.
        """
        if self.reasoner is not None:
            conj = make_conjunction(concept1, concept2)
            return self.reasoner.is_subsumed_by(conj, Bottom())
        
        # Without a reasoner, we use syntactic approximation
        # This is sound but incomplete
        if isinstance(concept1, Bottom) or isinstance(concept2, Bottom):
            return True
        
        if isinstance(concept1, AtomicConcept) and isinstance(concept2, Negation):
            if isinstance(concept2.operand, AtomicConcept):
                if concept1.name == concept2.operand.name:
                    return True
        
        if isinstance(concept2, AtomicConcept) and isinstance(concept1, Negation):
            if isinstance(concept1.operand, AtomicConcept):
                if concept2.name == concept1.operand.name:
                    return True
        
        return False
    
    def _combine_exists_forall(
        self, 
        pos: ClassifiedPremise, 
        neg: ClassifiedPremise
    ) -> List[Clause]:
        """Rule 1: C ⊔ V for D ⊓ W ⊑ ⊥"""
        return [Clause(pos.residue | neg.residue)]
    
    def _combine_exists_forall_inv(
        self, 
        pos: ClassifiedPremise, 
        neg: ClassifiedPremise
    ) -> List[Clause]:
        """Rule 2: C ⊔ Y for D ⊓ X ⊑ ⊥"""
        return [Clause(pos.residue | frozenset([neg.filler]))]
    
    def _combine_exists_inv_forall(
        self, 
        pos: ClassifiedPremise, 
        neg: ClassifiedPremise
    ) -> List[Clause]:
        """Rule 3: E ⊔ W for F ⊓ V ⊑ ⊥"""
        return [Clause(pos.residue | frozenset([neg.filler]))]
    
    def _combine_exists_inv_forall_inv(
        self, 
        pos: ClassifiedPremise, 
        neg: ClassifiedPremise
    ) -> List[Clause]:
        """Rule 4: E ⊔ X for F ⊓ Y ⊑ ⊥"""
        return [Clause(pos.residue | neg.residue)]


def eliminate_concept(
    kb: OntologyKB,
    concept_name: str,
    definer_manager: Optional[DefinerManager] = None,
    reasoner=None
) -> Tuple[OntologyKB, DefinerManager]:
    """
    Eliminate a concept name from a knowledge base.
    
    Args:
        kb: Input knowledge base
        concept_name: Concept to eliminate
        definer_manager: Optional DefinerManager
        reasoner: Optional DL reasoner
        
    Returns:
        Tuple of (resulting KB, DefinerManager)
    """
    dm = definer_manager or DefinerManager()
    normalizer = LinearNormalizer(dm)
    eliminator = ConceptEliminator(reasoner)
    
    # Normalize
    normalized_kb, a_clauses = normalizer.normalize_for_concept(kb, concept_name)
    
    # Get non-A clauses
    non_a_clauses = {c for c in normalized_kb.clauses if not c.contains_symbol(concept_name)}
    
    # Eliminate
    result_clauses = eliminator.eliminate(a_clauses, concept_name)
    
    # Combine
    all_clauses = non_a_clauses | result_clauses
    
    return OntologyKB.from_clauses(all_clauses), dm


def eliminate_role(
    kb: OntologyKB,
    role_name: str,
    definer_manager: Optional[DefinerManager] = None,
    reasoner=None
) -> Tuple[OntologyKB, DefinerManager]:
    """
    Eliminate a role name from a knowledge base.
    
    Args:
        kb: Input knowledge base
        role_name: Role to eliminate
        definer_manager: Optional DefinerManager
        reasoner: Optional DL reasoner
        
    Returns:
        Tuple of (resulting KB, DefinerManager)
    """
    dm = definer_manager or DefinerManager()
    normalizer = LinearNormalizer(dm)
    eliminator = RoleEliminator(reasoner)
    
    # Normalize
    normalized_kb, r_clauses = normalizer.normalize_for_role(kb, role_name)
    
    # Get non-r clauses
    non_r_clauses = {c for c in normalized_kb.clauses if not c.contains_symbol(role_name)}
    
    # Eliminate
    result_clauses = eliminator.eliminate(r_clauses, role_name)
    
    # Combine
    all_clauses = non_r_clauses | result_clauses
    
    return OntologyKB.from_clauses(all_clauses), dm
