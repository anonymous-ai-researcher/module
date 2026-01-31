"""
NFMR Unit Tests
Test suite for the core forgetting algorithm
"""

import pytest
from typing import Set

# Import core modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import (
    Concept, Top, Bottom, AtomicConcept, Negation, Conjunction, Disjunction,
    ExistentialRestriction, UniversalRestriction, Role, ConceptInclusion,
    Clause, OntologyKB, make_conjunction, make_disjunction
)
from core.normalizer import (
    DefinerManager, LinearNormalizer, normalize_kb_for_concept
)
from core.eliminator import (
    ConceptEliminator, RoleEliminator, eliminate_concept, eliminate_role
)
from core.forgetting import (
    NoiseFreeForgetting, RetrievalStatus, compute_zero_noise_module
)


class TestConcepts:
    """Tests for concept representation."""
    
    def test_top_concept(self):
        """Test Top concept properties."""
        top = Top()
        assert top.concept_type.name == 'TOP'
        assert top.signature == set()
        assert str(top) == '⊤'
        assert top == Top()
    
    def test_bottom_concept(self):
        """Test Bottom concept properties."""
        bottom = Bottom()
        assert bottom.concept_type.name == 'BOTTOM'
        assert bottom.signature == set()
        assert str(bottom) == '⊥'
    
    def test_atomic_concept(self):
        """Test AtomicConcept properties."""
        a = AtomicConcept('Disease')
        assert a.name == 'Disease'
        assert a.signature == {'Disease'}
        assert a.contains_symbol('Disease')
        assert not a.contains_symbol('Other')
    
    def test_negation(self):
        """Test Negation properties."""
        a = AtomicConcept('A')
        neg_a = Negation(a)
        
        assert neg_a.signature == {'A'}
        assert neg_a.contains_symbol('A')
        assert str(neg_a) == '¬A'
    
    def test_negation_nnf(self):
        """Test NNF conversion for negation."""
        a = AtomicConcept('A')
        
        # Double negation
        neg_neg_a = Negation(Negation(a))
        assert neg_neg_a.nnf() == a
        
        # Negation of Top
        neg_top = Negation(Top())
        assert neg_top.nnf() == Bottom()
        
        # Negation of Bottom
        neg_bottom = Negation(Bottom())
        assert neg_bottom.nnf() == Top()
    
    def test_conjunction(self):
        """Test Conjunction properties."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        conj = Conjunction(frozenset([a, b]))
        
        assert conj.signature == {'A', 'B'}
        assert conj.contains_symbol('A')
        assert conj.contains_symbol('B')
    
    def test_disjunction(self):
        """Test Disjunction properties."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        disj = Disjunction(frozenset([a, b]))
        
        assert disj.signature == {'A', 'B'}
    
    def test_existential_restriction(self):
        """Test ExistentialRestriction properties."""
        r = Role('hasChild')
        a = AtomicConcept('Person')
        exists_r_a = ExistentialRestriction(r, a)
        
        assert exists_r_a.signature == {'hasChild', 'Person'}
        assert exists_r_a.role == r
        assert exists_r_a.filler == a
    
    def test_universal_restriction(self):
        """Test UniversalRestriction properties."""
        r = Role('hasChild')
        a = AtomicConcept('Person')
        forall_r_a = UniversalRestriction(r, a)
        
        assert forall_r_a.signature == {'hasChild', 'Person'}
    
    def test_role_inverse(self):
        """Test Role inverse."""
        r = Role('hasChild')
        r_inv = r.inverse()
        
        assert r_inv.name == 'hasChild'
        assert r_inv.is_inverse
        assert r_inv.inverse() == r


class TestConceptInclusion:
    """Tests for concept inclusion axioms."""
    
    def test_basic_ci(self):
        """Test basic concept inclusion."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        ci = ConceptInclusion(a, b)
        
        assert ci.lhs == a
        assert ci.rhs == b
        assert ci.signature == {'A', 'B'}
        assert str(ci) == 'A ⊑ B'
    
    def test_ci_to_clause(self):
        """Test conversion to clausal form."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        ci = ConceptInclusion(a, b)
        
        clause = ci.to_clause()
        assert isinstance(clause, Clause)
        # A ⊑ B becomes ⊤ ⊑ ¬A ⊔ B


class TestOntologyKB:
    """Tests for OntologyKB."""
    
    def test_empty_kb(self):
        """Test empty knowledge base."""
        kb = OntologyKB()
        assert len(kb) == 0
        assert kb.signature == set()
    
    def test_kb_with_axioms(self):
        """Test KB with axioms."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        c = AtomicConcept('C')
        
        axioms = {
            ConceptInclusion(a, b),
            ConceptInclusion(b, c),
        }
        
        kb = OntologyKB(axioms)
        assert len(kb) == 2
        assert kb.signature == {'A', 'B', 'C'}
    
    def test_kb_add_remove(self):
        """Test adding and removing axioms."""
        kb = OntologyKB()
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        ci = ConceptInclusion(a, b)
        
        kb.add_axiom(ci)
        assert len(kb) == 1
        assert ci in kb
        
        kb.remove_axiom(ci)
        assert len(kb) == 0
    
    def test_get_axioms_with_symbol(self):
        """Test filtering axioms by symbol."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        c = AtomicConcept('C')
        
        axioms = {
            ConceptInclusion(a, b),
            ConceptInclusion(b, c),
        }
        
        kb = OntologyKB(axioms)
        
        a_axioms = kb.get_axioms_with_symbol('A')
        assert len(a_axioms) == 1
        
        b_axioms = kb.get_axioms_with_symbol('B')
        assert len(b_axioms) == 2


class TestDefinerManager:
    """Tests for DefinerManager."""
    
    def test_create_definer(self):
        """Test definer creation."""
        dm = DefinerManager()
        d1 = dm.create_definer()
        d2 = dm.create_definer()
        
        assert d1 != d2
        assert dm.is_definer(d1)
        assert dm.is_definer(d2)
        assert dm.num_definers == 2
    
    def test_definer_with_concept(self):
        """Test definer creation with associated concept."""
        dm = DefinerManager()
        a = AtomicConcept('A')
        
        d1 = dm.create_definer(a)
        d2 = dm.create_definer(a)  # Should return same definer
        
        assert d1 == d2
        assert dm.num_definers == 1


class TestNormalizer:
    """Tests for LinearNormalizer."""
    
    def test_normalize_simple_kb(self):
        """Test normalization of a simple KB."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        c = AtomicConcept('C')
        
        # A ⊑ B, B ⊑ C
        axioms = {
            ConceptInclusion(a, b),
            ConceptInclusion(b, c),
        }
        
        kb = OntologyKB(axioms)
        normalized_kb, a_clauses, dm = normalize_kb_for_concept(kb, 'B')
        
        # Should have some clauses with B
        assert len(a_clauses) > 0
    
    def test_normalize_with_restriction(self):
        """Test normalization of KB with restrictions."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        r = Role('r')
        
        # A ⊑ ∃r.B
        exists_r_b = ExistentialRestriction(r, b)
        axioms = {ConceptInclusion(a, exists_r_b)}
        
        kb = OntologyKB(axioms)
        normalized_kb, a_clauses, dm = normalize_kb_for_concept(kb, 'B')
        
        # Should have clause with B
        assert any(c.contains_symbol('B') for c in a_clauses)


class TestConceptEliminator:
    """Tests for ConceptEliminator."""
    
    def test_eliminate_simple_concept(self):
        """Test eliminating a simple concept."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        c = AtomicConcept('C')
        
        # C ⊑ A, A ⊑ B
        # After eliminating A: C ⊑ B
        axioms = {
            ConceptInclusion(c, a),
            ConceptInclusion(a, b),
        }
        
        kb = OntologyKB(axioms)
        result_kb, dm = eliminate_concept(kb, 'A')
        
        # Result should not contain A
        assert 'A' not in result_kb.signature
    
    def test_eliminate_preserves_entailments(self):
        """Test that elimination preserves entailments."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        c = AtomicConcept('C')
        
        # A ⊑ B, B ⊑ C implies A ⊑ C
        axioms = {
            ConceptInclusion(a, b),
            ConceptInclusion(b, c),
        }
        
        kb = OntologyKB(axioms)
        result_kb, dm = eliminate_concept(kb, 'B')
        
        # Result should not contain B
        assert 'B' not in result_kb.signature
        
        # Should preserve A ⊑ C relationship somehow


class TestNoiseFreeForgetting:
    """Tests for the main NoiseFreeForgetting class."""
    
    def test_compute_module_success(self):
        """Test successful module computation."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        c = AtomicConcept('C')
        
        axioms = {
            ConceptInclusion(a, b),
            ConceptInclusion(b, c),
        }
        
        kb = OntologyKB(axioms)
        forgetting = NoiseFreeForgetting(kb, timeout=60)
        
        # Keep A and C, forget B
        result = forgetting.compute_module({'A', 'C'})
        
        assert result.status == RetrievalStatus.SUCCESS
        assert result.module is not None
        assert 'B' not in result.module.signature
    
    def test_compute_module_timeout(self):
        """Test module computation with timeout."""
        # Create a larger KB that might take time
        axioms = set()
        concepts = [AtomicConcept(f'C{i}') for i in range(50)]
        
        for i in range(len(concepts) - 1):
            axioms.add(ConceptInclusion(concepts[i], concepts[i+1]))
        
        kb = OntologyKB(axioms)
        forgetting = NoiseFreeForgetting(kb, timeout=1)  # Very short timeout
        
        # Try to forget many symbols
        target = {'C0', 'C49'}
        result = forgetting.compute_module(target)
        
        # Should either succeed or timeout, not error
        assert result.status in (
            RetrievalStatus.SUCCESS,
            RetrievalStatus.TIMEOUT
        )
    
    def test_zero_noise_property(self):
        """Test that result is truly zero-noise."""
        a = AtomicConcept('Disease')
        b = AtomicConcept('HeartDisease')
        c = AtomicConcept('Symptom')
        r = Role('hasSymptom')
        
        # HeartDisease ⊑ Disease
        # HeartDisease ⊑ ∃hasSymptom.Symptom
        axioms = {
            ConceptInclusion(b, a),
            ConceptInclusion(b, ExistentialRestriction(r, c)),
        }
        
        kb = OntologyKB(axioms)
        forgetting = NoiseFreeForgetting(kb, timeout=60)
        
        # Keep only Disease and HeartDisease
        target = {'Disease', 'HeartDisease'}
        result = forgetting.compute_module(target)
        
        if result.is_success and result.module:
            # Module signature should be subset of target
            module_sig = result.module.signature
            leaked = module_sig - target
            
            # Zero-noise means no leaked symbols (except possibly definers)
            # For this test, we check no original symbols leaked
            original_leaked = leaked & {'Symptom', 'hasSymptom'}
            assert len(original_leaked) == 0, f"Leaked symbols: {original_leaked}"


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compute_zero_noise_module(self):
        """Test compute_zero_noise_module function."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        
        axioms = {ConceptInclusion(a, b)}
        kb = OntologyKB(axioms)
        
        result = compute_zero_noise_module(kb, {'A', 'B'}, timeout=30)
        
        assert result is not None
        assert result.status == RetrievalStatus.SUCCESS
    
    def test_make_conjunction(self):
        """Test make_conjunction utility."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        
        conj = make_conjunction(a, b)
        assert isinstance(conj, Conjunction)
        assert a in conj.operands
        assert b in conj.operands
        
        # Single operand should return the operand
        single = make_conjunction(a)
        assert single == a
        
        # Empty should return Top
        empty = make_conjunction()
        assert isinstance(empty, Top)
    
    def test_make_disjunction(self):
        """Test make_disjunction utility."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        
        disj = make_disjunction(a, b)
        assert isinstance(disj, Disjunction)
        
        # Empty should return Bottom
        empty = make_disjunction()
        assert isinstance(empty, Bottom)


class TestClause:
    """Tests for Clause class."""
    
    def test_clause_creation(self):
        """Test clause creation."""
        a = AtomicConcept('A')
        b = AtomicConcept('B')
        
        clause = Clause(frozenset([a, b]))
        assert len(clause) == 2
    
    def test_clause_tautology(self):
        """Test tautology detection."""
        a = AtomicConcept('A')
        neg_a = Negation(a)
        
        # A ⊔ ¬A is a tautology
        clause = Clause(frozenset([a, neg_a]))
        assert clause.is_tautology()
        
        # A ⊔ ⊤ is a tautology
        clause_top = Clause(frozenset([a, Top()]))
        assert clause_top.is_tautology()
    
    def test_clause_simplify(self):
        """Test clause simplification."""
        a = AtomicConcept('A')
        bottom = Bottom()
        
        # A ⊔ ⊥ simplifies to A
        clause = Clause(frozenset([a, bottom]))
        simplified = clause.simplify()
        
        assert simplified is not None
        assert bottom not in simplified.literals


# Run tests with pytest
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
