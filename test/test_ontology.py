"""
Tests for the core ontology module.
"""

import pytest
from nfmr.src.core.ontology import (
    Concept, Top, Bottom, AtomicConcept, Negation, Conjunction, Disjunction,
    ExistentialRestriction, UniversalRestriction, Role, ConceptInclusion,
    Clause, OntologyKB, make_conjunction, make_disjunction
)


class TestConcepts:
    """Test cases for concept classes."""
    
    def test_top_concept(self):
        """Test Top concept properties."""
        top = Top()
        assert top.concept_type.name == "TOP"
        assert top.signature == set()
        assert not top.contains_symbol("A")
        assert str(top) == "⊤"
    
    def test_bottom_concept(self):
        """Test Bottom concept properties."""
        bottom = Bottom()
        assert bottom.concept_type.name == "BOTTOM"
        assert bottom.signature == set()
        assert str(bottom) == "⊥"
    
    def test_atomic_concept(self):
        """Test AtomicConcept properties."""
        a = AtomicConcept("Disease")
        assert a.name == "Disease"
        assert a.signature == {"Disease"}
        assert a.contains_symbol("Disease")
        assert not a.contains_symbol("Symptom")
        assert str(a) == "Disease"
    
    def test_atomic_concept_equality(self):
        """Test AtomicConcept equality."""
        a1 = AtomicConcept("A")
        a2 = AtomicConcept("A")
        a3 = AtomicConcept("B")
        
        assert a1 == a2
        assert a1 != a3
        assert hash(a1) == hash(a2)
    
    def test_negation(self):
        """Test Negation concept."""
        a = AtomicConcept("Disease")
        neg_a = Negation(a)
        
        assert neg_a.operand == a
        assert neg_a.signature == {"Disease"}
        assert neg_a.contains_symbol("Disease")
        assert str(neg_a) == "¬Disease"
    
    def test_double_negation_nnf(self):
        """Test that double negation eliminates in NNF."""
        a = AtomicConcept("A")
        double_neg = Negation(Negation(a))
        
        nnf = double_neg.nnf()
        assert nnf == a
    
    def test_conjunction(self):
        """Test Conjunction concept."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        conj = Conjunction(frozenset([a, b]))
        
        assert conj.signature == {"A", "B"}
        assert conj.contains_symbol("A")
        assert conj.contains_symbol("B")
        assert not conj.contains_symbol("C")
    
    def test_disjunction(self):
        """Test Disjunction concept."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        disj = Disjunction(frozenset([a, b]))
        
        assert disj.signature == {"A", "B"}
        assert disj.contains_symbol("A")
    
    def test_existential_restriction(self):
        """Test ExistentialRestriction concept."""
        r = Role("hasSymptom")
        a = AtomicConcept("Fever")
        exists = ExistentialRestriction(r, a)
        
        assert exists.role == r
        assert exists.filler == a
        assert exists.signature == {"hasSymptom", "Fever"}
        assert str(exists) == "∃hasSymptom.Fever"
    
    def test_universal_restriction(self):
        """Test UniversalRestriction concept."""
        r = Role("affectsOrgan")
        a = AtomicConcept("Organ")
        forall = UniversalRestriction(r, a)
        
        assert forall.role == r
        assert forall.filler == a
        assert str(forall) == "∀affectsOrgan.Organ"
    
    def test_de_morgan_nnf(self):
        """Test De Morgan's laws in NNF transformation."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        
        # ¬(A ⊓ B) should become ¬A ⊔ ¬B
        conj = Conjunction(frozenset([a, b]))
        neg_conj = Negation(conj)
        nnf = neg_conj.nnf()
        
        assert isinstance(nnf, Disjunction)
        assert len(nnf.operands) == 2
    
    def test_exists_to_forall_nnf(self):
        """Test ¬∃R.C ≡ ∀R.¬C transformation."""
        r = Role("r")
        a = AtomicConcept("A")
        exists = ExistentialRestriction(r, a)
        neg_exists = Negation(exists)
        
        nnf = neg_exists.nnf()
        
        assert isinstance(nnf, UniversalRestriction)
        assert isinstance(nnf.filler, Negation)


class TestRole:
    """Test cases for Role class."""
    
    def test_role_creation(self):
        """Test basic role creation."""
        r = Role("hasSymptom")
        assert r.name == "hasSymptom"
        assert not r.is_inverse
        assert str(r) == "hasSymptom"
    
    def test_inverse_role(self):
        """Test inverse role."""
        r = Role("hasSymptom")
        r_inv = r.inverse()
        
        assert r_inv.name == "hasSymptom"
        assert r_inv.is_inverse
        assert str(r_inv) == "hasSymptom⁻"
    
    def test_double_inverse(self):
        """Test that double inverse returns original."""
        r = Role("r")
        r_inv_inv = r.inverse().inverse()
        
        assert r_inv_inv == r


class TestConceptInclusion:
    """Test cases for ConceptInclusion axioms."""
    
    def test_basic_ci(self):
        """Test basic concept inclusion."""
        a = AtomicConcept("Endocarditis")
        b = AtomicConcept("HeartDisease")
        ci = ConceptInclusion(a, b)
        
        assert ci.lhs == a
        assert ci.rhs == b
        assert ci.signature == {"Endocarditis", "HeartDisease"}
        assert str(ci) == "Endocarditis ⊑ HeartDisease"
    
    def test_ci_to_clause(self):
        """Test conversion to clause."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        ci = ConceptInclusion(a, b)
        
        clause = ci.to_clause()
        # A ⊑ B becomes ⊤ ⊑ ¬A ⊔ B
        assert isinstance(clause, Clause)
        assert len(clause.literals) == 2


class TestClause:
    """Test cases for Clause class."""
    
    def test_empty_clause(self):
        """Test empty clause."""
        clause = Clause(frozenset())
        assert len(clause) == 0
    
    def test_tautology_detection(self):
        """Test tautology detection."""
        a = AtomicConcept("A")
        neg_a = Negation(a)
        
        # A ⊔ ¬A is a tautology
        clause = Clause(frozenset([a, neg_a]))
        assert clause.is_tautology()
    
    def test_non_tautology(self):
        """Test non-tautology clause."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        
        clause = Clause(frozenset([a, b]))
        assert not clause.is_tautology()
    
    def test_simplify_removes_bottom(self):
        """Test that simplification removes bottom literals."""
        a = AtomicConcept("A")
        bottom = Bottom()
        
        clause = Clause(frozenset([a, bottom]))
        simplified = clause.simplify()
        
        assert simplified is not None
        assert Bottom() not in simplified.literals


class TestOntologyKB:
    """Test cases for OntologyKB class."""
    
    def test_empty_kb(self):
        """Test empty knowledge base."""
        kb = OntologyKB()
        assert len(kb) == 0
        assert kb.signature == set()
    
    def test_add_axiom(self):
        """Test adding axioms."""
        kb = OntologyKB()
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        ci = ConceptInclusion(a, b)
        
        kb.add_axiom(ci)
        
        assert len(kb) == 1
        assert ci in kb
        assert kb.signature == {"A", "B"}
    
    def test_remove_axiom(self):
        """Test removing axioms."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        ci = ConceptInclusion(a, b)
        kb = OntologyKB({ci})
        
        kb.remove_axiom(ci)
        assert len(kb) == 0
    
    def test_get_axioms_with_symbol(self):
        """Test filtering axioms by symbol."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        c = AtomicConcept("C")
        
        ci1 = ConceptInclusion(a, b)
        ci2 = ConceptInclusion(b, c)
        
        kb = OntologyKB({ci1, ci2})
        
        b_axioms = kb.get_axioms_with_symbol("B")
        assert len(b_axioms) == 2
        
        a_axioms = kb.get_axioms_with_symbol("A")
        assert len(a_axioms) == 1
    
    def test_concept_and_role_names(self):
        """Test extracting concept and role names."""
        disease = AtomicConcept("Disease")
        symptom = AtomicConcept("Symptom")
        has_symptom = Role("hasSymptom")
        
        restriction = ExistentialRestriction(has_symptom, symptom)
        ci = ConceptInclusion(disease, restriction)
        
        kb = OntologyKB({ci})
        
        assert "Disease" in kb.concept_names
        assert "Symptom" in kb.concept_names
        assert "hasSymptom" in kb.role_names
    
    def test_copy(self):
        """Test KB copying."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        ci = ConceptInclusion(a, b)
        
        kb1 = OntologyKB({ci})
        kb2 = kb1.copy()
        
        assert len(kb2) == 1
        assert ci in kb2
        
        # Modifications to copy shouldn't affect original
        kb2.remove_axiom(ci)
        assert ci in kb1


class TestMakeHelpers:
    """Test cases for make_conjunction and make_disjunction helpers."""
    
    def test_make_conjunction_flattening(self):
        """Test that nested conjunctions are flattened."""
        a = AtomicConcept("A")
        b = AtomicConcept("B")
        c = AtomicConcept("C")
        
        inner = Conjunction(frozenset([a, b]))
        result = make_conjunction(inner, c)
        
        assert isinstance(result, Conjunction)
        assert len(result.operands) == 3
    
    def test_make_conjunction_single(self):
        """Test that single operand returns the operand."""
        a = AtomicConcept("A")
        result = make_conjunction(a)
        
        assert result == a
    
    def test_make_conjunction_with_top(self):
        """Test that Top is removed from conjunctions."""
        a = AtomicConcept("A")
        top = Top()
        
        result = make_conjunction(a, top)
        assert result == a
    
    def test_make_conjunction_with_bottom(self):
        """Test that Bottom absorbs conjunction."""
        a = AtomicConcept("A")
        bottom = Bottom()
        
        result = make_conjunction(a, bottom)
        assert isinstance(result, Bottom)
    
    def test_make_disjunction_with_top(self):
        """Test that Top absorbs disjunction."""
        a = AtomicConcept("A")
        top = Top()
        
        result = make_disjunction(a, top)
        assert isinstance(result, Top)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
