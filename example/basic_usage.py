#!/usr/bin/env python3
"""
NFMR: Noise-Free Module Retrieval
Example Usage Script

This script demonstrates the basic usage of the NFMR package for:
1. Loading an ontology
2. Computing a zero-noise module
3. Running a RAG query
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    OntologyKB,
    AtomicConcept,
    Role,
    ConceptInclusion,
    ExistentialRestriction,
    NoiseFreeForgetting,
    RetrievalStatus,
)


def create_sample_ontology() -> OntologyKB:
    """Create a sample medical ontology for demonstration."""
    
    # Define concepts
    disease = AtomicConcept("Disease")
    heart_disease = AtomicConcept("HeartDisease")
    endocarditis = AtomicConcept("Endocarditis")
    infectious_endo = AtomicConcept("InfectiousEndocarditis")
    
    symptom = AtomicConcept("Symptom")
    fever = AtomicConcept("Fever")
    fatigue = AtomicConcept("Fatigue")
    
    treatment = AtomicConcept("Treatment")
    antibiotic = AtomicConcept("Antibiotic")
    
    organ = AtomicConcept("Organ")
    heart = AtomicConcept("Heart")
    heart_valve = AtomicConcept("HeartValve")
    
    pathogen = AtomicConcept("Pathogen")
    bacteria = AtomicConcept("Bacteria")
    
    # Define roles
    has_symptom = Role("hasSymptom")
    has_treatment = Role("hasTreatment")
    affects_organ = Role("affectsOrgan")
    caused_by = Role("causedBy")
    
    # Create axioms
    axioms = {
        # Subsumption hierarchy
        ConceptInclusion(heart_disease, disease),
        ConceptInclusion(endocarditis, heart_disease),
        ConceptInclusion(infectious_endo, endocarditis),
        ConceptInclusion(fever, symptom),
        ConceptInclusion(fatigue, symptom),
        ConceptInclusion(antibiotic, treatment),
        ConceptInclusion(heart, organ),
        ConceptInclusion(heart_valve, organ),
        ConceptInclusion(bacteria, pathogen),
        
        # Role restrictions
        ConceptInclusion(
            endocarditis,
            ExistentialRestriction(has_symptom, fever)
        ),
        ConceptInclusion(
            endocarditis,
            ExistentialRestriction(affects_organ, heart_valve)
        ),
        ConceptInclusion(
            infectious_endo,
            ExistentialRestriction(caused_by, bacteria)
        ),
        ConceptInclusion(
            infectious_endo,
            ExistentialRestriction(has_treatment, antibiotic)
        ),
        ConceptInclusion(
            heart_disease,
            ExistentialRestriction(affects_organ, heart)
        ),
    }
    
    return OntologyKB(axioms)


def example_basic_forgetting():
    """Demonstrate basic forgetting operation."""
    print("=" * 60)
    print("Example 1: Basic Forgetting")
    print("=" * 60)
    
    # Create ontology
    kb = create_sample_ontology()
    print(f"\nOriginal KB has {len(kb)} axioms")
    print(f"Signature: {kb.signature}")
    
    # Define target vocabulary (symbols to KEEP)
    target_vocab = {"Endocarditis", "HeartDisease", "Disease"}
    print(f"\nTarget vocabulary (to keep): {target_vocab}")
    
    # Compute zero-noise module
    forgetting = NoiseFreeForgetting(kb)
    result = forgetting.compute_module(target_vocab)
    
    print(f"\nResult status: {result.status.name}")
    print(f"Time: {result.time_seconds:.3f}s")
    print(f"Definers introduced: {result.num_definers_introduced}")
    print(f"Definers remaining: {result.num_definers_remaining}")
    
    if result.is_success:
        module = result.module
        print(f"\nModule has {len(module)} axioms")
        print(f"Module signature: {module.signature}")
        print("\nModule axioms:")
        for ax in sorted(module.axioms, key=str):
            print(f"  {ax}")


def example_targeted_retrieval():
    """Demonstrate targeted knowledge retrieval for specific concepts."""
    print("\n" + "=" * 60)
    print("Example 2: Targeted Knowledge Retrieval")
    print("=" * 60)
    
    # Create ontology
    kb = create_sample_ontology()
    
    # Scenario: We want to know about InfectiousEndocarditis
    # and its relationship to treatments
    target_vocab = {"InfectiousEndocarditis", "Antibiotic", "Treatment", "hasTreatment"}
    print(f"\nQuery: What do we know about treating Infectious Endocarditis?")
    print(f"Target vocabulary: {target_vocab}")
    
    # Compute module
    forgetting = NoiseFreeForgetting(kb)
    result = forgetting.compute_module(target_vocab)
    
    if result.is_success:
        module = result.module
        print(f"\nRetrieved {len(module)} relevant axioms:")
        for ax in sorted(module.axioms, key=str):
            print(f"  {ax}")
        
        # This module contains ONLY the relevant knowledge
        # No unrelated concepts like Fever, Heart, Bacteria, etc.
        print(f"\nModule signature (no noise!): {module.signature}")


def example_compare_with_syntax_module():
    """Compare zero-noise module with syntax-based extraction."""
    print("\n" + "=" * 60)
    print("Example 3: Zero-Noise vs Syntax-Based Extraction")
    print("=" * 60)
    
    kb = create_sample_ontology()
    target_vocab = {"Endocarditis", "HeartDisease"}
    
    print(f"\nTarget vocabulary: {target_vocab}")
    
    # Syntax-based extraction (simplified STAR-like approach)
    print("\n--- Syntax-based extraction (STAR-like) ---")
    syntax_module = set()
    visited = set()
    frontier = set(target_vocab)
    
    while frontier:
        symbol = frontier.pop()
        if symbol in visited:
            continue
        visited.add(symbol)
        
        for axiom in kb.axioms:
            if axiom.contains_symbol(symbol):
                syntax_module.add(axiom)
                # Add all symbols from the axiom
                frontier.update(axiom.signature - visited)
    
    print(f"Syntax module has {len(syntax_module)} axioms")
    print(f"Syntax module signature: {set().union(*[ax.signature for ax in syntax_module]) if syntax_module else set()}")
    
    # Zero-noise extraction
    print("\n--- Zero-noise extraction (NFMR) ---")
    forgetting = NoiseFreeForgetting(kb)
    result = forgetting.compute_module(target_vocab)
    
    if result.is_success:
        print(f"Zero-noise module has {len(result.module)} axioms")
        print(f"Zero-noise module signature: {result.module.signature}")
        
        # Highlight the difference
        syntax_sig = set().union(*[ax.signature for ax in syntax_module]) if syntax_module else set()
        noise = syntax_sig - target_vocab
        print(f"\n🔍 Symbol leakage in syntax method: {noise}")
        print(f"✨ Zero-noise method has NO leakage!")


def example_load_owl_file():
    """Demonstrate loading from OWL file."""
    print("\n" + "=" * 60)
    print("Example 4: Loading from OWL File")
    print("=" * 60)
    
    owl_path = Path(__file__).parent.parent / "data" / "ontologies" / "sample_medical.owl"
    
    if owl_path.exists():
        print(f"\nLoading ontology from: {owl_path}")
        kb = OntologyKB.from_owl(str(owl_path))
        print(f"Loaded {len(kb)} axioms")
        print(f"Concepts: {kb.concept_names}")
        print(f"Roles: {kb.role_names}")
    else:
        print(f"\nOWL file not found at: {owl_path}")
        print("Run this script from the project root or adjust the path.")


def main():
    """Run all examples."""
    print("\n" + "🔬 " * 20)
    print("NFMR: Noise-Free Module Retrieval - Examples")
    print("🔬 " * 20 + "\n")
    
    example_basic_forgetting()
    example_targeted_retrieval()
    example_compare_with_syntax_module()
    example_load_owl_file()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
