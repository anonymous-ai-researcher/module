"""
NFMR: Noise-Free Module Retrieval
OWL Parser Utility

This module provides functionality for parsing OWL ontology files
and converting them to the internal OntologyKB representation.
"""

from __future__ import annotations
from typing import Set, Optional, Dict, List, Any
from pathlib import Path
import logging
import re

from ..core.ontology import (
    Concept, Top, Bottom, AtomicConcept, Negation, Conjunction, Disjunction,
    ExistentialRestriction, UniversalRestriction, Role, ConceptInclusion,
    OntologyKB, make_conjunction, make_disjunction
)

logger = logging.getLogger(__name__)


class OWLParser:
    """
    Parser for OWL ontology files.
    
    Supports:
    - OWL/XML format
    - RDF/XML format
    - Functional syntax
    - Manchester syntax (partial)
    
    Extracts ALCI-expressible axioms and ignores unsupported constructs.
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize the parser.
        
        Args:
            strict: If True, raise errors on unsupported constructs
        """
        self.strict = strict
        self._concept_cache: Dict[str, AtomicConcept] = {}
        self._role_cache: Dict[str, Role] = {}
    
    def parse(self, path: str) -> OntologyKB:
        """
        Parse an OWL file and return an OntologyKB.
        
        Args:
            path: Path to the OWL file
            
        Returns:
            OntologyKB instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {path}")
        
        # Determine format and parse
        suffix = path.suffix.lower()
        content = path.read_text(encoding='utf-8')
        
        if suffix in ('.owl', '.xml', '.rdf'):
            return self._parse_xml(content)
        elif suffix == '.ofn':
            return self._parse_functional(content)
        elif suffix == '.omn':
            return self._parse_manchester(content)
        else:
            # Try to detect format from content
            if '<?xml' in content or '<Ontology' in content:
                return self._parse_xml(content)
            elif 'Prefix(' in content or 'Ontology(' in content:
                return self._parse_functional(content)
            else:
                return self._parse_manchester(content)
    
    def _parse_xml(self, content: str) -> OntologyKB:
        """Parse OWL/XML or RDF/XML format."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree required for XML parsing")
        
        axioms: Set[ConceptInclusion] = set()
        
        # Define namespaces
        namespaces = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xml': 'http://www.w3.org/XML/1998/namespace',
        }
        
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return OntologyKB(axioms)
        
        # Extract SubClassOf axioms
        for subclass in root.findall('.//owl:SubClassOf', namespaces):
            try:
                axiom = self._parse_subclass_axiom(subclass, namespaces)
                if axiom:
                    axioms.add(axiom)
            except Exception as e:
                if self.strict:
                    raise
                logger.debug(f"Skipping axiom due to: {e}")
        
        # Extract EquivalentClasses axioms
        for equiv in root.findall('.//owl:EquivalentClasses', namespaces):
            try:
                equiv_axioms = self._parse_equivalent_axiom(equiv, namespaces)
                axioms.update(equiv_axioms)
            except Exception as e:
                if self.strict:
                    raise
                logger.debug(f"Skipping axiom due to: {e}")
        
        # Also check for rdfs:subClassOf
        for elem in root.findall('.//*[@rdf:about]', namespaces):
            for subclass_of in elem.findall('rdfs:subClassOf', namespaces):
                try:
                    sub_name = self._extract_iri_name(elem.get(f'{{{namespaces["rdf"]}}}about'))
                    sup_ref = subclass_of.get(f'{{{namespaces["rdf"]}}}resource')
                    
                    if sub_name and sup_ref:
                        sup_name = self._extract_iri_name(sup_ref)
                        if sub_name and sup_name:
                            axiom = ConceptInclusion(
                                self._get_concept(sub_name),
                                self._get_concept(sup_name)
                            )
                            axioms.add(axiom)
                except Exception as e:
                    if self.strict:
                        raise
                    logger.debug(f"Skipping rdfs:subClassOf due to: {e}")
        
        logger.info(f"Parsed {len(axioms)} axioms from OWL/XML")
        return OntologyKB(axioms)
    
    def _parse_subclass_axiom(
        self, 
        elem, 
        namespaces: Dict[str, str]
    ) -> Optional[ConceptInclusion]:
        """Parse a SubClassOf axiom element."""
        children = list(elem)
        
        if len(children) < 2:
            return None
        
        lhs = self._parse_class_expression(children[0], namespaces)
        rhs = self._parse_class_expression(children[1], namespaces)
        
        if lhs and rhs:
            return ConceptInclusion(lhs, rhs)
        return None
    
    def _parse_equivalent_axiom(
        self, 
        elem, 
        namespaces: Dict[str, str]
    ) -> List[ConceptInclusion]:
        """Parse an EquivalentClasses axiom (creates two CIs)."""
        children = list(elem)
        
        if len(children) < 2:
            return []
        
        concepts = []
        for child in children:
            c = self._parse_class_expression(child, namespaces)
            if c:
                concepts.append(c)
        
        axioms = []
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                if i != j:
                    axioms.append(ConceptInclusion(concepts[i], concepts[j]))
        
        return axioms
    
    def _parse_class_expression(
        self, 
        elem, 
        namespaces: Dict[str, str]
    ) -> Optional[Concept]:
        """Parse a class expression element."""
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        
        if tag == 'Class':
            iri = elem.get('IRI') or elem.get(f'{{{namespaces["rdf"]}}}about')
            if iri:
                name = self._extract_iri_name(iri)
                return self._get_concept(name)
        
        elif tag == 'ObjectIntersectionOf':
            operands = []
            for child in elem:
                op = self._parse_class_expression(child, namespaces)
                if op:
                    operands.append(op)
            if operands:
                return make_conjunction(*operands)
        
        elif tag == 'ObjectUnionOf':
            operands = []
            for child in elem:
                op = self._parse_class_expression(child, namespaces)
                if op:
                    operands.append(op)
            if operands:
                return make_disjunction(*operands)
        
        elif tag == 'ObjectComplementOf':
            children = list(elem)
            if children:
                inner = self._parse_class_expression(children[0], namespaces)
                if inner:
                    return Negation(inner)
        
        elif tag == 'ObjectSomeValuesFrom':
            children = list(elem)
            if len(children) >= 2:
                role = self._parse_object_property(children[0], namespaces)
                filler = self._parse_class_expression(children[1], namespaces)
                if role and filler:
                    return ExistentialRestriction(role, filler)
        
        elif tag == 'ObjectAllValuesFrom':
            children = list(elem)
            if len(children) >= 2:
                role = self._parse_object_property(children[0], namespaces)
                filler = self._parse_class_expression(children[1], namespaces)
                if role and filler:
                    return UniversalRestriction(role, filler)
        
        # Handle named class reference via abbreviatedIRI
        abbreviated = elem.get('abbreviatedIRI')
        if abbreviated:
            name = abbreviated.split(':')[-1] if ':' in abbreviated else abbreviated
            return self._get_concept(name)
        
        return None
    
    def _parse_object_property(
        self, 
        elem, 
        namespaces: Dict[str, str]
    ) -> Optional[Role]:
        """Parse an object property element."""
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        
        if tag == 'ObjectProperty':
            iri = elem.get('IRI') or elem.get('abbreviatedIRI')
            if iri:
                name = self._extract_iri_name(iri)
                return self._get_role(name)
        
        elif tag == 'ObjectInverseOf':
            children = list(elem)
            if children:
                inner = self._parse_object_property(children[0], namespaces)
                if inner:
                    return inner.inverse()
        
        return None
    
    def _parse_functional(self, content: str) -> OntologyKB:
        """Parse OWL Functional Syntax."""
        axioms: Set[ConceptInclusion] = set()
        
        # Simple regex-based parsing for common patterns
        # SubClassOf(C D)
        subclass_pattern = r'SubClassOf\s*\(\s*([^)]+)\s+([^)]+)\s*\)'
        
        for match in re.finditer(subclass_pattern, content):
            try:
                lhs_str = match.group(1).strip()
                rhs_str = match.group(2).strip()
                
                lhs = self._parse_functional_expression(lhs_str)
                rhs = self._parse_functional_expression(rhs_str)
                
                if lhs and rhs:
                    axioms.add(ConceptInclusion(lhs, rhs))
            except Exception as e:
                if self.strict:
                    raise
                logger.debug(f"Skipping functional axiom: {e}")
        
        logger.info(f"Parsed {len(axioms)} axioms from Functional Syntax")
        return OntologyKB(axioms)
    
    def _parse_functional_expression(self, expr: str) -> Optional[Concept]:
        """Parse a functional syntax class expression."""
        expr = expr.strip()
        
        # Handle IRI reference
        if expr.startswith('<') and expr.endswith('>'):
            name = self._extract_iri_name(expr[1:-1])
            return self._get_concept(name)
        
        # Handle prefixed name
        if ':' in expr and not expr.startswith('ObjectSome'):
            name = expr.split(':')[-1]
            return self._get_concept(name)
        
        # Handle ObjectSomeValuesFrom
        if expr.startswith('ObjectSomeValuesFrom'):
            # Simplified parsing
            inner = expr[len('ObjectSomeValuesFrom'):].strip()
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1]
                parts = inner.split(None, 1)
                if len(parts) >= 2:
                    role = self._get_role(self._extract_iri_name(parts[0]))
                    filler = self._parse_functional_expression(parts[1])
                    if role and filler:
                        return ExistentialRestriction(role, filler)
        
        # Handle plain name
        if expr.isidentifier() or expr.replace('_', '').isalnum():
            return self._get_concept(expr)
        
        return None
    
    def _parse_manchester(self, content: str) -> OntologyKB:
        """Parse Manchester Syntax (basic support)."""
        axioms: Set[ConceptInclusion] = set()
        
        # Pattern for Class: Name SubClassOf: SuperClass
        lines = content.split('\n')
        current_class = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Class:'):
                current_class = line[6:].strip()
            
            elif line.startswith('SubClassOf:') and current_class:
                super_class = line[11:].strip()
                if current_class and super_class:
                    axiom = ConceptInclusion(
                        self._get_concept(current_class),
                        self._parse_manchester_expression(super_class)
                    )
                    if axiom.rhs:
                        axioms.add(axiom)
        
        logger.info(f"Parsed {len(axioms)} axioms from Manchester Syntax")
        return OntologyKB(axioms)
    
    def _parse_manchester_expression(self, expr: str) -> Optional[Concept]:
        """Parse a Manchester syntax class expression."""
        expr = expr.strip()
        
        # Handle 'some' restrictions: role some Filler
        if ' some ' in expr:
            parts = expr.split(' some ', 1)
            if len(parts) == 2:
                role = self._get_role(parts[0].strip())
                filler = self._parse_manchester_expression(parts[1])
                if role and filler:
                    return ExistentialRestriction(role, filler)
        
        # Handle 'only' restrictions: role only Filler
        if ' only ' in expr:
            parts = expr.split(' only ', 1)
            if len(parts) == 2:
                role = self._get_role(parts[0].strip())
                filler = self._parse_manchester_expression(parts[1])
                if role and filler:
                    return UniversalRestriction(role, filler)
        
        # Handle 'and' conjunctions
        if ' and ' in expr:
            parts = expr.split(' and ')
            operands = [self._parse_manchester_expression(p) for p in parts]
            operands = [o for o in operands if o]
            if operands:
                return make_conjunction(*operands)
        
        # Handle 'or' disjunctions
        if ' or ' in expr:
            parts = expr.split(' or ')
            operands = [self._parse_manchester_expression(p) for p in parts]
            operands = [o for o in operands if o]
            if operands:
                return make_disjunction(*operands)
        
        # Handle 'not' negation
        if expr.startswith('not '):
            inner = self._parse_manchester_expression(expr[4:])
            if inner:
                return Negation(inner)
        
        # Plain concept name
        return self._get_concept(expr)
    
    def _extract_iri_name(self, iri: str) -> str:
        """Extract the local name from an IRI."""
        if not iri:
            return ""
        
        # Remove angle brackets if present
        if iri.startswith('<') and iri.endswith('>'):
            iri = iri[1:-1]
        
        # Handle prefixed names
        if ':' in iri and not iri.startswith('http'):
            return iri.split(':')[-1]
        
        # Extract fragment or last path component
        if '#' in iri:
            return iri.split('#')[-1]
        elif '/' in iri:
            return iri.split('/')[-1]
        
        return iri
    
    def _get_concept(self, name: str) -> AtomicConcept:
        """Get or create an atomic concept."""
        name = name.strip()
        if name not in self._concept_cache:
            self._concept_cache[name] = AtomicConcept(name)
        return self._concept_cache[name]
    
    def _get_role(self, name: str) -> Role:
        """Get or create a role."""
        name = name.strip()
        if name not in self._role_cache:
            self._role_cache[name] = Role(name)
        return self._role_cache[name]


def parse_owl(path: str, strict: bool = False) -> OntologyKB:
    """
    Convenience function to parse an OWL file.
    
    Args:
        path: Path to the OWL file
        strict: If True, raise errors on unsupported constructs
        
    Returns:
        OntologyKB instance
    """
    parser = OWLParser(strict=strict)
    return parser.parse(path)
