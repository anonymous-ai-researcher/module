"""
NFMR: Noise-Free Module Retrieval
Bio-RAG-Bench - Biomedical RAG Benchmark

This module implements the Bio-RAG-Bench synthetic benchmark for evaluating
RAG systems on biomedical ontology-based question answering.
"""

from __future__ import annotations
from typing import Set, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import random
import logging

from ..core import OntologyKB, ConceptInclusion, AtomicConcept

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuestion:
    """A question in the Bio-RAG-Bench benchmark."""
    id: str
    question: str
    target_concepts: Set[str]
    ground_truth: str
    question_type: str  # 'subsumption', 'restriction', 'multi_hop'
    difficulty: str  # 'easy', 'medium', 'hard'
    source_axioms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'question': self.question,
            'target_concepts': list(self.target_concepts),
            'ground_truth': self.ground_truth,
            'question_type': self.question_type,
            'difficulty': self.difficulty,
            'source_axioms': self.source_axioms,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkQuestion':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            question=data['question'],
            target_concepts=set(data['target_concepts']),
            ground_truth=data['ground_truth'],
            question_type=data['question_type'],
            difficulty=data['difficulty'],
            source_axioms=data.get('source_axioms', []),
            metadata=data.get('metadata', {}),
        )


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: str
    predicted_answer: str
    ground_truth: str
    is_correct: bool
    is_hallucination: bool
    context_tokens: int
    retrieval_time_ms: float
    generation_time_ms: float
    retrieval_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BioRAGBenchGenerator:
    """
    Generator for Bio-RAG-Bench questions.
    
    Creates synthetic questions from biomedical ontologies that test:
    - Subsumption reasoning (Is A a type of B?)
    - Restriction understanding (What restrictions apply to A?)
    - Multi-hop reasoning (Combining multiple axioms)
    """
    
    def __init__(
        self,
        ontology: OntologyKB,
        llm_generator=None,
        seed: int = 42
    ):
        """
        Initialize the generator.
        
        Args:
            ontology: Source ontology for questions
            llm_generator: Optional LLM for generating natural questions
            seed: Random seed for reproducibility
        """
        self.ontology = ontology
        self.llm_generator = llm_generator
        self.random = random.Random(seed)
    
    def generate_benchmark(
        self,
        num_questions: int = 500,
        question_types: Optional[List[str]] = None
    ) -> List[BenchmarkQuestion]:
        """
        Generate a benchmark dataset.
        
        Args:
            num_questions: Number of questions to generate
            question_types: Types to include (default: all)
            
        Returns:
            List of BenchmarkQuestion objects
        """
        if question_types is None:
            question_types = ['subsumption', 'restriction', 'multi_hop']
        
        questions = []
        
        # Distribute questions across types
        per_type = num_questions // len(question_types)
        
        for qtype in question_types:
            type_questions = []
            
            if qtype == 'subsumption':
                type_questions = self._generate_subsumption_questions(per_type)
            elif qtype == 'restriction':
                type_questions = self._generate_restriction_questions(per_type)
            elif qtype == 'multi_hop':
                type_questions = self._generate_multihop_questions(per_type)
            
            questions.extend(type_questions)
        
        # Shuffle
        self.random.shuffle(questions)
        
        return questions[:num_questions]
    
    def _generate_subsumption_questions(
        self, 
        count: int
    ) -> List[BenchmarkQuestion]:
        """Generate subsumption-type questions."""
        questions = []
        
        # Find subsumption pairs from the ontology
        subsumption_pairs = []
        
        for axiom in self.ontology.axioms:
            lhs = axiom.lhs
            rhs = axiom.rhs
            
            if isinstance(lhs, AtomicConcept) and isinstance(rhs, AtomicConcept):
                subsumption_pairs.append((lhs.name, rhs.name, axiom))
        
        # Sample and create questions
        sampled = self.random.sample(
            subsumption_pairs, 
            min(count, len(subsumption_pairs))
        )
        
        for i, (sub, sup, axiom) in enumerate(sampled):
            # Positive question (true subsumption)
            q = BenchmarkQuestion(
                id=f"sub_pos_{i}",
                question=f"Is {sub} a type of {sup}?",
                target_concepts={sub, sup},
                ground_truth="Yes",
                question_type="subsumption",
                difficulty="easy",
                source_axioms=[str(axiom)],
            )
            questions.append(q)
            
            # Sometimes generate negative question (false subsumption)
            if self.random.random() < 0.3 and len(subsumption_pairs) > 1:
                # Pick a random unrelated concept
                other_pair = self.random.choice(subsumption_pairs)
                if other_pair[1] != sup:
                    q_neg = BenchmarkQuestion(
                        id=f"sub_neg_{i}",
                        question=f"Is {sub} a type of {other_pair[1]}?",
                        target_concepts={sub, other_pair[1]},
                        ground_truth="Cannot be determined from the context" if sub != other_pair[1] else "No",
                        question_type="subsumption",
                        difficulty="medium",
                        source_axioms=[],
                    )
                    questions.append(q_neg)
        
        return questions
    
    def _generate_restriction_questions(
        self, 
        count: int
    ) -> List[BenchmarkQuestion]:
        """Generate restriction-type questions."""
        questions = []
        
        # Find axioms with role restrictions
        from ..core import ExistentialRestriction, UniversalRestriction
        
        restriction_axioms = []
        
        for axiom in self.ontology.axioms:
            if isinstance(axiom.lhs, AtomicConcept):
                rhs = axiom.rhs
                if isinstance(rhs, (ExistentialRestriction, UniversalRestriction)):
                    restriction_axioms.append((axiom.lhs.name, rhs, axiom))
        
        sampled = self.random.sample(
            restriction_axioms,
            min(count, len(restriction_axioms))
        )
        
        for i, (concept, restriction, axiom) in enumerate(sampled):
            if isinstance(restriction, ExistentialRestriction):
                role = restriction.role
                filler = restriction.filler
                
                q = BenchmarkQuestion(
                    id=f"rest_{i}",
                    question=f"What is the {role.name} restriction for {concept}?",
                    target_concepts={concept, role.name},
                    ground_truth=f"{concept} has {role.name} some {filler}",
                    question_type="restriction",
                    difficulty="medium",
                    source_axioms=[str(axiom)],
                )
                questions.append(q)
        
        return questions
    
    def _generate_multihop_questions(
        self, 
        count: int
    ) -> List[BenchmarkQuestion]:
        """Generate multi-hop reasoning questions."""
        questions = []
        
        # Build a simple subsumption graph
        graph: Dict[str, Set[str]] = {}
        
        for axiom in self.ontology.axioms:
            if isinstance(axiom.lhs, AtomicConcept) and isinstance(axiom.rhs, AtomicConcept):
                sub = axiom.lhs.name
                sup = axiom.rhs.name
                
                if sub not in graph:
                    graph[sub] = set()
                graph[sub].add(sup)
        
        # Find multi-hop paths
        paths = []
        
        for start in graph:
            visited = {start}
            current = start
            path = [start]
            
            while current in graph and len(path) < 4:
                successors = graph[current] - visited
                if not successors:
                    break
                
                next_node = self.random.choice(list(successors))
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(path) >= 3:
                paths.append(path)
        
        # Create questions from paths
        sampled = self.random.sample(paths, min(count, len(paths)))
        
        for i, path in enumerate(sampled):
            start = path[0]
            end = path[-1]
            
            q = BenchmarkQuestion(
                id=f"multi_{i}",
                question=f"Is {start} a type of {end}? Explain the relationship.",
                target_concepts={start, end},
                ground_truth=f"Yes, {start} is a type of {end} through: {' -> '.join(path)}",
                question_type="multi_hop",
                difficulty="hard",
                source_axioms=[],
                metadata={"path": path, "hops": len(path) - 1}
            )
            questions.append(q)
        
        return questions
    
    def save_benchmark(
        self, 
        questions: List[BenchmarkQuestion], 
        path: str
    ) -> None:
        """Save benchmark to JSON file."""
        data = {
            'version': '1.0',
            'num_questions': len(questions),
            'questions': [q.to_dict() for q in questions]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(questions)} questions to {path}")
    
    @staticmethod
    def load_benchmark(path: str) -> List[BenchmarkQuestion]:
        """Load benchmark from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return [BenchmarkQuestion.from_dict(q) for q in data['questions']]


class BioRAGBenchEvaluator:
    """
    Evaluator for Bio-RAG-Bench.
    
    Computes metrics including:
    - Exact match accuracy
    - Hallucination rate
    - Context efficiency (tokens)
    - Retrieval/generation times
    """
    
    def __init__(
        self,
        rag_pipeline,
        hallucination_judge=None
    ):
        """
        Initialize the evaluator.
        
        Args:
            rag_pipeline: RAG pipeline to evaluate
            hallucination_judge: Optional LLM for judging hallucinations
        """
        self.rag_pipeline = rag_pipeline
        self.hallucination_judge = hallucination_judge
    
    def evaluate(
        self,
        benchmark: List[BenchmarkQuestion],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline on a benchmark.
        
        Args:
            benchmark: List of benchmark questions
            verbose: Whether to print progress
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = []
        
        for i, question in enumerate(benchmark):
            if verbose and i % 10 == 0:
                logger.info(f"Evaluating question {i+1}/{len(benchmark)}")
            
            # Get prediction
            response = self.rag_pipeline.query(
                question.question,
                target_concepts=question.target_concepts
            )
            
            # Check correctness
            is_correct = self._check_correctness(
                response.answer,
                question.ground_truth
            )
            
            # Check hallucination
            is_hallucination = self._check_hallucination(
                response.answer,
                response.context,
                question.target_concepts
            )
            
            result = EvaluationResult(
                question_id=question.id,
                predicted_answer=response.answer,
                ground_truth=question.ground_truth,
                is_correct=is_correct,
                is_hallucination=is_hallucination,
                context_tokens=response.context_tokens,
                retrieval_time_ms=response.retrieval_time_ms,
                generation_time_ms=response.generation_time_ms,
                retrieval_method=response.retrieval_method,
                metadata={
                    'question_type': question.question_type,
                    'difficulty': question.difficulty,
                }
            )
            results.append(result)
        
        return self._compute_metrics(results)
    
    def _check_correctness(
        self, 
        predicted: str, 
        ground_truth: str
    ) -> bool:
        """Check if prediction matches ground truth."""
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()
        
        # Exact match
        if pred_lower == gt_lower:
            return True
        
        # Check for key answer components
        if gt_lower.startswith("yes"):
            return "yes" in pred_lower and "no" not in pred_lower[:20]
        elif gt_lower.startswith("no"):
            return pred_lower.startswith("no") or "is not" in pred_lower
        
        # Partial match for longer answers
        gt_words = set(gt_lower.split())
        pred_words = set(pred_lower.split())
        overlap = len(gt_words & pred_words) / len(gt_words)
        
        return overlap > 0.7
    
    def _check_hallucination(
        self,
        answer: str,
        context: str,
        target_concepts: Set[str]
    ) -> bool:
        """Check if the answer contains hallucinations."""
        if self.hallucination_judge:
            # Use LLM to judge
            return self._llm_hallucination_check(answer, context)
        
        # Simple heuristic: check for concepts not in context
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        # Extract potential concept mentions from answer
        words = answer_lower.split()
        
        # Check for fabricated relationships
        fabrication_phrases = [
            "based on my knowledge",
            "i believe",
            "typically",
            "generally",
            "usually"
        ]
        
        for phrase in fabrication_phrases:
            if phrase in answer_lower:
                return True
        
        return False
    
    def _llm_hallucination_check(
        self, 
        answer: str, 
        context: str
    ) -> bool:
        """Use LLM to check for hallucinations."""
        prompt = f"""Determine if the following answer contains any information that is NOT supported by the given context.

Context:
{context}

Answer:
{answer}

Does the answer contain hallucinated information (claims not supported by the context)?
Respond with only 'YES' or 'NO'."""
        
        response = self.hallucination_judge.generate(
            prompt, 
            "", 
            "You are an expert at detecting hallucinations in text."
        )
        
        return "YES" in response.upper()
    
    def _compute_metrics(
        self, 
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        n = len(results)
        
        if n == 0:
            return {}
        
        accuracy = sum(1 for r in results if r.is_correct) / n
        hallucination_rate = sum(1 for r in results if r.is_hallucination) / n
        avg_tokens = sum(r.context_tokens for r in results) / n
        avg_retrieval_time = sum(r.retrieval_time_ms for r in results) / n
        avg_generation_time = sum(r.generation_time_ms for r in results) / n
        
        # Breakdown by question type
        type_metrics = {}
        for qtype in ['subsumption', 'restriction', 'multi_hop']:
            type_results = [r for r in results if r.metadata.get('question_type') == qtype]
            if type_results:
                type_metrics[qtype] = {
                    'accuracy': sum(1 for r in type_results if r.is_correct) / len(type_results),
                    'hallucination_rate': sum(1 for r in type_results if r.is_hallucination) / len(type_results),
                    'count': len(type_results),
                }
        
        # Breakdown by difficulty
        diff_metrics = {}
        for diff in ['easy', 'medium', 'hard']:
            diff_results = [r for r in results if r.metadata.get('difficulty') == diff]
            if diff_results:
                diff_metrics[diff] = {
                    'accuracy': sum(1 for r in diff_results if r.is_correct) / len(diff_results),
                    'count': len(diff_results),
                }
        
        return {
            'num_questions': n,
            'accuracy': accuracy,
            'hallucination_rate': hallucination_rate,
            'avg_context_tokens': avg_tokens,
            'avg_retrieval_time_ms': avg_retrieval_time,
            'avg_generation_time_ms': avg_generation_time,
            'by_type': type_metrics,
            'by_difficulty': diff_metrics,
            'retrieval_method': results[0].retrieval_method if results else None,
        }
