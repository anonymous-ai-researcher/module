#!/usr/bin/env python3
"""
NFMR RAG Evaluation Script

This script evaluates the RAG pipeline on Bio-RAG-Bench using different
retrieval methods to compare accuracy, hallucination rates, and efficiency.

Usage:
    python -m nfmr.experiments.run_rag_eval --model llama3-8b --methods nfmr,star,bm25
    python -m nfmr.experiments.run_rag_eval --benchmark data/benchmarks/bio_rag_bench.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import OntologyKB
from src.rag import RAGPipeline, RAGResponse
from src.evaluation import (
    BioRAGBenchGenerator,
    BioRAGBenchEvaluator,
    BenchmarkQuestion
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for RAG pipeline performance."""
    
    def __init__(
        self,
        ontology_path: str,
        llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        llm_backend: str = "huggingface"
    ):
        """Initialize the evaluator."""
        self.ontology_path = ontology_path
        self.llm_model = llm_model
        self.llm_backend = llm_backend
        
        # Load ontology
        logger.info(f"Loading ontology from {ontology_path}")
        if Path(ontology_path).exists():
            self.ontology = OntologyKB.from_owl(ontology_path)
        else:
            logger.warning("Ontology not found, creating dummy")
            self.ontology = self._create_dummy_ontology()
    
    def _create_dummy_ontology(self) -> OntologyKB:
        """Create a dummy biomedical ontology for testing."""
        from src.core import (
            AtomicConcept, Role, ConceptInclusion,
            ExistentialRestriction, Negation, make_disjunction
        )
        
        axioms = set()
        
        # Create biomedical-like concepts
        concepts = {
            'Disease': AtomicConcept('Disease'),
            'HeartDisease': AtomicConcept('HeartDisease'),
            'Endocarditis': AtomicConcept('Endocarditis'),
            'Myocarditis': AtomicConcept('Myocarditis'),
            'InfectiousDisease': AtomicConcept('InfectiousDisease'),
            'Symptom': AtomicConcept('Symptom'),
            'Fever': AtomicConcept('Fever'),
            'ChestPain': AtomicConcept('ChestPain'),
            'Treatment': AtomicConcept('Treatment'),
            'Antibiotic': AtomicConcept('Antibiotic'),
            'Surgery': AtomicConcept('Surgery'),
        }
        
        roles = {
            'hasSymptom': Role('hasSymptom'),
            'hasTreatment': Role('hasTreatment'),
            'causedBy': Role('causedBy'),
        }
        
        # Add subsumption axioms
        axioms.add(ConceptInclusion(concepts['HeartDisease'], concepts['Disease']))
        axioms.add(ConceptInclusion(concepts['Endocarditis'], concepts['HeartDisease']))
        axioms.add(ConceptInclusion(concepts['Myocarditis'], concepts['HeartDisease']))
        axioms.add(ConceptInclusion(concepts['Endocarditis'], concepts['InfectiousDisease']))
        axioms.add(ConceptInclusion(concepts['Fever'], concepts['Symptom']))
        axioms.add(ConceptInclusion(concepts['ChestPain'], concepts['Symptom']))
        axioms.add(ConceptInclusion(concepts['Antibiotic'], concepts['Treatment']))
        axioms.add(ConceptInclusion(concepts['Surgery'], concepts['Treatment']))
        
        # Add restriction axioms
        axioms.add(ConceptInclusion(
            concepts['HeartDisease'],
            ExistentialRestriction(roles['hasSymptom'], concepts['ChestPain'])
        ))
        axioms.add(ConceptInclusion(
            concepts['InfectiousDisease'],
            ExistentialRestriction(roles['hasSymptom'], concepts['Fever'])
        ))
        axioms.add(ConceptInclusion(
            concepts['InfectiousDisease'],
            ExistentialRestriction(roles['hasTreatment'], concepts['Antibiotic'])
        ))
        
        return OntologyKB(axioms)
    
    def evaluate_method(
        self,
        method: str,
        benchmark: List[BenchmarkQuestion],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Evaluate a single retrieval method."""
        logger.info(f"Evaluating method: {method}")
        
        # Create pipeline with this method
        # For testing without LLM, we'll mock the responses
        results = []
        
        for i, question in enumerate(benchmark):
            if i % 50 == 0:
                logger.info(f"  Processing question {i+1}/{len(benchmark)}")
            
            try:
                result = self._evaluate_question(question, method, timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Error on question {question.id}: {e}")
                results.append({
                    'question_id': question.id,
                    'error': str(e),
                    'is_correct': False,
                    'is_hallucination': True,
                })
        
        return self._compute_metrics(results, method)
    
    def _evaluate_question(
        self,
        question: BenchmarkQuestion,
        method: str,
        timeout: int
    ) -> Dict[str, Any]:
        """Evaluate a single question."""
        from src.rag import OntologyRetriever
        
        start_retrieval = time.time()
        
        # Retrieve context
        retriever = OntologyRetriever(self.ontology, method=method, timeout=timeout)
        retrieval_result = retriever.retrieve(question.target_concepts)
        
        retrieval_time = (time.time() - start_retrieval) * 1000  # ms
        
        # Serialize context
        context = ""
        context_tokens = 0
        
        if retrieval_result.module:
            context_lines = []
            for axiom in retrieval_result.module.axioms:
                context_lines.append(str(axiom))
            context = "\n".join(context_lines)
            context_tokens = len(context.split()) * 1.3  # rough estimate
        
        # For testing without LLM, use rule-based answer
        start_generation = time.time()
        answer = self._generate_mock_answer(question, context)
        generation_time = (time.time() - start_generation) * 1000
        
        # Check correctness and hallucination
        is_correct = self._check_correctness(answer, question.ground_truth)
        is_hallucination = self._check_hallucination(answer, context)
        
        return {
            'question_id': question.id,
            'question_type': question.question_type,
            'difficulty': question.difficulty,
            'answer': answer,
            'ground_truth': question.ground_truth,
            'is_correct': is_correct,
            'is_hallucination': is_hallucination,
            'context_tokens': int(context_tokens),
            'retrieval_time_ms': retrieval_time,
            'generation_time_ms': generation_time,
            'module_size': len(retrieval_result.module) if retrieval_result.module else 0,
            'is_zero_noise': retrieval_result.is_zero_noise if hasattr(retrieval_result, 'is_zero_noise') else False,
        }
    
    def _generate_mock_answer(
        self,
        question: BenchmarkQuestion,
        context: str
    ) -> str:
        """Generate a mock answer based on context (for testing without LLM)."""
        context_lower = context.lower()
        
        # Simple pattern matching for subsumption questions
        if question.question_type == 'subsumption':
            concepts = list(question.target_concepts)
            if len(concepts) >= 2:
                c1, c2 = concepts[0], concepts[1]
                
                # Check if subsumption is in context
                if f"{c1.lower()} ⊑ {c2.lower()}" in context_lower:
                    return f"Yes, {c1} is a type of {c2}."
                elif c1.lower() in context_lower and c2.lower() in context_lower:
                    return f"Yes, based on the ontology, {c1} appears to be related to {c2}."
        
        # Default: check if concepts appear in context
        found_concepts = sum(1 for c in question.target_concepts if c.lower() in context_lower)
        
        if found_concepts == len(question.target_concepts):
            return f"Based on the context, the answer involves: {', '.join(question.target_concepts)}"
        elif found_concepts > 0:
            return f"Partial information found. Some concepts appear in the context."
        else:
            return "Cannot determine from the provided context."
    
    def _check_correctness(self, answer: str, ground_truth: str) -> bool:
        """Check if the answer is correct."""
        answer_lower = answer.lower()
        gt_lower = ground_truth.lower()
        
        # Simple keyword matching
        if gt_lower.startswith("yes"):
            return "yes" in answer_lower and "no" not in answer_lower[:20]
        elif gt_lower.startswith("no"):
            return answer_lower.startswith("no") or "is not" in answer_lower
        elif "cannot" in gt_lower:
            return "cannot" in answer_lower or "not determine" in answer_lower
        
        # Overlap-based matching
        gt_words = set(gt_lower.split())
        answer_words = set(answer_lower.split())
        overlap = len(gt_words & answer_words) / len(gt_words) if gt_words else 0
        
        return overlap > 0.5
    
    def _check_hallucination(self, answer: str, context: str) -> bool:
        """Check if the answer contains hallucinations."""
        answer_lower = answer.lower()
        
        # Check for hedging phrases that indicate uncertainty
        uncertainty_phrases = [
            "based on my knowledge",
            "i believe",
            "typically",
            "generally speaking",
            "usually",
            "in most cases"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                return True
        
        # If context is empty but answer is confident, likely hallucination
        if not context.strip() and "yes" in answer_lower:
            return True
        
        return False
    
    def _compute_metrics(
        self,
        results: List[Dict[str, Any]],
        method: str
    ) -> Dict[str, Any]:
        """Compute aggregate metrics from results."""
        n = len(results)
        if n == 0:
            return {'method': method, 'error': 'No results'}
        
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'method': method, 'error': 'All evaluations failed'}
        
        n_valid = len(valid_results)
        
        accuracy = sum(1 for r in valid_results if r['is_correct']) / n_valid
        hallucination_rate = sum(1 for r in valid_results if r['is_hallucination']) / n_valid
        avg_context_tokens = sum(r['context_tokens'] for r in valid_results) / n_valid
        avg_retrieval_time = sum(r['retrieval_time_ms'] for r in valid_results) / n_valid
        avg_generation_time = sum(r['generation_time_ms'] for r in valid_results) / n_valid
        zero_noise_rate = sum(1 for r in valid_results if r.get('is_zero_noise', False)) / n_valid
        
        # Breakdown by question type
        type_metrics = {}
        for qtype in ['subsumption', 'restriction', 'multi_hop']:
            type_results = [r for r in valid_results if r.get('question_type') == qtype]
            if type_results:
                type_metrics[qtype] = {
                    'accuracy': sum(1 for r in type_results if r['is_correct']) / len(type_results),
                    'hallucination_rate': sum(1 for r in type_results if r['is_hallucination']) / len(type_results),
                    'count': len(type_results),
                }
        
        # Breakdown by difficulty
        diff_metrics = {}
        for diff in ['easy', 'medium', 'hard']:
            diff_results = [r for r in valid_results if r.get('difficulty') == diff]
            if diff_results:
                diff_metrics[diff] = {
                    'accuracy': sum(1 for r in diff_results if r['is_correct']) / len(diff_results),
                    'count': len(diff_results),
                }
        
        return {
            'method': method,
            'num_questions': n,
            'num_valid': n_valid,
            'accuracy': accuracy,
            'hallucination_rate': hallucination_rate,
            'avg_context_tokens': avg_context_tokens,
            'avg_retrieval_time_ms': avg_retrieval_time,
            'avg_generation_time_ms': avg_generation_time,
            'zero_noise_rate': zero_noise_rate,
            'by_type': type_metrics,
            'by_difficulty': diff_metrics,
        }


def run_comparison(
    ontology_path: str,
    benchmark_path: Optional[str],
    methods: List[str],
    num_questions: int,
    llm_model: str,
    output_path: str
):
    """Run comparison across multiple methods."""
    logger.info("Starting RAG evaluation comparison")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        ontology_path=ontology_path,
        llm_model=llm_model
    )
    
    # Load or generate benchmark
    if benchmark_path and Path(benchmark_path).exists():
        logger.info(f"Loading benchmark from {benchmark_path}")
        benchmark = BioRAGBenchGenerator.load_benchmark(benchmark_path)
    else:
        logger.info("Generating synthetic benchmark")
        generator = BioRAGBenchGenerator(evaluator.ontology)
        benchmark = generator.generate_benchmark(num_questions=num_questions)
        
        # Save generated benchmark
        if benchmark_path:
            Path(benchmark_path).parent.mkdir(parents=True, exist_ok=True)
            generator.save_benchmark(benchmark, benchmark_path)
    
    logger.info(f"Benchmark contains {len(benchmark)} questions")
    
    # Evaluate each method
    all_results = {}
    
    for method in methods:
        logger.info(f"\nEvaluating {method}...")
        results = evaluator.evaluate_method(method, benchmark)
        all_results[method] = results
    
    # Save results
    output = {
        'methods': all_results,
        'benchmark_size': len(benchmark),
        'llm_model': llm_model,
        'timestamp': datetime.now().isoformat(),
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return all_results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a comparison table of results."""
    print("\n" + "="*80)
    print("RAG EVALUATION RESULTS")
    print("="*80)
    
    # Header
    print(f"{'Method':<15} {'Accuracy':>10} {'Halluc.':>10} {'Tokens':>10} {'Ret.Time':>10} {'Zero-Noise':>12}")
    print("-"*80)
    
    for method, metrics in results.items():
        if 'error' in metrics:
            print(f"{method:<15} ERROR: {metrics['error']}")
            continue
        
        print(f"{method:<15} "
              f"{metrics['accuracy']*100:>9.1f}% "
              f"{metrics['hallucination_rate']*100:>9.1f}% "
              f"{metrics['avg_context_tokens']:>10.0f} "
              f"{metrics['avg_retrieval_time_ms']:>9.1f}ms "
              f"{metrics['zero_noise_rate']*100:>11.1f}%")
    
    print("="*80)
    
    # Detailed breakdown
    print("\nBREAKDOWN BY QUESTION TYPE:")
    print("-"*60)
    
    for method, metrics in results.items():
        if 'error' in metrics:
            continue
        
        print(f"\n{method}:")
        for qtype, type_metrics in metrics.get('by_type', {}).items():
            print(f"  {qtype}: accuracy={type_metrics['accuracy']*100:.1f}%, "
                  f"halluc.={type_metrics.get('hallucination_rate', 0)*100:.1f}%, "
                  f"n={type_metrics['count']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation on Bio-RAG-Bench"
    )
    parser.add_argument(
        '--ontology', '-o',
        type=str,
        default='data/ontologies/bioportal/sample.owl',
        help='Path to source ontology'
    )
    parser.add_argument(
        '--benchmark', '-b',
        type=str,
        default='data/benchmarks/bio_rag_bench.json',
        help='Path to benchmark file (will generate if not exists)'
    )
    parser.add_argument(
        '--methods', '-m',
        type=str,
        default='nfmr,star,bm25',
        help='Comma-separated list of methods to evaluate'
    )
    parser.add_argument(
        '--num-questions', '-n',
        type=int,
        default=500,
        help='Number of questions to generate if benchmark not provided'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='LLM model to use for generation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/rag_eval_results.json',
        help='Output file path'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    methods = args.methods.split(',')
    
    results = run_comparison(
        ontology_path=args.ontology,
        benchmark_path=args.benchmark,
        methods=methods,
        num_questions=args.num_questions,
        llm_model=args.model,
        output_path=args.output
    )
    
    print_comparison_table(results)


if __name__ == '__main__':
    main()
