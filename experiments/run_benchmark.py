#!/usr/bin/env python3
"""
NFMR Benchmark Runner

This script runs the full benchmark evaluation comparing NFMR against
baseline methods on the Oxford-ISG and BioPortal datasets.

Usage:
    python -m nfmr.experiments.run_benchmark --config configs/benchmark.yaml
    python -m nfmr.experiments.run_benchmark --methods nfmr,star --ratio 0.3
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import yaml
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import OntologyKB, NoiseFreeForgetting, RetrievalStatus
from src.rag import OntologyRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmark evaluation across multiple methods and datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the benchmark runner."""
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.seed = config.get('general', {}).get('seed', 42)
        random.seed(self.seed)
    
    def run(self) -> Dict[str, Any]:
        """Run the full benchmark."""
        logger.info("Starting benchmark evaluation")
        start_time = time.time()
        
        # Load datasets
        datasets = self._load_datasets()
        
        # Get enabled methods
        methods = self._get_enabled_methods()
        
        # Get evaluation parameters
        eval_config = self.config.get('evaluation', {})
        elimination_ratios = eval_config.get('elimination_ratios', [0.3])
        num_runs = eval_config.get('num_runs', 10)
        timeout = eval_config.get('timeout_seconds', 300)
        
        # Run evaluation
        for dataset_name, ontologies in datasets.items():
            logger.info(f"Evaluating on dataset: {dataset_name}")
            
            for ont_path, ontology in ontologies.items():
                logger.info(f"  Processing: {ont_path}")
                
                for ratio in elimination_ratios:
                    for method_name in methods:
                        for run_id in range(num_runs):
                            result = self._run_single_evaluation(
                                ontology=ontology,
                                ontology_name=ont_path,
                                dataset=dataset_name,
                                method=method_name,
                                elimination_ratio=ratio,
                                run_id=run_id,
                                timeout=timeout
                            )
                            self.results.append(result)
        
        # Compute aggregate metrics
        summary = self._compute_summary()
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.2f}s")
        
        return {
            'summary': summary,
            'detailed_results': self.results,
            'config': self.config,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_datasets(self) -> Dict[str, Dict[str, OntologyKB]]:
        """Load all configured datasets."""
        datasets = {}
        
        for name, config in self.config.get('datasets', {}).items():
            if not config.get('enabled', True):
                continue
            
            path = Path(config.get('path', f'data/ontologies/{name}/'))
            
            if not path.exists():
                logger.warning(f"Dataset path not found: {path}")
                # Create dummy ontology for testing
                datasets[name] = {
                    'dummy.owl': self._create_dummy_ontology()
                }
                continue
            
            ontologies = {}
            for owl_file in path.glob('**/*.owl'):
                try:
                    kb = OntologyKB.from_owl(str(owl_file))
                    
                    # Check size constraints
                    max_ci = config.get('max_concept_inclusions', float('inf'))
                    if len(kb) <= max_ci:
                        ontologies[str(owl_file.name)] = kb
                        
                except Exception as e:
                    logger.warning(f"Failed to load {owl_file}: {e}")
            
            datasets[name] = ontologies
            logger.info(f"Loaded {len(ontologies)} ontologies from {name}")
        
        return datasets
    
    def _create_dummy_ontology(self) -> OntologyKB:
        """Create a dummy ontology for testing."""
        from src.core import (
            AtomicConcept, Role, ConceptInclusion,
            ExistentialRestriction, make_disjunction
        )
        
        # Create some dummy axioms
        axioms = set()
        
        concepts = [AtomicConcept(f"Concept_{i}") for i in range(20)]
        roles = [Role(f"role_{i}") for i in range(5)]
        
        for i in range(len(concepts) - 1):
            # Subsumption: Concept_i ⊑ Concept_{i+1}
            axioms.add(ConceptInclusion(concepts[i], concepts[i+1]))
            
            # Restrictions: Concept_i ⊑ ∃role_j.Concept_k
            j = i % len(roles)
            k = (i + 2) % len(concepts)
            restriction = ExistentialRestriction(roles[j], concepts[k])
            axioms.add(ConceptInclusion(concepts[i], restriction))
        
        return OntologyKB(axioms)
    
    def _get_enabled_methods(self) -> List[str]:
        """Get list of enabled methods."""
        methods = []
        for name, config in self.config.get('methods', {}).items():
            if config.get('enabled', True):
                methods.append(name)
        return methods
    
    def _run_single_evaluation(
        self,
        ontology: OntologyKB,
        ontology_name: str,
        dataset: str,
        method: str,
        elimination_ratio: float,
        run_id: int,
        timeout: int
    ) -> Dict[str, Any]:
        """Run a single evaluation."""
        # Determine forgetting vocabulary
        signature = ontology.signature
        num_to_forget = int(len(signature) * elimination_ratio)
        
        forgetting_vocab = set(random.sample(list(signature), num_to_forget))
        target_vocab = signature - forgetting_vocab
        
        # Run the method
        result = {
            'ontology': ontology_name,
            'dataset': dataset,
            'method': method,
            'elimination_ratio': elimination_ratio,
            'run_id': run_id,
            'signature_size': len(signature),
            'forgetting_size': len(forgetting_vocab),
            'target_size': len(target_vocab),
        }
        
        try:
            if method == 'nfmr':
                ret = self._run_nfmr(ontology, target_vocab, timeout)
            elif method in ('star', 'bot', 'top'):
                ret = self._run_locality(ontology, target_vocab, method)
            elif method == 'bm25':
                ret = self._run_bm25(ontology, target_vocab)
            else:
                ret = self._run_external(ontology, forgetting_vocab, method, timeout)
            
            result.update(ret)
            
        except Exception as e:
            logger.error(f"Error in {method}: {e}")
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _run_nfmr(
        self, 
        ontology: OntologyKB, 
        target_vocab: Set[str],
        timeout: int
    ) -> Dict[str, Any]:
        """Run NFMR method."""
        forgetting = NoiseFreeForgetting(ontology, timeout=timeout)
        result = forgetting.compute_module(target_vocab)
        
        return {
            'status': result.status.name,
            'success': result.is_success,
            'is_zero_noise': result.is_zero_noise,
            'module_size': len(result.module) if result.module else 0,
            'time_seconds': result.time_seconds,
            'memory_mb': result.memory_mb,
            'definers_introduced': result.num_definers_introduced,
            'definers_remaining': result.num_definers_remaining,
            'symbol_leakage': self._compute_leakage(result.module, target_vocab) if result.module else None,
        }
    
    def _run_locality(
        self,
        ontology: OntologyKB,
        target_vocab: Set[str],
        method: str
    ) -> Dict[str, Any]:
        """Run locality-based method (STAR, BOT, TOP)."""
        retriever = OntologyRetriever(ontology, method=method)
        start = time.time()
        result = retriever.retrieve(target_vocab)
        elapsed = time.time() - start
        
        return {
            'status': result.status.name,
            'success': result.is_success,
            'is_zero_noise': False,  # Locality methods leak symbols
            'module_size': len(result.module) if result.module else 0,
            'time_seconds': elapsed,
            'memory_mb': 0,
            'definers_introduced': 0,
            'definers_remaining': 0,
            'symbol_leakage': self._compute_leakage(result.module, target_vocab) if result.module else None,
        }
    
    def _run_bm25(
        self,
        ontology: OntologyKB,
        target_vocab: Set[str]
    ) -> Dict[str, Any]:
        """Run BM25 baseline."""
        retriever = OntologyRetriever(ontology, method='bm25')
        start = time.time()
        result = retriever.retrieve(target_vocab, top_k=100)
        elapsed = time.time() - start
        
        return {
            'status': result.status.name,
            'success': result.is_success,
            'is_zero_noise': False,
            'module_size': len(result.module) if result.module else 0,
            'time_seconds': elapsed,
            'memory_mb': 0,
            'definers_introduced': 0,
            'definers_remaining': 0,
            'symbol_leakage': self._compute_leakage(result.module, target_vocab) if result.module else None,
        }
    
    def _run_external(
        self,
        ontology: OntologyKB,
        forgetting_vocab: Set[str],
        method: str,
        timeout: int
    ) -> Dict[str, Any]:
        """Run external tool (LETHE, FAME)."""
        # Placeholder for external tool integration
        # In practice, this would save the ontology to a temp file,
        # call the external tool, and parse the results
        
        return {
            'status': 'NOT_IMPLEMENTED',
            'success': False,
            'is_zero_noise': False,
            'module_size': 0,
            'time_seconds': 0,
            'memory_mb': 0,
            'definers_introduced': 0,
            'definers_remaining': 0,
            'symbol_leakage': None,
            'note': f'External tool {method} integration not implemented'
        }
    
    def _compute_leakage(
        self, 
        module: OntologyKB, 
        target_vocab: Set[str]
    ) -> float:
        """Compute symbol leakage ratio."""
        if module is None:
            return 1.0
        
        module_sig = module.signature
        leaked = module_sig - target_vocab
        
        if len(module_sig) == 0:
            return 0.0
        
        return len(leaked) / len(module_sig)
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {}
        
        # Group by method
        by_method: Dict[str, List[Dict]] = {}
        for r in self.results:
            method = r['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(r)
        
        for method, results in by_method.items():
            successful = [r for r in results if r.get('success', False)]
            
            summary[method] = {
                'total_runs': len(results),
                'successful_runs': len(successful),
                'success_rate': len(successful) / len(results) if results else 0,
                'avg_time': sum(r.get('time_seconds', 0) for r in successful) / len(successful) if successful else 0,
                'avg_module_size': sum(r.get('module_size', 0) for r in successful) / len(successful) if successful else 0,
                'avg_leakage': sum(r.get('symbol_leakage', 0) or 0 for r in successful) / len(successful) if successful else 0,
                'zero_noise_rate': sum(1 for r in results if r.get('is_zero_noise', False)) / len(results) if results else 0,
            }
        
        return summary
    
    def save_results(self, output_path: str, results: Dict[str, Any]):
        """Save results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run NFMR benchmark evaluation"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/benchmark.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--methods', '-m',
        type=str,
        default=None,
        help='Comma-separated list of methods to run'
    )
    parser.add_argument(
        '--ratio', '-r',
        type=float,
        default=None,
        help='Single elimination ratio to test'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/benchmark_results.json',
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
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {
            'general': {'seed': 42},
            'datasets': {'test': {'enabled': True, 'path': 'data/ontologies/test/'}},
            'evaluation': {'elimination_ratios': [0.3], 'num_runs': 5, 'timeout_seconds': 60},
            'methods': {'nfmr': {'enabled': True}, 'star': {'enabled': True}}
        }
    
    # Override with command-line arguments
    if args.methods:
        methods = args.methods.split(',')
        for name in config.get('methods', {}):
            config['methods'][name]['enabled'] = name in methods
    
    if args.ratio:
        config['evaluation']['elimination_ratios'] = [args.ratio]
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()
    
    # Save results
    runner.save_results(args.output, results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for method, stats in results['summary'].items():
        print(f"\n{method.upper()}:")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Zero-noise rate: {stats['zero_noise_rate']*100:.1f}%")
        print(f"  Avg time: {stats['avg_time']:.3f}s")
        print(f"  Avg module size: {stats['avg_module_size']:.0f}")
        print(f"  Avg leakage: {stats['avg_leakage']*100:.1f}%")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
