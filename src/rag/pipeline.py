"""
NFMR: Noise-Free Module Retrieval
RAG Pipeline Module

This module implements a complete RAG (Retrieval-Augmented Generation) pipeline
that uses zero-noise module retrieval for knowledge-intensive question answering.
"""

from __future__ import annotations
from typing import Set, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import os

from ..core import (
    OntologyKB, NoiseFreeForgetting, RetrievalResult, RetrievalStatus
)

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    query: str
    answer: str
    context: str
    context_tokens: int
    retrieval_method: str
    retrieval_time_ms: float
    generation_time_ms: float
    confidence: float
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'answer': self.answer,
            'context': self.context,
            'context_tokens': self.context_tokens,
            'retrieval_method': self.retrieval_method,
            'retrieval_time_ms': self.retrieval_time_ms,
            'generation_time_ms': self.generation_time_ms,
            'confidence': self.confidence,
            'sources': self.sources,
            'metadata': self.metadata,
        }


class OntologyRetriever:
    """
    Retriever that extracts relevant modules from ontologies.
    
    Supports multiple retrieval methods:
    - nfmr: Zero-noise module via forgetting
    - star: STAR locality-based module
    - bot: BOT locality-based module
    - top: TOP locality-based module
    - bm25: BM25 keyword matching on axioms
    """
    
    def __init__(
        self,
        ontology: OntologyKB,
        method: str = "nfmr",
        timeout: int = 300
    ):
        """
        Initialize the retriever.
        
        Args:
            ontology: The source ontology
            method: Retrieval method ('nfmr', 'star', 'bot', 'top', 'bm25')
            timeout: Timeout for retrieval operations
        """
        self.ontology = ontology
        self.method = method
        self.timeout = timeout
        
        # Initialize method-specific components
        if method == "nfmr":
            self.forgetting = NoiseFreeForgetting(ontology, timeout=timeout)
        elif method == "bm25":
            self._init_bm25_index()
    
    def _init_bm25_index(self):
        """Initialize BM25 index over axioms."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 required for BM25 retrieval: pip install rank-bm25")
        
        self.axiom_texts = []
        self.axiom_map = {}
        
        for i, axiom in enumerate(self.ontology.axioms):
            text = str(axiom)
            tokens = text.lower().split()
            self.axiom_texts.append(tokens)
            self.axiom_map[i] = axiom
        
        self.bm25 = BM25Okapi(self.axiom_texts)
    
    def retrieve(
        self,
        target_concepts: Set[str],
        top_k: int = 50
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge for the given concepts.
        
        Args:
            target_concepts: Set of concept/role names to retrieve for
            top_k: Number of axioms for BM25 (ignored for module methods)
            
        Returns:
            RetrievalResult with the retrieved module
        """
        import time
        start = time.time()
        
        if self.method == "nfmr":
            return self.forgetting.compute_module(target_concepts)
        
        elif self.method == "bm25":
            return self._retrieve_bm25(target_concepts, top_k, start)
        
        elif self.method in ("star", "bot", "top"):
            return self._retrieve_locality(target_concepts, start)
        
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")
    
    def _retrieve_bm25(
        self,
        target_concepts: Set[str],
        top_k: int,
        start_time: float
    ) -> RetrievalResult:
        """Retrieve using BM25."""
        import time
        
        query = " ".join(target_concepts).lower().split()
        scores = self.bm25.get_scores(query)
        
        # Get top-k axioms
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        retrieved_axioms = {self.axiom_map[i] for i in top_indices}
        module = OntologyKB(retrieved_axioms)
        
        return RetrievalResult(
            status=RetrievalStatus.SUCCESS,
            module=module,
            time_seconds=time.time() - start_time,
            memory_mb=0,
            num_definers_introduced=0,
            num_definers_remaining=0,
            symbols_eliminated=set(),
            message=f"Retrieved {len(retrieved_axioms)} axioms via BM25"
        )
    
    def _retrieve_locality(
        self,
        target_concepts: Set[str],
        start_time: float
    ) -> RetrievalResult:
        """Retrieve using locality-based module extraction."""
        import time
        
        # Implement STAR/BOT/TOP locality-based extraction
        # This is a simplified version - full implementation would use OWL API
        
        module_axioms = set()
        frontier = target_concepts.copy()
        visited = set()
        
        while frontier:
            symbol = frontier.pop()
            if symbol in visited:
                continue
            visited.add(symbol)
            
            # Get all axioms containing this symbol
            for axiom in self.ontology.axioms:
                if axiom.contains_symbol(symbol):
                    module_axioms.add(axiom)
                    
                    # Add new symbols to frontier based on locality type
                    if self.method == "star":
                        # STAR: most permissive
                        new_symbols = axiom.signature - visited
                    elif self.method == "bot":
                        # BOT: only from LHS
                        new_symbols = axiom.lhs.signature - visited
                    else:  # top
                        # TOP: only from RHS
                        new_symbols = axiom.rhs.signature - visited
                    
                    frontier.update(new_symbols)
        
        module = OntologyKB(module_axioms)
        
        return RetrievalResult(
            status=RetrievalStatus.SUCCESS,
            module=module,
            time_seconds=time.time() - start_time,
            memory_mb=0,
            num_definers_introduced=0,
            num_definers_remaining=0,
            symbols_eliminated=set(),
            message=f"Retrieved {len(module_axioms)} axioms via {self.method.upper()}"
        )


class LLMGenerator:
    """
    Generator component that uses an LLM to produce answers.
    
    Supports multiple backends:
    - openai: OpenAI API (GPT-4, etc.)
    - huggingface: HuggingFace models (Llama, etc.)
    - anthropic: Anthropic API (Claude)
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        backend: str = "huggingface",
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        """
        Initialize the generator.
        
        Args:
            model_name: Name of the model
            backend: Backend to use ('openai', 'huggingface', 'anthropic')
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._model = None
        self._tokenizer = None
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai required: pip install openai")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self._client = OpenAI(api_key=api_key)
    
    def _init_huggingface(self):
        """Initialize HuggingFace model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers required: pip install transformers torch")
        
        token = os.environ.get("HUGGINGFACE_TOKEN")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=token
        )
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic required: pip install anthropic")
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self._client = anthropic.Anthropic(api_key=api_key)
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer given a query and context.
        
        Args:
            query: The user's question
            context: Retrieved knowledge context
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the "
                "provided ontology context. Use only the information in the context "
                "to answer. If the answer cannot be determined from the context, "
                "say so clearly."
            )
        
        if self.backend == "openai":
            return self._generate_openai(query, context, system_prompt)
        elif self.backend == "huggingface":
            return self._generate_huggingface(query, context, system_prompt)
        elif self.backend == "anthropic":
            return self._generate_anthropic(query, context, system_prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _generate_openai(self, query: str, context: str, system_prompt: str) -> str:
        """Generate using OpenAI API."""
        if self._client is None:
            self._init_openai()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def _generate_huggingface(self, query: str, context: str, system_prompt: str) -> str:
        """Generate using HuggingFace model."""
        if self._model is None:
            self._init_huggingface()
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
Context:
{context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id
        )
        
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        
        return response.strip()
    
    def _generate_anthropic(self, query: str, context: str, system_prompt: str) -> str:
        """Generate using Anthropic API."""
        if self._client is None:
            self._init_anthropic()
        
        message = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        
        return message.content[0].text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough estimate
            return len(text.split()) * 1.3


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    This pipeline:
    1. Extracts target concepts from the query
    2. Retrieves a relevant module from the ontology
    3. Serializes the module as context
    4. Generates an answer using the LLM
    """
    
    def __init__(
        self,
        ontology_path: str,
        llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        llm_backend: str = "huggingface",
        retrieval_method: str = "nfmr",
        timeout: int = 300
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            ontology_path: Path to ontology file or directory
            llm_model: LLM model name
            llm_backend: LLM backend ('openai', 'huggingface', 'anthropic')
            retrieval_method: Retrieval method ('nfmr', 'star', 'bm25')
            timeout: Retrieval timeout
        """
        self.ontology_path = ontology_path
        self.retrieval_method = retrieval_method
        
        # Load ontology
        logger.info(f"Loading ontology from {ontology_path}")
        self.ontology = OntologyKB.from_owl(ontology_path)
        
        # Initialize components
        self.retriever = OntologyRetriever(
            self.ontology,
            method=retrieval_method,
            timeout=timeout
        )
        
        self.generator = LLMGenerator(
            model_name=llm_model,
            backend=llm_backend
        )
    
    def query(
        self,
        question: str,
        target_concepts: Optional[Set[str]] = None
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: The user's question
            target_concepts: Optional explicit target concepts
            
        Returns:
            RAGResponse with the answer and metadata
        """
        import time
        
        # Extract concepts if not provided
        if target_concepts is None:
            target_concepts = self._extract_concepts(question)
        
        # Retrieve
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(target_concepts)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        
        if not retrieval_result.is_success:
            return RAGResponse(
                query=question,
                answer=f"Retrieval failed: {retrieval_result.message}",
                context="",
                context_tokens=0,
                retrieval_method=self.retrieval_method,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0,
                confidence=0.0,
                metadata={"retrieval_status": retrieval_result.status.name}
            )
        
        # Serialize context
        context = self._serialize_module(retrieval_result.module)
        context_tokens = self.generator.count_tokens(context)
        
        # Generate
        gen_start = time.time()
        answer = self.generator.generate(question, context)
        gen_time_ms = (time.time() - gen_start) * 1000
        
        return RAGResponse(
            query=question,
            answer=answer,
            context=context,
            context_tokens=context_tokens,
            retrieval_method=self.retrieval_method,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=gen_time_ms,
            confidence=self._estimate_confidence(answer),
            sources=list(target_concepts),
            metadata={
                "retrieval_status": retrieval_result.status.name,
                "module_size": len(retrieval_result.module) if retrieval_result.module else 0,
                "is_zero_noise": retrieval_result.is_zero_noise
            }
        )
    
    def _extract_concepts(self, question: str) -> Set[str]:
        """Extract concept names from a question."""
        # Simple extraction: find words that match concept names
        words = set(question.replace("?", "").replace(",", "").split())
        concepts = words & self.ontology.signature
        
        # If no direct matches, try case-insensitive
        if not concepts:
            lower_sig = {s.lower(): s for s in self.ontology.signature}
            for word in words:
                if word.lower() in lower_sig:
                    concepts.add(lower_sig[word.lower()])
        
        return concepts if concepts else self.ontology.concept_names
    
    def _serialize_module(self, module: OntologyKB) -> str:
        """Serialize a module as text for the LLM context."""
        lines = []
        for axiom in sorted(module.axioms, key=str):
            # Convert to Manchester Syntax-like format
            lines.append(f"{axiom.lhs} SubClassOf {axiom.rhs}")
        return "\n".join(lines)
    
    def _estimate_confidence(self, answer: str) -> float:
        """Estimate confidence in the answer."""
        # Simple heuristic based on hedging language
        hedges = ["might", "may", "possibly", "unclear", "cannot determine", 
                  "not sure", "uncertain"]
        
        lower_answer = answer.lower()
        hedge_count = sum(1 for h in hedges if h in lower_answer)
        
        if hedge_count == 0:
            return 0.9
        elif hedge_count == 1:
            return 0.7
        else:
            return 0.5
