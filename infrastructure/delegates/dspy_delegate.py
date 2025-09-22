#!/usr/bin/env python3
"""
DSPy Delegate - Infrastructure Layer
====================================

Delegate for DSPy workflow operations following hexagonal architecture.
Provides clean interface for adapters without exposing infrastructure details.
"""

from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DSPyDelegate:
    """
    Delegate for DSPy workflow operations.

    Encapsulates all DSPy infrastructure access.
    Adapters use this delegate instead of direct imports.
    """

    def __init__(self):
        """Initialize DSPy delegate with lazy loading."""
        self._dspy = None
        self._lm = None
        self._rm = None
        self._initialized = False
        self._compiled_programs = {}

    def _initialize(self):
        """Lazy initialization of DSPy."""
        if self._initialized:
            return True

        try:
            # Import only when needed (lazy loading)
            import dspy

            self._dspy = dspy

            # Configure DSPy with default LM
            self._lm = dspy.OpenAI(
                model='gpt-3.5-turbo',
                temperature=0.7,
                max_tokens=1500
            )

            dspy.settings.configure(lm=self._lm)

            self._initialized = True
            logger.info("DSPy delegate initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"DSPy not available: {e}")
            # Use fallback implementation
            self._initialized = self._initialize_fallback()
            return self._initialized

    def _initialize_fallback(self):
        """Initialize with fallback workflow engine."""
        try:
            self._dspy = FallbackDSPy()
            logger.info("Using fallback DSPy implementation")
            return True
        except Exception as e:
            logger.error(f"Fallback initialization failed: {e}")
            return False

    def create_signature(self, inputs: List[str], outputs: List[str]) -> Any:
        """
        Create a DSPy signature.

        Args:
            inputs: Input field names
            outputs: Output field names

        Returns:
            DSPy signature
        """
        if not self._initialize():
            return None

        try:
            if hasattr(self._dspy, 'Signature'):
                # Real DSPy
                sig_str = ", ".join(inputs) + " -> " + ", ".join(outputs)
                return self._dspy.Signature(sig_str)
            else:
                # Fallback
                return self._dspy.create_signature(inputs, outputs)

        except Exception as e:
            logger.error(f"Failed to create signature: {e}")
            return None

    def create_chain_of_thought(self, signature: Any) -> Any:
        """
        Create a Chain-of-Thought module.

        Args:
            signature: DSPy signature

        Returns:
            CoT module
        """
        if not self._initialize():
            return None

        try:
            if hasattr(self._dspy, 'ChainOfThought'):
                return self._dspy.ChainOfThought(signature)
            else:
                return self._dspy.create_cot(signature)

        except Exception as e:
            logger.error(f"Failed to create CoT: {e}")
            return None

    def create_program_of_thought(self, signature: Any) -> Any:
        """
        Create a Program-of-Thought module.

        Args:
            signature: DSPy signature

        Returns:
            PoT module
        """
        if not self._initialize():
            return None

        try:
            if hasattr(self._dspy, 'ProgramOfThought'):
                return self._dspy.ProgramOfThought(signature)
            else:
                # Fallback to CoT
                return self.create_chain_of_thought(signature)

        except Exception as e:
            logger.error(f"Failed to create PoT: {e}")
            return None

    def create_retrieve(self, k: int = 3) -> Any:
        """
        Create a Retrieve module.

        Args:
            k: Number of passages to retrieve

        Returns:
            Retrieve module
        """
        if not self._initialize():
            return None

        try:
            if hasattr(self._dspy, 'Retrieve'):
                return self._dspy.Retrieve(k=k)
            else:
                return self._dspy.create_retrieve(k)

        except Exception as e:
            logger.error(f"Failed to create Retrieve: {e}")
            return None

    def execute_workflow(self, workflow: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a DSPy workflow.

        Args:
            workflow: Workflow definition
            inputs: Input data

        Returns:
            Workflow results
        """
        if not self._initialize():
            return {
                'success': False,
                'error': 'DSPy not available',
                'results': {}
            }

        try:
            # Build workflow from definition
            steps = workflow.get('steps', [])
            current_data = inputs.copy()

            results = {
                'steps': [],
                'final_output': None
            }

            for step in steps:
                step_type = step.get('type')
                step_config = step.get('config', {})

                if step_type == 'chain_of_thought':
                    # Execute CoT step
                    sig = self.create_signature(
                        step_config.get('inputs', ['question']),
                        step_config.get('outputs', ['answer'])
                    )
                    cot = self.create_chain_of_thought(sig)

                    if cot:
                        step_result = cot(**current_data)
                        current_data.update(step_result)
                        results['steps'].append({
                            'type': step_type,
                            'result': step_result
                        })

                elif step_type == 'retrieve':
                    # Execute retrieval step
                    retrieve = self.create_retrieve(step_config.get('k', 3))
                    if retrieve:
                        passages = retrieve(current_data.get('query', ''))
                        current_data['passages'] = passages
                        results['steps'].append({
                            'type': step_type,
                            'result': {'passages': passages}
                        })

                elif step_type == 'custom':
                    # Execute custom step
                    func_name = step_config.get('function')
                    if func_name and hasattr(self, func_name):
                        func = getattr(self, func_name)
                        step_result = func(current_data, step_config)
                        current_data.update(step_result)
                        results['steps'].append({
                            'type': step_type,
                            'result': step_result
                        })

            results['final_output'] = current_data
            results['success'] = True

            return results

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': {}
            }

    def compile_program(self, program: Any, training_set: List[Dict[str, Any]],
                       metric: Callable = None) -> Any:
        """
        Compile/optimize a DSPy program.

        Args:
            program: DSPy program
            training_set: Training examples
            metric: Evaluation metric

        Returns:
            Compiled program
        """
        if not self._initialize():
            return program

        try:
            if hasattr(self._dspy, 'BootstrapFewShot'):
                from dspy.teleprompt import BootstrapFewShot

                # Use default metric if none provided
                if metric is None:
                    def default_metric(example, prediction):
                        return len(prediction.get('answer', '')) > 0

                    metric = default_metric

                # Compile with BootstrapFewShot
                teleprompter = BootstrapFewShot(
                    metric=metric,
                    max_bootstrapped_demos=4,
                    max_labeled_demos=4
                )

                compiled = teleprompter.compile(
                    program,
                    trainset=training_set
                )

                return compiled
            else:
                # Fallback - return original program
                return program

        except Exception as e:
            logger.error(f"Program compilation failed: {e}")
            return program

    def create_rag_module(self, passages_per_hop: int = 3, max_hops: int = 2) -> Any:
        """
        Create a multi-hop RAG module.

        Args:
            passages_per_hop: Passages to retrieve per hop
            max_hops: Maximum number of hops

        Returns:
            RAG module
        """
        if not self._initialize():
            return None

        try:
            if hasattr(self._dspy, 'Module'):
                # Create custom RAG module
                class MultiHopRAG(self._dspy.Module):
                    def __init__(self, passages_per_hop=3, max_hops=2):
                        super().__init__()
                        self.retrieve = self._dspy.Retrieve(k=passages_per_hop)
                        self.generate_query = self._dspy.ChainOfThought("context, question -> query")
                        self.generate_answer = self._dspy.ChainOfThought("context, question -> answer")
                        self.max_hops = max_hops

                    def forward(self, question):
                        context = []

                        for hop in range(self.max_hops):
                            # Generate search query
                            query = self.generate_query(context=context, question=question).query

                            # Retrieve passages
                            passages = self.retrieve(query).passages
                            context.extend(passages)

                        # Generate final answer
                        answer = self.generate_answer(context=context, question=question)

                        return self._dspy.Prediction(
                            context=context,
                            answer=answer.answer
                        )

                return MultiHopRAG(passages_per_hop, max_hops)
            else:
                # Fallback implementation
                return self._dspy.create_rag_module(passages_per_hop, max_hops)

        except Exception as e:
            logger.error(f"Failed to create RAG module: {e}")
            return None

    def save_program(self, program: Any, path: str) -> bool:
        """
        Save compiled program.

        Args:
            program: DSPy program
            path: Save path

        Returns:
            Success status
        """
        try:
            import pickle
            from pathlib import Path

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'wb') as f:
                pickle.dump(program, f)

            logger.info(f"Saved program to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save program: {e}")
            return False

    def load_program(self, path: str) -> Any:
        """
        Load compiled program.

        Args:
            path: Load path

        Returns:
            DSPy program or None
        """
        try:
            import pickle

            with open(path, 'rb') as f:
                program = pickle.load(f)

            logger.info(f"Loaded program from {path}")
            return program

        except Exception as e:
            logger.error(f"Failed to load program: {e}")
            return None

    def is_available(self) -> bool:
        """Check if DSPy is available."""
        return self._initialize()


class FallbackDSPy:
    """Fallback DSPy implementation for basic workflow operations."""

    def __init__(self):
        """Initialize fallback DSPy."""
        self.signatures = {}
        self.modules = {}

    def create_signature(self, inputs: List[str], outputs: List[str]) -> Dict[str, Any]:
        """Create a simple signature representation."""
        return {
            'inputs': inputs,
            'outputs': outputs,
            'type': 'signature'
        }

    def create_cot(self, signature: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple CoT representation."""
        return {
            'signature': signature,
            'type': 'chain_of_thought'
        }

    def create_retrieve(self, k: int) -> Dict[str, Any]:
        """Create a simple retrieval representation."""
        return {
            'k': k,
            'type': 'retrieve'
        }

    def create_rag_module(self, passages_per_hop: int, max_hops: int) -> Dict[str, Any]:
        """Create a simple RAG module representation."""
        return {
            'passages_per_hop': passages_per_hop,
            'max_hops': max_hops,
            'type': 'multi_hop_rag'
        }


class DSPyDelegateFactory:
    """Factory for creating DSPy delegates."""

    _instance = None

    @classmethod
    def get_delegate(cls) -> DSPyDelegate:
        """Get singleton DSPy delegate instance."""
        if cls._instance is None:
            cls._instance = DSPyDelegate()
        return cls._instance


def get_dspy_delegate() -> DSPyDelegate:
    """Get DSPy delegate instance."""
    return DSPyDelegateFactory.get_delegate()