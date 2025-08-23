"""
Temporal Logic Synthesis Engine

Advanced temporal logic property synthesis for hardware verification
using LLM-guided pattern recognition and formal specification generation.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from enum import Enum
import re
from pathlib import Path

from ..core import CircuitVerifier
from ..parsers import CircuitAST, Module, Signal
from ..llm.llm_client import LLMManager
from ..monitoring.logger import get_logger


class TemporalOperator(Enum):
    """Temporal logic operators."""
    ALWAYS = "G"  # Globally
    EVENTUALLY = "F"  # Finally 
    NEXT = "X"  # Next
    UNTIL = "U"  # Until
    RELEASE = "R"  # Release
    WEAK_UNTIL = "W"  # Weak Until


class PropertyType(Enum):
    """Types of temporal properties."""
    SAFETY = "safety"
    LIVENESS = "liveness"
    FAIRNESS = "fairness"
    REACHABILITY = "reachability"
    INVARIANT = "invariant"
    RESPONSE = "response"


@dataclass
class TemporalProperty:
    """Represents a temporal logic property."""
    property_id: str
    name: str
    description: str
    ltl_formula: str
    psl_formula: str
    sva_assertion: str
    property_type: PropertyType
    applicable_signals: List[str]
    confidence_score: float
    synthesis_method: str
    validation_status: str = "pending"


@dataclass
class PropertyPattern:
    """Template for common temporal property patterns."""
    pattern_id: str
    name: str
    description: str
    ltl_template: str
    parameters: List[str]
    examples: List[str]
    complexity_level: int


class TemporalLogicSynthesis:
    """
    Advanced temporal logic property synthesis engine that automatically
    generates formal properties from circuit structure and behavioral analysis.
    """

    def __init__(self, verifier: CircuitVerifier):
        self.verifier = verifier
        self.llm_manager = LLMManager.create_default()
        self.logger = get_logger("temporal_logic_synthesis")
        
        # Property pattern library
        self.pattern_library = self._initialize_pattern_library()
        
        # Synthesis configuration
        self.max_property_complexity = 5
        self.min_confidence_threshold = 0.7
        
        self.logger.info("Temporal Logic Synthesis Engine initialized")

    def _initialize_pattern_library(self) -> Dict[str, PropertyPattern]:
        """Initialize library of common temporal property patterns."""
        patterns = {
            "safety_mutual_exclusion": PropertyPattern(
                pattern_id="safety_mutex",
                name="Mutual Exclusion Safety",
                description="Two signals cannot be active simultaneously",
                ltl_template="G(!(signal1 & signal2))",
                parameters=["signal1", "signal2"],
                examples=["G(!(grant1 & grant2))", "G(!(read & write))"],
                complexity_level=1
            ),
            
            "liveness_eventual_response": PropertyPattern(
                pattern_id="liveness_response",
                name="Eventual Response",
                description="Request eventually leads to response",
                ltl_template="G(request -> F(response))",
                parameters=["request", "response"],
                examples=["G(req -> F(ack))", "G(start -> F(done))"],
                complexity_level=2
            ),
            
            "safety_state_invariant": PropertyPattern(
                pattern_id="safety_invariant",
                name="State Invariant",
                description="Property that must always hold in valid states",
                ltl_template="G(valid_state -> condition)",
                parameters=["valid_state", "condition"],
                examples=["G(enabled -> counter <= MAX)", "G(active -> data_valid)"],
                complexity_level=2
            ),
            
            "liveness_progress": PropertyPattern(
                pattern_id="liveness_progress",
                name="Progress Guarantee",
                description="System makes progress from any state",
                ltl_template="G(F(progress_condition))",
                parameters=["progress_condition"],
                examples=["G(F(state_change))", "G(F(counter_increment))"],
                complexity_level=3
            ),
            
            "fairness_resource_access": PropertyPattern(
                pattern_id="fairness_access",
                name="Fair Resource Access",
                description="Fair access to shared resources",
                ltl_template="G(F(resource_grant1)) & G(F(resource_grant2))",
                parameters=["resource_grant1", "resource_grant2"],
                examples=["G(F(cpu1_grant)) & G(F(cpu2_grant))"],
                complexity_level=4
            ),
            
            "response_bounded_delay": PropertyPattern(
                pattern_id="response_bounded",
                name="Bounded Response Time",
                description="Response occurs within bounded time",
                ltl_template="G(request -> F[0:N](response))",
                parameters=["request", "response", "N"],
                examples=["G(interrupt -> F[0:10](service))"],
                complexity_level=4
            ),
            
            "reachability_state": PropertyPattern(
                pattern_id="reachability_state",
                name="State Reachability",
                description="Specific state is reachable",
                ltl_template="F(target_state)",
                parameters=["target_state"],
                examples=["F(error_state)", "F(final_state)"],
                complexity_level=1
            ),
            
            "stabilization": PropertyPattern(
                pattern_id="stabilization",
                name="System Stabilization",
                description="System eventually stabilizes to valid state",
                ltl_template="F(G(stable_condition))",
                parameters=["stable_condition"],
                examples=["F(G(!changing))", "F(G(converged))"],
                complexity_level=3
            )
        }
        
        return patterns

    async def synthesize_comprehensive_properties(
        self, 
        circuit_ast: CircuitAST,
        behavioral_hints: Optional[Dict[str, Any]] = None
    ) -> List[TemporalProperty]:
        """
        Synthesize comprehensive temporal properties for a circuit.
        
        Args:
            circuit_ast: Parsed circuit AST
            behavioral_hints: Optional behavioral analysis hints
            
        Returns:
            List of synthesized temporal properties
        """
        self.logger.info(f"Starting temporal property synthesis for {len(circuit_ast.modules)} modules")
        
        all_properties = []
        
        for module in circuit_ast.modules:
            self.logger.info(f"Synthesizing properties for module: {module.name}")
            
            # Phase 1: Pattern-based synthesis
            pattern_properties = await self._synthesize_pattern_based_properties(module)
            all_properties.extend(pattern_properties)
            
            # Phase 2: Structure-guided synthesis
            structure_properties = await self._synthesize_structure_guided_properties(module)
            all_properties.extend(structure_properties)
            
            # Phase 3: LLM-guided synthesis
            llm_properties = await self._synthesize_llm_guided_properties(
                module, behavioral_hints
            )
            all_properties.extend(llm_properties)
            
            # Phase 4: Cross-signal relationship synthesis
            relationship_properties = await self._synthesize_relationship_properties(module)
            all_properties.extend(relationship_properties)
        
        # Phase 5: Validation and ranking
        validated_properties = await self._validate_and_rank_properties(
            all_properties, circuit_ast
        )
        
        self.logger.info(f"Synthesized {len(validated_properties)} validated temporal properties")
        
        return validated_properties

    async def _synthesize_pattern_based_properties(
        self, module: Module
    ) -> List[TemporalProperty]:
        """Synthesize properties using common temporal patterns."""
        properties = []
        
        # Extract signals by category - using actual AST structure
        input_signals = [s.name for s in module.ports if s.direction == "input"]
        output_signals = [s.name for s in module.ports if s.direction == "output"]
        reg_signals = [s.name for s in module.signals if hasattr(s, 'signal_type') and s.signal_type == "reg"]
        
        # Identify control signals
        control_signals = self._identify_control_signals(module.signals + module.ports)
        data_signals = self._identify_data_signals(module.signals + module.ports)
        
        # Apply patterns based on signal analysis
        
        # 1. Mutual exclusion for control signals
        if len(control_signals) >= 2:
            for i, sig1 in enumerate(control_signals):
                for sig2 in control_signals[i+1:]:
                    if self._are_mutually_exclusive(sig1, sig2, module):
                        prop = self._apply_pattern(
                            "safety_mutual_exclusion", 
                            {"signal1": sig1, "signal2": sig2},
                            module.name
                        )
                        properties.append(prop)
        
        # 2. Request-response patterns
        req_resp_pairs = self._identify_request_response_pairs(module.signals)
        for req, resp in req_resp_pairs:
            prop = self._apply_pattern(
                "liveness_eventual_response",
                {"request": req, "response": resp},
                module.name
            )
            properties.append(prop)
        
        # 3. State invariants for registers
        for reg_signal in reg_signals:
            if self._has_bounded_range(reg_signal, module):
                bounds = self._get_signal_bounds(reg_signal, module)
                prop = self._create_bounds_invariant(reg_signal, bounds, module.name)
                properties.append(prop)
        
        # 4. Progress properties for state machines
        if self._is_state_machine(module):
            state_signal = self._get_state_signal(module)
            if state_signal:
                prop = self._apply_pattern(
                    "liveness_progress",
                    {"progress_condition": f"{state_signal}_changing"},
                    module.name
                )
                properties.append(prop)
        
        return properties

    async def _synthesize_structure_guided_properties(
        self, module: ModuleNode
    ) -> List[TemporalProperty]:
        """Synthesize properties based on circuit structure analysis."""
        properties = []
        
        # Analyze circuit structure
        structure_analysis = await self._analyze_circuit_structure(module)
        
        # Memory access properties
        if structure_analysis.get("has_memory", False):
            memory_props = await self._synthesize_memory_properties(module, structure_analysis)
            properties.extend(memory_props)
        
        # Pipeline properties
        if structure_analysis.get("is_pipeline", False):
            pipeline_props = await self._synthesize_pipeline_properties(module, structure_analysis)
            properties.extend(pipeline_props)
        
        # Clock domain properties
        if structure_analysis.get("multiple_clocks", False):
            clock_props = await self._synthesize_clock_domain_properties(module, structure_analysis)
            properties.extend(clock_props)
        
        # Bus protocol properties
        if structure_analysis.get("has_bus_interface", False):
            bus_props = await self._synthesize_bus_protocol_properties(module, structure_analysis)
            properties.extend(bus_props)
        
        return properties

    async def _synthesize_llm_guided_properties(
        self, 
        module: ModuleNode,
        behavioral_hints: Optional[Dict[str, Any]]
    ) -> List[TemporalProperty]:
        """Use LLM to synthesize domain-specific properties."""
        properties = []
        
        # Create context for LLM
        module_context = self._create_module_context(module, behavioral_hints)
        
        synthesis_prompt = f"""
        Analyze this hardware module and synthesize temporal logic properties:
        
        Module Context:
        {module_context}
        
        Generate 5-10 important temporal properties that should be verified.
        For each property, provide:
        1. Name and description
        2. LTL formula
        3. Property type (safety/liveness/fairness/etc.)
        4. Confidence score (0.0-1.0)
        
        Focus on:
        - Critical safety properties
        - Essential liveness guarantees
        - Performance requirements
        - Protocol compliance
        - Error handling behavior
        
        Format as JSON array.
        """
        
        try:
            response = await self.llm_manager.generate(
                synthesis_prompt, temperature=0.3, max_tokens=2000
            )
            
            # Parse LLM response
            llm_properties = await self._parse_llm_property_response(
                response.content, module.name
            )
            properties.extend(llm_properties)
            
        except Exception as e:
            self.logger.warning(f"LLM-guided synthesis failed: {e}")
        
        return properties

    async def _synthesize_relationship_properties(
        self, module: ModuleNode
    ) -> List[TemporalProperty]:
        """Synthesize properties based on cross-signal relationships."""
        properties = []
        
        # Analyze signal dependencies
        dependencies = self._analyze_signal_dependencies(module)
        
        # Data flow properties
        for producer, consumers in dependencies.items():
            for consumer in consumers:
                if self._is_critical_data_flow(producer, consumer, module):
                    prop = self._create_data_flow_property(
                        producer, consumer, module.name
                    )
                    properties.append(prop)
        
        # Timing relationship properties
        timing_constraints = self._extract_timing_constraints(module)
        for constraint in timing_constraints:
            prop = self._create_timing_property(constraint, module.name)
            properties.append(prop)
        
        return properties

    def _apply_pattern(
        self, 
        pattern_id: str, 
        parameters: Dict[str, str], 
        module_name: str
    ) -> TemporalProperty:
        """Apply a temporal pattern with specific parameters."""
        pattern = self.pattern_library[pattern_id]
        
        # Substitute parameters in template
        ltl_formula = pattern.ltl_template
        for param, value in parameters.items():
            ltl_formula = ltl_formula.replace(param, value)
        
        # Generate PSL and SVA equivalents
        psl_formula = self._ltl_to_psl(ltl_formula)
        sva_assertion = self._ltl_to_sva(ltl_formula)
        
        # Determine property type
        prop_type = self._infer_property_type(ltl_formula)
        
        return TemporalProperty(
            property_id=str(uuid.uuid4()),
            name=f"{pattern.name} - {module_name}",
            description=f"{pattern.description} in module {module_name}",
            ltl_formula=ltl_formula,
            psl_formula=psl_formula,
            sva_assertion=sva_assertion,
            property_type=prop_type,
            applicable_signals=list(parameters.values()),
            confidence_score=0.8,  # Pattern-based properties have high confidence
            synthesis_method="pattern_based"
        )

    def _identify_control_signals(self, signals: List[Signal]) -> List[str]:
        """Identify control signals based on naming patterns."""
        control_patterns = [
            r'.*enable.*', r'.*en$', r'.*valid.*', r'.*ready.*',
            r'.*start.*', r'.*stop.*', r'.*reset.*', r'.*rst.*',
            r'.*grant.*', r'.*req.*', r'.*ack.*', r'.*sel.*'
        ]
        
        control_signals = []
        for signal in signals:
            signal_name_lower = signal.name.lower()
            if any(re.match(pattern, signal_name_lower) for pattern in control_patterns):
                control_signals.append(signal.name)
        
        return control_signals

    def _identify_data_signals(self, signals: List[SignalNode]) -> List[str]:
        """Identify data signals based on naming patterns and width."""
        data_signals = []
        
        for signal in signals:
            # Wide signals are likely data
            if signal.width and signal.width > 1:
                data_signals.append(signal.name)
            
            # Common data signal names
            signal_name_lower = signal.name.lower()
            if any(keyword in signal_name_lower for keyword in 
                   ['data', 'addr', 'count', 'value', 'reg']):
                data_signals.append(signal.name)
        
        return data_signals

    def _identify_request_response_pairs(self, signals: List[SignalNode]) -> List[Tuple[str, str]]:
        """Identify request-response signal pairs."""
        pairs = []
        signal_names = [s.name.lower() for s in signals]
        
        # Common request-response patterns
        patterns = [
            ('req', 'ack'), ('request', 'response'), ('start', 'done'),
            ('enable', 'valid'), ('trigger', 'complete'), ('send', 'recv')
        ]
        
        for req_pattern, resp_pattern in patterns:
            req_signals = [s.name for s in signals if req_pattern in s.name.lower()]
            resp_signals = [s.name for s in signals if resp_pattern in s.name.lower()]
            
            # Match signals with similar base names
            for req in req_signals:
                for resp in resp_signals:
                    if self._signals_are_paired(req, resp):
                        pairs.append((req, resp))
        
        return pairs

    def _signals_are_paired(self, sig1: str, sig2: str) -> bool:
        """Check if two signals form a logical pair."""
        # Remove common prefixes/suffixes and compare base names
        base1 = re.sub(r'_(req|start|en|trigger)$', '', sig1.lower())
        base2 = re.sub(r'_(ack|done|valid|complete)$', '', sig2.lower())
        
        return base1 == base2

    def _are_mutually_exclusive(self, sig1: str, sig2: str, module: ModuleNode) -> bool:
        """Determine if two signals should be mutually exclusive."""
        # Heuristics based on naming patterns
        mutex_patterns = [
            (r'.*read.*', r'.*write.*'),
            (r'.*grant1.*', r'.*grant2.*'),
            (r'.*sel0.*', r'.*sel1.*'),
            (r'.*mode0.*', r'.*mode1.*')
        ]
        
        sig1_lower = sig1.lower()
        sig2_lower = sig2.lower()
        
        for pattern1, pattern2 in mutex_patterns:
            if (re.match(pattern1, sig1_lower) and re.match(pattern2, sig2_lower)) or \
               (re.match(pattern2, sig1_lower) and re.match(pattern1, sig2_lower)):
                return True
        
        return False

    def _has_bounded_range(self, signal: str, module: ModuleNode) -> bool:
        """Check if a signal has a bounded range that should be enforced."""
        # Find the signal in module
        for sig in module.signals:
            if sig.name == signal:
                # Multi-bit signals likely have meaningful bounds
                return sig.width and sig.width > 1
        return False

    def _get_signal_bounds(self, signal: str, module: ModuleNode) -> Dict[str, int]:
        """Get the bounds for a signal."""
        for sig in module.signals:
            if sig.name == signal and sig.width:
                return {
                    "min": 0,
                    "max": (2 ** sig.width) - 1
                }
        return {"min": 0, "max": 1}

    def _create_bounds_invariant(
        self, signal: str, bounds: Dict[str, int], module_name: str
    ) -> TemporalProperty:
        """Create a bounds invariant property for a signal."""
        ltl_formula = f"G({signal} >= {bounds['min']} & {signal} <= {bounds['max']})"
        psl_formula = self._ltl_to_psl(ltl_formula)
        sva_assertion = self._ltl_to_sva(ltl_formula)
        
        return TemporalProperty(
            property_id=str(uuid.uuid4()),
            name=f"Bounds Invariant - {signal}",
            description=f"Signal {signal} must stay within valid bounds",
            ltl_formula=ltl_formula,
            psl_formula=psl_formula,
            sva_assertion=sva_assertion,
            property_type=PropertyType.SAFETY,
            applicable_signals=[signal],
            confidence_score=0.9,
            synthesis_method="structure_guided"
        )

    def _is_state_machine(self, module: ModuleNode) -> bool:
        """Detect if module implements a state machine."""
        # Look for state-related signals
        state_indicators = ['state', 'current_state', 'next_state', 'fsm']
        
        for signal in module.signals:
            if any(indicator in signal.name.lower() for indicator in state_indicators):
                return True
        
        # Look for case statements or state transitions in assignments
        for assignment in module.assignments:
            if 'case' in assignment.get('type', '').lower():
                return True
        
        return False

    def _get_state_signal(self, module: ModuleNode) -> Optional[str]:
        """Get the main state signal for a state machine."""
        state_candidates = []
        
        for signal in module.signals:
            signal_name_lower = signal.name.lower()
            if 'state' in signal_name_lower:
                state_candidates.append(signal.name)
        
        # Prefer signals with 'current' or 'present'
        for candidate in state_candidates:
            if 'current' in candidate.lower() or 'present' in candidate.lower():
                return candidate
        
        # Return first state signal found
        return state_candidates[0] if state_candidates else None

    async def _analyze_circuit_structure(self, module: ModuleNode) -> Dict[str, Any]:
        """Analyze circuit structure to guide property synthesis."""
        analysis = {
            "has_memory": False,
            "is_pipeline": False,
            "multiple_clocks": False,
            "has_bus_interface": False,
            "complexity_score": 0.0
        }
        
        # Check for memory structures
        memory_indicators = ['mem', 'ram', 'rom', 'cache', 'buffer', 'fifo']
        for signal in module.signals:
            if any(indicator in signal.name.lower() for indicator in memory_indicators):
                analysis["has_memory"] = True
                break
        
        # Check for pipeline indicators
        pipeline_indicators = ['stage', 'pipe', 'valid', 'bubble']
        pipeline_count = sum(1 for signal in module.signals 
                           if any(indicator in signal.name.lower() for indicator in pipeline_indicators))
        analysis["is_pipeline"] = pipeline_count >= 3
        
        # Check for multiple clocks
        clock_signals = [s for s in module.signals if 'clk' in s.name.lower() or 'clock' in s.name.lower()]
        analysis["multiple_clocks"] = len(clock_signals) > 1
        
        # Check for bus interface
        bus_indicators = ['addr', 'data', 'strobe', 'cyc', 'ack', 'we', 'sel']
        bus_count = sum(1 for signal in module.signals 
                       if any(indicator in signal.name.lower() for indicator in bus_indicators))
        analysis["has_bus_interface"] = bus_count >= 3
        
        # Calculate complexity
        analysis["complexity_score"] = (
            len(module.signals) * 0.1 + 
            len(module.assignments) * 0.2 +
            (1.0 if analysis["has_memory"] else 0) +
            (1.0 if analysis["is_pipeline"] else 0) +
            (0.5 if analysis["multiple_clocks"] else 0)
        )
        
        return analysis

    async def _synthesize_memory_properties(
        self, module: ModuleNode, structure_analysis: Dict[str, Any]
    ) -> List[TemporalProperty]:
        """Synthesize properties specific to memory structures."""
        properties = []
        
        # Memory coherence properties
        memory_signals = self._identify_memory_signals(module.signals)
        
        for mem_group in memory_signals:
            if 'write_enable' in mem_group and 'read_enable' in mem_group:
                # Write-read exclusion
                prop = TemporalProperty(
                    property_id=str(uuid.uuid4()),
                    name="Memory Access Exclusion",
                    description="Memory read and write cannot occur simultaneously",
                    ltl_formula=f"G(!({mem_group['write_enable']} & {mem_group['read_enable']}))",
                    psl_formula="",
                    sva_assertion="",
                    property_type=PropertyType.SAFETY,
                    applicable_signals=[mem_group['write_enable'], mem_group['read_enable']],
                    confidence_score=0.9,
                    synthesis_method="memory_specific"
                )
                properties.append(prop)
        
        return properties

    async def _synthesize_pipeline_properties(
        self, module: ModuleNode, structure_analysis: Dict[str, Any]
    ) -> List[TemporalProperty]:
        """Synthesize properties specific to pipeline structures."""
        properties = []
        
        # Pipeline stage progression
        stage_signals = self._identify_pipeline_stages(module.signals)
        
        for i in range(len(stage_signals) - 1):
            current_stage = stage_signals[i]
            next_stage = stage_signals[i + 1]
            
            # Stage progression property
            prop = TemporalProperty(
                property_id=str(uuid.uuid4()),
                name=f"Pipeline Stage {i} Progression",
                description=f"Valid data in stage {i} eventually progresses to stage {i+1}",
                ltl_formula=f"G({current_stage}_valid -> F({next_stage}_valid))",
                psl_formula="",
                sva_assertion="",
                property_type=PropertyType.LIVENESS,
                applicable_signals=[f"{current_stage}_valid", f"{next_stage}_valid"],
                confidence_score=0.8,
                synthesis_method="pipeline_specific"
            )
            properties.append(prop)
        
        return properties

    async def _synthesize_clock_domain_properties(
        self, module: ModuleNode, structure_analysis: Dict[str, Any]
    ) -> List[TemporalProperty]:
        """Synthesize properties for multiple clock domains."""
        properties = []
        
        # Clock domain crossing properties
        # This is a simplified implementation
        clock_signals = [s.name for s in module.signals if 'clk' in s.name.lower()]
        
        if len(clock_signals) >= 2:
            prop = TemporalProperty(
                property_id=str(uuid.uuid4()),
                name="Clock Domain Synchronization",
                description="Data crossing clock domains must be properly synchronized",
                ltl_formula="G(cross_domain_valid -> X(sync_complete))",
                psl_formula="",
                sva_assertion="",
                property_type=PropertyType.SAFETY,
                applicable_signals=clock_signals,
                confidence_score=0.7,
                synthesis_method="clock_domain_specific"
            )
            properties.append(prop)
        
        return properties

    async def _synthesize_bus_protocol_properties(
        self, module: ModuleNode, structure_analysis: Dict[str, Any]
    ) -> List[TemporalProperty]:
        """Synthesize properties for bus protocol compliance."""
        properties = []
        
        # Wishbone-style bus properties
        bus_signals = self._identify_bus_signals(module.signals)
        
        if bus_signals:
            # Bus handshake property
            prop = TemporalProperty(
                property_id=str(uuid.uuid4()),
                name="Bus Handshake Protocol",
                description="Bus requests must be acknowledged",
                ltl_formula=f"G({bus_signals.get('strobe', 'stb')} -> F({bus_signals.get('ack', 'ack')}))",
                psl_formula="",
                sva_assertion="",
                property_type=PropertyType.LIVENESS,
                applicable_signals=list(bus_signals.values()),
                confidence_score=0.85,
                synthesis_method="bus_protocol_specific"
            )
            properties.append(prop)
        
        return properties

    def _identify_memory_signals(self, signals: List[SignalNode]) -> List[Dict[str, str]]:
        """Identify memory-related signal groups."""
        memory_groups = []
        # Simplified implementation
        signal_names = [s.name for s in signals]
        
        # Look for common memory signal patterns
        if any('we' in name.lower() or 'write_enable' in name.lower() for name in signal_names):
            group = {}
            for name in signal_names:
                name_lower = name.lower()
                if 'we' in name_lower or 'write_enable' in name_lower:
                    group['write_enable'] = name
                elif 're' in name_lower or 'read_enable' in name_lower:
                    group['read_enable'] = name
            
            if len(group) >= 2:
                memory_groups.append(group)
        
        return memory_groups

    def _identify_pipeline_stages(self, signals: List[SignalNode]) -> List[str]:
        """Identify pipeline stage signals."""
        stages = []
        
        for signal in signals:
            name_lower = signal.name.lower()
            if 'stage' in name_lower:
                stages.append(signal.name.replace('_valid', '').replace('_stage', ''))
        
        return sorted(list(set(stages)))

    def _identify_bus_signals(self, signals: List[SignalNode]) -> Dict[str, str]:
        """Identify bus protocol signals."""
        bus_signals = {}
        
        for signal in signals:
            name_lower = signal.name.lower()
            if 'stb' in name_lower or 'strobe' in name_lower:
                bus_signals['strobe'] = signal.name
            elif 'ack' in name_lower:
                bus_signals['ack'] = signal.name
            elif 'cyc' in name_lower:
                bus_signals['cycle'] = signal.name
        
        return bus_signals

    def _create_module_context(
        self, module: ModuleNode, behavioral_hints: Optional[Dict[str, Any]]
    ) -> str:
        """Create context description for LLM analysis."""
        context = f"""
        Module Name: {module.name}
        
        Signals:
        """
        
        for signal in module.signals[:20]:  # Limit for context size
            context += f"  - {signal.name} ({signal.signal_type}"
            if signal.width:
                context += f", width: {signal.width}"
            context += ")\n"
        
        if behavioral_hints:
            context += "\nBehavioral Hints:\n"
            for key, value in behavioral_hints.items():
                context += f"  - {key}: {value}\n"
        
        return context

    async def _parse_llm_property_response(
        self, response: str, module_name: str
    ) -> List[TemporalProperty]:
        """Parse LLM response into TemporalProperty objects."""
        properties = []
        
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                property_data = json.loads(json_str)
                
                for prop_dict in property_data:
                    prop = TemporalProperty(
                        property_id=str(uuid.uuid4()),
                        name=prop_dict.get('name', 'LLM Generated Property'),
                        description=prop_dict.get('description', ''),
                        ltl_formula=prop_dict.get('ltl_formula', ''),
                        psl_formula=self._ltl_to_psl(prop_dict.get('ltl_formula', '')),
                        sva_assertion=self._ltl_to_sva(prop_dict.get('ltl_formula', '')),
                        property_type=PropertyType(prop_dict.get('property_type', 'safety')),
                        applicable_signals=prop_dict.get('applicable_signals', []),
                        confidence_score=float(prop_dict.get('confidence_score', 0.7)),
                        synthesis_method="llm_guided"
                    )
                    properties.append(prop)
                    
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM property response: {e}")
        
        return properties

    def _analyze_signal_dependencies(self, module: ModuleNode) -> Dict[str, List[str]]:
        """Analyze signal dependencies in the module."""
        dependencies = {}
        
        # Simplified dependency analysis based on assignments
        for assignment in module.assignments:
            target = assignment.get('target', '')
            sources = assignment.get('sources', [])
            
            for source in sources:
                if source not in dependencies:
                    dependencies[source] = []
                if target not in dependencies[source]:
                    dependencies[source].append(target)
        
        return dependencies

    def _is_critical_data_flow(self, producer: str, consumer: str, module: ModuleNode) -> bool:
        """Determine if a data flow relationship is critical for verification."""
        # Heuristics for critical data flows
        critical_patterns = ['data', 'addr', 'control', 'state', 'count']
        
        producer_lower = producer.lower()
        consumer_lower = consumer.lower()
        
        return any(pattern in producer_lower or pattern in consumer_lower 
                  for pattern in critical_patterns)

    def _create_data_flow_property(
        self, producer: str, consumer: str, module_name: str
    ) -> TemporalProperty:
        """Create a data flow property."""
        ltl_formula = f"G({producer}_valid -> F({consumer}_ready))"
        
        return TemporalProperty(
            property_id=str(uuid.uuid4()),
            name=f"Data Flow: {producer} -> {consumer}",
            description=f"Data from {producer} flows correctly to {consumer}",
            ltl_formula=ltl_formula,
            psl_formula=self._ltl_to_psl(ltl_formula),
            sva_assertion=self._ltl_to_sva(ltl_formula),
            property_type=PropertyType.LIVENESS,
            applicable_signals=[producer, consumer],
            confidence_score=0.75,
            synthesis_method="data_flow_analysis"
        )

    def _extract_timing_constraints(self, module: ModuleNode) -> List[Dict[str, Any]]:
        """Extract timing constraints from module structure."""
        constraints = []
        
        # Look for delay-related signals
        for signal in module.signals:
            if 'delay' in signal.name.lower() or 'timeout' in signal.name.lower():
                constraints.append({
                    'type': 'timeout',
                    'signal': signal.name,
                    'description': f"Timeout constraint for {signal.name}"
                })
        
        return constraints

    def _create_timing_property(
        self, constraint: Dict[str, Any], module_name: str
    ) -> TemporalProperty:
        """Create a timing-related property."""
        signal = constraint['signal']
        ltl_formula = f"G({signal}_start -> F[0:MAX_DELAY]({signal}_end))"
        
        return TemporalProperty(
            property_id=str(uuid.uuid4()),
            name=f"Timing Constraint: {signal}",
            description=constraint['description'],
            ltl_formula=ltl_formula,
            psl_formula=self._ltl_to_psl(ltl_formula),
            sva_assertion=self._ltl_to_sva(ltl_formula),
            property_type=PropertyType.SAFETY,
            applicable_signals=[signal],
            confidence_score=0.8,
            synthesis_method="timing_analysis"
        )

    def _ltl_to_psl(self, ltl_formula: str) -> str:
        """Convert LTL formula to PSL (Property Specification Language)."""
        # Simplified conversion
        psl = ltl_formula
        psl = psl.replace('G(', 'always (')
        psl = psl.replace('F(', 'eventually (')
        psl = psl.replace('X(', 'next (')
        psl = psl.replace(' & ', ' and ')
        psl = psl.replace(' | ', ' or ')
        psl = psl.replace(' -> ', ' -> ')
        return psl

    def _ltl_to_sva(self, ltl_formula: str) -> str:
        """Convert LTL formula to SystemVerilog Assertion."""
        # Simplified conversion
        sva = ltl_formula
        sva = sva.replace('G(', 'always @(posedge clk) (')
        sva = sva.replace('F(', 'eventually (')
        sva = sva.replace('X(', 'nexttime (')
        sva = sva.replace(' & ', ' && ')
        sva = sva.replace(' | ', ' || ')
        sva = sva.replace(' -> ', ' |-> ')
        return f"assert property ({sva});"

    def _infer_property_type(self, ltl_formula: str) -> PropertyType:
        """Infer property type from LTL formula structure."""
        if ltl_formula.startswith('G(') and 'F(' not in ltl_formula:
            return PropertyType.SAFETY
        elif 'F(' in ltl_formula:
            return PropertyType.LIVENESS
        elif 'G(F(' in ltl_formula:
            return PropertyType.FAIRNESS
        elif ltl_formula.startswith('F('):
            return PropertyType.REACHABILITY
        else:
            return PropertyType.SAFETY

    async def _validate_and_rank_properties(
        self, properties: List[TemporalProperty], circuit_ast: CircuitAST
    ) -> List[TemporalProperty]:
        """Validate synthesized properties and rank by importance."""
        validated_properties = []
        
        for prop in properties:
            # Basic validation
            if self._is_valid_property(prop):
                # Calculate importance score
                importance_score = self._calculate_importance_score(prop, circuit_ast)
                prop.confidence_score *= importance_score
                
                if prop.confidence_score >= self.min_confidence_threshold:
                    validated_properties.append(prop)
        
        # Sort by confidence score
        validated_properties.sort(key=lambda p: p.confidence_score, reverse=True)
        
        return validated_properties

    def _is_valid_property(self, prop: TemporalProperty) -> bool:
        """Validate a temporal property for basic correctness."""
        # Check if LTL formula is non-empty and has basic structure
        if not prop.ltl_formula or len(prop.ltl_formula.strip()) < 3:
            return False
        
        # Check for balanced parentheses
        if prop.ltl_formula.count('(') != prop.ltl_formula.count(')'):
            return False
        
        # Check if applicable signals are specified
        if not prop.applicable_signals:
            return False
        
        return True

    def _calculate_importance_score(
        self, prop: TemporalProperty, circuit_ast: CircuitAST
    ) -> float:
        """Calculate importance score for a property."""
        score = 1.0
        
        # Safety properties are generally more important
        if prop.property_type == PropertyType.SAFETY:
            score *= 1.2
        elif prop.property_type == PropertyType.LIVENESS:
            score *= 1.1
        
        # Properties covering more signals are more important
        signal_coverage = len(prop.applicable_signals) / max(1, len(circuit_ast.modules[0].signals))
        score *= (1.0 + signal_coverage * 0.5)
        
        # Pattern-based properties have higher confidence
        if prop.synthesis_method == "pattern_based":
            score *= 1.1
        
        return min(score, 1.5)  # Cap the multiplier

    def export_properties(
        self, 
        properties: List[TemporalProperty], 
        output_dir: str, 
        formats: List[str] = ["json", "psl", "sva"]
    ):
        """Export synthesized properties in various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if "json" in formats:
            json_data = [asdict(prop) for prop in properties]
            with open(output_path / "synthesized_properties.json", 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
        
        if "psl" in formats:
            psl_content = "-- Synthesized PSL Properties\n\n"
            for prop in properties:
                psl_content += f"-- {prop.name}\n"
                psl_content += f"-- {prop.description}\n"
                psl_content += f"{prop.psl_formula}\n\n"
            
            with open(output_path / "properties.psl", 'w') as f:
                f.write(psl_content)
        
        if "sva" in formats:
            sva_content = "// Synthesized SystemVerilog Assertions\n\n"
            for prop in properties:
                sva_content += f"// {prop.name}\n"
                sva_content += f"// {prop.description}\n"
                sva_content += f"{prop.sva_assertion}\n\n"
            
            with open(output_path / "properties.sv", 'w') as f:
                f.write(sva_content)
        
        self.logger.info(f"Properties exported to {output_dir} in formats: {formats}")