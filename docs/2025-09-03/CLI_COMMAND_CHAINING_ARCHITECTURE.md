# CLI Command Chaining Architecture for Triple Gateway System

**Document Version:** 1.0  
**Date:** 2025-09-03  
**Target Audience:** DSPy Development Team  
**Purpose:** Technical specification for CLI-level command chaining with multi-level orchestration

---

## 📋 **Executive Summary**

This document outlines the implementation strategy for adding CLI-level command chaining to the proposed Triple Gateway Architecture (HeirOSGateway + LLMPromptGateway + LLMGateway). The design enables sophisticated command pipelines with event-driven orchestration, where HeirOS acts as an intelligent listener that learns from CLI usage patterns.

---

## 🎯 **Core Concept: Multi-Level Orchestration**

Rather than a single orchestration layer, we propose **multiple orchestration levels** that enable:
- CLI commands to trigger complex workflows
- HeirOS to listen and learn from CLI patterns
- Event-driven communication between gateways
- Automatic workflow tagging and optimization

```
CLI Input → Level 1: CLI Orchestrator → Event Bus
                                            ↓
                        Level 2: HeirOS Orchestrator (Listening)
                                            ↓
                        Level 3: Gateway Orchestrators (Execution)
```

---

## 🏗️ **Architecture Components**

### **1. Command Chain Parser & Executor**

The foundation for parsing and executing chained gateway commands:

```python
class CommandChain:
    """Parse and execute chained gateway commands"""
    
    def parse_chain(self, command_string: str) -> List[CommandNode]:
        # Parse pipe operators:
        # | for sequential
        # & for parallel
        # > for output redirect
        # && for conditional execution
        # || for fallback execution
        
        # Example: "heiros create-workflow doc-analysis | prompt enhance few-shot | llm chat --audit='testing'"
        
    def execute_chain(self, chain: List[CommandNode], context: dict = None):
        # Execute commands with context passing between stages
        for cmd in chain:
            context = self.execute_command(cmd, context)
            if cmd.operator == '|':
                # Pass output to next command
                continue
            elif cmd.operator == '&':
                # Execute in parallel
                self.execute_parallel(cmd)
            elif cmd.operator == '&&':
                # Execute only if previous succeeded
                if not context.success:
                    break
```

### **2. Unified CLI Interface**

Single entry point that routes to appropriate gateways:

```python
class GatewayCLI:
    """Unified CLI for all three gateways"""
    
    @click.group(chain=True)  # Enable command chaining
    def cli():
        pass
    
    @cli.command()
    @click.option('--workflow', help='Workflow ID or definition')
    @click.pass_context
    def heiros(ctx, workflow):
        # HeirOSGateway commands
        ctx.obj['pipeline'].append(('heiros', workflow))
    
    @cli.command()
    @click.option('--strategy', default='auto', help='Enhancement strategy')
    @click.pass_context
    def prompt(ctx, strategy):
        # LLMPromptGateway commands
        ctx.obj['pipeline'].append(('prompt', strategy))
    
    @cli.command()
    @click.option('--audit', required=True, help='Audit reason (mandatory)')
    @click.option('--model', default='claude', help='Model selection')
    @click.pass_context
    def llm(ctx, audit, model):
        # LLMGateway commands (mandatory)
        ctx.obj['pipeline'].append(('llm', {'audit': audit, 'model': model}))
```

### **3. Context Propagation System**

Context object that flows through the command chain:

```python
@dataclass
class ChainContext:
    """Context that flows through command chain"""
    messages: List[dict] = field(default_factory=list)
    workflow_id: Optional[str] = None
    enhanced_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    results: List[Any] = field(default_factory=list)
    errors: List[Exception] = field(default_factory=list)
    success: bool = True
    
    def pipe_forward(self) -> dict:
        """Format context for next command in chain"""
        return {
            'input': self.results[-1] if self.results else None,
            'metadata': self.metadata,
            'workflow_id': self.workflow_id,
            'previous_errors': self.errors
        }
    
    def merge_parallel(self, other: 'ChainContext'):
        """Merge results from parallel execution"""
        self.results.extend(other.results)
        self.errors.extend(other.errors)
        self.metadata.update(other.metadata)
```

---

## 🔄 **Multi-Level Orchestration System**

### **Level 1: CLI Command Orchestrator (Entry Point)**

```python
class CLIOrchestrator:
    """Parses and initiates command chains"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.command_history = []
        
    def execute_command(self, cmd: str):
        # Parse command
        chain = self.parse_chain(cmd)
        
        # Record for history
        self.command_history.append({
            'command': cmd,
            'timestamp': datetime.now(),
            'user': self.get_current_user()
        })
        
        # Emit CLI event for HeirOS to listen
        event = CLIEvent(
            command=cmd,
            chain=chain,
            timestamp=datetime.now(),
            session_id=self.get_session_id(),
            user_context=self.get_context()
        )
        self.event_bus.emit(event)
        
        # Execute chain
        result = self.execute_chain(chain)
        
        # Emit completion event
        self.event_bus.emit(CLICompletionEvent(
            original_event=event,
            result=result,
            execution_time=self.calculate_duration()
        ))
        
        return result
    
    def parse_chain(self, cmd: str) -> List[CommandNode]:
        """Parse command string into executable nodes"""
        # Implementation for parsing shell-like syntax
        nodes = []
        segments = self.tokenize(cmd)
        
        for segment in segments:
            node = CommandNode(
                command=segment.command,
                args=segment.args,
                options=segment.options,
                operator=segment.operator
            )
            nodes.append(node)
            
        return nodes
```

### **Level 2: HeirOS Workflow Orchestrator (Listener & Tagger)**

```python
class HeirOSOrchestrator:
    """Listens to CLI events and manages workflow lifecycle"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.listeners = {}
        self.workflow_registry = {}
        self.pattern_cache = {}
        self.learning_engine = WorkflowLearningEngine()
        
    @event_listener("cli.command.executed")
    def on_cli_command(self, event: CLIEvent):
        """Tag and track CLI-triggered workflows"""
        
        # Create workflow from CLI event
        workflow_id = self.create_workflow_from_cli(event)
        
        # Tag workflow with CLI metadata
        tags = self.analyze_and_tag(event)
        self.tag_workflow(workflow_id, {
            'trigger': 'cli',
            'command': event.command,
            'chain_depth': len(event.chain),
            'timestamp': event.timestamp,
            'session_id': event.session_id,
            'auto_tags': tags,
            'pattern_match': self.match_patterns(event.command)
        })
        
        # Start monitoring
        self.monitor_workflow(workflow_id)
        
        # Learn from this execution
        self.learning_engine.record_pattern(event)
        
    def create_workflow_from_cli(self, event: CLIEvent) -> str:
        """Convert CLI chain into HeirOS workflow"""
        
        # Check if we've seen this pattern before
        if self.is_known_pattern(event.command):
            template = self.get_workflow_template(event.command)
            dag = self.instantiate_from_template(template, event)
        else:
            # Build new DAG from chain
            dag = self.build_dag_from_chain(event.chain)
        
        workflow = Workflow(
            dag=dag,
            metadata={
                'source': 'cli',
                'auto_generated': True,
                'chain_pattern': self.analyze_pattern(event.chain),
                'optimization_hints': self.suggest_optimizations(event.chain)
            }
        )
        
        return self.register_workflow(workflow)
    
    def analyze_and_tag(self, event: CLIEvent) -> List[str]:
        """Intelligent tagging based on command analysis"""
        tags = []
        
        # Structural analysis
        if len(event.chain) > 3:
            tags.append('complex_pipeline')
        if self.has_parallel_execution(event.chain):
            tags.append('parallel_processing')
            
        # Gateway usage analysis
        gateways_used = self.extract_gateways(event.chain)
        if all(g in gateways_used for g in ['heiros', 'prompt', 'llm']):
            tags.append('full_stack')
        elif gateways_used == ['llm']:
            tags.append('minimal_path')
            
        # Content analysis
        if 'compliance' in event.command.lower():
            tags.append('compliance_workflow')
        if 'analysis' in event.command.lower():
            tags.append('analytical_workflow')
            
        return tags
```

### **Level 3: Gateway-Level Orchestrators (Execution)**

```python
class GatewayOrchestrator:
    """Individual gateway orchestration with event emission"""
    
    def __init__(self, gateway_type: str, gateway_instance, event_bus: EventBus):
        self.gateway_type = gateway_type
        self.gateway = gateway_instance
        self.event_bus = event_bus
        self.metrics = MetricsCollector()
        
    def execute(self, operation: Operation, context: dict):
        # Emit start event
        start_event = GatewayEvent(
            type='start',
            gateway=self.gateway_type,
            operation=operation,
            context_size=len(str(context))
        )
        self.event_bus.emit(start_event)
        
        # Start metrics collection
        self.metrics.start_timer()
        
        try:
            # Execute gateway operation
            result = self.gateway.execute(operation, context)
            
            # Record success metrics
            self.metrics.record_success()
            
            # Emit completion event
            self.event_bus.emit(GatewayEvent(
                type='complete',
                gateway=self.gateway_type,
                operation=operation,
                result=result,
                duration=self.metrics.get_duration()
            ))
            
            return result
            
        except Exception as e:
            # Record failure metrics
            self.metrics.record_failure(e)
            
            # Emit error event
            self.event_bus.emit(GatewayEvent(
                type='error',
                gateway=self.gateway_type,
                operation=operation,
                error=e
            ))
            
            # Decide whether to propagate or handle
            if self.should_fallback(e):
                return self.execute_fallback(operation, context)
            else:
                raise
```

---

## 🎭 **Event-Driven Architecture**

### **Core Event System**

```python
class EventDrivenChainExecutor:
    """Orchestrates command chains through event system"""
    
    def __init__(self):
        self.event_bus = EventBus()
        
        # Initialize orchestrators
        self.orchestrators = {
            'cli': CLIOrchestrator(self.event_bus),
            'heiros': HeirOSOrchestrator(self.event_bus),
            'prompt': PromptOrchestrator(self.event_bus),
            'llm': LLMOrchestrator(self.event_bus)
        }
        
        # Setup cross-orchestrator communication
        self.setup_listeners()
        
    def setup_listeners(self):
        """Configure cross-orchestrator communication"""
        
        # HeirOS listens to ALL CLI commands
        self.orchestrators['heiros'].listen_to('cli.*')
        
        # HeirOS can intercept and modify chains
        self.orchestrators['heiros'].add_interceptor(
            pattern='cli.chain.*',
            handler=self.optimize_chain
        )
        
        # Prompt gateway listens to HeirOS workflow events
        self.orchestrators['prompt'].listen_to('heiros.workflow.*')
        
        # LLM gateway (mandatory) listens to everything for governance
        self.orchestrators['llm'].listen_to('*')
        
        # Setup metric aggregation
        self.setup_metrics_aggregation()
        
    def optimize_chain(self, event: ChainEvent) -> ChainEvent:
        """HeirOS optimization of command chains"""
        
        # Analyze chain for optimization opportunities
        optimizations = self.orchestrators['heiros'].suggest_optimizations(event.chain)
        
        if optimizations:
            # Apply optimizations
            event.chain = self.apply_optimizations(event.chain, optimizations)
            event.metadata['optimized'] = True
            event.metadata['optimizations_applied'] = optimizations
            
        return event
```

### **Event Types and Flow**

```python
@dataclass
class CLIEvent:
    """Base event for CLI operations"""
    command: str
    chain: List[CommandNode]
    timestamp: datetime
    session_id: str
    user_context: dict
    
@dataclass
class WorkflowEvent:
    """Workflow lifecycle events"""
    workflow_id: str
    event_type: Literal['created', 'started', 'completed', 'failed']
    metadata: dict
    
@dataclass
class GatewayEvent:
    """Gateway execution events"""
    gateway: str
    operation: str
    event_type: Literal['start', 'complete', 'error']
    duration: Optional[float] = None
    error: Optional[Exception] = None
```

---

## 🎯 **HeirOS Listening & Tagging System**

### **Advanced Pattern Recognition**

```python
class HeirOSListener:
    """Advanced listening and workflow tagging"""
    
    def __init__(self):
        self.patterns = {}
        self.workflow_tags = defaultdict(set)
        self.pattern_matchers = []
        self.ml_tagger = MLBasedTagger()  # ML model for tag prediction
        
    def register_cli_pattern(self, pattern: str, handler: Callable, tags: List[str]):
        """Register patterns to listen for"""
        matcher = PatternMatcher(pattern, handler, tags)
        self.pattern_matchers.append(matcher)
        
    @listen_for_cli
    def tag_workflow_from_cli(self, cli_event: CLIEvent):
        """Auto-tag workflows based on CLI patterns"""
        
        tags = set()
        
        # Rule-based tagging
        tags.update(self.rule_based_tags(cli_event))
        
        # ML-based tagging
        tags.update(self.ml_tagger.predict_tags(cli_event.command))
        
        # Pattern matching
        for matcher in self.pattern_matchers:
            if matcher.matches(cli_event.command):
                tags.update(matcher.tags)
                # Execute pattern handler
                additional_tags = matcher.handler(cli_event)
                tags.update(additional_tags)
                
        # Create workflow with tags
        workflow = self.create_workflow(
            source='cli',
            tags=list(tags),
            metadata={
                'cli_command': cli_event.command,
                'user': cli_event.user,
                'session': cli_event.session_id,
                'timestamp': cli_event.timestamp,
                'confidence_scores': self.ml_tagger.get_confidence_scores()
            }
        )
        
        return workflow
    
    def rule_based_tags(self, event: CLIEvent) -> Set[str]:
        """Apply rule-based tagging logic"""
        tags = set()
        
        # Command structure analysis
        if self.is_complex_chain(event.chain):
            tags.add('complex_pipeline')
            
        # Gateway usage analysis
        if self.uses_all_gateways(event.chain):
            tags.add('full_stack')
        elif self.only_uses_mandatory(event.chain):
            tags.add('minimal_path')
            
        # Performance characteristics
        if self.is_parallel_execution(event.chain):
            tags.add('parallel_execution')
        if self.has_fallback_logic(event.chain):
            tags.add('resilient_workflow')
            
        return tags
```

### **Learning Engine**

```python
class WorkflowLearningEngine:
    """Learn from CLI patterns to improve future executions"""
    
    def __init__(self):
        self.pattern_database = PatternDatabase()
        self.optimization_model = OptimizationModel()
        self.template_generator = TemplateGenerator()
        
    def record_pattern(self, event: CLIEvent):
        """Record CLI pattern for learning"""
        pattern = Pattern(
            command=event.command,
            chain=event.chain,
            timestamp=event.timestamp,
            user=event.user_context.get('user'),
            session=event.session_id
        )
        
        self.pattern_database.store(pattern)
        
        # Analyze for optimization opportunities
        if self.should_optimize(pattern):
            optimization = self.optimization_model.generate(pattern)
            self.store_optimization(pattern, optimization)
            
        # Check if we should create a template
        if self.should_create_template(pattern):
            template = self.template_generator.create(pattern)
            self.store_template(template)
    
    def suggest_optimization(self, command: str) -> Optional[Optimization]:
        """Suggest optimization for a command"""
        similar_patterns = self.pattern_database.find_similar(command)
        
        if similar_patterns:
            # Use historical data to suggest optimization
            return self.optimization_model.predict(command, similar_patterns)
            
        return None
    
    def should_create_template(self, pattern: Pattern) -> bool:
        """Determine if pattern is common enough for template"""
        frequency = self.pattern_database.get_frequency(pattern)
        complexity = self.calculate_complexity(pattern)
        
        # Create template for frequently used complex patterns
        return frequency > 10 and complexity > 3
```

---

## 🚀 **Command Chain Execution Examples**

### **Basic Chain Patterns**

```bash
# Sequential chaining (pipe |)
tidyllm heiros create-dag financial-analysis \
    | prompt enhance few-shot \
    | llm chat --audit="Q4 analysis"

# Parallel execution (&)
tidyllm heiros workflow-1 & heiros workflow-2 \
    | prompt merge \
    | llm summarize --audit="Parallel workflow summary"

# Conditional chaining (&&)
tidyllm prompt validate \
    && llm chat --model=claude --audit="Validated prompt" \
    || llm chat --model=gpt-4 --audit="Fallback execution"

# Output redirection (>)
tidyllm heiros execute doc-pipeline \
    | llm analyze --audit="Document analysis" \
    > results.json

# Complex pipeline with error handling
tidyllm heiros create-workflow regulatory-check \
    --sparse-agreement=SOX_COMPLIANCE \
    | prompt enhance chain-of-thought \
    --examples=financial_examples.json \
    | llm chat \
    --model=claude-3 \
    --audit="SOX compliance review" \
    --max-tokens=4000 \
    || heiros log-error \
    && heiros save-result
```

### **Advanced Patterns with HeirOS Listening**

```bash
# Pattern that triggers automatic workflow creation
tidyllm --pattern=compliance-check \
    prompt validate \
    | llm review --model=claude --audit="Compliance validation"
# HeirOS recognizes pattern and creates pre-defined compliance workflow

# Learning pattern - HeirOS optimizes after multiple executions
tidyllm heiros analyze-document \
    | prompt enhance \
    | llm summarize --audit="Document summary"
# After 10 executions, HeirOS suggests: "Use cached prompt template for 30% speed improvement"

# Workflow template invocation
tidyllm --use-template=financial-analysis-v2 \
    --input=quarterly_report.pdf \
    --audit="Q4 financial analysis"
# HeirOS expands template to full workflow based on learned patterns
```

---

## 🔧 **Implementation Components**

### **Shell Integration**

```python
class ShellIntegration:
    """Native shell feature support"""
    
    def enable_autocomplete(self):
        """Register commands for shell autocomplete"""
        commands = self.get_all_commands()
        
        # Generate completion script
        completion_script = self.generate_completion_script(commands)
        
        # Register with shell
        self.register_with_shell(completion_script)
        
    def export_aliases(self):
        """Export common command patterns as aliases"""
        aliases = {
            'tlm-full': 'tidyllm heiros | prompt | llm',
            'tlm-quick': 'tidyllm llm',
            'tlm-analyze': 'tidyllm heiros analyze | llm',
            'tlm-compliance': 'tidyllm --pattern=compliance-check'
        }
        
        for alias, command in aliases.items():
            self.create_alias(alias, command)
            
    def support_environment_vars(self):
        """Use environment variables for configuration"""
        env_vars = {
            'TIDYLLM_DEFAULT_AUDIT': 'Default audit reason',
            'TIDYLLM_WORKFLOW_PATH': 'Path to workflow definitions',
            'TIDYLLM_CACHE_DIR': 'Cache directory for templates',
            'TIDYLLM_DEFAULT_MODEL': 'Default LLM model',
            'TIDYLLM_PARALLEL_LIMIT': 'Max parallel executions'
        }
        
        return self.read_env_vars(env_vars)
```

### **Stream Processing**

```python
class StreamChain:
    """Handle streaming responses through the chain"""
    
    async def stream_chain(self, commands: List[str]):
        """Stream results through command chain"""
        
        pipeline = self.build_pipeline(commands)
        
        async for chunk in self.execute_streaming(pipeline):
            if chunk.type == 'intermediate':
                # Log progress without blocking
                await self.log_progress(chunk)
                
            elif chunk.type == 'result':
                # Yield result to caller
                yield chunk.data
                
            elif chunk.type == 'error':
                # Handle error in stream
                if self.can_recover(chunk.error):
                    await self.attempt_recovery(chunk)
                else:
                    raise chunk.error
    
    async def execute_streaming(self, pipeline: Pipeline):
        """Execute pipeline with streaming"""
        
        for stage in pipeline.stages:
            async for result in stage.execute_async():
                yield StreamChunk(
                    type='intermediate',
                    stage=stage.name,
                    data=result
                )
                
                # Pass to next stage if needed
                if stage.has_next():
                    stage.next.add_input(result)
```

### **State Management**

```python
class ChainStateManager:
    """Maintain state between chained commands"""
    
    def __init__(self):
        self.sessions = {}
        self.global_context = {}
        
    def create_session(self, session_id: str) -> Session:
        """Create new chain session"""
        session = Session(
            id=session_id,
            context=ChainContext(),
            start_time=datetime.now(),
            commands=[]
        )
        self.sessions[session_id] = session
        return session
    
    def update_session(self, session_id: str, command: str, result: Any):
        """Update session with command result"""
        session = self.sessions[session_id]
        session.commands.append(command)
        session.context.results.append(result)
        session.last_updated = datetime.now()
        
    def get_session_context(self, session_id: str) -> ChainContext:
        """Get current session context"""
        return self.sessions[session_id].context
        
    def cleanup_expired_sessions(self, ttl_seconds: int = 3600):
        """Clean up old sessions"""
        cutoff = datetime.now() - timedelta(seconds=ttl_seconds)
        expired = [
            sid for sid, session in self.sessions.items()
            if session.last_updated < cutoff
        ]
        for sid in expired:
            del self.sessions[sid]
```

---

## 📊 **Metrics and Monitoring**

### **Performance Tracking**

```python
class ChainMetrics:
    """Track chain execution metrics"""
    
    def __init__(self):
        self.metrics = {
            'execution_times': [],
            'chain_lengths': [],
            'gateway_usage': defaultdict(int),
            'error_rates': defaultdict(float),
            'pattern_frequency': defaultdict(int)
        }
        
    def record_execution(self, event: CLICompletionEvent):
        """Record execution metrics"""
        self.metrics['execution_times'].append(event.duration)
        self.metrics['chain_lengths'].append(len(event.chain))
        
        for gateway in event.gateways_used:
            self.metrics['gateway_usage'][gateway] += 1
            
        pattern = self.extract_pattern(event.command)
        self.metrics['pattern_frequency'][pattern] += 1
        
    def get_optimization_suggestions(self) -> List[str]:
        """Suggest optimizations based on metrics"""
        suggestions = []
        
        # Check for slow patterns
        slow_patterns = self.identify_slow_patterns()
        for pattern in slow_patterns:
            suggestions.append(f"Pattern '{pattern}' is slow. Consider caching or optimization.")
            
        # Check for redundant gateway usage
        redundant = self.find_redundant_usage()
        for usage in redundant:
            suggestions.append(f"Redundant use of {usage}. Consider simplifying chain.")
            
        return suggestions
```

---

## 🔒 **Security and Validation**

### **Chain Validation**

```python
class ChainValidator:
    """Validate command chains before execution"""
    
    def validate_chain(self, chain: List[CommandNode]) -> ValidationResult:
        """Ensure chain is valid and safe"""
        
        # Check for mandatory LLM gateway
        if not self.has_llm_gateway(chain):
            return ValidationResult(
                valid=False,
                error="Chain must include mandatory LLM gateway for governance"
            )
            
        # Check for circular dependencies
        if self.has_circular_dependency(chain):
            return ValidationResult(
                valid=False,
                error="Chain contains circular dependencies"
            )
            
        # Validate permissions
        for node in chain:
            if not self.has_permission(node):
                return ValidationResult(
                    valid=False,
                    error=f"Insufficient permissions for {node.command}"
                )
                
        # Check resource limits
        if self.exceeds_resource_limits(chain):
            return ValidationResult(
                valid=False,
                error="Chain exceeds resource limits"
            )
            
        return ValidationResult(valid=True)
```

---

## 🚦 **Benefits of This Architecture**

### **For Development Teams**
1. **Flexible Command Composition**: Build complex workflows using simple shell-like syntax
2. **Progressive Enhancement**: Start with simple commands, evolve to complex workflows
3. **Reusable Patterns**: HeirOS learns and suggests workflow templates
4. **Clear Debugging**: Event system provides complete execution trace

### **For Operations Teams**
1. **Workflow Learning**: HeirOS automatically identifies and optimizes common patterns
2. **Auto-Tagging**: Intelligent classification of workflows for better organization
3. **Performance Insights**: Metrics-driven optimization suggestions
4. **Audit Trail**: Complete tracking from CLI input to execution result

### **For Enterprise**
1. **Governance Maintained**: Mandatory LLM gateway ensures compliance
2. **Cost Optimization**: Pattern recognition reduces redundant LLM calls
3. **Risk Management**: Multi-level orchestration provides control points
4. **Scalability**: Event-driven architecture supports high-volume operations

---

## 📈 **Implementation Roadmap**

### **Phase 1: Foundation (Week 1-2)**
- [ ] Implement basic command chain parser
- [ ] Create CLI orchestrator with event emission
- [ ] Setup basic HeirOS listener
- [ ] Implement context propagation

### **Phase 2: Integration (Week 3-4)**
- [ ] Wire HeirOS orchestrator to CLI events
- [ ] Implement gateway-level orchestrators
- [ ] Add state management system
- [ ] Create validation framework

### **Phase 3: Intelligence (Week 5-6)**
- [ ] Build pattern recognition system
- [ ] Implement workflow learning engine
- [ ] Add auto-tagging capabilities
- [ ] Create template generation

### **Phase 4: Optimization (Week 7-8)**
- [ ] Add streaming support
- [ ] Implement parallel execution
- [ ] Build metrics collection
- [ ] Create optimization suggestions

---

## 🎯 **Key Implementation Requirements**

1. **State Management**: Maintain context between chained commands
2. **Error Handling**: Graceful degradation with fallback options
3. **Validation**: Ensure mandatory LLM gateway inclusion
4. **Performance**: Support parallel execution where possible
5. **Logging**: Complete event trail for debugging
6. **Output Formats**: Support JSON, YAML, plain text outputs
7. **Learning**: Pattern recognition for workflow optimization
8. **Security**: Permission validation at each orchestration level

---

## 📝 **Example Implementation**

```python
class TidyLLMCLI:
    """Main CLI with multi-level orchestration"""
    
    def __init__(self):
        # Initialize event system
        self.event_bus = EventBus()
        
        # Initialize orchestration layers
        self.cli_orchestrator = CLIOrchestrator(self.event_bus)
        self.heiros_orchestrator = HeirOSOrchestrator(self.event_bus)
        self.gateway_orchestrators = self.init_gateway_orchestrators()
        
        # Setup HeirOS listening
        self.heiros_orchestrator.start_listening()
        
        # Initialize support systems
        self.state_manager = ChainStateManager()
        self.validator = ChainValidator()
        self.metrics = ChainMetrics()
        
    @click.command()
    @click.argument('command')
    @click.option('--track/--no-track', default=True, help='Enable HeirOS tracking')
    @click.option('--optimize/--no-optimize', default=True, help='Enable optimization')
    @click.option('--session', help='Session ID for stateful execution')
    def execute(self, command: str, track: bool, optimize: bool, session: str):
        """Execute CLI command with orchestration"""
        
        # Get or create session
        if not session:
            session = self.state_manager.create_session(str(uuid.uuid4()))
        
        # Level 1: CLI parsing
        chain = self.cli_orchestrator.parse(command)
        
        # Validate chain
        validation = self.validator.validate_chain(chain)
        if not validation.valid:
            raise ValueError(f"Invalid chain: {validation.error}")
        
        # Emit event for HeirOS listening (Level 2)
        if track:
            event = CLIExecutionEvent(
                command=command,
                chain=chain,
                session_id=session,
                tracking_enabled=True
            )
            self.event_bus.emit(event)
            
            # Wait for HeirOS optimization if enabled
            if optimize:
                optimization = self.heiros_orchestrator.get_optimization(command)
                if optimization:
                    chain = optimization.apply(chain)
        
        # Execute with full orchestration
        context = self.state_manager.get_session_context(session)
        result = self.cli_orchestrator.execute_chain(chain, context)
        
        # Update session state
        self.state_manager.update_session(session, command, result)
        
        # Record metrics
        self.metrics.record_execution(CLICompletionEvent(
            command=command,
            chain=chain,
            result=result,
            duration=self.calculate_duration(),
            gateways_used=self.extract_gateways(chain)
        ))
        
        # HeirOS captures result for learning
        if track:
            self.event_bus.emit(CLICompletionEvent(
                command=command,
                result=result,
                metrics=self.metrics.get_current()
            ))
            
        return result
    
    def init_gateway_orchestrators(self):
        """Initialize gateway-specific orchestrators"""
        return {
            'heiros': GatewayOrchestrator('heiros', HeirOSGateway(), self.event_bus),
            'prompt': GatewayOrchestrator('prompt', LLMPromptGateway(), self.event_bus),
            'llm': GatewayOrchestrator('llm', LLMGateway(), self.event_bus)
        }
```

---

## 🎯 **Conclusion**

This multi-level orchestration architecture with CLI command chaining provides:

1. **Sophisticated Command Pipelines**: Shell-like syntax for complex AI workflows
2. **Intelligent Learning**: HeirOS learns from usage patterns to optimize future executions
3. **Event-Driven Flexibility**: Loosely coupled components communicate through events
4. **Progressive Enhancement**: Each orchestration level adds value without tight coupling
5. **Enterprise Governance**: Mandatory LLM gateway ensures compliance while allowing flexibility

The key innovation is that **HeirOS acts not just as an executor but as an intelligent observer**, learning from CLI patterns to continuously improve workflow execution while maintaining enterprise governance requirements.

---

**Next Steps for DSPy Team:**
1. Review and provide feedback on architecture
2. Prioritize implementation phases
3. Identify integration points with existing DSPy systems
4. Define specific requirements for HeirOS listening patterns
5. Determine desired CLI syntax and command structure