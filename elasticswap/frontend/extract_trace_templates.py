# Add to pipeline.py (before generate functions)

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import glob
import json
from collections import defaultdict
from dateutil import parser as date_parser

@dataclass
class ToolExecution:
    """Individual tool execution with timing"""
    name: str
    start_timestamp: str
    duration: float  # seconds
    
    @property
    def start_time_epoch(self) -> float:
        """Convert timestamp to seconds since epoch for comparison"""
        from dateutil import parser as date_parser
        return date_parser.parse(self.start_timestamp).timestamp()
    
    @property
    def end_time_epoch(self) -> float:
        return self.start_time_epoch + self.duration

@dataclass
class TurnTokens:
    """Token breakdown for a single LLM turn"""
    cache_read: int      # Tokens read from cache (prefix)
    new_input: int       # New tokens added (cache_creation + input)
    output: int          # Generated output tokens
    tool_executions: List[ToolExecution] = None
    
    def __post_init__(self):
        if self.tool_executions is None:
            self.tool_executions = []
    
    @property
    def total_input(self):
        return self.cache_read + self.new_input
    
    @property
    def num_tools(self):
        return len(self.tool_executions)
    
    @property
    def total_tool_time_with_overlap(self) -> float:
        """Compute actual wall-clock time accounting for parallel tool execution"""
        if not self.tool_executions:
            return 0.0
        
        # Sort by start time
        sorted_tools = sorted(self.tool_executions, key=lambda t: t.start_time_epoch)
        
        # Merge overlapping intervals to get actual wall-clock time
        total_time = 0.0
        current_start = sorted_tools[0].start_time_epoch
        current_end = sorted_tools[0].end_time_epoch
        
        for tool in sorted_tools[1:]:
            if tool.start_time_epoch <= current_end:
                # Overlapping - extend current interval
                current_end = max(current_end, tool.end_time_epoch)
            else:
                # Gap - add completed interval and start new one
                total_time += (current_end - current_start)
                current_start = tool.start_time_epoch
                current_end = tool.end_time_epoch
        
        # Add final interval
        total_time += (current_end - current_start)
        return total_time

@dataclass
class TraceTemplate:
    """Skeleton of a Claude trace with turn-by-turn evolution"""
    template_id: str
    session_id: str
    turns: List[TurnTokens]  # Each turn now contains its own tools
    
    @property
    def num_turns(self):
        return len(self.turns)
    
    @property
    def total_tools(self):
        return sum(len(t.tool_executions) for t in self.turns)
    
    @property
    def total_context_accumulated(self):
        """Total context size at the end of the trace (last turn's full context)"""
        if not self.turns:
            return 0
        last_turn = self.turns[-1]
        return last_turn.cache_read + last_turn.new_input
    
    def __repr__(self):
        return f"Template({self.num_turns} turns, {self.total_tools} tools, {self.total_context_accumulated} ctx tokens)"
    
    def print_sequence(self):
        """Print the chronological execution sequence of this trace"""
        print(f"\n{'='*70}")
        print(f"Template: {self.template_id} | Session: {self.session_id[:16]}...")
        print(f"Total: {self.num_turns} turns, {self.total_tools} tools, {self.total_context_accumulated} context tokens")
        print(f"{'='*70}\n")
        
        for i, turn in enumerate(self.turns, 1):
            # Print LLM call
            print(f"Turn {i}: LLMCall(cache_read={turn.cache_read}, new_input={turn.new_input}, output={turn.output})")
            
            # Print associated tool calls
            if turn.tool_executions:
                total_sequential = sum(t.duration for t in turn.tool_executions)
                total_wallclock = turn.total_tool_time_with_overlap
                overlap_pct = ((total_sequential - total_wallclock) / total_sequential * 100) if total_sequential > 0 else 0
                
                print(f"  Tools: {len(turn.tool_executions)} total | "
                      f"Sequential: {total_sequential:.3f}s | "
                      f"Wall-clock: {total_wallclock:.3f}s | "
                      f"Overlap: {overlap_pct:.1f}%")
                
                for j, tool_exec in enumerate(turn.tool_executions, 1):
                    print(f"    └─→ {j}. {tool_exec.name}(duration={tool_exec.duration:.3f}s)")
            print()  # Blank line between turns


def split_session_at_llm_gaps(session_events: List[dict], gap_threshold_seconds: float = 300) -> List[List[dict]]:
    """
    Split a session into multiple sub-sessions at large gaps between LLM calls.
    
    This handles human pauses where users step away and come back later.
    
    Args:
        session_events: List of events for a single session
        gap_threshold_seconds: Split when LLM-to-LLM gap exceeds this (default: 300s = 5min)
    
    Returns:
        List of sub-session event lists
    """
    if not session_events:
        return []
    
    sub_sessions = []
    current_sub_session = []
    last_llm_time = None
    seen_message_ids = set()  # Track unique LLM calls
    
    for event in session_events:
        if event['event_type'] == 'llm_call':
            msg_id = event['message_id']
            
            # Skip duplicates in gap calculation
            if msg_id not in seen_message_ids:
                seen_message_ids.add(msg_id)
                current_time = date_parser.parse(event['timestamp']).timestamp()
                
                # Check for large gap
                if last_llm_time is not None:
                    gap = current_time - last_llm_time
                    
                    if gap > gap_threshold_seconds:
                        # Large gap detected - save current sub-session and start new one
                        if current_sub_session:
                            sub_sessions.append(current_sub_session)
                        current_sub_session = []
                
                last_llm_time = current_time
        
        current_sub_session.append(event)
    
    # Don't forget the last sub-session
    if current_sub_session:
        sub_sessions.append(current_sub_session)
    
    return sub_sessions


def extract_trace_templates_with_evolution(
    claude_dataset_dir: str,
    trace_filter_percentile: float = 90.0,  # Filter traces above this percentile
    split_at_gap_seconds: float = 300,       # Split traces at gaps > 5 minutes
    max_trace_duration_minutes: float = 20   # Filter traces longer than 20 minutes
) -> List[TraceTemplate]:
    """
    Extract trace templates from Claude tool call dataset, filtering traces with outlier tool times.
    
    The Claude dataset includes user idle time in tool execution times. Instead of capping
    individual tools, this function filters out entire traces that have abnormally high
    average tool execution times:
    1. Computes the distribution of all tool execution times
    2. Calculates the p90 threshold
    3. For each trace, computes its average tool execution time
    4. Removes traces where avg_tool_time > p90 threshold
    
    Additionally, this function:
    5. Splits traces at large gaps (> split_at_gap_seconds) between LLM calls to handle human pauses
    6. Filters out traces that exceed max_trace_duration_minutes after splitting
    
    This preserves the natural distribution of tool times while removing problematic traces
    where users stepped away during execution.
    
    Args:
        claude_dataset_dir: Directory with *_events.json files
        trace_filter_percentile: Percentile for trace filtering (default 90.0)
        split_at_gap_seconds: Split traces at gaps larger than this (default 300s = 5min)
        max_trace_duration_minutes: Filter traces longer than this (default 20 minutes)
    
    Returns:
        List of TraceTemplate objects with turn-by-turn token evolution
    """
    json_files = sorted(glob.glob(f"{claude_dataset_dir}/*_events.json"))
    
    print(f"Loading traces from {claude_dataset_dir}...")
    
    # First pass: collect all tool durations to compute threshold
    all_tool_durations = []
    for json_file in json_files:
        with open(json_file) as f:
            events = json.load(f)
        
        for event in events:
            if event['event_type'] == 'tool_call':
                exec_time = event.get('execution_time_ms')
                if exec_time is not None and exec_time > 0:
                    all_tool_durations.append(exec_time / 1000.0)
    
    # Compute threshold for trace filtering
    if all_tool_durations:
        threshold_s = np.percentile(all_tool_durations, trace_filter_percentile)
        print(f"  Total tools across all traces: {len(all_tool_durations)}")
        print(f"  Duration p50: {np.percentile(all_tool_durations, 50):.3f}s")
        print(f"  Duration p75: {np.percentile(all_tool_durations, 75):.3f}s")
        print(f"  Duration p90: {np.percentile(all_tool_durations, 90):.3f}s")
        print(f"  Duration p95: {np.percentile(all_tool_durations, 95):.3f}s")
        print(f"  Duration p99: {np.percentile(all_tool_durations, 99):.3f}s")
        print(f"  Duration max: {np.max(all_tool_durations):.3f}s")
        print(f"\n  Trace filtering threshold (p{trace_filter_percentile:.0f}): {threshold_s:.3f}s")
        print(f"  Will remove traces where avg_tool_time > {threshold_s:.3f}s")
    else:
        threshold_s = 30.0  # Fallback
        print(f"  No tools found, using default threshold: {threshold_s}s")
    
    # Second pass: extract templates, filtering traces with high avg tool times
    templates = []
    filtered_traces = 0
    total_traces = 0
    traces_split = 0
    traces_too_long = 0
    
    for json_file in json_files:
        with open(json_file) as f:
            events = json.load(f)
        
        # SORT BY TIMESTAMP to ensure chronological order
        events.sort(key=lambda e: e['timestamp'])
        
        # Group by session
        sessions = defaultdict(list)
        
        for event in events:
            session_id = event['session_id']
            sessions[session_id].append(event)
        
        # Process each session (with splitting at large gaps)
        # Sort sessions by ID to ensure deterministic order
        for session_id, session_events in sorted(sessions.items()):
            if not session_events:
                continue
            
            # Split session at large gaps between LLM calls (human pauses)
            sub_sessions = split_session_at_llm_gaps(session_events, split_at_gap_seconds)
            
            if len(sub_sessions) > 1:
                traces_split += 1
            
            # Process each sub-session as a separate trace
            for sub_idx, sub_session_events in enumerate(sub_sessions):
                if not sub_session_events:
                    continue
                
                # Calculate trace duration (first to last event)
                timestamps = [date_parser.parse(e['timestamp']).timestamp() 
                             for e in sub_session_events]
                duration_minutes = (timestamps[-1] - timestamps[0]) / 60
                
                # Filter out traces that are too long
                if duration_minutes > max_trace_duration_minutes:
                    traces_too_long += 1
                    continue
                
                turns = []
                seen_message_ids = set()  # Track deduplicated LLM calls
                current_turn = None
                current_tools = []
                
                for event in sub_session_events:
                    if event['event_type'] == 'llm_call':
                        msg_id = event['message_id']
                        
                        # Skip duplicate LLM calls with same message_id
                        if msg_id in seen_message_ids:
                            continue
                        
                        seen_message_ids.add(msg_id)
                        
                        # Get token counts for this call
                        cache_read = event.get('cache_read_input_tokens', 0)
                        new_input = event.get('cache_creation_input_tokens', 0) + event.get('input_tokens', 0)
                        output = event.get('output_tokens', 0)
                        
                        # Check if this is truly a NEW turn using context evolution heuristic
                        is_new_turn = True
                        if current_turn is not None:
                            prev_context = current_turn.cache_read + current_turn.new_input
                            # Allow 10% tolerance for tokenization differences
                            if abs(cache_read - prev_context) < prev_context * 0.1 and \
                               cache_read == current_turn.cache_read and \
                               new_input == current_turn.new_input and \
                               output == current_turn.output:
                                # This is a duplicate with identical stats - skip it
                                is_new_turn = False
                        
                        if is_new_turn:
                            # Save previous turn if exists
                            if current_turn is not None:
                                current_turn.tool_executions = current_tools
                                turns.append(current_turn)
                                current_tools = []
                            
                            # Start new turn
                            current_turn = TurnTokens(
                                cache_read=cache_read,
                                new_input=new_input,
                                output=output
                            )
                    
                    elif event['event_type'] == 'tool_call':
                        exec_time = event.get('execution_time_ms')
                        if exec_time is not None and exec_time > 0:  # Skip 0.0 and None
                            duration_seconds = exec_time / 1000.0
                            # No artificial cap - preserve actual tool durations
                            tool_exec = ToolExecution(
                                name=event['tool_name'],
                                start_timestamp=event['timestamp'],
                                duration=duration_seconds
                            )
                            current_tools.append(tool_exec)
                
                # Don't forget the last turn
                if current_turn is not None:
                    current_turn.tool_executions = current_tools
                    turns.append(current_turn)
                
                if turns:
                    total_traces += 1
                    
                    # Calculate average tool execution time for this trace
                    all_trace_tools = []
                    for turn in turns:
                        for tool in turn.tool_executions:
                            all_trace_tools.append(tool.duration)
                    
                    if all_trace_tools:
                        avg_tool_time = np.mean(all_trace_tools)
                        
                        # Filter trace if average tool time exceeds threshold
                        if avg_tool_time > threshold_s:
                            filtered_traces += 1
                            continue  # Skip this trace
                    
                    # Trace passes filter - add to templates
                    # Use sub_idx to create unique template_id for split traces
                    template_id_suffix = f"-{sub_idx}" if len(sub_sessions) > 1 else ""
                    template = TraceTemplate(
                        template_id=f"{session_id[:8]}{template_id_suffix}",
                        session_id=session_id,
                        turns=turns
                    )
                    templates.append(template)
    
    print(f"\n  ===== Trace Processing Summary =====")
    print(f"  Original sessions split: {traces_split}")
    print(f"  Total sub-traces after splitting: {total_traces}")
    if total_traces > 0:
        print(f"  Filtered (tool time too high): {filtered_traces} ({filtered_traces/total_traces*100:.1f}%)")
        print(f"  Filtered (duration > {max_trace_duration_minutes} min): {traces_too_long} ({traces_too_long/total_traces*100:.1f}%)")
        print(f"  Kept traces: {len(templates)} ({len(templates)/total_traces*100:.1f}%)")
    else:
        print(f"  Filtered (tool time too high): {filtered_traces}")
        print(f"  Filtered (duration > {max_trace_duration_minutes} min): {traces_too_long}")
        print(f"  Kept traces: {len(templates)}")
    print(f"  Gap threshold used: {split_at_gap_seconds}s ({split_at_gap_seconds/60:.0f} minutes)")
    return templates

@dataclass
class TurnExecution:
    """Concrete execution of a single turn with actual token IDs"""
    turn_idx: int
    prompt_token_ids: List[int]  # Full context for this turn
    target_output_tokens: int    # Expected output length
    tool_wait_time: float        # Time to wait after this turn (wall-clock)
    tool_names: List[str]        # Tool names for reference

@dataclass
class RequestTrajectory:
    """A concrete request with unique token IDs following a template"""
    request_id: str
    template_id: str
    source_template: TraceTemplate  # Add reference to source template
    turns: List[TurnExecution]
    
    def __repr__(self):
        return f"Request({self.request_id}, {len(self.turns)} turns, template={self.template_id})"


def generate_unique_token_sequence(base_seed: int, length: int) -> List[int]:
    """
    Generate a unique sequence of token IDs.
    
    Uses base_seed to ensure different requests get different sequences.
    Token IDs are in valid range (assuming vocab size ~100k).
    """
    rng = np.random.RandomState(base_seed)
    # Generate tokens in range [1000, 99999] to avoid special tokens
    return rng.randint(1000, 100000, size=length).tolist()


def instantiate_trajectory_from_template(
    template: TraceTemplate, 
    request_id: str, 
    base_seed: int
) -> RequestTrajectory:
    """
    Create a concrete request trajectory with unique token IDs from a template.
    
    Maintains the evolving context pattern for prefix cache hits:
    - Turn 0: [input_tokens] -> generates [output_tokens]
    - Turn 1: [Turn0_input + Turn0_output + new_input] -> generates [output_tokens]
    - Turn N: [All previous tokens + new_input] -> generates [output_tokens]
    
    This ensures vLLM can match the prefix cache across turns.
    """
    turns = []
    accumulated_tokens = []  # Full conversation history (inputs + outputs)
    
    for turn_idx, turn_template in enumerate(template.turns):
        # Compute how many tokens should be in the input for this turn
        total_input_tokens = turn_template.cache_read + turn_template.new_input
        
        if turn_idx == 0:
            # First turn: generate all input tokens fresh
            prompt_tokens = generate_unique_token_sequence(
                base_seed + turn_idx, 
                total_input_tokens
            )
            accumulated_tokens = prompt_tokens.copy()
        else:
            # Subsequent turns: reuse prefix from accumulated context
            # cache_read tells us how many tokens from previous context are reused
            if turn_template.cache_read <= len(accumulated_tokens):
                # Take exact prefix from accumulated context
                cached_prefix = accumulated_tokens[:turn_template.cache_read]
            else:
                # Shouldn't happen, but handle gracefully
                cached_prefix = accumulated_tokens.copy()
            
            # Generate new tokens to append (user input + tool results)
            new_tokens = generate_unique_token_sequence(
                base_seed + turn_idx, 
                turn_template.new_input
            )
            
            prompt_tokens = cached_prefix + new_tokens
            
            # Update accumulated context with new input
            accumulated_tokens = prompt_tokens.copy()
        
        # Create turn execution
        turn_exec = TurnExecution(
            turn_idx=turn_idx,
            prompt_token_ids=prompt_tokens,
            target_output_tokens=turn_template.output,
            tool_wait_time=turn_template.total_tool_time_with_overlap,
            tool_names=[t.name for t in turn_template.tool_executions]
        )
        turns.append(turn_exec)
        
        # After this turn executes, output tokens would be generated
        # Add placeholder output tokens to accumulated context for next turn
        # (These won't be used as input, but affect the prefix length calculation)
        if turn_template.output > 0:
            output_tokens = generate_unique_token_sequence(
                base_seed + turn_idx + 5000,  # Different seed for outputs
                turn_template.output
            )
            accumulated_tokens.extend(output_tokens)
    
    return RequestTrajectory(
        request_id=request_id,
        template_id=template.template_id,
        source_template=template,  # Store reference
        turns=turns
    )


def generate_request_trajectories(
    templates: List[TraceTemplate], 
    num_requests: int,
    base_seed: int = 42
) -> List[RequestTrajectory]:
    """
    Generate N request trajectories by sampling from templates.
    
    Each request gets:
    - A sampled template (with replacement)
    - Unique token ID sequences (no cross-request cache hits)
    - Full turn-by-turn execution plan
    
    Args:
        templates: List of extracted templates
        num_requests: Number of requests to generate
        base_seed: Random seed for reproducibility
        
    Returns:
        List of RequestTrajectory objects ready for execution
    """
    rng = np.random.RandomState(base_seed)
    trajectories = []
    
    print(f"\nGenerating {num_requests} request trajectories from {len(templates)} templates...")
    
    # Track template usage for diversity statistics
    from collections import Counter
    template_usage = Counter()
    
    for i in range(num_requests):
        # Sample a template (with replacement)
        template = rng.choice(templates)
        template_usage[template.template_id] += 1
        
        # Generate unique trajectory
        request_id = f"req_{i:06d}"
        # Use request index * 10000 as base seed to ensure non-overlapping token sequences
        trajectory = instantiate_trajectory_from_template(
            template, 
            request_id, 
            base_seed=base_seed + i * 10000
        )
        
        trajectories.append(trajectory)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_requests} trajectories...")
    
    # Calculate template diversity statistics
    unique_templates_used = len(template_usage)
    max_usage = max(template_usage.values())
    min_usage = min(template_usage.values())
    avg_usage = np.mean(list(template_usage.values()))
    
    print(f"✓ Generated {num_requests} trajectories")
    print(f"  Avg turns per request: {np.mean([len(t.turns) for t in trajectories]):.1f}")
    print(f"  Avg tokens per request: {np.mean([sum(len(turn.prompt_token_ids) for turn in t.turns) for t in trajectories]):.1f}")
    print(f"\n  Template diversity:")
    print(f"    Available templates: {len(templates)}")
    print(f"    Unique templates used: {unique_templates_used} ({unique_templates_used/len(templates)*100:.1f}%)")
    print(f"    Usage per template: min={min_usage}, max={max_usage}, avg={avg_usage:.1f}")
    
    # Show top 10 most-used templates
    if unique_templates_used > 0:
        print(f"\n    Top templates used:")
        for template_id, count in template_usage.most_common(min(10, unique_templates_used)):
            percentage = (count / num_requests) * 100
            print(f"      {template_id}: {count} times ({percentage:.1f}%)")
    
    if unique_templates_used < len(templates):
        unused = len(templates) - unique_templates_used
        print(f"\n    ⚠ Note: {unused} templates were not sampled (increase num_requests for more diversity)")
    
    return trajectories

def trajectory_to_dag_nodes(trajectory: RequestTrajectory, 
                                annotate: bool = True, 
                                prefill_only: bool = False):
    """
    Convert a RequestTrajectory into a list of Node objects for DAG execution.
    
    Creates a sequential chain:
    LLMCall_0 -> [ToolCall_1, ToolCall_2, ...] -> LLMCall_1 -> [Tools] -> ...
    
    Args:
        trajectory: RequestTrajectory to convert
        annotate: Whether to annotate with expected durations (default True)
    
    Returns:
        List of nodes in dependency order
    """
    from node import LLMCallNode, ToolCallNode
    
    nodes = []
    prev_llm_node = None
    
    for turn in trajectory.turns:
        # Create LLMCall node with raw token IDs
        llm_node = LLMCallNode(
            prompt_template=None,  # Not used with token IDs
            prompt_token_ids=turn.prompt_token_ids,
            target_output_tokens=1 if prefill_only else turn.target_output_tokens,
            turn_idx=turn.turn_idx,
            llm_name=f"turn_{turn.turn_idx}",
            name=f"{trajectory.request_id}_llm_{turn.turn_idx}",
        )
        
        # Link to previous LLM node (sequential dependency)
        if prev_llm_node is not None:
            llm_node.add_input("prev_turn", prev_llm_node)
        
        nodes.append(llm_node)
        
        # Create tool nodes after this LLM call
        # Tools execute in parallel, all depending on the LLM call
        for tool_idx, tool_name in enumerate(turn.tool_names):
            # Use total wait time divided by number of tools as approximation
            # (actual overlap is computed in template, but we need per-tool time here)
            tool_duration = turn.tool_wait_time / len(turn.tool_names) if turn.tool_names else 0.0
            
            tool_node = ToolCallNode(
                tool_fn=lambda **kwargs: f"Tool {tool_name} result",
                name=f"{trajectory.request_id}_tool_{turn.turn_idx}_{tool_idx}",
                expected_time=tool_duration
            )
            
            # Tool depends on LLM call
            tool_node.add_input("llm_output", llm_node)
            nodes.append(tool_node)
        
        prev_llm_node = llm_node
    
    # Annotate with expected durations for scheduler hints
    # For Claude traces, we want immediate tool wait time, not cumulative downstream
    if annotate:
        # Custom annotation for sequential multi-turn traces
        # Each LLM node should only know about its immediate tool wait time
        for node in nodes:
            if hasattr(node, 'prompt_token_ids'):  # LLM node
                # Find immediate downstream tool nodes
                immediate_tools = [n for n in node.downstream if hasattr(n, 'expected_time')]
                
                if immediate_tools:
                    # Use max tool time (they execute in parallel)
                    max_tool_time = max(t.expected_time for t in immediate_tools)
                    node.metadata["kv_reuse_expected_duration_s"] = max_tool_time
                else:
                    # No tools after this turn - it's a leaf (final turn)
                    node.metadata["kv_reuse_expected_duration_s"] = 9999999.0
            else:  # Tool node
                # Tools don't need this metadata
                node.metadata["kv_reuse_expected_duration_s"] = 0.0
    
    return nodes


def generate_claude_trace_workload(
    num_requests: int,
    claude_dataset_dir: str = "/vllm/vllm/elasticswap/toolcall_dataset_claude",
    arrival_rate: float = 0.5,
    seed: int = 42,
    trace_filter_percentile: float = 90.0,
    prefill_only: bool = False
):
    """
    Generate workload from Claude traces for use with pipeline.py.
    
    Args:
        num_requests: Number of requests to generate
        claude_dataset_dir: Directory with Claude trace JSON files
        arrival_rate: Poisson arrival rate (requests per second)
        seed: Random seed
        trace_filter_percentile: Percentile for filtering outlier traces
    
    Returns:
        Tuple of (dags, dag_names, arrival_times, requests_meta) for run_dags_with_arrival_times
    """
    # Extract templates
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir,
        trace_filter_percentile=trace_filter_percentile
    )
    
    # Generate trajectories (returns trajectories with template usage tracked)
    trajectories = generate_request_trajectories(templates, num_requests, base_seed=seed)
    
    # Collect template usage statistics for metadata
    from collections import Counter
    template_usage_stats = Counter()
    for traj in trajectories:
        template_usage_stats[traj.template_id] += 1
    
    # Convert to DAGs with unique request IDs and build metadata
    dags = []
    dag_names = []
    requests_meta = []
    
    for i, traj in enumerate(trajectories):
        dag = trajectory_to_dag_nodes(traj, annotate=True, prefill_only=prefill_only)
        dags.append(dag)
        dag_names.append(traj.request_id)
        
        # Build metadata for this request
        total_tokens = sum(len(turn.prompt_token_ids) for turn in traj.turns)
        total_output_tokens = sum(turn.target_output_tokens for turn in traj.turns)
        total_tools = sum(len(turn.tool_names) for turn in traj.turns)
        
        # Per-turn breakdown
        turn_info = []
        for turn in traj.turns:
            turn_info.append({
                'turn_idx': turn.turn_idx,
                'input_tokens': len(turn.prompt_token_ids),
                'output_tokens': turn.target_output_tokens,
                'num_tools': len(turn.tool_names),
                'tool_wait_time': turn.tool_wait_time,
                'tool_names': turn.tool_names
            })
        
        requests_meta.append({
            'id': i,
            'request_id': traj.request_id,
            'template_id': traj.template_id,
            'num_turns': len(traj.turns),
            'total_input_tokens': total_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tools': total_tools,
            'turns': turn_info
        })
    
    # Generate Poisson arrival times
    rng = np.random.RandomState(seed)
    inter_arrival_times = rng.exponential(scale=1.0/arrival_rate, size=num_requests)
    arrival_times = list(inter_arrival_times)
    
    # Compute determinism verification hash
    import hashlib
    hash_data = []
    for i, traj in enumerate(trajectories):
        for turn in traj.turns:
            # Hash the token IDs to verify determinism
            hash_data.extend(turn.prompt_token_ids)
    hash_data.extend([int(t * 1000000) for t in arrival_times])  # Include arrival times
    hash_bytes = np.array(hash_data, dtype=np.int64).tobytes()
    workload_hash = hashlib.sha256(hash_bytes).hexdigest()[:16]
    
    print(f"\n✓ Generated Claude trace workload:")
    print(f"  Requests: {num_requests}")
    print(f"  Arrival rate: {arrival_rate} req/s")
    print(f"  Seed: {seed}")
    print(f"  Total duration: {sum(arrival_times):.1f}s")
    print(f"  Avg turns per request: {np.mean([len(t.turns) for t in trajectories]):.1f}")
    print(f"  Avg tokens per request: {np.mean([m['total_input_tokens'] for m in requests_meta]):.0f}")
    print(f"  Total DAG nodes: {sum(len(dag) for dag in dags)}")
    print(f"  Determinism hash: {workload_hash}")
    print(f"    (This hash should be identical across runs with same seed)")
    
    return dags, dag_names, arrival_times, requests_meta


def main():
    # Extract templates
    templates = extract_trace_templates_with_evolution("/vllm/vllm/elasticswap/toolcall_dataset_claude")
    print(f"\nExtracted {len(templates)} templates total\n")
    
    # Show first template as example
    if templates:
        templates[0].print_sequence()
    
    # Generate synthetic trajectories
    trajectories = generate_request_trajectories(templates, num_requests=10, base_seed=42)
    
    # Show first trajectory
    print("\n" + "="*70)
    print("EXAMPLE GENERATED TRAJECTORY")
    print("="*70)
    traj = trajectories[0]
    print(f"\n{traj}")
    
    # Show the source template for comparison
    print("\n" + "-"*70)
    print("SOURCE TEMPLATE:")
    print("-"*70)
    for i, turn in enumerate(traj.source_template.turns):
        tools_info = f"{len(turn.tool_executions)} tools, wait={turn.total_tool_time_with_overlap:.3f}s"
        print(f"  Turn {i}: tokens={turn.cache_read + turn.new_input}, output={turn.output}, {tools_info}")
    print("-"*70 + "\n")
    
    # Show generated trajectory details
    for turn in traj.turns:
        print(f"\nTurn {turn.turn_idx}:")
        print(f"  Prompt tokens: {len(turn.prompt_token_ids)} tokens (first 10: {turn.prompt_token_ids[:10]})")
        print(f"  Target output: {turn.target_output_tokens} tokens")
        print(f"  Tool wait time: {turn.tool_wait_time:.3f}s")
        if turn.tool_names:
            print(f"  Tools: {', '.join(turn.tool_names)}")

if __name__ == "__main__":
    main()