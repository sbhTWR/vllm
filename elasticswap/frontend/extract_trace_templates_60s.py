# Extract trace templates from 60s-annotated Claude dataset
# Maps <60s -> "persist" and >=60s -> "evict"

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import glob
import json
from collections import defaultdict
from dateutil import parser as date_parser

@dataclass
class ModelPrediction:
    """Model prediction for a tool execution"""
    predicted_label: str  # "evict" or "persist"
    probabilities: Dict[str, float]  # {"persist": 0.x, "evict": 0.y}

@dataclass
class ToolExecution:
    """Individual tool execution with timing and model prediction"""
    name: str
    start_timestamp: str
    duration: float  # seconds
    model_prediction: Optional[ModelPrediction] = None
    
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
                    pred_str = ""
                    if tool_exec.model_prediction:
                        pred_str = f" | pred={tool_exec.model_prediction.predicted_label}"
                    print(f"    └─→ {j}. {tool_exec.name}(duration={tool_exec.duration:.3f}s{pred_str})")
            print()  # Blank line between turns


def map_duration_bucket_to_prediction(duration_bucket_data: dict) -> Optional[ModelPrediction]:
    """
    Map duration bucket prediction to persist/evict prediction.
    
    Maps:
    - "<60s" -> "persist"
    - ">=60s" -> "evict"
    
    Args:
        duration_bucket_data: Dict with 'predicted_bucket' and 'probabilities'
    
    Returns:
        ModelPrediction with persist/evict labels, or None if invalid
    """
    if not duration_bucket_data:
        return None
    
    predicted_bucket = duration_bucket_data.get('predicted_bucket')
    bucket_probs = duration_bucket_data.get('probabilities', {})
    
    if not predicted_bucket or not bucket_probs:
        return None
    
    # Map buckets to persist/evict
    if predicted_bucket == '<60s':
        predicted_label = 'persist'
    elif predicted_bucket == '>=60s':
        predicted_label = 'evict'
    else:
        # Unknown bucket, skip
        return None
    
    # Map probabilities
    persist_prob = bucket_probs.get('<60s', 0.0)
    evict_prob = bucket_probs.get('>=60s', 0.0)
    
    return ModelPrediction(
        predicted_label=predicted_label,
        probabilities={
            'persist': persist_prob,
            'evict': evict_prob
        }
    )


def split_session_into_time_windows(session_events: List[dict], window_duration_seconds: float = 600) -> List[List[dict]]:
    """
    Split a session into multiple sub-sessions based on time windows from the first LLM call.
    
    This ensures that when running experiments with a fixed duration (e.g., 10 minutes),
    all kinds of patterns from longer traces will appear.
    
    Args:
        session_events: List of events for a single session (must be sorted by timestamp)
        window_duration_seconds: Duration of each window in seconds (default: 600s = 10min)
    
    Returns:
        List of sub-session event lists, each representing a time window
    """
    if not session_events:
        return []
    
    # Find the first LLM call to use as reference point
    first_llm_time = None
    for event in session_events:
        if event['event_type'] == 'llm_call':
            first_llm_time = date_parser.parse(event['timestamp']).timestamp()
            break
    
    if first_llm_time is None:
        # No LLM calls in this session, return as-is
        return [session_events] if session_events else []
    
    # Split events into time windows
    sub_sessions = []
    current_window_idx = 0
    current_sub_session = []
    
    for event in session_events:
        event_time = date_parser.parse(event['timestamp']).timestamp()
        time_since_start = event_time - first_llm_time
        
        # Determine which window this event belongs to
        event_window_idx = int(time_since_start / window_duration_seconds)
        
        # If we've moved to a new window, save the current sub-session
        if event_window_idx > current_window_idx:
            if current_sub_session:
                sub_sessions.append(current_sub_session)
            current_sub_session = []
            current_window_idx = event_window_idx
        
        current_sub_session.append(event)
    
    # Don't forget the last sub-session
    if current_sub_session:
        sub_sessions.append(current_sub_session)
    
    return sub_sessions


def extract_trace_templates_with_evolution(
    claude_dataset_dir: str,
    window_duration_minutes: float = 10,      # Split traces into N-minute windows
    max_trace_duration_minutes: float = None  # No longer used, kept for compatibility
) -> List[TraceTemplate]:
    """
    Extract trace templates from Claude tool call dataset with 60s bucket predictions.
    
    This version maps duration bucket predictions to persist/evict labels:
    - "<60s" -> "persist"
    - ">=60s" -> "evict"
    
    Args:
        claude_dataset_dir: Directory with *_events.json files (60s annotated)
        window_duration_minutes: Duration of each window in minutes (default 10 min)
        max_trace_duration_minutes: Deprecated, kept for compatibility
    
    Returns:
        List of TraceTemplate objects with turn-by-turn token evolution and persist/evict predictions
    """
    json_files = sorted(glob.glob(f"{claude_dataset_dir}/*_events.json"))
    
    print(f"Loading 60s-annotated traces from {claude_dataset_dir}...")
    print(f"Mapping: <60s -> persist, >=60s -> evict")
    
    # Collect all tool durations for statistics
    all_tool_durations = []
    all_predictions = []
    for json_file in json_files:
        with open(json_file) as f:
            events = json.load(f)
        
        for event in events:
            if event['event_type'] == 'tool_call':
                exec_time = event.get('execution_time_ms')
                if exec_time is not None and exec_time > 0:
                    all_tool_durations.append(exec_time / 1000.0)
                
                # Track model predictions (after mapping)
                duration_bucket = event.get('duration_bucket_prediction')
                if duration_bucket:
                    model_pred = map_duration_bucket_to_prediction(duration_bucket)
                    if model_pred:
                        all_predictions.append(model_pred.predicted_label)
    
    # Print statistics
    if all_tool_durations:
        print(f"\n  Total tools across all traces: {len(all_tool_durations)}")
        print(f"  Duration p50: {np.percentile(all_tool_durations, 50):.3f}s")
        print(f"  Duration p75: {np.percentile(all_tool_durations, 75):.3f}s")
        print(f"  Duration p90: {np.percentile(all_tool_durations, 90):.3f}s")
        print(f"  Duration p95: {np.percentile(all_tool_durations, 95):.3f}s")
        print(f"  Duration p99: {np.percentile(all_tool_durations, 99):.3f}s")
        print(f"  Duration max: {np.max(all_tool_durations):.3f}s")
    
    if all_predictions:
        from collections import Counter
        pred_counts = Counter(all_predictions)
        print(f"\n  Model prediction distribution (after mapping):")
        for label, count in sorted(pred_counts.items()):
            print(f"    {label}: {count} ({count/len(all_predictions)*100:.1f}%)")
    
    print(f"\n  Note: NO filtering based on tool duration (keeping all traces)")
    
    # Extract templates without filtering
    templates = []
    total_traces = 0
    traces_split = 0
    window_duration_seconds = window_duration_minutes * 60
    
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
        
        # Process each session (with splitting into time windows)
        # Sort sessions by ID to ensure deterministic order
        for session_id, session_events in sorted(sessions.items()):
            if not session_events:
                continue
            
            # Split session into fixed time windows from first LLM call
            sub_sessions = split_session_into_time_windows(session_events, window_duration_seconds)
            
            if len(sub_sessions) > 1:
                traces_split += 1
            
            # Process each sub-session as a separate trace
            for sub_idx, sub_session_events in enumerate(sub_sessions):
                if not sub_session_events:
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
                            
                            # Extract and map duration bucket prediction
                            model_pred = None
                            duration_bucket = event.get('duration_bucket_prediction')
                            if duration_bucket:
                                model_pred = map_duration_bucket_to_prediction(duration_bucket)
                            
                            # No artificial cap - preserve actual tool durations
                            tool_exec = ToolExecution(
                                name=event['tool_name'],
                                start_timestamp=event['timestamp'],
                                duration=duration_seconds,
                                model_prediction=model_pred
                            )
                            current_tools.append(tool_exec)
                
                # Don't forget the last turn
                if current_turn is not None:
                    current_turn.tool_executions = current_tools
                    turns.append(current_turn)
                
                if turns:
                    total_traces += 1
                    
                    # NO FILTERING - keep all traces
                    # Use sub_idx to create unique template_id for split traces
                    template_id_suffix = f"-{sub_idx}" if len(sub_sessions) > 1 else ""
                    template = TraceTemplate(
                        template_id=f"{session_id[:8]}{template_id_suffix}",
                        session_id=session_id,
                        turns=turns
                    )
                    templates.append(template)
    
    print(f"\n  ===== Trace Processing Summary =====")
    print(f"  Original sessions split into windows: {traces_split}")
    print(f"  Total sub-traces after splitting: {total_traces}")
    print(f"  Templates extracted: {len(templates)}")
    print(f"  Window duration used: {window_duration_minutes} minutes ({window_duration_seconds}s)")
    print(f"  Note: Each window starts from the first LLM call of the original session")
    return templates

@dataclass
class TurnExecution:
    """Concrete execution of a single turn with actual token IDs"""
    turn_idx: int
    prompt_token_ids: List[int]  # Full context for this turn
    target_output_tokens: int    # Expected output length
    tool_wait_time: float        # Time to wait after this turn (wall-clock)
    tool_names: List[str]        # Tool names for reference
    tool_predictions: List[Optional[ModelPrediction]]  # Model predictions for each tool

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
    Also preserves model predictions from the template.
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
        
        # Extract tool predictions from template
        tool_predictions = [t.model_prediction for t in turn_template.tool_executions]
        
        # Create turn execution
        turn_exec = TurnExecution(
            turn_idx=turn_idx,
            prompt_token_ids=prompt_tokens,
            target_output_tokens=turn_template.output,
            tool_wait_time=turn_template.total_tool_time_with_overlap,
            tool_names=[t.name for t in turn_template.tool_executions],
            tool_predictions=tool_predictions
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
    - Model predictions from the template
    
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


def _aggregate_predictions(predictions: List[Optional[ModelPrediction]], strategy: str = 'avg_probabilities') -> Optional[Dict]:
    """
    Aggregate tool predictions using the specified strategy.
    
    Args:
        predictions: List of ModelPrediction objects (may contain None)
        strategy: Aggregation strategy to use
            - 'avg_probabilities': Average probabilities, select max (default)
            - 'majority_vote': Most common label
            - 'conservative': Prefer persist > evict
            - 'aggressive': Prefer evict > persist
            - 'max_confidence': Use prediction with highest confidence
    
    Returns:
        Dict with aggregated prediction or None if no valid predictions
    """
    # Filter out None predictions
    valid_preds = [p for p in predictions if p is not None]
    if not valid_preds:
        return None
    
    from collections import Counter
    
    if strategy == 'majority_vote':
        # Majority vote on labels
        labels = [p.predicted_label for p in valid_preds]
        label_counts = Counter(labels)
        aggregated_label = label_counts.most_common(1)[0][0]
        
        # Average probabilities
        avg_probs = {
            'persist': sum(p.probabilities['persist'] for p in valid_preds) / len(valid_preds),
            'evict': sum(p.probabilities['evict'] for p in valid_preds) / len(valid_preds)
        }
        
        return {
            'predicted_label': aggregated_label,
            'probabilities': avg_probs,
            'aggregation_method': 'majority_vote',
            'num_tools_aggregated': len(valid_preds)
        }
    
    elif strategy == 'conservative':
        # Prefer persist > evict
        labels = [p.predicted_label for p in valid_preds]
        
        if 'persist' in labels:
            aggregated_label = 'persist'
        else:
            aggregated_label = 'evict'
        
        avg_probs = {
            'persist': sum(p.probabilities['persist'] for p in valid_preds) / len(valid_preds),
            'evict': sum(p.probabilities['evict'] for p in valid_preds) / len(valid_preds)
        }
        
        return {
            'predicted_label': aggregated_label,
            'probabilities': avg_probs,
            'aggregation_method': 'conservative',
            'num_tools_aggregated': len(valid_preds)
        }
    
    elif strategy == 'aggressive':
        # Prefer evict > persist
        labels = [p.predicted_label for p in valid_preds]
        
        if 'evict' in labels:
            aggregated_label = 'evict'
        else:
            aggregated_label = 'persist'
        
        avg_probs = {
            'persist': sum(p.probabilities['persist'] for p in valid_preds) / len(valid_preds),
            'evict': sum(p.probabilities['evict'] for p in valid_preds) / len(valid_preds)
        }
        
        return {
            'predicted_label': aggregated_label,
            'probabilities': avg_probs,
            'aggregation_method': 'aggressive',
            'num_tools_aggregated': len(valid_preds)
        }
    
    elif strategy == 'max_confidence':
        # Use prediction with highest confidence
        max_conf_pred = max(valid_preds, key=lambda p: p.probabilities[p.predicted_label])
        
        return {
            'predicted_label': max_conf_pred.predicted_label,
            'probabilities': max_conf_pred.probabilities,
            'aggregation_method': 'max_confidence',
            'num_tools_aggregated': len(valid_preds),
            'max_confidence': max_conf_pred.probabilities[max_conf_pred.predicted_label]
        }
    
    else:  # Default: 'avg_probabilities'
        # Average probabilities
        avg_probs = {
            'persist': sum(p.probabilities['persist'] for p in valid_preds) / len(valid_preds),
            'evict': sum(p.probabilities['evict'] for p in valid_preds) / len(valid_preds)
        }
        
        # Select label with highest average probability
        aggregated_label = max(avg_probs.items(), key=lambda x: x[1])[0]
        
        return {
            'predicted_label': aggregated_label,
            'probabilities': avg_probs,
            'aggregation_method': 'avg_probabilities',
            'num_tools_aggregated': len(valid_preds)
        }


def compute_predicted_tool_duration(tool_predictions: List[Optional[ModelPrediction]], 
                                   persist_bin_avg: float = 30.0,
                                   evict_bin_avg: float = 120.0) -> float:
    """
    Compute total predicted duration by summing individual tool expected durations.
    
    For each tool:
        expected_duration = P(persist) * persist_bin_avg + P(evict) * evict_bin_avg
    
    Total duration = sum of all individual expected durations
    
    Args:
        tool_predictions: List of ModelPrediction objects for each tool
        persist_bin_avg: Average duration for persist bin (<60s), default 30s
        evict_bin_avg: Average duration for evict bin (>=60s), default 120s
    
    Returns:
        Total predicted duration in seconds
    """
    total_duration = 0.0
    
    for pred in tool_predictions:
        if pred is not None:
            # Expected value for this specific tool
            persist_prob = pred.probabilities.get('persist', 0.0)
            evict_prob = pred.probabilities.get('evict', 0.0)
            
            expected_duration = persist_prob * persist_bin_avg + evict_prob * evict_bin_avg
            total_duration += expected_duration
    
    return total_duration


def trajectory_to_dag_nodes(trajectory: RequestTrajectory, 
                                annotate: bool = True, 
                                prefill_only: bool = False,
                                aggregation_strategy: str = 'avg_probabilities',
                                oracle: bool = True,
                                persist_bin_avg: float = 30.0,
                                evict_bin_avg: float = 120.0):
    """
    Convert a RequestTrajectory into a list of Node objects for DAG execution.
    
    Creates a sequential chain:
    LLMCall_0 -> [ToolCall_1, ToolCall_2, ...] -> LLMCall_1 -> [Tools] -> ...
    
    IMPORTANT: 
    - If an LLM call has no tools, a dummy tool node (duration=0) is inserted
    - If an LLM call has no tools, dummy prediction (persist=1.0) is added
    - For the last LLM call, dummy prediction (evict=1.0) is added if no tools
    
    Args:
        trajectory: RequestTrajectory to convert
        annotate: Whether to annotate with expected durations (default True)
        prefill_only: If True, only generate prefill requests (1 output token)
        aggregation_strategy: Strategy for aggregating tool predictions (for labels)
            Options: 'avg_probabilities', 'majority_vote', 'conservative', 'aggressive', 'max_confidence'
        oracle: If True, use actual tool execution times (ORACLE - ground truth)
                If False, use ML predictions from model (REALISTIC scenario)
        persist_bin_avg: Average duration for persist bin (<60s), default 30s
        evict_bin_avg: Average duration for evict bin (>=60s), default 120s
    
    Returns:
        List of nodes in dependency order
    """
    from node import LLMCallNode, ToolCallNode
    
    nodes = []
    prev_turn_tool_nodes = []  # Track tool nodes from previous turn
    num_turns = len(trajectory.turns)
    
    for turn_idx, turn in enumerate(trajectory.turns):
        is_last_turn = (turn_idx == num_turns - 1)
        has_tools = len(turn.tool_names) > 0
        
        # Create LLMCall node with raw token IDs
        llm_node = LLMCallNode(
            prompt_template=None,  # Not used with token IDs
            prompt_token_ids=turn.prompt_token_ids,
            target_output_tokens=1 if prefill_only else turn.target_output_tokens,
            turn_idx=turn.turn_idx,
            llm_name=f"turn_{turn.turn_idx}",
            name=f"{trajectory.request_id}_llm_{turn.turn_idx}",
        )
        
        # Link to previous turn's tool nodes (if they exist)
        # This ensures proper dependency: Tools must complete before next LLM call
        if prev_turn_tool_nodes:
            for tool_node in prev_turn_tool_nodes:
                llm_node.add_input(f"prev_tool_{tool_node.name}", tool_node)
        
        # Add aggregated prediction metadata to LLM node
        if has_tools:
            # Regular case: has tools, aggregate their predictions
            aggregated_pred = _aggregate_predictions(turn.tool_predictions, aggregation_strategy)
            if aggregated_pred:
                llm_node.metadata["aggregated_tool_prediction"] = aggregated_pred
        else:
            # No tools - add dummy prediction
            if is_last_turn:
                # Last turn without tools -> evict=1.0
                llm_node.metadata["aggregated_tool_prediction"] = {
                    'predicted_label': 'evict',
                    'probabilities': {
                        'persist': 0.0,
                        'evict': 1.0
                    },
                    'aggregation_method': 'dummy_last_turn',
                    'num_tools_aggregated': 0
                }
            else:
                # Non-last turn without tools -> persist=1.0
                llm_node.metadata["aggregated_tool_prediction"] = {
                    'predicted_label': 'persist',
                    'probabilities': {
                        'persist': 1.0,
                        'evict': 0.0
                    },
                    'aggregation_method': 'dummy_no_tools',
                    'num_tools_aggregated': 0
                }
        
        nodes.append(llm_node)
        
        # Create ONE combined tool node per turn (not individual nodes per tool)
        # This simplifies the DAG while preserving wall-clock execution time
        current_turn_tool_nodes = []
        
        if has_tools:
            # Regular case: create single combined tool node for all tools in this turn
            # The duration is the total wall-clock time (already accounts for parallel execution)
            combined_tool_node = ToolCallNode(
                tool_fn=lambda **kwargs: f"Combined tools result",
                name=f"{trajectory.request_id}_tools_{turn.turn_idx}",
                expected_time=turn.tool_wait_time  # Wall-clock time with overlap
            )
            
            # Add metadata about the combined tools
            combined_tool_node.metadata["num_tools"] = len(turn.tool_names)
            combined_tool_node.metadata["tool_names"] = turn.tool_names
            combined_tool_node.metadata["is_combined"] = True
            
            # Tool depends on LLM call
            combined_tool_node.add_input("llm_output", llm_node)
            nodes.append(combined_tool_node)
            current_turn_tool_nodes.append(combined_tool_node)
        else:
            # No tools - insert dummy tool node with duration=0
            dummy_tool_node = ToolCallNode(
                tool_fn=lambda **kwargs: "Dummy tool result",
                name=f"{trajectory.request_id}_tools_{turn.turn_idx}_dummy",
                expected_time=0.0
            )
            
            # Mark as dummy in metadata
            dummy_tool_node.metadata["is_dummy"] = True
            dummy_tool_node.metadata["num_tools"] = 0
            dummy_tool_node.metadata["tool_names"] = []
            
            # Tool depends on LLM call
            dummy_tool_node.add_input("llm_output", llm_node)
            nodes.append(dummy_tool_node)
            current_turn_tool_nodes.append(dummy_tool_node)
        
        # Update for next iteration
        prev_turn_tool_nodes = current_turn_tool_nodes
    
    # Annotate with expected durations for scheduler hints
    # For Claude traces, we want immediate tool wait time, not cumulative downstream
    if annotate:
        # First, identify all LLM nodes and find the last one
        llm_nodes = [n for n in nodes if hasattr(n, 'prompt_token_ids')]
        last_llm_node = llm_nodes[-1] if llm_nodes else None
        
        # Custom annotation for sequential multi-turn traces
        # Each LLM node should only know about its immediate tool wait time
        for idx, node in enumerate(llm_nodes):
            turn = trajectory.turns[idx]
            is_final = (node is last_llm_node)
            
            # Find immediate downstream tool nodes (excluding dummy tools)
            immediate_tools = [n for n in node.downstream if hasattr(n, 'expected_time')]
            real_tools = [t for t in immediate_tools if not t.metadata.get('is_dummy', False)]
            
            if is_final:
                # Final LLM call: KV cache won't be reused after this, even if there are tools
                node.metadata["kv_reuse_expected_duration_s"] = 9999999.0
                
            elif oracle:
                # ORACLE MODE: Use ground truth (actual tool execution time)
                # This is what a perfect predictor would know
                if real_tools:
                    max_tool_time = max(t.expected_time for t in real_tools)
                    node.metadata["kv_reuse_expected_duration_s"] = max_tool_time
                else:
                    node.metadata["kv_reuse_expected_duration_s"] = 0.0
                    
            else:
                # PREDICTION MODE: Sum of per-tool expected durations
                # This is what's available in a realistic deployment
                if len(turn.tool_predictions) > 0:
                    # Use the new per-tool aggregation method
                    predicted_duration = compute_predicted_tool_duration(
                        turn.tool_predictions,
                        persist_bin_avg=persist_bin_avg,
                        evict_bin_avg=evict_bin_avg
                    )
                    node.metadata["kv_reuse_expected_duration_s"] = predicted_duration
                    
                    # Optional: Store for debugging
                    node.metadata["predicted_duration_breakdown"] = {
                        'num_tools': len(turn.tool_predictions),
                        'total_predicted': predicted_duration,
                        'persist_bin_avg': persist_bin_avg,
                        'evict_bin_avg': evict_bin_avg
                    }
                else:
                    # No tools - immediate reuse
                    node.metadata["kv_reuse_expected_duration_s"] = 0.0
        
        # Tool nodes don't need kv_reuse hints
        for node in nodes:
            if not hasattr(node, 'prompt_token_ids'):
                node.metadata["kv_reuse_expected_duration_s"] = 0.0
    
    return nodes


def generate_claude_trace_workload(
    num_requests: int,
    claude_dataset_dir: str = "/vllm/vllm/elasticswap/toolcall_dataset_claude_annotated_60s",
    arrival_rate: float = 0.5,
    seed: int = 42,
    prefill_only: bool = False,
    window_duration_minutes: float = 10,
    aggregation_strategy: str = 'avg_probabilities',
    oracle: bool = True,
    persist_bin_avg: float = 30.0,
    evict_bin_avg: float = 120.0
):
    """
    Generate workload from 60s-annotated Claude traces for use with pipeline.py.
    
    This version maps duration bucket predictions to persist/evict labels:
    - "<60s" -> "persist" 
    - ">=60s" -> "evict"
    
    Args:
        num_requests: Number of requests to generate
        claude_dataset_dir: Directory with 60s-annotated Claude trace JSON files
        arrival_rate: Poisson arrival rate (requests per second)
        seed: Random seed
        prefill_only: If True, only generate prefill requests (1 output token)
        window_duration_minutes: Duration of each window in minutes (default 10)
        aggregation_strategy: Strategy for aggregating parallel tool predictions (for labels)
            Options: 'avg_probabilities', 'majority_vote', 'conservative', 'aggressive', 'max_confidence'
        oracle: If True, use actual tool execution times (ORACLE - ground truth upper bound)
                If False, use per-tool ML predictions (REALISTIC - practical performance)
        persist_bin_avg: Average duration for persist bin (<60s), default 30s
        evict_bin_avg: Average duration for evict bin (>=60s), default 120s
    
    Returns:
        Tuple of (dags, dag_names, arrival_times, requests_meta) for run_dags_with_arrival_times
    """
    # Extract templates (no percentile filtering, window-based splitting)
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir,
        window_duration_minutes=window_duration_minutes
    )
    
    # Generate trajectories (returns trajectories with template usage tracked)
    trajectories = generate_request_trajectories(templates, num_requests, base_seed=seed)
    
    # Collect template usage statistics for metadata
    from collections import Counter
    template_usage_stats = Counter()
    prediction_stats = Counter()
    
    for traj in trajectories:
        template_usage_stats[traj.template_id] += 1
        # Count predictions across all tools
        for turn in traj.turns:
            for pred in turn.tool_predictions:
                if pred:
                    prediction_stats[pred.predicted_label] += 1
    
    # Convert to DAGs with unique request IDs and build metadata
    dags = []
    dag_names = []
    requests_meta = []
    
    for i, traj in enumerate(trajectories):
        dag = trajectory_to_dag_nodes(
            traj, 
            annotate=True, 
            prefill_only=prefill_only, 
            aggregation_strategy=aggregation_strategy,
            oracle=oracle,
            persist_bin_avg=persist_bin_avg,
            evict_bin_avg=evict_bin_avg
        )
        dags.append(dag)
        dag_names.append(traj.request_id)
        
        # Build metadata for this request
        total_tokens = sum(len(turn.prompt_token_ids) for turn in traj.turns)
        total_output_tokens = sum(turn.target_output_tokens for turn in traj.turns)
        total_tools = sum(len(turn.tool_names) for turn in traj.turns)
        
        # Per-turn breakdown with predictions
        turn_info = []
        for turn in traj.turns:
            tool_pred_info = []
            for tool_name, pred in zip(turn.tool_names, turn.tool_predictions):
                pred_dict = None
                if pred:
                    pred_dict = {
                        'predicted_label': pred.predicted_label,
                        'probabilities': pred.probabilities
                    }
                tool_pred_info.append({
                    'tool_name': tool_name,
                    'prediction': pred_dict
                })
            
            turn_info.append({
                'turn_idx': turn.turn_idx,
                'input_tokens': len(turn.prompt_token_ids),
                'output_tokens': turn.target_output_tokens,
                'num_tools': len(turn.tool_names),
                'tool_wait_time': turn.tool_wait_time,
                'tools': tool_pred_info
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
    
    mode_str = "ORACLE (ground truth)" if oracle else f"PREDICTED (persist={persist_bin_avg}s, evict={evict_bin_avg}s)"
    print(f"\n✓ Generated Claude trace workload ({mode_str}):")
    print(f"  Requests: {num_requests}")
    print(f"  Arrival rate: {arrival_rate} req/s")
    print(f"  Seed: {seed}")
    print(f"  Total duration: {sum(arrival_times):.1f}s")
    print(f"  Avg turns per request: {np.mean([len(t.turns) for t in trajectories]):.1f}")
    print(f"  Avg tokens per request: {np.mean([m['total_input_tokens'] for m in requests_meta]):.0f}")
    print(f"  Total DAG nodes: {sum(len(dag) for dag in dags)}")
    print(f"\n  Model prediction distribution (persist/evict):")
    total_preds = sum(prediction_stats.values())
    if total_preds > 0:
        for label in sorted(prediction_stats.keys()):
            count = prediction_stats[label]
            print(f"    {label}: {count} ({count/total_preds*100:.1f}%)")
    print(f"  Determinism hash: {workload_hash}")
    print(f"    (This hash should be identical across runs with same seed)")
    
    return dags, dag_names, arrival_times, requests_meta


def main():
    # Extract templates
    templates = extract_trace_templates_with_evolution("/vllm/vllm/elasticswap/toolcall_dataset_claude_annotated_60s")
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
            for tool_name, pred in zip(turn.tool_names, turn.tool_predictions):
                if pred:
                    print(f"    - {tool_name}: prediction={pred.predicted_label} (probs={pred.probabilities})")

if __name__ == "__main__":
    main()


