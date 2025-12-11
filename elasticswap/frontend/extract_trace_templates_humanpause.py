# Extract trace templates from 60s-annotated Claude dataset with HumanPause splitting
# Maps <60s -> "persist" and >=60s -> "evict"
# NEW: Splits traces at HumanPause boundaries instead of using windowed heuristics

import os
from dataclasses import dataclass
from typing import Any,List, Dict, Tuple, Optional, Set
import numpy as np
import glob
import json
from collections import defaultdict
from dateutil import parser as date_parser


TOOL_DURATION_MEANS: Dict[str, Dict[str, float]] = {}
GLOBAL_DURATION_MEANS: Dict[str, float] = {
    'persist': 30.0,
    'evict': 120.0,
}

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
    tool_input: Dict[str, Any] = None
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


def split_session_at_human_pause(session_events: List[dict], min_split_duration_seconds: float = 300.0) -> List[List[dict]]:
    """
    Split a session into multiple sub-sessions at HumanPause boundaries >= min_split_duration_seconds.
    
    Logic:
    - Iterate through events from start
    - HumanPause < min_split_duration_seconds: treat as regular tool call, continue
    - HumanPause >= min_split_duration_seconds: split here (this HumanPause ends the trace)
    - Start new trace with the next LLMCall
    - All traces end with HumanPause >= min_split_duration_seconds
    
    Each trace:
    - Starts with an LLMCall
    - Ends at a HumanPause toolcall >= min_split_duration_seconds (inclusive)
    - Includes all events between start and end
    - Includes HumanPause toolcalls < min_split_duration_seconds as regular tool calls
    
    Args:
        session_events: List of events for a single session (must be sorted by timestamp)
        min_split_duration_seconds: Minimum HumanPause duration (in seconds) to trigger a split (default: 300.0)
    
    Returns:
        List of sub-session event lists, each representing a trace from LLMCall to HumanPause (>= threshold)
    """
    if not session_events:
        return []
    
    sub_sessions = []
    current_trace = []
    waiting_for_llm_start = True  # Need to find first LLM call to start a trace
    
    for i, event in enumerate(session_events):
        if event['event_type'] == 'llm_call':
            # If we were waiting for an LLM call, start a new trace
            if waiting_for_llm_start:
                current_trace = []
                waiting_for_llm_start = False
            
            # Add LLM call to current trace
            current_trace.append(event)
        
        elif event['event_type'] == 'tool_call':
            tool_name = event.get('tool_name', '')
            
            if tool_name == 'HumanPause':
                # Get HumanPause duration
                exec_time = event.get('execution_time_ms')
                if exec_time is not None:
                    duration_seconds = exec_time / 1000.0
                else:
                    duration_seconds = 0.0
                
                # Always include HumanPause in the current trace
                current_trace.append(event)
                
                # Split if duration >= threshold
                if duration_seconds >= min_split_duration_seconds:
                    # Save this trace if it has at least one LLM call
                    if current_trace and any(e['event_type'] == 'llm_call' for e in current_trace):
                        sub_sessions.append(current_trace)
                    
                    # Reset for next trace
                    current_trace = []
                    waiting_for_llm_start = True
                # If duration < threshold, continue in same trace (treat as regular tool call)
            else:
                # Regular tool call - add to current trace if we have an active trace
                if not waiting_for_llm_start:
                    current_trace.append(event)
        else:
            # Other event types - add to current trace if we have an active trace
            if not waiting_for_llm_start:
                current_trace.append(event)
    
    # Don't include any remaining trace that doesn't end with HumanPause >= 300s
    # (Such traces would be incomplete and are skipped)
    
    return sub_sessions


def calculate_template_duration(template: TraceTemplate) -> float:
    """
    Calculate end-to-end duration of a template in seconds.
    
    Returns the time difference between the first event and the last event.
    """
    from dateutil import parser as date_parser
    
    all_timestamps = []
    
    for turn in template.turns:
        for tool_exec in turn.tool_executions:
            try:
                # Parse timestamp and add start time
                start_epoch = date_parser.parse(tool_exec.start_timestamp).timestamp()
                all_timestamps.append(start_epoch)
                
                # Add end time (start + duration)
                end_epoch = start_epoch + tool_exec.duration
                all_timestamps.append(end_epoch)
            except (ValueError, TypeError):
                # Skip invalid timestamps
                continue
    
    if len(all_timestamps) < 2:
        return 0.0
    
    # Return duration from first to last event
    return max(all_timestamps) - min(all_timestamps)


def extract_trace_templates_with_evolution(
    claude_dataset_dir: str,
    window_duration_minutes: float = None,  # Not used, kept for API compatibility
    max_trace_duration_minutes: float = None,  # Not used, kept for API compatibility
    training_tool_calls_path: Optional[str] = "/vllm/vllm/elasticswap/training_tool_calls_sessionwise.json",
    min_humanpause_duration_seconds: float = 300.0,
    max_template_duration_seconds: Optional[float] = None
) -> List[TraceTemplate]:
    """
    Extract trace templates from Claude tool call dataset with HumanPause-based splitting.
    
    This version:
    - Splits traces at HumanPause boundaries (no windowed heuristics)
    - Each trace starts with an LLMCall
    - Each trace ends at a HumanPause toolcall
    - Maps duration bucket predictions to persist/evict labels:
      - "<60s" -> "persist"
      - ">=60s" -> "evict"
    
    Args:
        claude_dataset_dir: Directory with *_events.json files (60s annotated with HumanPause)
        window_duration_minutes: Not used (kept for API compatibility)
        max_trace_duration_minutes: Not used (kept for API compatibility)
        training_tool_calls_path: Path to training tool calls for filtering
        min_humanpause_duration_seconds: Minimum HumanPause duration (in seconds) to trigger a split (default: 300.0)
        max_template_duration_seconds: Maximum end-to-end duration (in seconds) for templates to include.
                                      If None, no filtering by duration (default: None)
    
    Returns:
        List of TraceTemplate objects with turn-by-turn token evolution and persist/evict predictions
    """
    json_files = sorted(glob.glob(f"{claude_dataset_dir}/*_events.json"))
    
    print(f"Loading 60s-annotated traces with HumanPause from {claude_dataset_dir}...")
    print(f"Mapping: <60s -> persist, >=60s -> evict")
    print(f"Splitting strategy: HumanPause boundaries >= {min_humanpause_duration_seconds} seconds")
    print(f"  - HumanPause < {min_humanpause_duration_seconds}s: Included as regular tool calls")
    print(f"  - HumanPause >= {min_humanpause_duration_seconds}s: Marks trace boundary (included in trace)")
    if max_template_duration_seconds is not None:
        print(f"  - Filtering: Only templates with end-to-end duration < {max_template_duration_seconds}s ({max_template_duration_seconds/60:.1f} minutes)")

    SKIP_TRAINING_FILTER = os.environ.get("SKIP_TRAINING_FILTER", "0") == "1"
    if SKIP_TRAINING_FILTER:
        training_tool_calls_path = None
        print(f"  Skipping training tool call filtering")
    else:
        print(f"  Using training tool call filtering")

    # Load training tool call IDs to filter
    training_tool_call_ids: Set[Tuple[str, int]] = set()
    session_alias_to_id = {}
    
    if training_tool_calls_path:
        with open(training_tool_calls_path) as f:
            training_tool_calls = json.load(f)
        manifest_path = os.path.join(claude_dataset_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as mf:
                manifest_data = json.load(mf)
            for entry in manifest_data:
                session_name = entry.get("session_name")
                session_id = entry.get("session_id")
                if session_name and session_id:
                    session_alias_to_id[session_name] = session_id

        print(f"  Loaded {len(training_tool_calls)} training tool calls.")
    
    dataset_alias_map: Dict[str, str] = {}

    # Collect all tool durations for statistics (across entire dataset, without filtering)
    all_tool_durations = []
    all_predictions = []
    tool_duration_accumulators = defaultdict(lambda: {
        'persist': {'sum': 0.0, 'count': 0},
        'evict': {'sum': 0.0, 'count': 0},
    })
    global_duration_accumulators = {
        'persist': {'sum': 0.0, 'count': 0},
        'evict': {'sum': 0.0, 'count': 0},
    }
    for json_file in json_files:
        with open(json_file) as f:
            events = json.load(f)
        
        alias_name = os.path.basename(json_file).replace("_events.json", "")
        if alias_name and alias_name not in dataset_alias_map and events:
            first_session_id = events[0].get('session_id')
            if first_session_id:
                dataset_alias_map[alias_name] = first_session_id
        
        session_event_counters = defaultdict(int)
        for event in events:
            session_id = event.get('session_id')
            if session_id is None:
                continue
            event['_event_index'] = session_event_counters[session_id]
            session_event_counters[session_id] += 1
        
        for event in events:
            if event['event_type'] == 'tool_call':
                exec_time = event.get('execution_time_ms')
                duration_seconds = None
                if exec_time is not None and exec_time > 0:
                    duration_seconds = exec_time / 1000.0
                    all_tool_durations.append(duration_seconds)
                
                # Track model predictions (after mapping)
                duration_bucket = event.get('duration_bucket_prediction')
                if duration_bucket:
                    model_pred = map_duration_bucket_to_prediction(duration_bucket)
                    if model_pred:
                        all_predictions.append(model_pred.predicted_label)
                        # Only accumulate duration statistics if we have a valid duration
                        if duration_seconds is not None:
                            bucket_label = model_pred.predicted_label
                            tool_stats = tool_duration_accumulators[event['tool_name']]
                            tool_stats[bucket_label]['sum'] += duration_seconds
                            tool_stats[bucket_label]['count'] += 1
                            global_duration_accumulators[bucket_label]['sum'] += duration_seconds
                            global_duration_accumulators[bucket_label]['count'] += 1

    # Incorporate dataset-derived alias map if manifest didn't cover it
    for alias, session_id in dataset_alias_map.items():
        session_alias_to_id.setdefault(alias, session_id)

    # Resolve training tool call identifiers using combined alias map
    if training_tool_calls_path:
        for entry in training_tool_calls:
            event_index = entry.get("event_index")
            if event_index is None:
                continue
            possible_session_ids = set()
            for key in (entry.get("session_id"), entry.get("session_name")):
                if not key:
                    continue
                possible_session_ids.add(key)
                mapped = session_alias_to_id.get(key)
                if mapped:
                    possible_session_ids.add(mapped)
            if not possible_session_ids:
                continue
            for session_id in possible_session_ids:
                training_tool_call_ids.add((session_id, event_index))
        print(f"  Resolved identifiers for filtering: {len(training_tool_call_ids)}")

    # Compute per-tool and global duration means for persist/evict buckets
    tool_duration_means: Dict[str, Dict[str, float]] = {}
    for tool_name, bucket_stats in tool_duration_accumulators.items():
        tool_duration_means[tool_name] = {}
        for bucket_label in ('persist', 'evict'):
            bucket = bucket_stats[bucket_label]
            if bucket['count'] > 0:
                tool_duration_means[tool_name][bucket_label] = bucket['sum'] / bucket['count']

    global_duration_means = {}
    for bucket_label in ('persist', 'evict'):
        bucket = global_duration_accumulators[bucket_label]
        if bucket['count'] > 0:
            global_duration_means[bucket_label] = bucket['sum'] / bucket['count']

    # Update module-level means with computed statistics
    global TOOL_DURATION_MEANS, GLOBAL_DURATION_MEANS
    TOOL_DURATION_MEANS = tool_duration_means
    GLOBAL_DURATION_MEANS = {
        'persist': global_duration_means.get('persist', GLOBAL_DURATION_MEANS.get('persist', 30.0)),
        'evict': global_duration_means.get('evict', GLOBAL_DURATION_MEANS.get('evict', 120.0)),
    }
    if GLOBAL_DURATION_MEANS['persist'] and GLOBAL_DURATION_MEANS['evict']:
        print(f"  Global duration means (s): persist={GLOBAL_DURATION_MEANS['persist']:.3f}, evict={GLOBAL_DURATION_MEANS['evict']:.3f}")
    
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
    
    # Extract templates with HumanPause splitting
    templates = []
    total_traces = 0
    traces_split = 0
    filtered_tool_call_count = 0
    filtered_templates = 0
    humanpause_count = 0
    
    for json_file in json_files:
        with open(json_file) as f:
            events = json.load(f)
        
        session_event_counters = defaultdict(int)
        for event in events:
            session_id = event.get('session_id')
            if session_id is None:
                continue
            event['_event_index'] = session_event_counters[session_id]
            session_event_counters[session_id] += 1
        
        # SORT BY TIMESTAMP to ensure chronological order
        events.sort(key=lambda e: e['timestamp'])
        
        # Count HumanPause events
        for event in events:
            if event['event_type'] == 'tool_call' and event.get('tool_name') == 'HumanPause':
                humanpause_count += 1
        
        # Group by session
        sessions = defaultdict(list)
        
        for event in events:
            session_id = event['session_id']
            sessions[session_id].append(event)
        
        # Process each session (with splitting at HumanPause boundaries)
        # Sort sessions by ID to ensure deterministic order
        for session_id, session_events in sorted(sessions.items()):
            if not session_events:
                continue
            
            # Split session at HumanPause boundaries >= min_humanpause_duration_seconds
            sub_sessions = split_session_at_human_pause(session_events, min_split_duration_seconds=min_humanpause_duration_seconds)
            
            if len(sub_sessions) > 1:
                traces_split += 1
            
            # Process each sub-session as a separate trace
            for sub_idx, sub_session_events in enumerate(sub_sessions):
                if not sub_session_events:
                    continue
                
                matching_training_tools = [
                    event for event in sub_session_events
                    if event['event_type'] == 'tool_call'
                    and (event['session_id'], event.get('_event_index')) in training_tool_call_ids
                ]
                if matching_training_tools:
                    filtered_tool_call_count += len(matching_training_tools)
                    filtered_templates += 1
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
                        tool_name = event.get('tool_name', '')
                        
                        exec_time = event.get('execution_time_ms')
                        # HumanPause may have 0 execution time, but we still want to include it
                        if exec_time is not None:
                            duration_seconds = exec_time / 1000.0
                        else:
                            duration_seconds = 0.0
                        
                        # Extract and map duration bucket prediction
                        # HumanPause typically doesn't have predictions, but check anyway
                        model_pred = None
                        duration_bucket = event.get('duration_bucket_prediction')
                        if duration_bucket:
                            model_pred = map_duration_bucket_to_prediction(duration_bucket)
                        
                        # Include HumanPause as a tool in the trace (marks the end)
                        tool_exec = ToolExecution(
                            name=tool_name,
                            start_timestamp=event['timestamp'],
                            duration=duration_seconds,
                            tool_input=event.get('tool_input', {}),
                            model_prediction=model_pred
                        )
                        current_tools.append(tool_exec)
                
                # Don't forget the last turn
                if current_turn is not None:
                    current_turn.tool_executions = current_tools
                    turns.append(current_turn)
                
                # Only create template if trace ends with HumanPause
                # Check if the last turn has a HumanPause tool
                if turns:
                    last_turn = turns[-1]
                    has_humanpause = any(
                        tool.name == 'HumanPause' 
                        for tool in last_turn.tool_executions
                    )
                    
                    if has_humanpause:
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
                    # else: Skip traces that don't end with HumanPause
    
    # Filter templates by end-to-end duration if threshold is specified
    filtered_by_duration = 0
    if max_template_duration_seconds is not None:
        filtered_templates_list = []
        for template in templates:
            duration = calculate_template_duration(template)
            if duration < max_template_duration_seconds:
                filtered_templates_list.append(template)
            else:
                filtered_by_duration += 1
        templates = filtered_templates_list
    
    print(f"\n  ===== Trace Processing Summary =====")
    print(f"  Total HumanPause events found: {humanpause_count}")
    print(f"  Original sessions split at HumanPause: {traces_split}")
    print(f"  Total traces after splitting: {total_traces}")
    print(f"  Tool calls filtered (training split): {filtered_tool_call_count}")
    print(f"  Templates skipped (training split): {filtered_templates}")
    if max_template_duration_seconds is not None:
        print(f"  Templates filtered by duration (< {max_template_duration_seconds}s): {filtered_by_duration}")
    print(f"  Templates extracted after filter: {len(templates)}")
    print(f"  Note: Each trace starts with LLMCall and ends at HumanPause >= {min_humanpause_duration_seconds}s")
    print(f"  Note: HumanPause < {min_humanpause_duration_seconds}s are included as regular tool calls in traces")
    print(f"  Note: Traces that don't end with HumanPause >= {min_humanpause_duration_seconds}s are skipped")
    if max_template_duration_seconds is not None:
        print(f"  Note: Only templates with end-to-end duration < {max_template_duration_seconds}s are included")
    return templates

@dataclass
class TurnExecution:
    """Concrete execution of a single turn with actual token IDs"""
    turn_idx: int
    prompt_token_ids: List[int]  # Full context for this turn
    target_output_tokens: int    # Expected output length
    tool_wait_time: float        # Time to wait after this turn (wall-clock)
    tool_names: List[str]        # Tool names for reference
    tool_inputs: List[Any]  # Tool inputs for reference
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
            tool_predictions=tool_predictions,
            tool_inputs=[t.tool_input for t in turn_template.tool_executions],
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
                                   tool_names: List[str],
                                   persist_bin_avg: float = 30.0,
                                   evict_bin_avg: float = 120.0) -> Tuple[float, List[float]]:
    """
    Compute predicted duration statistics for tool executions.
    
    For each tool:
        expected_duration = P(persist) * persist_mean(tool) + P(evict) * evict_mean(tool)
    
    Args:
        tool_predictions: List of ModelPrediction objects for each tool
        tool_names: Names for each tool prediction, used to select bucket means
        persist_bin_avg: Fallback average duration for persist bin (<60s), default 30s
        evict_bin_avg: Fallback average duration for evict bin (>=60s), default 120s
    
    Returns:
        Tuple of (max_expected_duration, list_of_per_tool_expected_durations)
    """
    per_tool_expected: List[float] = []
    
    for idx, pred in enumerate(tool_predictions):
        if pred is not None:
            # Expected value for this specific tool
            persist_prob = pred.probabilities.get('persist', 0.0)
            evict_prob = pred.probabilities.get('evict', 0.0)
            
            tool_name = tool_names[idx] if idx < len(tool_names) else None
            tool_means = TOOL_DURATION_MEANS.get(tool_name, {}) if tool_name else {}
            persist_mean = tool_means.get('persist', GLOBAL_DURATION_MEANS.get('persist', persist_bin_avg))
            evict_mean = tool_means.get('evict', GLOBAL_DURATION_MEANS.get('evict', evict_bin_avg))
            
            expected_duration = persist_prob * persist_mean + evict_prob * evict_mean
            per_tool_expected.append(expected_duration)
    
    max_duration = max(per_tool_expected) if per_tool_expected else 0.0
    return max_duration, per_tool_expected


def trajectory_to_dag_nodes(trajectory: RequestTrajectory, 
                                annotate: bool = True, 
                                prefill_only: bool = False,
                                aggregation_strategy: str = 'avg_probabilities',
                                oracle: bool = True,
                                persist_bin_avg: float = 30.0,
                                evict_bin_avg: float = 120.0,
                                rate_controller = None):
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
            rate_controller=rate_controller
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

            # add next tools to the metadata
            tool_hints = []
            for name, raw_input in zip(turn.tool_names, turn.tool_inputs):
                if raw_input is None:
                    serialized = ""
                elif isinstance(raw_input, str):
                    serialized = raw_input
                else:
                    serialized = json.dumps(raw_input, sort_keys=True)
                tool_hints.append({
                    "tool_name": name,
                    "tool_args": serialized,
                })
            
            # Set next_tools after building the complete list
            llm_node.metadata["next_tools"] = tool_hints
        else:
            llm_node.metadata["next_tools"] = []
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
        
        # Explicitly ensure last turn has HumanPause in next_tools
        # This is critical because the last LLMCall should always have HumanPause
        # even if has_tools was False (which shouldn't happen, but we ensure it)
        if is_last_turn:
            # Check if HumanPause is already in next_tools
            has_humanpause = any(
                tool.get("tool_name") == "HumanPause" 
                for tool in llm_node.metadata.get("next_tools", [])
            )
            
            if not has_humanpause:
                # HumanPause should be in turn.tool_names for the last turn
                # Find it and add it to next_tools
                humanpause_idx = None
                for idx, tool_name in enumerate(turn.tool_names):
                    if tool_name == "HumanPause":
                        humanpause_idx = idx
                        break
                
                if humanpause_idx is not None and humanpause_idx < len(turn.tool_inputs):
                    # HumanPause exists in turn.tool_names, add it to next_tools
                    raw_input = turn.tool_inputs[humanpause_idx]
                    if raw_input is None:
                        serialized = ""
                    elif isinstance(raw_input, str):
                        serialized = raw_input
                    else:
                        serialized = json.dumps(raw_input, sort_keys=True)
                    
                    humanpause_hint = {
                        "tool_name": "HumanPause",
                        "tool_args": serialized,
                    }
                    
                    # Initialize next_tools if it's empty
                    if not llm_node.metadata.get("next_tools"):
                        llm_node.metadata["next_tools"] = []
                    llm_node.metadata["next_tools"].append(humanpause_hint)
                else:
                    # HumanPause not found in turn.tool_names, but we know it should be there
                    # Add a default HumanPause hint (fallback case)
                    if not llm_node.metadata.get("next_tools"):
                        llm_node.metadata["next_tools"] = []
                    llm_node.metadata["next_tools"].append({
                        "tool_name": "HumanPause",
                        "tool_args": "{}",  # Default empty args
                    })
        
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
                # PREDICTION MODE: Use per-tool expected durations from the model
                if len(turn.tool_predictions) > 0:
                    # Use the new per-tool aggregation method
                    predicted_duration, per_tool_expected = compute_predicted_tool_duration(
                        turn.tool_predictions,
                        turn.tool_names,
                        persist_bin_avg=persist_bin_avg,
                        evict_bin_avg=evict_bin_avg
                    )
                    node.metadata["kv_reuse_expected_duration_s"] = predicted_duration
                    
                    # Optional: Store for debugging
                    node.metadata["predicted_duration_breakdown"] = {
                        'num_tools': len(turn.tool_predictions),
                        'max_predicted': predicted_duration,
                        'per_tool_expected': per_tool_expected,
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
    claude_dataset_dir: str = "/vllm/vllm/elasticswap/toolcall_dataset_claude_annotated_60s_sessionwise",
    arrival_rate: float = 0.5,
    seed: int = 42,
    prefill_only: bool = False,
    window_duration_minutes: float = None,  # Not used, kept for API compatibility
    aggregation_strategy: str = 'avg_probabilities',
    oracle: bool = True,
    persist_bin_avg: float = 30.0,
    evict_bin_avg: float = 120.0,
    rate_controller = None,
    min_humanpause_duration_seconds: float = 300.0,
    max_template_duration_seconds: Optional[float] = None
):
    """
    Generate workload from 60s-annotated Claude traces with HumanPause splitting.
    
    This version:
    - Splits traces at HumanPause boundaries (no windowed heuristics)
    - Maps duration bucket predictions to persist/evict labels:
      - "<60s" -> "persist" 
      - ">=60s" -> "evict"
    
    Args:
        num_requests: Number of requests to generate
        claude_dataset_dir: Directory with 60s-annotated Claude trace JSON files (with HumanPause)
        arrival_rate: Poisson arrival rate (requests per second)
        seed: Random seed
        prefill_only: If True, only generate prefill requests (1 output token)
        window_duration_minutes: Not used (kept for API compatibility)
        aggregation_strategy: Strategy for aggregating parallel tool predictions (for labels)
            Options: 'avg_probabilities', 'majority_vote', 'conservative', 'aggressive', 'max_confidence'
        oracle: If True, use actual tool execution times (ORACLE - ground truth upper bound)
                If False, use per-tool ML predictions (REALISTIC - practical performance)
        persist_bin_avg: Average duration for persist bin (<60s), default 30s
        evict_bin_avg: Average duration for evict bin (>=60s), default 120s
        min_humanpause_duration_seconds: Minimum HumanPause duration (in seconds) to trigger a split (default: 300.0)
        max_template_duration_seconds: Maximum end-to-end duration (in seconds) for templates to include.
                                      If None, no filtering by duration (default: None)
    
    Returns:
        Tuple of (dags, dag_names, arrival_times, requests_meta) for run_dags_with_arrival_times
    """
    # Extract templates (HumanPause-based splitting, no windowed heuristics)
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir,
        window_duration_minutes=window_duration_minutes,
        min_humanpause_duration_seconds=min_humanpause_duration_seconds,
        max_template_duration_seconds=max_template_duration_seconds
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
            evict_bin_avg=evict_bin_avg,
            rate_controller=rate_controller
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
    print(f"\n✓ Generated Claude trace workload with HumanPause splitting ({mode_str}):")
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

