#!/usr/bin/env python3
"""
Dump example traces to files for inspection.
"""

import os
import sys
import json
from collections import defaultdict

# Add the frontend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_trace_templates_humanpause import (
    extract_trace_templates_with_evolution,
    TraceTemplate
)

def template_to_dict(template: TraceTemplate) -> dict:
    """Convert a TraceTemplate to a dictionary for JSON serialization."""
    return {
        'template_id': template.template_id,
        'session_id': template.session_id,
        'num_turns': template.num_turns,
        'total_tools': template.total_tools,
        'total_context_accumulated': template.total_context_accumulated,
        'turns': [
            {
                'turn_idx': i + 1,
                'cache_read': turn.cache_read,
                'new_input': turn.new_input,
                'output': turn.output,
                'total_input': turn.total_input,
                'num_tools': turn.num_tools,
                'total_tool_time_sequential': sum(t.duration for t in turn.tool_executions),
                'total_tool_time_wallclock': turn.total_tool_time_with_overlap,
                'tools': [
                    {
                        'name': tool.name,
                        'duration_seconds': tool.duration,
                        'start_timestamp': tool.start_timestamp,
                        'prediction': {
                            'predicted_label': tool.model_prediction.predicted_label,
                            'probabilities': tool.model_prediction.probabilities
                        } if tool.model_prediction else None,
                        'tool_input': tool.tool_input
                    }
                    for tool in turn.tool_executions
                ]
            }
            for i, turn in enumerate(template.turns)
        ]
    }

def main():
    # Dataset path
    dataset_dir = "/vllm/vllm/elasticswap/toolcall_dataset_claude_annotated_60s_sessionwise_v3_clustered/toolcall_dataset_claude_annotated_60s_sessionwise_v3_bfcl_benchmark"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print("Extracting trace templates with HumanPause-based splitting...")
    
    # Extract templates
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir=dataset_dir,
        window_duration_minutes=None,
        training_tool_calls_path=None  # Skip training filter for demo
    )
    
    print(f"✓ Extracted {len(templates)} templates")
    
    # Create output directory
    output_dir = "/vllm/vllm/elasticswap/frontend/example_traces"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dump all templates to JSON
    all_templates_file = os.path.join(output_dir, "all_templates.json")
    print(f"\nDumping all {len(templates)} templates to {all_templates_file}...")
    with open(all_templates_file, 'w') as f:
        json.dump([template_to_dict(t) for t in templates], f, indent=2)
    print(f"✓ Saved all templates")
    
    # Select diverse examples
    examples = []
    
    # 1. Shortest template (1 turn)
    shortest = min(templates, key=lambda t: t.num_turns)
    examples.append(("shortest_template", shortest))
    
    # 2. Longest template (most turns)
    longest = max(templates, key=lambda t: t.num_turns)
    examples.append(("longest_template", longest))
    
    # 3. Template with most tools
    most_tools = max(templates, key=lambda t: t.total_tools)
    examples.append(("most_tools_template", most_tools))
    
    # 4. Template with no tools
    no_tools = [t for t in templates if t.total_tools == 0]
    if no_tools:
        examples.append(("no_tools_template", no_tools[0]))
    
    # 5. Medium-length templates (around median)
    sorted_by_turns = sorted(templates, key=lambda t: t.num_turns)
    median_idx = len(sorted_by_turns) // 2
    examples.append(("median_length_template", sorted_by_turns[median_idx]))
    
    # 6. A few random diverse examples
    import random
    random.seed(42)
    diverse_samples = random.sample(templates, min(5, len(templates)))
    for i, sample in enumerate(diverse_samples, 1):
        examples.append((f"random_sample_{i}", sample))
    
    # Dump examples to individual files
    print(f"\nDumping {len(examples)} example templates...")
    for name, template in examples:
        # JSON format
        json_file = os.path.join(output_dir, f"{name}.json")
        with open(json_file, 'w') as f:
            json.dump(template_to_dict(template), f, indent=2)
        
        # Human-readable text format
        txt_file = os.path.join(output_dir, f"{name}.txt")
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Template: {template.template_id}\n")
            f.write(f"Session: {template.session_id}\n")
            f.write(f"Total Turns: {template.num_turns}\n")
            f.write(f"Total Tools: {template.total_tools}\n")
            f.write(f"Total Context (final): {template.total_context_accumulated} tokens\n")
            f.write("=" * 80 + "\n\n")
            
            for i, turn in enumerate(template.turns, 1):
                f.write(f"Turn {i}:\n")
                f.write(f"  Cache Read: {turn.cache_read} tokens\n")
                f.write(f"  New Input: {turn.new_input} tokens\n")
                f.write(f"  Output: {turn.output} tokens\n")
                f.write(f"  Total Input: {turn.total_input} tokens\n")
                f.write(f"  Tools: {turn.num_tools}\n")
                
                if turn.tool_executions:
                    total_seq = sum(t.duration for t in turn.tool_executions)
                    total_wall = turn.total_tool_time_with_overlap
                    overlap_pct = ((total_seq - total_wall) / total_seq * 100) if total_seq > 0 else 0
                    
                    f.write(f"  Tool Execution Time:\n")
                    f.write(f"    Sequential: {total_seq:.3f}s\n")
                    f.write(f"    Wall-clock (with overlap): {total_wall:.3f}s\n")
                    f.write(f"    Overlap: {overlap_pct:.1f}%\n")
                    
                    f.write(f"  Tool Details:\n")
                    for j, tool in enumerate(turn.tool_executions, 1):
                        f.write(f"    {j}. {tool.name}\n")
                        f.write(f"       Duration: {tool.duration:.3f}s\n")
                        f.write(f"       Timestamp: {tool.start_timestamp}\n")
                        if tool.model_prediction:
                            f.write(f"       Prediction: {tool.model_prediction.predicted_label}\n")
                            f.write(f"       Probabilities: {tool.model_prediction.probabilities}\n")
                        if tool.tool_input:
                            f.write(f"       Input: {json.dumps(tool.tool_input, indent=8)}\n")
                else:
                    f.write(f"  No tools in this turn\n")
                
                f.write("\n")
        
        print(f"  ✓ {name}: {template.num_turns} turns, {template.total_tools} tools")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("TRACE TEMPLATE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Templates: {len(templates)}\n\n")
        
        turns_per_template = [t.num_turns for t in templates]
        tools_per_template = [t.total_tools for t in templates]
        context_per_template = [t.total_context_accumulated for t in templates]
        
        f.write("Turns per Template:\n")
        f.write(f"  Min: {min(turns_per_template)}\n")
        f.write(f"  Max: {max(turns_per_template)}\n")
        f.write(f"  Mean: {sum(turns_per_template) / len(turns_per_template):.2f}\n")
        f.write(f"  Median: {sorted(turns_per_template)[len(turns_per_template) // 2]}\n\n")
        
        f.write("Tools per Template:\n")
        f.write(f"  Min: {min(tools_per_template)}\n")
        f.write(f"  Max: {max(tools_per_template)}\n")
        f.write(f"  Mean: {sum(tools_per_template) / len(tools_per_template):.2f}\n")
        f.write(f"  Median: {sorted(tools_per_template)[len(tools_per_template) // 2]}\n\n")
        
        f.write("Context Tokens per Template (final accumulated):\n")
        f.write(f"  Min: {min(context_per_template)}\n")
        f.write(f"  Max: {max(context_per_template)}\n")
        f.write(f"  Mean: {sum(context_per_template) / len(context_per_template):.2f}\n")
        f.write(f"  Median: {sorted(context_per_template)[len(context_per_template) // 2]}\n\n")
        
        # Tool name distribution
        tool_counter = defaultdict(int)
        for template in templates:
            for turn in template.turns:
                for tool in turn.tool_executions:
                    tool_counter[tool.name] += 1
        
        f.write("Most Common Tools:\n")
        for tool_name, count in sorted(tool_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"  {tool_name}: {count}\n")
    
    print(f"\n✓ Summary saved to {summary_file}")
    print(f"\nAll files saved to: {output_dir}/")
    print(f"  - all_templates.json: All {len(templates)} templates in JSON format")
    print(f"  - {len(examples)} example template files (both .json and .txt formats)")
    print(f"  - summary.txt: Overall statistics")


if __name__ == "__main__":
    main()


