#!/usr/bin/env python3
"""
Test script to generate example templates using HumanPause-based splitting
and generate template statistics.
"""

import os
import sys
from collections import Counter
import numpy as np

# Add the frontend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_trace_templates_humanpause import (
    extract_trace_templates_with_evolution,
    generate_request_trajectories,
    TraceTemplate
)

def print_template_statistics(templates: list):
    """Print comprehensive statistics about the extracted templates."""
    if not templates:
        print("No templates found!")
        return
    
    print("\n" + "="*80)
    print("TEMPLATE STATISTICS")
    print("="*80)
    
    # Basic counts
    print(f"\nTotal templates: {len(templates)}")
    
    # Turns per template
    turns_per_template = [t.num_turns for t in templates]
    print(f"\nTurns per template:")
    print(f"  Min: {min(turns_per_template)}")
    print(f"  Max: {max(turns_per_template)}")
    print(f"  Mean: {np.mean(turns_per_template):.2f}")
    print(f"  Median: {np.median(turns_per_template):.2f}")
    print(f"  Std: {np.std(turns_per_template):.2f}")
    
    # Tools per template
    tools_per_template = [t.total_tools for t in templates]
    print(f"\nTools per template:")
    print(f"  Min: {min(tools_per_template)}")
    print(f"  Max: {max(tools_per_template)}")
    print(f"  Mean: {np.mean(tools_per_template):.2f}")
    print(f"  Median: {np.median(tools_per_template):.2f}")
    print(f"  Std: {np.std(tools_per_template):.2f}")
    
    # Context tokens per template
    context_per_template = [t.total_context_accumulated for t in templates]
    print(f"\nContext tokens per template (final accumulated):")
    print(f"  Min: {min(context_per_template)}")
    print(f"  Max: {max(context_per_template)}")
    print(f"  Mean: {np.mean(context_per_template):.2f}")
    print(f"  Median: {np.median(context_per_template):.2f}")
    print(f"  Std: {np.std(context_per_template):.2f}")
    
    # Distribution of turns
    turns_dist = Counter(turns_per_template)
    print(f"\nDistribution of turns per template:")
    for num_turns in sorted(turns_dist.keys())[:20]:  # Show top 20
        count = turns_dist[num_turns]
        pct = (count / len(templates)) * 100
        print(f"  {num_turns} turns: {count} templates ({pct:.1f}%)")
    
    # Distribution of tools
    tools_dist = Counter(tools_per_template)
    print(f"\nDistribution of tools per template:")
    for num_tools in sorted(tools_dist.keys())[:20]:  # Show top 20
        count = tools_dist[num_tools]
        pct = (count / len(templates)) * 100
        print(f"  {num_tools} tools: {count} templates ({pct:.1f}%)")
    
    # Tools per turn
    all_tools_per_turn = []
    for template in templates:
        for turn in template.turns:
            all_tools_per_turn.append(turn.num_tools)
    
    if all_tools_per_turn:
        print(f"\nTools per turn (across all templates):")
        print(f"  Min: {min(all_tools_per_turn)}")
        print(f"  Max: {max(all_tools_per_turn)}")
        print(f"  Mean: {np.mean(all_tools_per_turn):.2f}")
        print(f"  Median: {np.median(all_tools_per_turn):.2f}")
        
        tools_per_turn_dist = Counter(all_tools_per_turn)
        print(f"\n  Distribution:")
        for num_tools in sorted(tools_per_turn_dist.keys())[:15]:
            count = tools_per_turn_dist[num_tools]
            pct = (count / len(all_tools_per_turn)) * 100
            print(f"    {num_tools} tools: {count} turns ({pct:.1f}%)")
    
    # Tool names distribution
    tool_name_counter = Counter()
    for template in templates:
        for turn in template.turns:
            for tool_exec in turn.tool_executions:
                tool_name_counter[tool_exec.name] += 1
    
    if tool_name_counter:
        print(f"\nMost common tool names (top 15):")
        for tool_name, count in tool_name_counter.most_common(15):
            pct = (count / sum(tool_name_counter.values())) * 100
            print(f"  {tool_name}: {count} ({pct:.1f}%)")
    
    # Prediction distribution
    prediction_counter = Counter()
    for template in templates:
        for turn in template.turns:
            for tool_exec in turn.tool_executions:
                if tool_exec.model_prediction:
                    prediction_counter[tool_exec.model_prediction.predicted_label] += 1
    
    if prediction_counter:
        print(f"\nPrediction distribution:")
        total = sum(prediction_counter.values())
        for label, count in sorted(prediction_counter.items()):
            pct = (count / total) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*80)


def main():
    # Dataset path
    dataset_dir = "/vllm/vllm/elasticswap/toolcall_dataset_claude_annotated_60s_sessionwise_v3_clustered/toolcall_dataset_claude_annotated_60s_sessionwise_v3_bfcl_benchmark"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please update the dataset_dir path in the script.")
        return
    
    print("Extracting trace templates with HumanPause-based splitting...")
    print(f"Dataset: {dataset_dir}")
    
    # Extract templates
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir=dataset_dir,
        window_duration_minutes=None,  # Not used with HumanPause splitting
        training_tool_calls_path=None  # Skip training filter for demo
    )
    
    print(f"\n✓ Extracted {len(templates)} templates")
    
    # Print statistics
    print_template_statistics(templates)
    
    # Show example templates
    print("\n" + "="*80)
    print("EXAMPLE TEMPLATES")
    print("="*80)
    
    # Show first 5 templates
    for i, template in enumerate(templates[:5], 1):
        print(f"\n--- Example Template {i} ---")
        template.print_sequence()
    
    # Show a template with many turns if available
    templates_by_turns = sorted(templates, key=lambda t: t.num_turns, reverse=True)
    if templates_by_turns and templates_by_turns[0].num_turns > 5:
        print("\n" + "="*80)
        print("LONGEST TEMPLATE (by number of turns)")
        print("="*80)
        templates_by_turns[0].print_sequence()
    
    # Show a template with many tools if available
    templates_by_tools = sorted(templates, key=lambda t: t.total_tools, reverse=True)
    if templates_by_tools and templates_by_tools[0].total_tools > 10:
        print("\n" + "="*80)
        print("TEMPLATE WITH MOST TOOLS")
        print("="*80)
        templates_by_tools[0].print_sequence()
    
    # Test trajectory generation
    print("\n" + "="*80)
    print("TESTING TRAJECTORY GENERATION")
    print("="*80)
    
    print(f"\nGenerating 10 sample trajectories from {len(templates)} templates...")
    trajectories = generate_request_trajectories(templates, num_requests=10, base_seed=42)
    
    print(f"\n✓ Generated {len(trajectories)} trajectories")
    print(f"\nSample trajectory details:")
    for i, traj in enumerate(trajectories[:3], 1):
        print(f"\n  Trajectory {i}:")
        print(f"    Request ID: {traj.request_id}")
        print(f"    Template ID: {traj.template_id}")
        print(f"    Number of turns: {len(traj.turns)}")
        print(f"    Total input tokens: {sum(len(turn.prompt_token_ids) for turn in traj.turns)}")
        print(f"    Total output tokens: {sum(turn.target_output_tokens for turn in traj.turns)}")
        print(f"    Total tools: {sum(len(turn.tool_names) for turn in traj.turns)}")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()


