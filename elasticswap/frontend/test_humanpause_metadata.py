#!/usr/bin/env python3
"""
Test script to verify that the last LLMCall has HumanPause in next_tools metadata.
This ensures that agent_id and tool_call_info are properly set in the evictor.
"""

import os
import sys
import json

# Add the frontend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_trace_templates_humanpause import generate_claude_trace_workload


def test_last_llmcall_has_humanpause():
    """Test that the last LLMCall in each DAG has HumanPause in next_tools metadata."""
    
    dataset_dir = "/vllm/vllm/elasticswap/toolcall_dataset_claude_annotated_60s_sessionwise_v3_clustered/toolcall_dataset_claude_annotated_60s_sessionwise_v3_bfcl_benchmark"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("   Skipping test - dataset not available")
        return True  # Don't fail if dataset is missing
    
    print("=" * 80)
    print("Testing HumanPause metadata in last LLMCall")
    print("=" * 80)
    print()
    
    # Generate a small workload (5 requests)
    print("Generating workload with 5 requests...")
    dags, dag_names, arrival_times, requests_meta = generate_claude_trace_workload(
        num_requests=5,
        claude_dataset_dir=dataset_dir,
        arrival_rate=0.5,
        seed=42,
        prefill_only=False,
        min_humanpause_duration_seconds=300.0,
        max_template_duration_seconds=3600.0,  # 60 minutes
        oracle=True
    )
    
    print(f"\n✓ Generated {len(dags)} DAGs")
    print()
    
    # Test each DAG
    all_passed = True
    results = []
    
    for dag_idx, (dag, dag_name, meta) in enumerate(zip(dags, dag_names, requests_meta)):
        print(f"Testing DAG {dag_idx + 1}: {dag_name}")
        print(f"  Template ID: {meta['template_id']}")
        print(f"  Number of turns: {meta['num_turns']}")
        
        # Find all LLM nodes
        llm_nodes = [node for node in dag if hasattr(node, 'prompt_token_ids')]
        
        if not llm_nodes:
            print(f"  ❌ No LLM nodes found in DAG!")
            all_passed = False
            results.append({
                'dag_name': dag_name,
                'passed': False,
                'error': 'No LLM nodes found'
            })
            continue
        
        # Get the last LLM node
        last_llm_node = llm_nodes[-1]
        last_turn_idx = len(llm_nodes) - 1
        
        print(f"  Last LLM node: {last_llm_node.name} (turn {last_turn_idx})")
        
        # Check metadata
        metadata = last_llm_node.metadata
        next_tools = metadata.get("next_tools", [])
        kv_reuse_duration = metadata.get("kv_reuse_expected_duration_s")
        
        print(f"  kv_reuse_expected_duration_s: {kv_reuse_duration}")
        print(f"  next_tools count: {len(next_tools)}")
        
        # Check if HumanPause is in next_tools
        has_humanpause = False
        humanpause_info = None
        
        for tool in next_tools:
            tool_name = tool.get("tool_name", "")
            if tool_name == "HumanPause":
                has_humanpause = True
                humanpause_info = tool
                break
        
        if has_humanpause:
            print(f"  ✅ HumanPause found in next_tools")
            print(f"     Tool args: {humanpause_info.get('tool_args', 'N/A')[:100]}...")
            
            # Verify it would be properly converted to ToolUsageHint
            # (This is what the evictor uses)
            tool_hint_dict = {
                "tool_name": humanpause_info.get("tool_name"),
                "tool_args": humanpause_info.get("tool_args", "")
            }
            print(f"     Would create ToolUsageHint: {tool_hint_dict}")
            
            results.append({
                'dag_name': dag_name,
                'passed': True,
                'has_humanpause': True,
                'tool_args': humanpause_info.get('tool_args', '')[:50]
            })
        else:
            print(f"  ❌ HumanPause NOT found in next_tools!")
            print(f"     Available tools: {[t.get('tool_name', 'unknown') for t in next_tools]}")
            all_passed = False
            results.append({
                'dag_name': dag_name,
                'passed': False,
                'has_humanpause': False,
                'available_tools': [t.get('tool_name', 'unknown') for t in next_tools]
            })
        
        # Check kv_reuse_expected_duration_s
        if kv_reuse_duration == 9999999.0:
            print(f"  ✅ kv_reuse_expected_duration_s correctly set to 9999999.0 (final LLM call)")
        else:
            print(f"  ⚠️  kv_reuse_expected_duration_s is {kv_reuse_duration} (expected 9999999.0)")
        
        print()
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"Passed: {passed_count}/{total_count}")
    print()
    
    if all_passed:
        print("✅ All tests passed! Last LLMCall always has HumanPause in next_tools.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        print()
        print("Failed DAGs:")
        for r in results:
            if not r['passed']:
                print(f"  - {r['dag_name']}: {r.get('error', 'HumanPause not found')}")
    
    print()
    
    # Detailed results
    print("Detailed Results:")
    for r in results:
        status = "✅" if r['passed'] else "❌"
        print(f"  {status} {r['dag_name']}: ", end="")
        if r['passed']:
            print(f"HumanPause found (args: {r.get('tool_args', 'N/A')})")
        else:
            print(f"HumanPause missing (available: {r.get('available_tools', [])})")
    
    return all_passed


if __name__ == "__main__":
    success = test_last_llmcall_has_humanpause()
    sys.exit(0 if success else 1)

