#!/usr/bin/env python3
"""
Integration module for fencer profile graph generation.

This module integrates the fencer profile plotting system with the existing
fencer analysis workflow, allowing automatic generation of profile graphs
when fencer analysis is completed.
"""

import os
import json
import logging
import sys
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append('/workspace/Project/your_scripts')

# Import the fencer profile plotting module
from fencer_profile_plotting import save_fencer_profile_plots


def integrate_fencer_profile_graphs_with_analysis(upload_id: int, base_results_dir: str = "/workspace/Project/results") -> Dict[str, Any]:
    """
    Integrate fencer profile graph generation with existing fencer analysis.
    
    This function should be called after fencer analysis is completed to automatically
    generate comprehensive profile visualizations.
    
    Args:
        upload_id: The upload ID for which to generate profile graphs
        base_results_dir: Base directory containing analysis results
        
    Returns:
        Dictionary containing graph generation results and file paths
    """
    try:
        results_dir = os.path.join(base_results_dir, str(upload_id))
        fencer_analysis_dir = os.path.join(results_dir, 'fencer_analysis')
        
        if not os.path.exists(fencer_analysis_dir):
            logging.error(f"Fencer analysis directory not found: {fencer_analysis_dir}")
            return {"success": False, "error": "Fencer analysis directory not found"}
        
        # Generate profile graphs directly
        profile_results = {}
        
        # Process each fencer
        for fencer_name in ['Fencer_Left', 'Fencer_Right']:
            fencer_file = os.path.join(fencer_analysis_dir, f'fencer_{fencer_name}_analysis.json')
            
            if os.path.exists(fencer_file):
                try:
                    with open(fencer_file, 'r', encoding='utf-8') as f:
                        fencer_data = json.load(f)
                    
                    # Create output directory for this fencer's plots
                    fencer_output_dir = os.path.join(fencer_analysis_dir, 'profile_plots', fencer_name)
                    
                    # Generate plots
                    plot_files = save_fencer_profile_plots(fencer_data, fencer_name, fencer_output_dir)
                    profile_results[fencer_name] = plot_files
                    
                    logging.info(f"Generated profile plots for {fencer_name}")
                    
                except Exception as e:
                    logging.error(f"Error processing {fencer_name}: {str(e)}")
            else:
                logging.warning(f"Fencer analysis file not found: {fencer_file}")
        
        if not profile_results:
            return {"success": False, "error": "No fencer profile graphs generated"}
        
        # Update cross-bout analysis with graph file paths
        cross_bout_file = os.path.join(fencer_analysis_dir, 'cross_bout_analysis.json')
        
        if os.path.exists(cross_bout_file):
            try:
                with open(cross_bout_file, 'r', encoding='utf-8') as f:
                    cross_bout_data = json.load(f)
                
                # Add profile graph paths to the analysis
                if 'fencer_profile_plots' not in cross_bout_data:
                    cross_bout_data['fencer_profile_plots'] = {}
                
                cross_bout_data['fencer_profile_plots'].update(profile_results)
                
                # Save updated cross-bout analysis
                with open(cross_bout_file, 'w', encoding='utf-8') as f:
                    json.dump(cross_bout_data, f, indent=2, ensure_ascii=False)
                
                logging.info(f"Updated cross-bout analysis with profile plot paths")
                
            except Exception as e:
                logging.warning(f"Could not update cross-bout analysis: {str(e)}")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "profile_plots": profile_results,
            "total_graphs": sum(len(files) for files in profile_results.values()),
            "fencers_processed": list(profile_results.keys())
        }
        
    except Exception as e:
        logging.error(f"Error integrating fencer profile graphs: {str(e)}")
        return {"success": False, "error": str(e)}


def update_fencer_analysis_with_enhanced_tags(upload_id: int, base_results_dir: str = "/workspace/Project/results") -> Dict[str, Any]:
    """
    Update fencer analysis files with enhanced tagging system results.
    
    This function applies the enhanced tagging system we created earlier to existing
    fencer analysis data and updates the files with tag information.
    
    Args:
        upload_id: The upload ID to process
        base_results_dir: Base directory containing analysis results
        
    Returns:
        Dictionary with tagging results
    """
    try:
        # Import our enhanced tagging system
        import sys
        sys.path.append('/workspace/Project/your_scripts')
        from tagging import extract_tags_from_bout_analysis
        
        results_dir = os.path.join(base_results_dir, str(upload_id))
        fencer_analysis_dir = os.path.join(results_dir, 'fencer_analysis')
        match_analysis_dir = os.path.join(results_dir, 'match_analysis')
        
        if not os.path.exists(fencer_analysis_dir) or not os.path.exists(match_analysis_dir):
            return {"success": False, "error": "Required analysis directories not found"}
        
        # Process each match and extract tags
        all_tags = {"Fencer_Left": set(), "Fencer_Right": set()}
        bout_count = 0
        
        for file_name in os.listdir(match_analysis_dir):
            if file_name.endswith('_analysis.json'):
                match_file = os.path.join(match_analysis_dir, file_name)
                
                try:
                    with open(match_file, 'r', encoding='utf-8') as f:
                        match_data = json.load(f)
                    
                    # Extract tags for this bout
                    bout_tags = extract_tags_from_bout_analysis(match_data)
                    
                    # Accumulate tags
                    all_tags["Fencer_Left"].update(bout_tags.get('left', set()))
                    all_tags["Fencer_Right"].update(bout_tags.get('right', set()))
                    bout_count += 1
                    
                except Exception as e:
                    logging.warning(f"Error processing {file_name}: {str(e)}")
        
        # Update fencer analysis files with tag data
        for fencer_name in ["Fencer_Left", "Fencer_Right"]:
            fencer_file = os.path.join(fencer_analysis_dir, f'fencer_{fencer_name}_analysis.json')
            
            if os.path.exists(fencer_file):
                try:
                    with open(fencer_file, 'r', encoding='utf-8') as f:
                        fencer_data = json.load(f)
                    
                    # Add tag information
                    fencer_data['performance_tags'] = {
                        'all_tags': list(all_tags[fencer_name]),
                        'positive_tags': [tag for tag in all_tags[fencer_name] 
                                        if not any(neg in tag for neg in ['no_', 'poor_', 'excessive_', 'slow_', 'low_', 'missed_', 'failed_', 'limited_', 'insufficient_', 'inconsistent_'])],
                        'negative_tags': [tag for tag in all_tags[fencer_name] 
                                        if any(neg in tag for neg in ['no_', 'poor_', 'excessive_', 'slow_', 'low_', 'missed_', 'failed_', 'limited_', 'insufficient_', 'inconsistent_'])],
                        'total_bouts_analyzed': bout_count
                    }
                    
                    # Calculate tag statistics
                    total_tags = len(all_tags[fencer_name])
                    positive_count = len(fencer_data['performance_tags']['positive_tags'])
                    negative_count = len(fencer_data['performance_tags']['negative_tags'])
                    
                    fencer_data['performance_tags']['statistics'] = {
                        'total_unique_tags': total_tags,
                        'positive_tag_count': positive_count,
                        'negative_tag_count': negative_count,
                        'strength_weakness_ratio': positive_count / max(negative_count, 1)
                    }
                    
                    # Save updated fencer analysis
                    with open(fencer_file, 'w', encoding='utf-8') as f:
                        json.dump(fencer_data, f, indent=2, ensure_ascii=False)
                    
                    logging.info(f"Updated {fencer_name} analysis with {total_tags} performance tags")
                    
                except Exception as e:
                    logging.error(f"Error updating {fencer_name} analysis: {str(e)}")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "bouts_processed": bout_count,
            "fencer_tags": {fencer: len(tags) for fencer, tags in all_tags.items()},
            "total_unique_tags": len(set().union(*all_tags.values()))
        }
        
    except Exception as e:
        logging.error(f"Error updating fencer analysis with tags: {str(e)}")
        return {"success": False, "error": str(e)}


def generate_comprehensive_fencer_profile_report(upload_id: int, base_results_dir: str = "/workspace/Project/results") -> Dict[str, Any]:
    """
    Generate comprehensive fencer profile report including graphs, tags, and analysis.
    
    This is the main function that orchestrates the complete fencer profile generation process.
    
    Args:
        upload_id: The upload ID to process
        base_results_dir: Base directory containing analysis results
        
    Returns:
        Comprehensive report dictionary
    """
    try:
        logging.info(f"Generating comprehensive fencer profile report for upload {upload_id}")
        
        # Step 1: Update fencer analysis with enhanced tags
        tagging_result = update_fencer_analysis_with_enhanced_tags(upload_id, base_results_dir)
        
        # Step 2: Generate profile graphs
        graph_result = integrate_fencer_profile_graphs_with_analysis(upload_id, base_results_dir)
        
        # Step 3: Compile comprehensive report
        report = {
            "upload_id": upload_id,
            "generation_timestamp": None,  # Would be filled with current timestamp
            "tagging_analysis": tagging_result,
            "profile_visualization": graph_result,
            "summary": {
                "success": tagging_result.get("success", False) and graph_result.get("success", False),
                "total_graphs_generated": graph_result.get("total_graphs", 0),
                "fencers_analyzed": [],
                "key_insights": []
            }
        }
        
        # Add summary information
        if graph_result.get("success"):
            report["summary"]["fencers_analyzed"] = graph_result.get("fencers_processed", [])
        
        if tagging_result.get("success"):
            fencer_tags = tagging_result.get("fencer_tags", {})
            for fencer, tag_count in fencer_tags.items():
                insight = f"{fencer}: {tag_count} performance characteristics identified"
                report["summary"]["key_insights"].append(insight)
        
        # Save comprehensive report
        results_dir = os.path.join(base_results_dir, str(upload_id), 'fencer_analysis')
        os.makedirs(results_dir, exist_ok=True)
        report_file = os.path.join(results_dir, 'comprehensive_profile_report.json')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved comprehensive fencer profile report to {report_file}")
        
        return report
        
    except Exception as e:
        logging.error(f"Error generating comprehensive fencer profile report: {str(e)}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test with upload ID 101 (which has multiple matches)
    test_upload_id = 101
    base_dir = "/workspace/Project/results/1"  # Correct base directory
    
    print("Testing Fencer Profile Integration")
    print("=" * 50)
    
    # Test comprehensive report generation
    result = generate_comprehensive_fencer_profile_report(test_upload_id, base_dir)
    
    if result.get("success"):
        print("✅ Successfully generated comprehensive fencer profile report")
        print(f"   📊 Graphs: {result.get('profile_visualization', {}).get('total_graphs', 0)}")
        print(f"   🏷️  Tags: {result.get('tagging_analysis', {}).get('total_unique_tags', 0)} unique tags")
        print(f"   👥 Fencers: {len(result.get('summary', {}).get('fencers_analyzed', []))}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print(f"\nReport details: {json.dumps(result, indent=2)}")