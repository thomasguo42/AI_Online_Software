#!/usr/bin/env python3
"""
Fencer-Centric Profile System

This module creates and manages fencer profiles independently of specific uploads/matches.
Profiles aggregate data across ALL of a fencer's performances and store in fencer-specific directories.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

# Add current directory to path for imports
sys.path.insert(0, '/workspace/Project/your_scripts')
sys.path.insert(0, '/workspace/Project')

from fencer_profile_plotting import save_fencer_profile_plots
from tagging import extract_tags_from_bout_analysis

def get_fencer_profile_directory(user_id: int, fencer_id: int, base_dir: str = "/workspace/Project") -> str:
    """
    Get the directory path for a fencer's profile.
    
    Args:
        user_id: User ID who owns the fencer
        fencer_id: Fencer ID
        base_dir: Base directory for the project
        
    Returns:
        Path to fencer's profile directory
    """
    return os.path.join(base_dir, 'fencer_profiles', str(user_id), str(fencer_id))

def collect_fencer_data_across_uploads(fencer_id: int, user_id: int, base_results_dir: str = "/workspace/Project/results") -> Dict[str, Any]:
    """
    Collect all data for a fencer across all their uploads/matches.
    
    Args:
        fencer_id: Fencer ID to collect data for
        user_id: User ID who owns the fencer
        base_results_dir: Base directory containing upload results
        
    Returns:
        Aggregated fencer data across all performances
    """
    import sys
    sys.path.insert(0, '/workspace/Project')
    from models import Upload, db
    from app import create_app
    
    # Create Flask app context to access database
    app = create_app()
    aggregated_data = {
        "fencer_id": fencer_id,
        "user_id": user_id,
        "last_updated": datetime.now().isoformat(),
        "total_uploads": 0,
        "total_bouts": 0,
        "aggregated_metrics": {},
        "all_bouts": [],
        "performance_tags": set(),
        "upload_sources": []
    }
    
    with app.app_context():
        # Find all uploads where this fencer participated
        uploads = Upload.query.filter(
            (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
        ).filter_by(user_id=user_id, status='completed').all()
        
        logging.info(f"Found {len(uploads)} uploads for fencer {fencer_id}")
        
        all_bout_data = []
        all_tags = set()
        
        for upload in uploads:
            fencer_side = 'left' if upload.left_fencer_id == fencer_id else 'right'
            fencer_side_key = f'{fencer_side}_data'
            
            # Check for analysis data
            upload_results_dir = os.path.join(base_results_dir, str(user_id), str(upload.id))
            
            # Look for fencer analysis data
            fencer_analysis_dir = os.path.join(upload_results_dir, 'fencer_analysis')
            if os.path.exists(fencer_analysis_dir):
                fencer_file = os.path.join(fencer_analysis_dir, f'fencer_Fencer_{fencer_side.title()}_analysis.json')
                
                if os.path.exists(fencer_file):
                    try:
                        with open(fencer_file, 'r', encoding='utf-8') as f:
                            fencer_upload_data = json.load(f)
                        
                        # Add this upload's data to aggregated data
                        all_bout_data.extend(fencer_upload_data.get('bouts', []))
                        aggregated_data['upload_sources'].append({
                            'upload_id': upload.id,
                            'fencer_side': fencer_side,
                            'bout_count': len(fencer_upload_data.get('bouts', []))
                        })
                        
                        logging.info(f"Loaded {len(fencer_upload_data.get('bouts', []))} bouts from upload {upload.id}")
                        
                    except Exception as e:
                        logging.error(f"Error loading fencer data from upload {upload.id}: {e}")
            
            # Look for match analysis to extract tags
            match_analysis_dir = os.path.join(upload_results_dir, 'match_analysis')
            if os.path.exists(match_analysis_dir):
                for file_name in os.listdir(match_analysis_dir):
                    if file_name.endswith('_analysis.json'):
                        match_file = os.path.join(match_analysis_dir, file_name)
                        try:
                            with open(match_file, 'r', encoding='utf-8') as f:
                                match_data = json.load(f)
                            
                            # Extract tags for this bout
                            bout_tags = extract_tags_from_bout_analysis(match_data)
                            if fencer_side in bout_tags:
                                all_tags.update(bout_tags[fencer_side])
                                
                        except Exception as e:
                            logging.error(f"Error processing match analysis {match_file}: {e}")
        
        # Aggregate all the bout data
        if all_bout_data:
            aggregated_data['total_uploads'] = len(uploads)
            aggregated_data['total_bouts'] = len(all_bout_data)
            aggregated_data['all_bouts'] = all_bout_data
            aggregated_data['performance_tags'] = list(all_tags)
            
            # Calculate aggregated metrics
            aggregated_data['aggregated_metrics'] = calculate_aggregated_metrics(all_bout_data)
            
            # Create a structure compatible with the plotting system
            aggregated_data['bouts'] = all_bout_data
            aggregated_data['metrics'] = aggregated_data['aggregated_metrics']
            
        return aggregated_data

def calculate_aggregated_metrics(all_bouts: List[Dict]) -> Dict[str, Any]:
    """
    Calculate aggregated metrics across all bouts for a fencer.
    
    Args:
        all_bouts: List of bout data dictionaries
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not all_bouts:
        return {}
    
    # Collect all metrics
    velocities = []
    accelerations = []
    advance_ratios = []
    pause_ratios = []
    first_step_times = []
    arm_extension_counts = []
    launch_counts = 0
    
    for bout in all_bouts:
        metrics = bout.get('metrics', {})
        
        if 'velocity' in metrics:
            velocities.append(metrics['velocity'])
        if 'acceleration' in metrics:
            accelerations.append(metrics['acceleration'])
        if 'advance_ratio' in metrics:
            advance_ratios.append(metrics['advance_ratio'])
        if 'pause_ratio' in metrics:
            pause_ratios.append(metrics['pause_ratio'])
        if 'first_step' in metrics and 'init_time' in metrics['first_step']:
            first_step_times.append(metrics['first_step']['init_time'])
        if 'arm_extension_freq' in metrics:
            arm_extension_counts.append(metrics['arm_extension_freq'])
        if metrics.get('has_launch'):
            launch_counts += 1
    
    # Calculate aggregated statistics
    aggregated = {
        'total_bouts': len(all_bouts),
        'avg_velocity': sum(velocities) / len(velocities) if velocities else 0,
        'avg_acceleration': sum(accelerations) / len(accelerations) if accelerations else 0,
        'avg_advance_ratio': sum(advance_ratios) / len(advance_ratios) if advance_ratios else 0,
        'avg_pause_ratio': sum(pause_ratios) / len(pause_ratios) if pause_ratios else 0,
        'avg_first_step_init': sum(first_step_times) / len(first_step_times) if first_step_times else 0,
        'total_arm_extensions': sum(arm_extension_counts),
        'avg_arm_extension_duration': 0.5,  # Default value
        'launch_success_rate': launch_counts / len(all_bouts) if all_bouts else 0,
        'attacking_ratio': 0.5,  # Default value - could be calculated from bout analysis
    }
    
    return aggregated

def generate_fencer_profile(fencer_id: int, user_id: int, force_regenerate: bool = False) -> Dict[str, Any]:
    """
    Generate or update a complete fencer profile with graphs.
    
    Args:
        fencer_id: Fencer ID to generate profile for
        user_id: User ID who owns the fencer
        force_regenerate: If True, regenerate even if recent profile exists
        
    Returns:
        Dictionary with generation results and file paths
    """
    try:
        # Get fencer profile directory
        profile_dir = get_fencer_profile_directory(user_id, fencer_id)
        os.makedirs(profile_dir, exist_ok=True)
        
        profile_data_file = os.path.join(profile_dir, 'profile_data.json')
        last_updated_file = os.path.join(profile_dir, 'last_updated.json')
        
        # Check if we need to regenerate
        should_generate = force_regenerate
        if not should_generate:
            if not os.path.exists(profile_data_file) or not os.path.exists(last_updated_file):
                should_generate = True
            else:
                # Check if data is recent (less than 1 day old)
                try:
                    with open(last_updated_file, 'r') as f:
                        last_updated = json.load(f)
                    
                    last_update_time = datetime.fromisoformat(last_updated['timestamp'])
                    if (datetime.now() - last_update_time).days > 0:
                        should_generate = True
                        logging.info(f"Profile data is {(datetime.now() - last_update_time).days} days old, regenerating")
                except Exception:
                    should_generate = True
        
        if should_generate:
            logging.info(f"Generating fencer profile for fencer {fencer_id}")
            
            # Collect all fencer data across uploads
            fencer_data = collect_fencer_data_across_uploads(fencer_id, user_id)
            
            if not fencer_data.get('all_bouts'):
                # Create an empty profile for new fencers
                return {
                    "success": True,
                    "fencer_id": fencer_id,
                    "user_id": user_id,
                    "profile_directory": profile_dir,
                    "plot_files": {},
                    "total_bouts": 0,
                    "total_uploads": 0,
                    "performance_tags": 0,
                    "message": "No analysis data available yet. Upload and analyze videos to generate profile graphs."
                }
            
            # Save profile data
            with open(profile_data_file, 'w', encoding='utf-8') as f:
                json.dump(fencer_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate profile graphs
            plots_dir = os.path.join(profile_dir, 'profile_plots')
            fencer_name = f"Fencer_{fencer_id}"
            
            plot_files = save_fencer_profile_plots(fencer_data, fencer_name, plots_dir)
            
            # Update last updated timestamp
            with open(last_updated_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'fencer_id': fencer_id,
                    'user_id': user_id,
                    'total_bouts': fencer_data['total_bouts'],
                    'total_uploads': fencer_data['total_uploads']
                }, f, indent=2)
            
            logging.info(f"Generated fencer profile with {len(plot_files)} graphs")
            
            return {
                "success": True,
                "fencer_id": fencer_id,
                "user_id": user_id,
                "profile_directory": profile_dir,
                "plot_files": plot_files,
                "total_bouts": fencer_data['total_bouts'],
                "total_uploads": fencer_data['total_uploads'],
                "performance_tags": len(fencer_data['performance_tags'])
            }
        else:
            # Load existing data
            with open(profile_data_file, 'r', encoding='utf-8') as f:
                fencer_data = json.load(f)
            
            # Check for existing plots
            plots_dir = os.path.join(profile_dir, 'profile_plots')
            plot_files = {}
            
            if os.path.exists(plots_dir):
                for file_name in os.listdir(plots_dir):
                    if file_name.endswith('.png'):
                        plot_type = file_name.replace('.png', '').replace(f'fencer_{fencer_id}_', '')
                        plot_files[plot_type] = os.path.join(plots_dir, file_name)
            
            return {
                "success": True,
                "fencer_id": fencer_id,
                "user_id": user_id,
                "profile_directory": profile_dir,
                "plot_files": plot_files,
                "cached": True,
                "total_bouts": fencer_data.get('total_bouts', 0),
                "total_uploads": fencer_data.get('total_uploads', 0),
                "performance_tags": len(fencer_data.get('performance_tags', []))
            }
            
    except Exception as e:
        logging.error(f"Error generating fencer profile: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "fencer_id": fencer_id
        }

def update_all_fencer_profiles(user_id: int, force_regenerate: bool = False) -> Dict[str, Any]:
    """
    Update profiles for all fencers belonging to a user.
    
    Args:
        user_id: User ID to update all fencers for
        force_regenerate: If True, regenerate all profiles even if recent
        
    Returns:
        Summary of update results
    """
    import sys
    sys.path.insert(0, '/workspace/Project')
    from models import Fencer
    from app import create_app
    
    app = create_app()
    results = {
        "user_id": user_id,
        "updated_fencers": [],
        "errors": [],
        "total_processed": 0
    }
    
    with app.app_context():
        fencers = Fencer.query.filter_by(user_id=user_id).all()
        
        for fencer in fencers:
            try:
                result = generate_fencer_profile(fencer.id, user_id, force_regenerate)
                if result['success']:
                    results['updated_fencers'].append({
                        'fencer_id': fencer.id,
                        'fencer_name': fencer.name,
                        'total_bouts': result['total_bouts'],
                        'cached': result.get('cached', False)
                    })
                else:
                    results['errors'].append({
                        'fencer_id': fencer.id,
                        'fencer_name': fencer.name,
                        'error': result['error']
                    })
                results['total_processed'] += 1
            except Exception as e:
                results['errors'].append({
                    'fencer_id': fencer.id,
                    'fencer_name': fencer.name,
                    'error': str(e)
                })
    
    return results

if __name__ == "__main__":
    # Test with user 1's fencers
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Fencer-Centric Profile System")
    print("=" * 50)
    
    # Update all profiles for user 1
    results = update_all_fencer_profiles(user_id=1, force_regenerate=True)
    
    print(f"Processed {results['total_processed']} fencers for user {results['user_id']}")
    print(f"Successfully updated: {len(results['updated_fencers'])}")
    print(f"Errors: {len(results['errors'])}")
    
    for fencer in results['updated_fencers']:
        print(f"  ✅ {fencer['fencer_name']} (ID: {fencer['fencer_id']}) - {fencer['total_bouts']} bouts")
        
    for error in results['errors']:
        print(f"  ❌ {error['fencer_name']} (ID: {error['fencer_id']}) - {error['error']}")