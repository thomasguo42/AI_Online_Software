#!/usr/bin/env python3
"""
Direct profile generator that works like holistic analysis - checks all videos every time.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import case

# Set Chinese font (prefer Simplified Chinese)
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Noto Sans SC',
    'SimHei',
    'Microsoft YaHei',
    'DejaVu Sans',
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, '/workspace/Project')
sys.path.insert(0, '/workspace/Project/your_scripts')

def generate_fencer_profile_directly(fencer_id: int, user_id: int, fencer_name: str, base_dir: str = "/workspace/Project", weapon_type: str = None) -> Dict[str, Any]:
    """
    Generate fencer profile directly by scanning all completed uploads, like holistic analysis.
    
    Args:
        fencer_id: Fencer ID to generate profile for
        user_id: User ID who owns the fencer
        fencer_name: Name of the fencer
        base_dir: Base directory for the project
        weapon_type: Filter uploads by weapon type ('saber', 'foil', or 'epee'). If None, uses all uploads.
        
    Returns:
        Dictionary with generation results and file paths
    """
    
    sys.path.insert(0, '/workspace/Project')
    from models import Upload, db
    from app import create_app
    from tagging import extract_tags_from_bout_analysis
    
    app = create_app()
    
    try:
        with app.app_context():
            weapon_filter_text = f" ({weapon_type})" if weapon_type else ""
            print(f"🔍 Generating profile for {fencer_name} (ID: {fencer_id}, User: {user_id}){weapon_filter_text}")
            
            # Find all completed uploads where this fencer participated
            query = Upload.query.filter(
                (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
            ).filter_by(user_id=user_id, status='completed')
            
            # Filter by weapon type if specified
            if weapon_type:
                query = query.filter_by(weapon_type=weapon_type)

            query = query.order_by(
                case((Upload.match_datetime.is_(None), 1), else_=0),
                Upload.match_datetime.asc(),
                Upload.id.asc()
            )

            uploads = query.all()
            
            print(f"   Found {len(uploads)} completed uploads for this fencer")
            
            # Collect all bout data and analysis
            all_bout_data = []
            all_tags = set()
            upload_sources = []
            
            for upload in uploads:
                fencer_side = 'left' if upload.left_fencer_id == fencer_id else 'right'
                fencer_side_key = f'{fencer_side}_data'
                
                print(f"   Processing Upload {upload.id}: {fencer_name} is {fencer_side} fencer")
                
                # Check for fencer analysis data
                upload_results_dir = os.path.join(base_dir, 'results', str(user_id), str(upload.id))
                fencer_analysis_dir = os.path.join(upload_results_dir, 'fencer_analysis')
                
                if os.path.exists(fencer_analysis_dir):
                    # Load individual fencer analysis
                    fencer_file = os.path.join(fencer_analysis_dir, f'fencer_Fencer_{fencer_side.title()}_analysis.json')
                    
                    if os.path.exists(fencer_file):
                        try:
                            with open(fencer_file, 'r', encoding='utf-8') as f:
                                fencer_upload_data = json.load(f)
                            
                            bout_data = fencer_upload_data.get('bouts', [])
                            match_datetime_iso = upload.match_datetime.isoformat() if upload.match_datetime else None
                            for bout in bout_data:
                                bout['upload_id'] = upload.id
                                bout['fencer_side'] = fencer_side
                                bout['match_datetime'] = match_datetime_iso
                                bout['match_title'] = upload.match_title
                                bout['weapon_type'] = upload.weapon_type
                            all_bout_data.extend(bout_data)
                            
                            upload_sources.append({
                                'upload_id': upload.id,
                                'fencer_side': fencer_side,
                                'bout_count': len(bout_data),
                                'match_datetime': match_datetime_iso,
                                'match_title': upload.match_title,
                                'weapon_type': upload.weapon_type,
                                'is_multi_video': upload.is_multi_video
                            })
                            
                            print(f"     ✅ Loaded {len(bout_data)} bouts from fencer analysis")
                            
                        except Exception as e:
                            print(f"     ❌ Error loading fencer analysis: {e}")
                    else:
                        print(f"     ❌ No fencer analysis file: {fencer_file}")
                    
                    # Extract tags from match analysis files
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
                                    print(f"     ❌ Error processing match analysis {match_file}: {e}")
                else:
                    print(f"     ❌ No analysis directory: {fencer_analysis_dir}")
            
            print(f"   Total bouts collected: {len(all_bout_data)}")
            print(f"   Total unique tags: {len(all_tags)}")
            
            if len(all_bout_data) == 0:
                return {
                    "success": False,
                    "error": "No bout data found for this fencer",
                    "fencer_id": fencer_id,
                    "message": "Upload and analyze videos with this fencer to generate profile graphs."
                }
            
            # Create profile directory (weapon-specific if weapon type specified)
            if weapon_type:
                profile_dir = os.path.join(base_dir, 'fencer_profiles', str(user_id), str(fencer_id), weapon_type)
            else:
                profile_dir = os.path.join(base_dir, 'fencer_profiles', str(user_id), str(fencer_id))
            os.makedirs(profile_dir, exist_ok=True)
            
            plots_dir = os.path.join(profile_dir, 'profile_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Calculate aggregated metrics
            aggregated_data = {
                "fencer_id": fencer_id,
                "user_id": user_id,
                "fencer_name": fencer_name,
                "last_updated": datetime.now().isoformat(),
                "total_uploads": len(uploads),
                "total_bouts": len(all_bout_data),
                "all_bouts": all_bout_data,
                "performance_tags": list(all_tags),
                "upload_sources": upload_sources,
                "bouts": all_bout_data,  # For compatibility with plotting functions
                "metrics": calculate_aggregated_metrics(all_bout_data)
            }
            
            # Save profile data
            profile_data_file = os.path.join(profile_dir, 'profile_data.json')
            with open(profile_data_file, 'w', encoding='utf-8') as f:
                json.dump(aggregated_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate graphs
            plot_files = {}
            
            try:
                # Generate radar chart
                radar_file = create_fencer_radar_chart(aggregated_data, plots_dir, fencer_id, weapon_type)
                if radar_file and os.path.exists(radar_file):
                    plot_files['radar_profile'] = radar_file
                    print(f"   ✅ Generated radar chart: {os.path.basename(radar_file)}")
                
                # Generate comprehensive analysis chart
                analysis_file = create_fencer_analysis_chart(aggregated_data, plots_dir, fencer_id, weapon_type)
                if analysis_file and os.path.exists(analysis_file):
                    plot_files['profile_analysis'] = analysis_file
                    plot_files['comprehensive_profile'] = analysis_file  # Alias
                    print(f"   ✅ Generated analysis chart: {os.path.basename(analysis_file)}")
                    
            except Exception as e:
                print(f"   ❌ Error generating charts: {e}")
                import traceback
                traceback.print_exc()
            
            # Update timestamp
            timestamp_file = os.path.join(profile_dir, 'last_updated.json')
            with open(timestamp_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'fencer_id': fencer_id,
                    'user_id': user_id,
                    'fencer_name': fencer_name,
                    'total_bouts': len(all_bout_data),
                    'total_uploads': len(uploads)
                }, f, indent=2)
            
            print(f"✅ Profile generation completed for {fencer_name}")
            print(f"   Directory: {profile_dir}")
            print(f"   Generated {len(plot_files)} graph files")
            
            return {
                "success": True,
                "fencer_id": fencer_id,
                "user_id": user_id,
                "fencer_name": fencer_name,
                "profile_directory": profile_dir,
                "plot_files": plot_files,
                "total_bouts": len(all_bout_data),
                "total_uploads": len(uploads),
                "performance_tags": len(all_tags)
            }
            
    except Exception as e:
        print(f"❌ Error generating profile for {fencer_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "fencer_id": fencer_id
        }

def calculate_aggregated_metrics(all_bouts: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregated metrics from all bouts"""
    if not all_bouts:
        return {}
    
    # Collect all metrics
    velocities = []
    accelerations = []
    advance_ratios = []
    pause_ratios = []
    
    for bout in all_bouts:
        metrics = bout.get('metrics', {})
        
        if 'velocity' in metrics and metrics['velocity'] is not None:
            velocities.append(float(metrics['velocity']))
        if 'acceleration' in metrics and metrics['acceleration'] is not None:
            accelerations.append(float(metrics['acceleration']))
        if 'advance_ratio' in metrics and metrics['advance_ratio'] is not None:
            advance_ratios.append(float(metrics['advance_ratio']))
        if 'pause_ratio' in metrics and metrics['pause_ratio'] is not None:
            pause_ratios.append(float(metrics['pause_ratio']))
    
    # Calculate aggregated statistics with defaults
    return {
        'total_bouts': len(all_bouts),
        'avg_velocity': np.mean(velocities) if velocities else 2.5,
        'avg_acceleration': np.mean(accelerations) if accelerations else 1.2,
        'avg_advance_ratio': np.mean(advance_ratios) if advance_ratios else 0.4,
        'avg_pause_ratio': np.mean(pause_ratios) if pause_ratios else 0.3,
        'avg_first_step_init': 0.5,  # Default
        'total_arm_extensions': len(all_bouts) * 3,  # Estimate
        'avg_arm_extension_duration': 0.8,
        'launch_success_rate': 0.6,  # Default
        'attacking_ratio': 0.5,
    }

def create_fencer_radar_chart(fencer_data: Dict[str, Any], output_dir: str, fencer_id: int, weapon_type: str = None) -> str:
    """Create radar chart for fencer performance"""
    try:
        metrics = fencer_data.get('metrics', {})
        
        # Define the 8 dimensions for radar chart
        categories = [
            'Velocity', 'Acceleration', 'Advance Ratio', 'Pause Control',
            'First Step', 'Arm Extensions', 'Launch Success', 'Attack Ratio'
        ]
        
        # Normalize values to 0-1 scale for radar chart
        values = [
            min(1.0, metrics.get('avg_velocity', 2.5) / 5.0),
            min(1.0, metrics.get('avg_acceleration', 1.2) / 3.0),
            metrics.get('avg_advance_ratio', 0.4),
            1.0 - metrics.get('avg_pause_ratio', 0.3),  # Invert pause ratio
            1.0 - min(1.0, metrics.get('avg_first_step_init', 0.5)),  # Invert (faster is better)
            min(1.0, metrics.get('total_arm_extensions', 10) / 20.0),
            metrics.get('launch_success_rate', 0.6),
            metrics.get('attacking_ratio', 0.5)
        ]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each category
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=fencer_data.get('fencer_name', f'Fencer {fencer_id}'))
        ax.fill(angles, values, alpha=0.25)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.title(f'{fencer_data.get("fencer_name", f"Fencer {fencer_id}")} - Performance Radar\n({metrics.get("total_bouts", 0)} bouts analyzed)', 
                 size=16, fontweight='bold', pad=20)
        
        # Save
        weapon_suffix = f"_{weapon_type}" if weapon_type else ""
        output_file = os.path.join(output_dir, f'fencer_{fencer_id}_radar_profile{weapon_suffix}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file
        
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        return None

def create_fencer_analysis_chart(fencer_data: Dict[str, Any], output_dir: str, fencer_id: int, weapon_type: str = None) -> str:
    """Create comprehensive analysis chart"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{fencer_data.get("fencer_name", f"Fencer {fencer_id}")} - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        metrics = fencer_data.get('metrics', {})
        
        # 1. Performance Metrics Bar Chart
        metric_names = ['Velocity', 'Acceleration', 'Advance Ratio', 'Launch Success']
        metric_values = [
            metrics.get('avg_velocity', 2.5),
            metrics.get('avg_acceleration', 1.2),
            metrics.get('avg_advance_ratio', 0.4),
            metrics.get('launch_success_rate', 0.6)
        ]
        
        bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_title('Key Performance Metrics')
        ax1.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. Tag Distribution
        tags = fencer_data.get('performance_tags', [])
        if tags:
            # Count positive vs negative tags (simple heuristic)
            positive_tags = [t for t in tags if not any(neg in t for neg in ['poor', 'no_', 'low_', 'failed', 'insufficient', 'missed', 'broken', 'excessive'])]
            negative_tags = [t for t in tags if t not in positive_tags]
            
            tag_counts = [len(positive_tags), len(negative_tags)]
            tag_labels = ['Positive Traits', 'Areas for Improvement']
            
            ax2.pie(tag_counts, labels=tag_labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax2.set_title(f'Performance Tag Distribution\n({len(tags)} total tags)')
        else:
            ax2.text(0.5, 0.5, 'No tags available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Performance Tag Distribution')
        
        # 3. Bout Progression (if multiple bouts)
        bouts = fencer_data.get('all_bouts', [])
        if len(bouts) > 1:
            bout_numbers = list(range(1, len(bouts) + 1))
            velocities = [bout.get('metrics', {}).get('velocity', 2.5) for bout in bouts]
            
            ax3.plot(bout_numbers, velocities, 'o-', color='blue')
            ax3.set_title('Performance Progression (Velocity)')
            ax3.set_xlabel('Bout Number')
            ax3.set_ylabel('Velocity (m/s)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, f'Single bout\n(Need multiple bouts for progression)', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Performance Progression')
        
        # 4. Upload Sources
        upload_sources = fencer_data.get('upload_sources', [])
        if upload_sources:
            upload_ids = [f"Upload {src['upload_id']}" for src in upload_sources]
            bout_counts = [src['bout_count'] for src in upload_sources]
            
            ax4.bar(upload_ids, bout_counts, color='lightblue')
            ax4.set_title('Bouts per Upload')
            ax4.set_ylabel('Number of Bouts')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No upload data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Bouts per Upload')
        
        plt.tight_layout()
        
        # Save
        weapon_suffix = f"_{weapon_type}" if weapon_type else ""
        output_file = os.path.join(output_dir, f'fencer_{fencer_id}_profile_analysis{weapon_suffix}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file
        
    except Exception as e:
        print(f"Error creating analysis chart: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test the direct generator
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Direct Profile Generator")
    print("=" * 50)
    
    # Test with user 2, fencer 2 (leo)
    result = generate_fencer_profile_directly(2, 2, "leo")
    
    if result.get('success'):
        print(f"\n✅ SUCCESS: Generated profile for leo")
        print(f"   Total bouts: {result['total_bouts']}")
        print(f"   Generated files: {list(result['plot_files'].keys())}")
    else:
        print(f"\n❌ FAILED: {result.get('error')}")
