#!/usr/bin/env python3
"""Test script to verify the fencer profile fix works correctly."""

import sys
sys.path.insert(0, '/workspace/Project')

from app import create_app
from your_scripts.professional_report_generator import generate_professional_report

def test_fencer_profile_fix():
    """Test that fencer 23's profile now has consistent data."""
    app = create_app()
    
    with app.app_context():
        fencer_id = 23
        user_id = 1
        fencer_name = "yt"
        
        print("=" * 80)
        print(f"Testing fencer profile fix for {fencer_name} (ID: {fencer_id})")
        print("=" * 80)
        
        # Regenerate the professional report
        print("\n📊 Regenerating professional report...")
        result = generate_professional_report(
            fencer_id=fencer_id,
            user_id=user_id,
            fencer_name=fencer_name
        )
        
        if not result.get('success'):
            print(f"❌ Failed to generate report: {result.get('error')}")
            return False
        
        print(f"✅ Report generated successfully")
        print(f"   Report path: {result['report_path']}")
        print(f"   Matches processed: {result['matches_processed']}")
        
        # Read and check the generated profile
        import json
        with open(result['report_path'], 'r') as f:
            profile = json.load(f)
        
        print("\n" + "=" * 80)
        print("VERIFICATION: Checking data consistency")
        print("=" * 80)
        
        # Get performance dashboard stats
        dashboard = profile.get('performance_dashboard', {})
        style_profile = dashboard.get('style_profile', [])
        
        # Get tactical analysis stats
        tactical = profile.get('tactical_analysis', {})
        
        print("\n📊 Performance Dashboard (style_profile):")
        for cat_data in style_profile:
            label = cat_data['label']
            count = cat_data['count']
            wins = cat_data['wins']
            losses = cat_data['losses']
            win_rate = cat_data['win_rate']
            print(f"   {label:8s}: {count:2d} bouts, {wins:2d} wins, {losses:2d} losses, {win_rate:5.1f}% win rate")
        
        print("\n🎯 Tactical Analysis (key_metrics):")
        for category in ['attack', 'defense', 'in_box']:
            cat_data = tactical.get(category, {})
            metrics = cat_data.get('key_metrics', {})
            count = metrics.get('bout_count', 0)
            wins = metrics.get('wins', 0)
            losses = metrics.get('losses', 0)
            win_rate = metrics.get('win_rate', 0)
            print(f"   {category:8s}: {count:2d} bouts, {wins:2d} wins, {losses:2d} losses, {win_rate:5.1f}% win rate")
        
        # Check for inconsistencies
        print("\n" + "=" * 80)
        print("CONSISTENCY CHECK")
        print("=" * 80)
        
        all_consistent = True
        
        for cat_data in style_profile:
            label = cat_data['label']
            dashboard_count = cat_data['count']
            dashboard_wins = cat_data['wins']
            dashboard_losses = cat_data['losses']
            dashboard_win_rate = cat_data['win_rate']
            
            tactical_data = tactical.get(label, {})
            tactical_metrics = tactical_data.get('key_metrics', {})
            tactical_count = tactical_metrics.get('bout_count', 0)
            tactical_wins = tactical_metrics.get('wins', 0)
            tactical_losses = tactical_metrics.get('losses', 0)
            tactical_win_rate = tactical_metrics.get('win_rate', 0)
            
            # Check consistency
            is_consistent = (
                dashboard_count == tactical_count and
                dashboard_wins == tactical_wins and
                dashboard_losses == tactical_losses and
                abs(dashboard_win_rate - tactical_win_rate) < 0.1  # Allow small floating point difference
            )
            
            status = "✅" if is_consistent else "❌"
            print(f"\n{status} {label}:")
            print(f"   Dashboard: {dashboard_count} bouts, {dashboard_wins}W {dashboard_losses}L, {dashboard_win_rate:.1f}%")
            print(f"   Tactical:  {tactical_count} bouts, {tactical_wins}W {tactical_losses}L, {tactical_win_rate:.1f}%")
            
            if not is_consistent:
                all_consistent = False
                print(f"   ⚠️  INCONSISTENCY DETECTED!")
        
        print("\n" + "=" * 80)
        if all_consistent:
            print("✅ ALL DATA IS CONSISTENT! Fix successful!")
        else:
            print("❌ INCONSISTENCIES REMAIN! Further investigation needed.")
        print("=" * 80)
        
        return all_consistent

if __name__ == '__main__':
    success = test_fencer_profile_fix()
    sys.exit(0 if success else 1)
