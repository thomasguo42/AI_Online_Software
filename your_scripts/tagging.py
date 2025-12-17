"""
Tag extraction module for fencing bout analysis.

This module analyzes bout analysis JSON data and extracts relevant tags
for both left and right fencers based on their performance metrics.
"""

import json
import logging
from typing import Dict, List, Set, Any


def extract_tags_from_bout_analysis(analysis_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Extract tags for both fencers from bout analysis data.
    
    Args:
        analysis_data: The parsed JSON data from match_*_analysis.json
        
    Returns:
        Dictionary with 'left' and 'right' keys, each containing a set of English tag names
    """
    tags = {
        'left': set(),
        'right': set()
    }
    
    try:
        # Extract tags for left fencer
        left_data = analysis_data.get('left_data', {})
        english_tags_left = _extract_fencer_tags(left_data, 'left')
        tags['left'] = _translate_tags_to_english(english_tags_left)

        # Extract tags for right fencer
        right_data = analysis_data.get('right_data', {})
        english_tags_right = _extract_fencer_tags(right_data, 'right')
        tags['right'] = _translate_tags_to_english(english_tags_right)
        
        logging.info(f"Extracted tags - Left: {tags['left']}, Right: {tags['right']}")
        
    except Exception as e:
        logging.error(f"Error extracting tags from bout analysis: {str(e)}")
        
    return tags


def _translate_tags_to_english(english_tags: Set[str]) -> Set[str]:
    """
    Translate English tag names to display-friendly English using the translation map.

    Args:
        english_tags: Set of English tag names

    Returns:
        Set of display-friendly English tag names
    """
    translation_map = get_tag_translation_map()
    display_tags = set()

    for english_tag in english_tags:
        display_tag = translation_map.get(english_tag, english_tag)
        display_tags.add(display_tag)
        if display_tag == english_tag and english_tag in translation_map.values():
            # Tag was already in display format, no need to warn
            pass
        elif display_tag == english_tag:
            # No translation found, log a warning
            logging.warning(f"No display translation found for tag: {english_tag}")

    return display_tags


def _extract_fencer_tags(fencer_data: Dict[str, Any], side: str) -> Set[str]:
    """
    Extract tags for a single fencer based on their performance data.
    
    Args:
        fencer_data: The fencer's data from the analysis JSON
        side: 'left' or 'right' - used for logging
        
    Returns:
        Set of tag names applicable to this fencer
    """
    tags = set()
    
    try:
        # 1. Launch detection
        if fencer_data.get('has_launch', False):
            tags.add('launch')
        else:
            tags.add('no_launch')  # Negative tag for lack of launching
            
        # 2. Arm extension detection
        arm_extension_freq = fencer_data.get('arm_extension_freq', 0)
        if arm_extension_freq > 0:
            tags.add('arm_extension')
            # Check for excessive arm extensions
            if arm_extension_freq > 3:
                tags.add('excessive_arm_extensions')
        else:
            tags.add('no_arm_extension')  # Negative tag for lack of arm extensions
            
        # 3. Attack types from interval analysis
        interval_analysis = fencer_data.get('interval_analysis', {})
        
        # Check advance analyses for attack types
        advance_analyses = interval_analysis.get('advance_analyses', [])
        has_any_attack = False
        for analysis in advance_analyses:
            attack_info = analysis.get('attack_info', {})
            if attack_info.get('has_attack', False):
                has_any_attack = True
                attack_type = attack_info.get('attack_type', '')
                if attack_type:
                    tags.add(attack_type)  # e.g., 'simple_attack', 'simple_preparation'
                    
            # Check for poor attack distance management
            if not analysis.get('good_attack_distance', False) and attack_info.get('has_attack', False):
                tags.add('poor_attack_distance')
                
        # If no attacks were found, add negative tag
        if not has_any_attack:
            tags.add('no_attacks')
                    
        # Check retreat analyses for defensive qualities
        retreat_analyses = interval_analysis.get('retreat_analyses', [])
        has_good_defense = False
        for analysis in retreat_analyses:
            # Good defensive quality
            if analysis.get('defensive_quality') == 'good':
                tags.add('good_defensive_quality')
                has_good_defense = True
            elif analysis.get('defensive_quality') == 'poor':
                tags.add('poor_defensive_quality')
                
            # Distance management during retreats
            if analysis.get('maintained_safe_distance', False):
                tags.add('maintain_safe_distance')
            else:
                tags.add('poor_distance_maintenance')
                
            # Spacing consistency
            if analysis.get('consistent_spacing', False):
                tags.add('consistent_spacing')
            else:
                tags.add('inconsistent_spacing')
                
            # Missed counter-attack opportunities
            missed_opportunities = analysis.get('opportunities_missed', 0)
            if missed_opportunities > 0:
                tags.add('missed_counter_opportunities')
                
            # Failed distance pulls
            if analysis.get('successful_distance_pulls', 0) == 0 and len(analysis.get('launch_responses', [])) > 0:
                tags.add('failed_distance_pulls')
        
        # If no good defensive qualities found but retreats exist
        if retreat_analyses and not has_good_defense:
            tags.add('poor_defensive_quality')
        
        # 4. Tempo analysis
        tempo_quality_good = False
        for analysis in advance_analyses:
            tempo_type = analysis.get('tempo_type', '')
            if tempo_type:
                tags.add(tempo_type)  # e.g., 'steady_tempo', 'variable_tempo', 'broken_tempo'
                if tempo_type == 'steady_tempo':
                    tempo_quality_good = True
            
            # Check for excessive tempo changes (indicates poor rhythm control)
            tempo_changes = analysis.get('tempo_changes', 0)
            if tempo_changes > 5:
                tags.add('excessive_tempo_changes')
                
        # 5. Distance management
        poor_distance_count = 0
        for analysis in advance_analyses:
            if analysis.get('good_attack_distance', False):
                tags.add('good_attack_distance')
            elif analysis.get('attack_info', {}).get('has_attack', False):
                poor_distance_count += 1
                
        if poor_distance_count > 0:
            tags.add('poor_attack_distance')
                
        # 6. Additional attack characteristics
        summary = interval_analysis.get('summary', {})
        attacks = summary.get('attacks', {})
        
        total_attacks = attacks.get('total', 0)
        
        # Attack type analysis
        if attacks.get('simple', 0) > 0:
            tags.add('simple_attack')
        if attacks.get('compound', 0) > 0:
            tags.add('compound_attack')
        if attacks.get('holding', 0) > 0:
            tags.add('holding_attack')
        if attacks.get('preparations', 0) > 0:
            tags.add('preparation_attack')
            
        # Check for lack of attack variety (only one type of attack)
        attack_types_used = sum([1 for attack_type in ['simple', 'compound', 'holding', 'preparations'] 
                                if attacks.get(attack_type, 0) > 0])
        if total_attacks > 1 and attack_types_used == 1:
            tags.add('limited_attack_variety')
            
        # 7. Tempo variations from summary
        tempo = summary.get('tempo', {})
        if tempo.get('steady', 0) > 0:
            tags.add('steady_tempo')
        if tempo.get('variable', 0) > 0:
            tags.add('variable_tempo')
        if tempo.get('broken', 0) > 0:
            tags.add('broken_tempo')
            
        # 8. Performance ratios analysis
        advance_ratio = fencer_data.get('advance_ratio', 0)
        pause_ratio = fencer_data.get('pause_ratio', 0)
        
        # Excessive pausing
        if pause_ratio > 0.7:
            tags.add('excessive_pausing')
        
        # Insufficient forward pressure
        if advance_ratio < 0.2:
            tags.add('insufficient_forward_pressure')
            
        # 9. Velocity and acceleration analysis
        velocity = fencer_data.get('velocity', 0)
        acceleration = fencer_data.get('acceleration', 0)
        
        # Low velocity/speed
        if velocity < 1.0:
            tags.add('low_speed')
        elif velocity > 3.0:
            tags.add('high_speed')
            
        # Poor acceleration
        if acceleration < 5.0:
            tags.add('poor_acceleration')
        elif acceleration > 25.0:
            tags.add('excellent_acceleration')
            
        # 10. First step analysis
        first_step = fencer_data.get('first_step', {})
        if first_step.get('init_time', 0) > 0.2:
            tags.add('slow_reaction_time')
        elif first_step.get('init_time', 0) < 0.1:
            tags.add('fast_reaction_time')
            
        logging.debug(f"Extracted {len(tags)} tags for {side} fencer: {tags}")
        
    except Exception as e:
        logging.error(f"Error extracting tags for {side} fencer: {str(e)}")
        
    return tags


def get_tag_translation_map() -> Dict[str, str]:
    """
    Get mapping from English tag names to display-friendly English tag names.

    Returns:
        Dictionary mapping English names to display-friendly English names
    """
    return {
        # Launch and extensions
        'launch': 'Launch',
        'arm_extension': 'Arm Extension',
        'fast_reaction_time': 'Fast Reaction Time',
        'high_speed': 'High Speed',
        'excellent_acceleration': 'Excellent Acceleration',
        'no_launch': 'No Launch',
        'no_arm_extension': 'No Arm Extension',
        'excessive_arm_extensions': 'Excessive Arm Extensions',
        'slow_reaction_time': 'Slow Reaction Time',
        'low_speed': 'Low Speed',
        'poor_acceleration': 'Poor Acceleration',

        # Attack types
        'simple_attack': 'Simple Attack',
        'compound_attack': 'Compound Attack',
        'holding_attack': 'Holding Attack',
        'preparation_attack': 'Preparation Attack',
        'simple_preparation': 'Simple Preparation',
        'no_attacks': 'No Attacks',
        'limited_attack_variety': 'Limited Attack Variety',

        # Tempo types
        'steady_tempo': 'Steady Tempo',
        'variable_tempo': 'Variable Tempo',
        'broken_tempo': 'Broken Tempo',
        'excessive_tempo_changes': 'Excessive Tempo Changes',

        # Distance and spacing
        'good_attack_distance': 'Good Attack Distance',
        'maintain_safe_distance': 'Maintain Safe Distance',
        'consistent_spacing': 'Consistent Spacing',
        'poor_attack_distance': 'Poor Attack Distance',
        'poor_distance_maintenance': 'Poor Distance Maintenance',
        'inconsistent_spacing': 'Inconsistent Spacing',

        # Defense
        'good_defensive_quality': 'Good Defensive Quality',
        'poor_defensive_quality': 'Poor Defensive Quality',
        'missed_counter_opportunities': 'Missed Counter Opportunities',
        'failed_distance_pulls': 'Failed Distance Pulls',

        # Movement and pressure
        'excessive_pausing': 'Excessive Pausing',
        'insufficient_forward_pressure': 'Insufficient Forward Pressure'
    }

def get_predefined_tags() -> List[str]:
    """
    Get the list of all possible tags that can be assigned.

    This is useful for initializing the Tag table in the database.

    Returns:
        List of all possible tag names
    """
    return [
        # Launch and extensions (positive)
        'Launch',
        'Arm Extension',
        'Fast Reaction Time',
        'High Speed',
        'Excellent Acceleration',

        # Launch and extensions (negative)
        'No Launch',
        'No Arm Extension',
        'Excessive Arm Extensions',
        'Slow Reaction Time',
        'Low Speed',
        'Poor Acceleration',

        # Attack types (positive)
        'Simple Attack',
        'Compound Attack',
        'Holding Attack',
        'Preparation Attack',
        'Simple Preparation',

        # Attack types (negative)
        'No Attacks',
        'Limited Attack Variety',

        # Tempo types (positive)
        'Steady Tempo',
        'Variable Tempo',
        'Broken Tempo',

        # Tempo types (negative)
        'Excessive Tempo Changes',

        # Distance and spacing (positive)
        'Good Attack Distance',
        'Maintain Safe Distance',
        'Consistent Spacing',

        # Distance and spacing (negative)
        'Poor Attack Distance',
        'Poor Distance Maintenance',
        'Inconsistent Spacing',

        # Defense (positive)
        'Good Defensive Quality',

        # Defense (negative)
        'Poor Defensive Quality',
        'Missed Counter Opportunities',
        'Failed Distance Pulls',

        # Movement and pressure (positive)
        # (existing positive tags cover this)

        # Movement and pressure (negative)
        'Excessive Pausing',
        'Insufficient Forward Pressure'
    ]


def initialize_tags_in_database(db):
    """
    Initialize all predefined tags in the database.
    
    This should be called once to populate the Tag table with all possible tags.
    
    Args:
        db: Flask-SQLAlchemy database instance
    """
    from models import Tag
    
    predefined_tags = get_predefined_tags()
    
    for tag_name in predefined_tags:
        # Check if tag already exists
        existing_tag = Tag.query.filter_by(name=tag_name).first()
        if not existing_tag:
            new_tag = Tag(name=tag_name)
            db.session.add(new_tag)
            logging.info(f"Added new tag: {tag_name}")
    
    try:
        db.session.commit()
        logging.info(f"Successfully initialized {len(predefined_tags)} tags in database")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error initializing tags in database: {str(e)}")
        raise