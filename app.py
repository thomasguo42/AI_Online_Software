from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify, flash, current_app, abort
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import logging
import subprocess
import json
import traceback
import re
import threading
from datetime import datetime
import shutil
import sqlite3
import cv2
import pandas as pd
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Ensure Gemini config available from environment; do not hardcode keys
if os.getenv('GEMINI_API_KEY'):
    # Make available via Flask app config later
    pass
from celery_config import celery, init_celery
from celery.result import AsyncResult
from models import db, User, Upload, UploadVideo, Fencer, Bout, Tag, BoutTag, HolisticAnalysis, VideoAnalysis
from your_scripts.video_analysis import process_first_frame
from your_scripts.professional_report_generator import generate_professional_report
from your_scripts.gemini_rest import generate_text as gemini_generate_text
from flask import session  # Ensure this import is at the top
from prompts import generate_holistic_gpt_prompt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Set font to support Chinese characters (prefer Simplified Chinese)
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
import time
import numpy as np
import sys
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/Project/app.log'),
        logging.StreamHandler()
    ]
)

DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash-lite'
os.environ.setdefault('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)

# Ensure SQLite engine uses WAL and sane busy timeout to reduce writer lock errors
@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    try:
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")  # 30s
            finally:
                cursor.close()
    except Exception:
        # Best-effort; do not block app startup if PRAGMA fails
        pass


def _extract_candidate_payload(data: Dict) -> Tuple[str, List[Any]]:
    """Extract textual and JSON payload fragments from Gemini response."""
    if not isinstance(data, dict):
        return '', []

    candidates = data.get('candidates') or []
    if not candidates:
        return '', []

    text_parts: List[str] = []
    json_parts: List[Any] = []

    for candidate in candidates:
        content_obj = candidate.get('content')
        if isinstance(content_obj, dict):
            parts = content_obj.get('parts') or []
        elif isinstance(content_obj, list):
            parts = content_obj
        else:
            parts = []

        for part in parts:
            if not isinstance(part, dict):
                continue
            if 'jsonValue' in part and part['jsonValue'] is not None:
                json_parts.append(part['jsonValue'])
            if 'text' in part and isinstance(part['text'], str):
                text_parts.append(part['text'])

        # Fallback support for legacy fields
        legacy_text = candidate.get('text') or candidate.get('output')
        if isinstance(legacy_text, str) and legacy_text.strip():
            text_parts.append(legacy_text)

    combined_text = ''.join(text_parts).strip()

    parsed_fragments: List[Any] = []
    for item in json_parts:
        if isinstance(item, str):
            try:
                parsed_fragments.append(json.loads(item))
            except Exception:
                parsed_fragments.append(item)
        else:
            parsed_fragments.append(item)

    return combined_text, parsed_fragments


def _extract_ai_video_analysis(video_analysis):
    """Parse cached AI analysis from VideoAnalysis record and assess completeness."""
    section_label_map = {
        'loss_analysis': 'Loss Analysis',
        'win_analysis': 'Win Pattern Analysis',
        'category_performance': 'Category Performance Analysis',
        'overall_performance': 'Overall Performance Analysis',
        'reason_reports': 'Detailed Win/Loss Reports'
    }

    def _is_empty(obj):
        if obj is None:
            return True
        if isinstance(obj, (list, tuple, set)):
            return len(obj) == 0
        if isinstance(obj, dict):
            return len(obj) == 0
        return False

    result = {
        'analysis_status': 'none',
        'analysis_generated_at': None,
        'analysis_ready': False,
        'analysis_complete': False,
        'loss_analysis': {},
        'win_analysis': {},
        'category_performance_analysis': {'left_fencer': {}, 'right_fencer': {}},
        'overall_performance_analysis': {'left_fencer': {}, 'right_fencer': {}},
        'loss_reason_reports': {},
        'win_reason_reports': {},
        'reason_summary_bullets': {},
        'ai_section_status': {key: False for key in section_label_map.keys()},
        'missing_section_labels': list(section_label_map.values())
    }

    if not video_analysis:
        return result

    result['analysis_status'] = video_analysis.status or 'none'
    result['analysis_generated_at'] = video_analysis.generated_at

    if video_analysis.status != 'completed':
        return result

    result['analysis_ready'] = True

    if video_analysis.loss_analysis:
        try:
            parsed_outcomes = json.loads(video_analysis.loss_analysis)
            if isinstance(parsed_outcomes, dict) and ('loss' in parsed_outcomes or 'win' in parsed_outcomes):
                result['loss_analysis'] = parsed_outcomes.get('loss', {}) or {}
                result['win_analysis'] = parsed_outcomes.get('win', {}) or {}
            else:
                result['loss_analysis'] = parsed_outcomes or {}
                result['win_analysis'] = {}
        except json.JSONDecodeError:
            logging.warning("Failed to parse cached loss analysis JSON for upload %s", video_analysis.upload_id)
            result['analysis_ready'] = False

    try:
        if video_analysis.left_category_analysis:
            left_category = json.loads(video_analysis.left_category_analysis)
            if isinstance(left_category, dict):
                result['category_performance_analysis']['left_fencer'] = left_category
        if video_analysis.right_category_analysis:
            right_category = json.loads(video_analysis.right_category_analysis)
            if isinstance(right_category, dict):
                result['category_performance_analysis']['right_fencer'] = right_category
    except json.JSONDecodeError:
        logging.warning("Failed to parse cached category analysis JSON for upload %s", video_analysis.upload_id)
        result['analysis_ready'] = False

    try:
        if video_analysis.left_overall_analysis:
            left_overall = json.loads(video_analysis.left_overall_analysis)
            if isinstance(left_overall, dict):
                result['overall_performance_analysis']['left_fencer'] = left_overall
        if video_analysis.right_overall_analysis:
            right_overall = json.loads(video_analysis.right_overall_analysis)
            if isinstance(right_overall, dict):
                result['overall_performance_analysis']['right_fencer'] = right_overall
    except json.JSONDecodeError:
        logging.warning("Failed to parse cached overall analysis JSON for upload %s", video_analysis.upload_id)
        result['analysis_ready'] = False

    if video_analysis.detailed_analysis:
        try:
            reason_cache = json.loads(video_analysis.detailed_analysis)
        except json.JSONDecodeError:
            logging.warning("Failed to parse cached reason reports JSON for upload %s", video_analysis.upload_id)
            reason_cache = {}
            result['analysis_ready'] = False
        if isinstance(reason_cache, dict):
            result['loss_reason_reports'] = reason_cache.get('loss_reason_reports', {}) or {}
            result['win_reason_reports'] = reason_cache.get('win_reason_reports', {}) or {}
            result['reason_summary_bullets'] = reason_cache.get('reason_summary_bullets', {}) or {}

    result['ai_section_status']['loss_analysis'] = not _is_empty(result['loss_analysis'])
    result['ai_section_status']['win_analysis'] = not _is_empty(result['win_analysis'])

    category_left = result['category_performance_analysis'].get('left_fencer')
    category_right = result['category_performance_analysis'].get('right_fencer')
    result['ai_section_status']['category_performance'] = not (_is_empty(category_left) or _is_empty(category_right))

    overall_left = result['overall_performance_analysis'].get('left_fencer')
    overall_right = result['overall_performance_analysis'].get('right_fencer')
    result['ai_section_status']['overall_performance'] = not (_is_empty(overall_left) or _is_empty(overall_right))

    reason_complete = not (_is_empty(result['loss_reason_reports']) or _is_empty(result['win_reason_reports']))
    result['ai_section_status']['reason_reports'] = reason_complete

    result['analysis_complete'] = result['analysis_ready'] and all(result['ai_section_status'].values())
    result['missing_section_labels'] = [
        section_label_map[key]
        for key, ready in result['ai_section_status'].items()
        if not ready
    ]

    return result


def _dispatch_video_view_regeneration(upload_id: int):
    """Queue or execute video view regeneration and return (task_id, ran_inline)."""
    try:
        prefer_async = bool(current_app.config.get('VIDEO_VIEW_REGEN_ASYNC', True))
    except RuntimeError:
        # Outside Flask context; assume async is allowed
        prefer_async = True

    from tasks import regenerate_video_analysis_task

    if prefer_async:
        try:
            async_result = regenerate_video_analysis_task.apply_async(args=[upload_id])
            return async_result.id, False
        except Exception as exc:
            logging.warning(
                "Falling back to inline video view regeneration for upload %s due to Celery error: %s",
                upload_id,
                exc,
                exc_info=True
            )
    else:
        logging.info("Video view regeneration configured for inline execution; skipping Celery queue for upload %s", upload_id)

    from models import VideoAnalysis

    try:
        now = datetime.utcnow()
        analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
        if analysis:
            analysis.status = 'pending'
            analysis.error_message = None
            analysis.generated_at = now
        else:
            analysis = VideoAnalysis(upload_id=upload_id, status='pending', generated_at=now)
            db.session.add(analysis)
        db.session.commit()
    except Exception as status_exc:
        db.session.rollback()
        logging.warning(
            "Failed to stage pending status for upload %s before inline regeneration: %s",
            upload_id,
            status_exc
        )

    def _run_inline():
        try:
            regenerate_video_analysis_task.apply(args=[upload_id])
        except Exception as inline_exc:
            logging.error(
                "Inline video view regeneration failed for upload %s: %s",
                upload_id,
                inline_exc,
                exc_info=True
            )

    threading.Thread(
        target=_run_inline,
        name=f"video-view-regen-{upload_id}",
        daemon=True
    ).start()

    return None, True


def _shorten_text(text, limit=400):
    """Compact whitespace and truncate long text for table display."""
    if not text:
        return ''
    if isinstance(text, (list, tuple, set)):
        text = '；'.join(str(item) for item in text if item)
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + '…'
    return text


def _extract_reason_mode(reports, fencer_key, category):
    """Pick the most representative success/failure description from reason reports."""
    try:
        category_reports = (reports or {}).get(fencer_key, {}).get(category, []) or []
    except Exception:
        category_reports = []

    if not category_reports:
        return ''

    sorted_reports = sorted(
        category_reports,
        key=lambda item: ((item or {}).get('touch_count') or 0, len((item or {}).get('touches') or [])),
        reverse=True
    )
    top_report = sorted_reports[0] or {}

    for key in ('summary_bullet', 'analysis_headline', 'core_narrative'):
        text = _shorten_text(top_report.get(key))
        if text:
            return text

    for list_key in ('key_sequences', 'focus_points', 'validation_checks'):
        for entry in top_report.get(list_key, []) or []:
            text = _shorten_text(entry)
            if text:
                return text

    return ''


def _build_immediate_adjustments(bullets_map, max_items: int = 6):
    """Generate rapid tactical pointers from cached reason summary bullets."""
    if not isinstance(bullets_map, dict):
        return []

    adjustments = []

    def _append_entries(source_map, prefix):
        if isinstance(source_map, list):
            normalized = {'generic': source_map}
        elif isinstance(source_map, dict):
            normalized = source_map
        else:
            return

        for category in ('attack', 'in_box', 'defense', 'generic'):
            for entry in normalized.get(category, []) or []:
                if not entry:
                    continue
                text = str(entry).strip()
                if not text:
                    continue
                lower = text.lower()
                if any(token in lower for token in ('training', 'drill', 'conditioning')):
                    continue
                formatted = f"{prefix}{text}".strip()
                if not formatted.endswith(('.', '!', '?')):
                    formatted += '.'
                if formatted not in adjustments:
                    adjustments.append(formatted)
                if len(adjustments) >= max_items:
                    return

    _append_entries(bullets_map.get('loss'), 'Adjust: ')
    if len(adjustments) < max_items:
        _append_entries(bullets_map.get('win'), 'Keep leveraging: ')

    return adjustments[:max_items]


def _determine_priority(win_rate, net_points, attempts):
    """Assign strategic priority label and style based on performance."""
    if attempts == 0:
        return 'Evaluate', 'priority-evaluate'
    if win_rate >= 55 and net_points >= 0:
        return 'Optimize', 'priority-optimize'
    if win_rate >= 35 or net_points >= 0:
        return 'Improve', 'priority-improve'
    return 'Rebuild - Critical', 'priority-rebuild'


def _build_tactical_summary(bout_type_stats, win_reports, loss_reports):
    """Create enriched tactical summary table data for each fencer."""
    category_meta = [
        ('attack', 'Initiated Attack', 'bout-type-attack'),
        ('in_box', 'Simultaneous/Counter-Attack', 'bout-type-inbox'),
        ('defense', 'Defensive Riposte', 'bout-type-defense')
    ]

    summary = {'left_fencer': [], 'right_fencer': []}

    for fencer_key in ('left_fencer', 'right_fencer'):
        fencer_stats = (bout_type_stats or {}).get(fencer_key, {})

        for category, label, row_class in category_meta:
            category_stats = fencer_stats.get(category, {}) or {}
            attempts = int(category_stats.get('count') or 0)
            wins = int(category_stats.get('wins') or 0)
            losses = max(attempts - wins, 0)
            net_points = wins - losses
            win_rate = float(category_stats.get('win_rate') or 0)

            rate_display = f"{int(round(win_rate))}%" if attempts else '—'
            attempt_display = f"{attempts} attempts"
            if net_points > 0:
                net_display = f"+{net_points} ({wins}W, {losses}L)"
            elif net_points < 0:
                net_display = f"{net_points} ({wins}W, {losses}L)"
            else:
                net_display = f"0 ({wins}W, {losses}L)"

            success_mode = _shorten_text(_extract_reason_mode(win_reports, fencer_key, category))
            failure_mode = _shorten_text(_extract_reason_mode(loss_reports, fencer_key, category))

            priority_label, priority_class = _determine_priority(win_rate, net_points, attempts)

            if win_rate >= 60:
                rate_class = 'win-rate-good'
            elif win_rate >= 40:
                rate_class = 'win-rate-average'
            else:
                rate_class = 'win-rate-poor'

            summary[fencer_key].append({
                'category': category,
                'label': label,
                'row_class': row_class,
                'win_rate_display': rate_display,
                'win_rate_value': win_rate,
                'rate_class': rate_class,
                'attempt_display': attempt_display,
                'net_display': net_display,
                'success_mode': success_mode,
                'failure_mode': failure_mode,
                'priority_label': priority_label,
                'priority_class': priority_class
            })

    return summary

def create_fencer_radar_chart(fencer_data, output_dir, fencer_id):
    """Create radar chart for fencer performance"""
    try:
        metrics = fencer_data.get('metrics', {})
        
        # Define the 8 dimensions for radar chart
        categories = [
            'Speed', 'Acceleration', 'Advance Ratio', 'Pause Control',
            'First Step', 'Arm Extension', 'Launch Success', 'Attack Ratio'
        ]
        
        # Normalize values to 0-1 scale for radar chart
        values = [
            min(1.0, metrics.get('avg_velocity', 0.0) / 5.0) if metrics.get('avg_velocity', 0.0) > 0 else 0.0,
            min(1.0, metrics.get('avg_acceleration', 0.0) / 3.0) if metrics.get('avg_acceleration', 0.0) > 0 else 0.0,
            metrics.get('avg_advance_ratio', 0.0),
            1.0 - metrics.get('avg_pause_ratio', 0.0),  # Invert pause ratio
            1.0 - min(1.0, metrics.get('avg_first_step_init', 0.0)),  # Invert (faster is better)
            min(1.0, metrics.get('total_arm_extensions', 0) / 20.0) if metrics.get('total_arm_extensions', 0) > 0 else 0.0,
            metrics.get('launch_success_rate', 0.0),
            metrics.get('attacking_ratio', 0.0)
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
        
        # Check if all values are zero (no data case)
        if all(v == 0.0 for v in values[:-1]):  # Exclude the duplicate first value
            plt.title(f'{fencer_data.get("fencer_name", f"Fencer {fencer_id}")} - Performance Radar Chart\n(No Data Available - Upload and Analyze Videos)',
                     size=16, fontweight='bold', pad=20)
            # Add text in center explaining no data
            ax.text(0, 0.5, 'No Performance Data\nUpload and Analyze Videos\nto View Radar Chart',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        else:
            plt.title(f'{fencer_data.get("fencer_name", f"Fencer {fencer_id}")} - Performance Radar Chart\n({metrics.get("total_bouts", 0)} Bouts Analyzed)',
                     size=16, fontweight='bold', pad=20)
        
        # Save
        output_file = os.path.join(output_dir, f'fencer_{fencer_id}_radar_profile.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file
        
    except Exception as e:
        logging.error(f"Error creating radar chart: {e}")
        return None

def create_fencer_analysis_chart(fencer_data, output_dir, fencer_id):
    """Create comprehensive analysis chart"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{fencer_data.get("fencer_name", f"Fencer {fencer_id}")} - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        metrics = fencer_data.get('metrics', {})
        
        # 1. Performance Metrics Bar Chart
        metric_names = ['Speed', 'Acceleration', 'Advance Ratio', 'Launch Success']
        metric_values = [
            metrics.get('avg_velocity', 0.0),
            metrics.get('avg_acceleration', 0.0),
            metrics.get('avg_advance_ratio', 0.0),
            metrics.get('launch_success_rate', 0.0)
        ]

        bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_title('Key Performance Metrics')
        ax1.set_ylabel('Value')

        # Check if all values are zero
        if all(v == 0.0 for v in metric_values):
            ax1.text(0.5, 0.5, 'No Performance Data\nUpload and Analyze Videos\nto View Metrics',
                    ha='center', va='center', transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        else:
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
            tag_labels = ['Positive Attributes', 'Areas for Improvement']

            ax2.pie(tag_counts, labels=tag_labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax2.set_title(f'Performance Tag Distribution\n(Total {len(tags)} Tags)')
        else:
            ax2.text(0.5, 0.5, 'No Tags Available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Performance Tag Distribution')
        
        # 3. Bout Progression (if multiple bouts)
        bouts = fencer_data.get('all_bouts', [])
        if len(bouts) > 1:
            bout_numbers = [bout.get('match_idx', i+1) for i, bout in enumerate(bouts)]
            velocities = [bout.get('metrics', {}).get('velocity', 0) or 0 for bout in bouts]
            advance_ratios = [bout.get('metrics', {}).get('advance_ratio', 0) or 0 for bout in bouts]
            
            # Plot velocity and advance ratio
            ax3_twin = ax3.twinx()
            line1 = ax3.plot(bout_numbers, velocities, 'o-', color='blue', linewidth=2, markersize=6, label='Speed')
            line2 = ax3_twin.plot(bout_numbers, advance_ratios, 's-', color='red', linewidth=2, markersize=6, label='Advance Ratio')

            ax3.set_title('Bout Progression')
            ax3.set_xlabel('Bout Number')
            ax3.set_ylabel('Speed (m/s)', color='blue')
            ax3_twin.set_ylabel('Advance Ratio', color='red')
            ax3.grid(True, alpha=0.3)

            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left')
        else:
            ax3.text(0.5, 0.5, f'Single Bout\n(Multiple bouts needed to show progression)',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Performance Progression')
        
        # 4. Upload Sources with Win/Loss
        upload_sources = fencer_data.get('upload_sources', [])
        if upload_sources:
            upload_labels = []
            bout_counts = []
            win_counts = []
            
            for us in upload_sources:
                upload_labels.append(f"Video {us['upload_id']}")
                bout_counts.append(us['bout_count'])
                
                # Count wins for this upload
                upload_bouts = [b for b in bouts if b.get('upload_id') == us['upload_id']]
                wins = sum(1 for b in upload_bouts if b.get('result') in ['left', 'right'] and 
                          ((b.get('result') == 'left' and us.get('fencer_side') == 'left') or
                           (b.get('result') == 'right' and us.get('fencer_side') == 'right')))
                win_counts.append(wins)
            
            x = range(len(upload_labels))
            width = 0.35
            
            ax4.bar([i - width/2 for i in x], bout_counts, width, label='Total Bouts', color='lightblue', alpha=0.8)
            ax4.bar([i + width/2 for i in x], win_counts, width, label='Wins', color='lightgreen', alpha=0.8)

            ax4.set_title('Performance by Video')
            ax4.set_xlabel('Video')
            ax4.set_ylabel('Bout Count')
            ax4.set_xticks(x)
            ax4.set_xticklabels(upload_labels)
            ax4.legend()

            # Rotate labels if many uploads
            if len(upload_labels) > 3:
                ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No Upload Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance by Video')
        
        plt.tight_layout()
        
        # Save
        output_file = os.path.join(output_dir, f'fencer_{fencer_id}_profile_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file
        
    except Exception as e:
        logging.error(f"Error creating analysis chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_aggregated_metrics(all_bouts):
    """Calculate aggregated metrics from all bouts"""
    if not all_bouts:
        return {
            'total_bouts': 0,
            'avg_velocity': 0.0,
            'avg_acceleration': 0.0,
            'avg_advance_ratio': 0.0,
            'avg_pause_ratio': 0.0,
            'avg_first_step_init': 0.0,
            'total_arm_extensions': 0,
            'avg_arm_extension_duration': 0.0,
            'launch_success_rate': 0.0,
            'attacking_ratio': 0.0,
        }
    
    # Collect all metrics
    velocities = []
    accelerations = []
    advance_ratios = []
    pause_ratios = []
    first_steps = []
    arm_extensions_counts = []
    arm_extension_durations = []
    launch_flags = []
    attacking_flags = []
    
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
        if 'first_step_init' in metrics and metrics['first_step_init'] is not None:
            first_steps.append(float(metrics['first_step_init']))
        if 'arm_extension_freq' in metrics and metrics['arm_extension_freq'] is not None:
            arm_extensions_counts.append(int(metrics['arm_extension_freq']))
        if 'avg_arm_extension_duration' in metrics and metrics['avg_arm_extension_duration'] is not None:
            arm_extension_durations.append(float(metrics['avg_arm_extension_duration']))
        if 'has_launch' in metrics and metrics['has_launch'] is not None:
            launch_flags.append(metrics['has_launch'])
        if 'is_attacking' in metrics and metrics['is_attacking'] is not None:
            attacking_flags.append(metrics['is_attacking'])
    
    # Calculate aggregated statistics - use 0.0 when no data available
    return {
        'total_bouts': len(all_bouts),
        'avg_velocity': np.mean(velocities) if velocities else 0.0,
        'avg_acceleration': np.mean(accelerations) if accelerations else 0.0,
        'avg_advance_ratio': np.mean(advance_ratios) if advance_ratios else 0.0,
        'avg_pause_ratio': np.mean(pause_ratios) if pause_ratios else 0.0,
        'avg_first_step_init': np.mean(first_steps) if first_steps else 0.0,
        'total_arm_extensions': sum(arm_extensions_counts),
        'avg_arm_extension_duration': np.mean(arm_extension_durations) if arm_extension_durations else 0.0,
        'launch_success_rate': (sum(1 for f in launch_flags if f) / len(launch_flags)) if launch_flags else 0.0,
        'attacking_ratio': (sum(1 for a in attacking_flags if a) / len(attacking_flags)) if attacking_flags else 0.0,
    }

def extract_tags_from_uploads(uploads, fencer_id):
    """Extract performance tags for a fencer from all their uploads"""
    all_tags = set()
    
    for upload in uploads:
        result_dir = os.path.join('results', str(upload.user_id), str(upload.id))
        match_analysis_dir = os.path.join(result_dir, 'match_analysis')
        
        if os.path.exists(match_analysis_dir):
            for file_name in os.listdir(match_analysis_dir):
                if file_name.endswith('_analysis.json'):
                    match_file = os.path.join(match_analysis_dir, file_name)
                    try:
                        with open(match_file, 'r', encoding='utf-8') as f:
                            match_data = json.load(f)
                        
                        # Import tagging function
                        sys.path.insert(0, '/workspace/Project/your_scripts')
                        from tagging import extract_tags_from_bout_analysis
                        
                        # Extract tags for this bout
                        fencer_side = 'left' if upload.left_fencer_id == fencer_id else 'right'
                        bout_tags = extract_tags_from_bout_analysis(match_data)
                        if fencer_side in bout_tags:
                            all_tags.update(bout_tags[fencer_side])
                            
                    except Exception as e:
                        logging.warning(f"Error processing match analysis {match_file}: {e}")
    
    return list(all_tags)

def get_video_metadata(video_path):
    """Return video resolution (width, height), duration (seconds), and bitrate (kbps) using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration,bit_rate",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        streams = metadata.get("streams", [])
        if not streams:
            logging.error(f"No video streams found in {video_path}")
            return None, None, None
        width = streams[0].get("width")
        height = streams[0].get("height")
        duration = float(metadata.get("format", {}).get("duration", 0))
        bitrate = int(metadata.get("format", {}).get("bit_rate", 0)) / 1000  # Convert to kbps
        return (width, height), duration, bitrate
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
        logging.error(f"Error getting metadata for {video_path}: {str(e)}")
        return None, None, None

def compress_video(input_path, output_path, max_resolution="1280:720", max_size_mb=100, max_bitrate_kbps=5000, bitrate="1500k"):
    """Compress video if resolution, file size, or bitrate exceeds thresholds."""
    try:
        # Check file size
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        resolution, duration, input_bitrate_kbps = get_video_metadata(input_path)
        if not resolution:
            logging.error(f"Could not determine metadata for {input_path}")
            return False
        width, height = resolution
        needs_compression = width > 1280 or height > 720 or file_size_mb > max_size_mb or input_bitrate_kbps > max_bitrate_kbps
        if not needs_compression:
            logging.info(f"No compression needed for {input_path}: {width}x{height}, {file_size_mb:.2f}MB, {input_bitrate_kbps:.2f}kbps")
            return True  # Use original video
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", f"scale={max_resolution}:force_original_aspect_ratio=decrease,pad={max_resolution}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-b:v", bitrate, "-preset", "veryfast", "-crf", "28",
            "-c:a", "aac", "-b:a", "96k",
            "-y", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Compressed {input_path} to {output_path}: {width}x{height}, {file_size_mb:.2f}MB, {input_bitrate_kbps:.2f}kbps")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Compression failed for {input_path}: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error during compression: {str(e)}")
        return False

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.getcwd(), 'instance', 'site.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # Engine options to play nicely with SQLite under Celery concurrency
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'connect_args': {
            'timeout': 30,  # seconds; complements PRAGMA busy_timeout
        },
    }
    app.config['UPLOAD_FOLDER'] = 'Uploads'
    app.config['RESULT_FOLDER'] = 'results'
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    app.config['result_backend'] = 'redis://localhost:6379/0'
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Increased to 500 MB
    app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
    # Gemini API key (prefer environment variable)
    app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY') or 'AIzaSyDy_RRq8hd8rTYILt_mYtMH8GtM41GFp6I'
    app.config['GEMINI_MODEL'] = os.getenv('GEMINI_MODEL') or DEFAULT_GEMINI_MODEL
    app.config['VIDEO_VIEW_REGEN_ASYNC'] = os.getenv('VIDEO_VIEW_REGEN_ASYNC', '1').lower() not in ('0', 'false', 'no')

    TAG_TRANSLATIONS = {
        'launch': 'Launch',
        'no_launch': 'No Launch',
        'arm_extension': 'Arm Extension',
        'no_arm_extension': 'No Arm Extension',
        'over_extension': 'Over Extension',
        'simple_attack': 'Simple Attack',
        'compound_attack': 'Compound Attack',
        'holding_attack': 'Holding Attack',
        'preparation_attack': 'Preparation Attack',
        'simple_preparation': 'Simple Preparation',
        'no_attacks': 'No Attacks',
        'limited_attack_variety': 'Limited Attack Variety',
        'steady_tempo': 'Steady Tempo',
        'variable_tempo': 'Variable Tempo',
        'broken_tempo': 'Broken Tempo',
        'excessive_pausing': 'Excessive Pausing',
        'excessive_tempo_changes': 'Excessive Tempo Changes',
        'good_attack_distance': 'Good Attack Distance',
        'poor_attack_distance': 'Poor Attack Distance',
        'maintain_safe_distance': 'Maintain Safe Distance',
        'poor_distance_maintaining': 'Poor Distance Maintaining',
        'poor_distance_maintenance': 'Poor Distance Maintenance',
        'consistent_spacing': 'Consistent Spacing',
        'inconsistent_spacing': 'Inconsistent Spacing',
        'good_defensive_quality': 'Good Defensive Quality',
        'poor_defensive_quality': 'Poor Defensive Quality',
        'missed_counter_opportunities': 'Missed Counter Opportunities',
        'failed_space_opening': 'Failed Space Opening',
        'failed_distance_pulls': 'Failed Distance Pulls',
        'low_speed': 'Low Speed',
        'poor_acceleration': 'Poor Acceleration',
        'fast_reaction_time': 'Fast Reaction Time',
    }

    @app.context_processor
    def inject_translations():
        return dict(TAG_TRANSLATIONS=TAG_TRANSLATIONS)

    db.init_app(app)
    login_manager = LoginManager(app)
    login_manager.login_view = 'login'

    init_celery(app)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

    # Attempt to auto-apply DB migration for multi-video support to avoid missing columns
    try:
        with app.app_context():
            try:
                # Import and run migration script programmatically
                from migrations.add_multi_video_support import run_migration as _run_multi_video_migration
                multi_applied = _run_multi_video_migration()
                if multi_applied:
                    logging.info("Multi-video migration check executed successfully.")
                else:
                    logging.info("Multi-video migration not applied (possibly already up-to-date).")

                from migrations.add_match_datetime import run_migration as _run_match_datetime_migration
                datetime_applied = _run_match_datetime_migration()
                if datetime_applied:
                    logging.info("Match datetime migration check executed successfully.")
                else:
                    logging.info("Match datetime migration not applied (possibly already up-to-date).")
            except Exception as e:
                logging.warning(f"Multi-video migration check failed: {e}")
    except Exception as e:
        logging.warning(f"Unable to run startup migrations: {e}")

    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('upload'))
        return redirect(url_for('login'))

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            if not username or not password:
                return render_template('register.html', error='Username and password are required')
            if len(username) > 150 or len(password) > 150:
                return render_template('register.html', error='Username or password too long')
            if User.query.filter_by(username=username).first():
                return render_template('register.html', error='Username already exists')
            try:
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username=username, password=hashed_password)
                db.session.add(new_user)
                db.session.commit()
                logging.info(f"User {username} registered successfully")
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                logging.error(f"Registration failed: {str(e)}")
                return render_template('register.html', error=f'Registration failed: {str(e)}')
        return render_template('register.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                logging.info(f"User {username} logged in")
                return redirect(url_for('upload'))
            return render_template('login.html', error='Invalid credentials')
        return render_template('login.html')

    @app.route('/logout')
    @login_required
    def logout():
        username = current_user.username
        logout_user()
        logging.info(f"User {username} logged out")
        return redirect(url_for('login'))


    @app.route('/upload', methods=['GET', 'POST'])
    @login_required
    def upload():
        if request.method == 'POST':
            # Check if this is a multi-video upload
            is_multi_video = request.form.get('is_multi_video') == 'true'
            
            if is_multi_video:
                return handle_multi_video_upload()
            else:
                return handle_single_video_upload()
        
        # GET request - show upload form
        uploads, fencers, error = get_safe_uploads_and_fencers(current_user.id)
        return render_template('upload.html', uploads=uploads, fencers=fencers, error=error)

    @app.route('/upload/<int:upload_id>/delete', methods=['POST'])
    @login_required
    def delete_upload(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403

        def _abs_path(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            return path if os.path.isabs(path) else os.path.join(app.root_path, path)

        result_dir_abs = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload.id))
        file_paths = []

        if upload.video_path:
            file_paths.append(_abs_path(upload.video_path))
        if upload.output_video_path:
            file_paths.append(_abs_path(upload.output_video_path))
        if upload.detection_image_path:
            file_paths.append(_abs_path(upload.detection_image_path))

        related_videos = list(upload.videos)
        for video in related_videos:
            if video.video_path:
                file_paths.append(_abs_path(video.video_path))
            if video.detection_image_path:
                file_paths.append(_abs_path(video.detection_image_path))

        try:
            bout_records = list(Bout.query.filter_by(upload_id=upload_id))
            bout_ids = [bout.id for bout in bout_records]
            if bout_ids:
                BoutTag.query.filter(BoutTag.bout_id.in_(bout_ids)).delete(synchronize_session=False)
                for bout in bout_records:
                    db.session.delete(bout)

            video_records = list(upload.videos)
            for video in video_records:
                db.session.delete(video)

            VideoAnalysis.query.filter_by(upload_id=upload_id).delete(synchronize_session=False)
            db.session.delete(upload)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.error('Failed to delete upload %s: %s', upload_id, exc, exc_info=True)
            flash('Failed to delete upload. Please try again.', 'danger')
            return redirect(url_for('upload'))

        for path_abs in file_paths:
            if path_abs and os.path.exists(path_abs):
                try:
                    if os.path.isdir(path_abs):
                        shutil.rmtree(path_abs, ignore_errors=True)
                    else:
                        os.remove(path_abs)
                except Exception as cleanup_exc:
                    logging.warning('Failed to remove %s during upload deletion: %s', path_abs, cleanup_exc)

        if os.path.isdir(result_dir_abs):
            shutil.rmtree(result_dir_abs, ignore_errors=True)

        flash('Upload deleted successfully.', 'success')
        return redirect(url_for('upload'))

    def handle_single_video_upload():
        """Handle traditional single video upload"""
        if 'video' not in request.files:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='No video file uploaded', uploads=uploads, fencers=fencers)
        video_file = request.files['video']
        if video_file.filename == '':
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='No file selected', uploads=uploads, fencers=fencers)

        match_date_str = request.form.get('match_date')
        match_time_str = request.form.get('match_time')
        if not match_date_str or not match_time_str:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Please provide both match date and time for the bout.', uploads=uploads, fencers=fencers)
        try:
            match_datetime = datetime.strptime(f"{match_date_str} {match_time_str}", '%Y-%m-%d %H:%M')
        except ValueError:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Invalid match date or time format. Please re-enter.', uploads=uploads, fencers=fencers)
        filename = secure_filename(video_file.filename)
        if not filename.rsplit('.', 1)[-1].lower() in app.config['ALLOWED_EXTENSIONS']:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Invalid file format. Allowed formats: mp4, avi, mov', uploads=uploads, fencers=fencers)
        if filename.lower() in {ext.lower() for ext in app.config['ALLOWED_EXTENSIONS']}:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Filename cannot be same as extension', uploads=uploads, fencers=fencers)
        max_file_size_mb = 500
        max_duration_sec = 600
        file_size_mb = video_file.seek(0, os.SEEK_END) / (1024 * 1024)
        video_file.seek(0)
        if file_size_mb > max_file_size_mb:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error=f'Video file too large ({file_size_mb:.2f}MB, max {max_file_size_mb}MB)', uploads=uploads, fencers=fencers)

        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        os.makedirs(upload_dir, exist_ok=True)
        video_path = os.path.join(upload_dir, filename)
        video_file.save(video_path)

        resolution, duration, bitrate = get_video_metadata(video_path)
        if duration and duration > max_duration_sec:
            os.remove(video_path)
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error=f'Video too long ({duration:.2f}s, max {max_duration_sec}s)', uploads=uploads, fencers=fencers)

        compressed_path = os.path.join(upload_dir, f"compressed_{filename}")
        if not compress_video(video_path, compressed_path):
            logging.warning(f"Compression failed, proceeding with original video: {video_path}")
            final_video_path = video_path
        else:
            final_video_path = compressed_path if os.path.exists(compressed_path) else video_path

        try:
            new_upload = Upload(
                user_id=current_user.id,
                video_path=final_video_path,
                match_datetime=match_datetime,
                status='processing_detection',
                detection_image_path=None,
                is_multi_video=False
            )
            db.session.add(new_upload)
            db.session.commit()
            new_upload.cross_bout_analysis_path = os.path.join(
                str(current_user.id),
                str(new_upload.id),
                'fencer_analysis',
                'cross_bout_analysis.json'
            )
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logging.error(
                "Error creating upload record for %s: %s",
                final_video_path,
                e,
                exc_info=True
            )
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error=f'Video processing error: {str(e)}', uploads=uploads, fencers=fencers)

        session['latest_upload_id'] = new_upload.id
        session['force_upload_redirect'] = True  # Flag to force redirect after upload
        logging.info(
            "Video uploaded by user %s: %s (upload_id=%s) awaiting detection",
            current_user.username,
            final_video_path,
            new_upload.id
        )

        try:
            from tasks import generate_initial_detection_task
            async_result = generate_initial_detection_task.delay(new_upload.id)
            logging.info(
                "Scheduled initial detection task %s for upload %s",
                async_result.id,
                new_upload.id
            )
        except Exception as task_error:
            logging.error(
                "Failed to schedule detection task for upload %s, falling back to synchronous processing: %s",
                new_upload.id,
                task_error,
                exc_info=True
            )
            try:
                detection_image_path, _ = process_first_frame(final_video_path, upload_dir)
                new_upload.detection_image_path = detection_image_path
                new_upload.status = 'awaiting_selection'
                db.session.commit()
                logging.info(
                    "Synchronous fallback completed for upload %s -> %s",
                    new_upload.id,
                    detection_image_path
                )
                return redirect(url_for('select_fencers', upload_id=new_upload.id))
            except Exception as fallback_error:
                db.session.rollback()
                logging.error(
                    "Fallback detection failed for upload %s: %s",
                    new_upload.id,
                    fallback_error,
                    exc_info=True
                )
                # Update status to error and clean up files
                errored_upload = Upload.query.get(new_upload.id)
                if errored_upload:
                    errored_upload.status = 'error'
                    errored_upload.detection_image_path = None
                    try:
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)
                uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
                return render_template('upload.html', error='Initial detection failed, please retry upload', uploads=uploads, fencers=fencers)

        return redirect(url_for('upload_wait', upload_id=new_upload.id))
    
    def handle_multi_video_upload():
        """Handle multi-video upload for complete matches"""
        match_title = request.form.get('match_title', '').strip()
        if not match_title:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Please enter match title', uploads=uploads, fencers=fencers)

        match_date_str = request.form.get('match_date')
        match_time_str = request.form.get('match_time')
        if not match_date_str or not match_time_str:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Please provide both match date and time for the match.', uploads=uploads, fencers=fencers)
        try:
            match_datetime = datetime.strptime(f"{match_date_str} {match_time_str}", '%Y-%m-%d %H:%M')
        except ValueError:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Invalid match date or time format. Please re-enter.', uploads=uploads, fencers=fencers)

        # Get uploaded videos
        video_files = request.files.getlist('videos[]')

        if not video_files or len([f for f in video_files if f.filename]) == 0:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='Please upload at least one video file', uploads=uploads, fencers=fencers)
        
        # Validate all videos
        valid_videos = []
        for i, video_file in enumerate(video_files):
            if video_file.filename == '':
                continue
                
            filename = secure_filename(video_file.filename)
            if not filename.rsplit('.', 1)[-1].lower() in app.config['ALLOWED_EXTENSIONS']:
                uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
                return render_template('upload.html', error=f'Video {i+1} format invalid. Allowed formats: mp4, avi, mov', uploads=uploads, fencers=fencers)

            file_size_mb = video_file.seek(0, os.SEEK_END) / (1024 * 1024)
            video_file.seek(0)
            if file_size_mb > 500:
                uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
                return render_template('upload.html', error=f'Video {i+1} too large ({file_size_mb:.2f}MB, max 500MB)', uploads=uploads, fencers=fencers)
            
            valid_videos.append((video_file, filename, ''))
        
        if len(valid_videos) == 0:
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error='No valid video files', uploads=uploads, fencers=fencers)
        
        try:
            # Create main upload record
            new_upload = Upload(
                user_id=current_user.id,
                video_path='',  # Multi-video doesn't have a single path, use empty string
                match_datetime=match_datetime,
                status='awaiting_multi_video_setup',
                is_multi_video=True,
                match_title=match_title
            )
            db.session.add(new_upload)
            db.session.commit()
            
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id), str(new_upload.id))
            os.makedirs(upload_dir, exist_ok=True)
            
            # Process and save each video
            for sequence_order, (video_file, filename, tracking_index) in enumerate(valid_videos, 1):
                # Save video file
                video_path = os.path.join(upload_dir, f"video_{sequence_order}_{filename}")
                video_file.save(video_path)
                
                # Compress video
                compressed_path = os.path.join(upload_dir, f"compressed_video_{sequence_order}_{filename}")
                if not compress_video(video_path, compressed_path):
                    logging.warning(f"Compression failed for video {sequence_order}, proceeding with original")
                    final_video_path = video_path
                else:
                    final_video_path = compressed_path if os.path.exists(compressed_path) else video_path
                
                # Process first frame for detection
                detection_dir = os.path.join(upload_dir, f"video_{sequence_order}")
                os.makedirs(detection_dir, exist_ok=True)
                detection_image_path, detections = process_first_frame(final_video_path, detection_dir)
                
                # Create UploadVideo record
                upload_video = UploadVideo(
                    upload_id=new_upload.id,
                    video_path=final_video_path,
                    sequence_order=sequence_order,
                    selected_indexes=tracking_index,
                    detection_image_path=detection_image_path,
                    status='pending'
                )
                db.session.add(upload_video)
            
            new_upload.cross_bout_analysis_path = os.path.join(str(current_user.id), str(new_upload.id), 'fencer_analysis', 'cross_bout_analysis.json')
            db.session.commit()
            
            session['latest_upload_id'] = new_upload.id
            session['force_upload_redirect'] = True
            logging.info(f"Multi-video upload created by user {current_user.username}: {match_title} with {len(valid_videos)} videos")
            return redirect(url_for('select_fencers', upload_id=new_upload.id))
            
        except Exception as e:
            # Rollback the database session to clear any pending transactions
            db.session.rollback()
            logging.error(f"Error processing multi-video upload: {str(e)}\n{traceback.format_exc()}")
            # Clean up any uploaded files
            if 'upload_dir' in locals() and os.path.exists(upload_dir):
                import shutil
                shutil.rmtree(upload_dir, ignore_errors=True)
            uploads, fencers, db_error = get_safe_uploads_and_fencers(current_user.id)
            return render_template('upload.html', error=f'Multi-video upload error: {str(e)}', uploads=uploads, fencers=fencers)

    @app.route('/admin/migrate', methods=['POST'])
    @login_required
    def admin_run_migration():
        """Admin route to manually trigger database migration"""
        if not current_user.is_authenticated or current_user.username != 'admin':
            return 'Unauthorized', 403
        try:
            from migrations.add_multi_video_support import run_migration
            success_multi = run_migration()
            from migrations.add_match_datetime import run_migration as run_match_datetime_migration
            success_datetime = run_match_datetime_migration()
            success = success_multi and success_datetime
            return ('Migration completed successfully', 200) if success else ('Migration failed', 500)
        except Exception as e:
            logging.error(f"Manual migration failed: {e}")
            return f'Migration error: {str(e)}', 500

    @app.route('/select_multi_video_indexes/<int:upload_id>', methods=['GET', 'POST'])
    @login_required
    def select_multi_video_indexes(upload_id):
        """Allow user to select tracking indexes for each video in a multi-video upload"""
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403
        
        if not upload.is_multi_video:
            return redirect(url_for('select_fencers', upload_id=upload_id))
        
        if request.method == 'POST':
            # Update tracking indexes for each video
            for video in upload.videos:
                tracking_key = f'tracking_indexes_{video.id}'
                if tracking_key in request.form:
                    video.selected_indexes = request.form[tracking_key]
                    db.session.add(video)
            
            db.session.commit()
            return redirect(url_for('select_fencers', upload_id=upload_id))
        
        return render_template('select_multi_video_indexes.html', upload=upload)


    @app.route('/analysis_processing_wait/<int:upload_id>')
    @login_required
    def analysis_processing_wait(upload_id):
        """Display a waiting screen while analysis jobs finish for an upload."""
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403

        if upload.status == 'awaiting_selection':
            return redirect(url_for('select_fencers', upload_id=upload_id))
        if upload.status == 'awaiting_multi_video_setup':
            return redirect(url_for('select_multi_video_indexes', upload_id=upload_id))

        if upload.status == 'awaiting_user_input':
            return redirect(url_for('select_bout_winners', upload_id=upload_id))

        if upload.status == 'completed':
            return redirect(url_for('results', upload_id=upload_id))

        status_url = url_for('check_status', upload_id=upload_id)
        winners_url = url_for('select_bout_winners', upload_id=upload_id)
        results_url = url_for('results', upload_id=upload_id)

        return render_template(
            'analysis_processing_wait.html',
            upload=upload,
            status_url=status_url,
            winners_url=winners_url,
            results_url=results_url,
            upload_url=url_for('upload'),
            is_multi=upload.is_multi_video
        )


    @app.route('/manage_fencers', methods=['GET', 'POST'])
    @login_required
    def manage_fencers():
        if request.method == 'POST':
            name = request.form.get('name')
            if name and len(name) <= 150:
                new_fencer = Fencer(user_id=current_user.id, name=name)
                db.session.add(new_fencer)
                db.session.commit()
                logging.info(f"Fencer {name} added by user {current_user.username}")
                return redirect(url_for('manage_fencers'))

        fencers = Fencer.query.filter_by(user_id=current_user.id).all()

        # Add tag summary for each fencer
        for fencer in fencers:
            # Get all uploads where this fencer participated
            uploads = Upload.query.filter(
                (Upload.left_fencer_id == fencer.id) | (Upload.right_fencer_id == fencer.id)
            ).filter_by(user_id=current_user.id, status='completed').all()
            
            # Get all unique tags for this fencer with counts
            tag_data = {}
            for upload in uploads:
                fencer_side = 'left' if upload.left_fencer_id == fencer.id else 'right'
                
                # Get all bouts for this upload
                bouts = Bout.query.filter_by(upload_id=upload.id).all()
                for bout in bouts:
                    for bout_tag in bout.tags:
                        if bout_tag.fencer_side == fencer_side:
                            tag_name = bout_tag.tag.name
                            if tag_name not in tag_data:
                                tag_data[tag_name] = 0
                            tag_data[tag_name] += 1
            
            # Sort tags by count (descending) and convert to list of tuples
            fencer.tag_summary = sorted(tag_data.items(), key=lambda x: x[1], reverse=True)

        return render_template('manage_fencers.html', fencers=fencers)

    @app.route('/fencer/<int:fencer_id>/delete', methods=['POST'])
    @login_required
    def delete_fencer(fencer_id):
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return 'Unauthorized', 403

        analysis_dir = os.path.join(
            app.config['RESULT_FOLDER'],
            str(current_user.id),
            'fencer',
            str(fencer_id)
        )
        profile_dir = _get_fencer_profile_dir(current_user.id, fencer_id)

        try:
            uploads = Upload.query.filter(
                (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
            ).filter_by(user_id=current_user.id).all()

            for upload in uploads:
                cleared = False
                if upload.left_fencer_id == fencer_id:
                    upload.left_fencer_id = None
                    cleared = True
                if upload.right_fencer_id == fencer_id:
                    upload.right_fencer_id = None
                    cleared = True
                if cleared:
                    logging.info(
                        "Removed fencer %s association from upload %s during deletion",
                        fencer_id,
                        upload.id
                    )

            holistic_entries = HolisticAnalysis.query.filter_by(fencer_id=fencer_id).all()
            for entry in holistic_entries:
                db.session.delete(entry)

            db.session.delete(fencer)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.error("Failed to delete fencer %s: %s", fencer_id, exc, exc_info=True)
            flash('Failed to delete fencer. Please try again.', 'error')
            return redirect(url_for('manage_fencers'))

        cleanup_warning = False
        for target_dir in {analysis_dir, profile_dir}:
            try:
                if os.path.isdir(target_dir):
                    shutil.rmtree(target_dir)
            except FileNotFoundError:
                continue
            except Exception as exc:
                cleanup_warning = True
                logging.error(
                    "Failed to remove directory %s for fencer %s: %s",
                    target_dir,
                    fencer_id,
                    exc,
                    exc_info=True
                )

        if cleanup_warning:
            flash('Fencer removed, but some cached files could not be deleted.', 'warning')
        else:
            flash('Fencer removed successfully.', 'success')

        return redirect(url_for('manage_fencers'))

    @app.route('/select_fencers/<int:upload_id>', methods=['GET', 'POST'])
    @login_required
    def select_fencers(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403
        if upload.status == 'processing_detection':
            flash('Initial detection still running, please wait a moment.', 'info')
            return redirect(url_for('upload_wait', upload_id=upload_id))
        if upload.status == 'error' and not upload.detection_image_path:
            flash('Initial detection failed, please re-upload video.', 'error')
            return redirect(url_for('upload'))
        if request.method == 'POST':
            weapon_type = request.form.get('weapon_type', 'saber')  # Default to saber if not provided
            try:
                if upload.is_multi_video:
                    # Collect per-video indexes
                    any_set = False
                    for video in upload.videos:
                        key = f"video_indexes_{video.id}"
                        if key in request.form:
                            raw = (request.form.get(key) or '').strip()
                            parts = raw.split()
                            if len(parts) != 2:
                                raise ValueError
                            # store on UploadVideo
                            video.selected_indexes = ' '.join(parts)
                            any_set = True
                    if not any_set:
                        raise ValueError
                    upload.weapon_type = weapon_type
                    upload.status = 'processing'
                    db.session.commit()
                    from tasks import analyze_multi_video_task
                    analyze_multi_video_task.delay(upload_id)
                    logging.info(f"Per-video fencer selection submitted for upload {upload_id} (weapon: {weapon_type})")
                    return redirect(url_for('analysis_processing_wait', upload_id=upload_id))
                else:
                    indexes = request.form['indexes']
                    idx_list = [int(i) for i in indexes.split()]
                    if len(idx_list) != 2:
                        raise ValueError
                    upload.selected_indexes = ' '.join(map(str, idx_list))
                    upload.weapon_type = weapon_type
                    upload.status = 'processing'
                    db.session.commit()
                    from tasks import analyze_video_task
                    analyze_video_task.delay(upload_id)
                    logging.info(f"Fencer selection submitted for upload {upload_id}: {indexes} (weapon: {weapon_type})")
                    return redirect(url_for('analysis_processing_wait', upload_id=upload_id))
            except ValueError:
                return render_template('select_fencers.html', upload=upload, error='Please enter two valid indices separated by space')
        return render_template('select_fencers.html', upload=upload)

    @app.route('/upload_wait/<int:upload_id>')
    @login_required
    def upload_wait(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403

        if upload.status in ('awaiting_selection', 'awaiting_multi_video_setup'):
            return redirect(url_for('select_fencers', upload_id=upload_id))
        if upload.status == 'processing':
            return redirect(url_for('results', upload_id=upload_id))
        if upload.status == 'completed':
            return redirect(url_for('results', upload_id=upload_id))

        return render_template(
            'upload_wait.html',
            upload=upload,
            status_url=url_for('check_status', upload_id=upload_id),
            select_url=url_for('select_fencers', upload_id=upload_id),
            results_url=url_for('results', upload_id=upload_id),
            upload_url=url_for('upload')
        )

    @app.route('/select_bout_winners/<int:upload_id>', methods=['GET', 'POST'])
    @login_required
    def select_bout_winners(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403

        # Get bouts for both single and multi-video uploads
        if upload.is_multi_video:
            # For multi-video uploads, get all bouts from all videos in sequence order
            bouts = db.session.query(Bout).join(UploadVideo, Bout.upload_video_id == UploadVideo.id)\
                .filter(UploadVideo.upload_id == upload_id)\
                .order_by(UploadVideo.sequence_order, Bout.match_idx).all()
        else:
            # For single video uploads, get bouts directly by upload_id
            bouts = Bout.query.filter_by(upload_id=upload_id).order_by(Bout.match_idx).all()

        if not bouts:
            return render_template('select_bout_winners.html', upload=upload, bouts=[], error='No bouts found.')

        result_dir_rel = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))
        result_dir_abs = result_dir_rel if os.path.isabs(result_dir_rel) else os.path.join(app.root_path, result_dir_rel)

        def absolute_path(path: str) -> str:
            if not path:
                return None
            return path if os.path.isabs(path) else os.path.join(app.root_path, path)

        def ensure_parent(path: str) -> None:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

        def get_video_metadata(video_path: str) -> Tuple[float, int, int, int]:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                raise FileNotFoundError(f"Could not open video file {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0:
                fps = 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            return fps, frame_count, width, height

        def write_video_segment(source_path: str, start_frame: int, end_frame: int, output_path: str, fps: float, size: Tuple[int, int]) -> None:
            if start_frame > end_frame:
                raise ValueError('Segment end frame must be greater than or equal to start frame')
            ensure_parent(output_path)
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video file {source_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            width, height = size
            if width <= 0 or height <= 0:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            codecs = ['mp4v', 'avc1', 'H264']
            writer = None
            for codec in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                candidate = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if candidate.isOpened():
                    writer = candidate
                    break
                candidate.release()
            if writer is None:
                cap.release()
                raise RuntimeError(f"Unable to create video writer for {output_path}")
            frame_idx = start_frame
            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                frame_idx += 1
            writer.release()
            cap.release()

        def ensure_web_playable_mp4(input_path: str) -> None:
            """Re-encode MP4 to H.264/AAC with faststart to maximize browser compatibility."""
            try:
                tmp_out = f"{input_path}.tmp.mp4"
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '23',
                    '-movflags', '+faststart',
                    '-c:a', 'aac', '-b:a', '96k', tmp_out
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                try:
                    os.replace(tmp_out, input_path)
                except Exception:
                    # Fallback if atomic replace not available
                    shutil.copyfile(tmp_out, input_path)
                    os.remove(tmp_out)
            except Exception as reencode_exc:
                logging.warning(f"Failed to re-encode {input_path} for web playback: {reencode_exc}")

        def get_source_video_path(bout_obj: Bout) -> str:
            if upload.is_multi_video and bout_obj.upload_video:
                return bout_obj.upload_video.video_path
            return upload.video_path

        def get_csv_dir(bout_obj: Bout) -> str:
            if upload.is_multi_video and bout_obj.upload_video:
                candidate = os.path.join(result_dir_abs, f'csv_video_{bout_obj.upload_video.sequence_order}')
                if os.path.isdir(candidate):
                    return candidate
            if upload.csv_dir:
                candidate = absolute_path(upload.csv_dir)
                if candidate and os.path.isdir(candidate):
                    return candidate
            return os.path.join(result_dir_abs, 'csv')

        def load_scaling_factor(csv_dir: str) -> float:
            meta_path = os.path.join(csv_dir, 'meta.csv')
            if os.path.exists(meta_path):
                try:
                    meta_df = pd.read_csv(meta_path)
                    if 'c' in meta_df.columns and not meta_df['c'].empty:
                        return float(meta_df['c'].iloc[0])
                except Exception as exc:
                    logging.warning(f"Failed to read scaling factor from {meta_path}: {exc}")
            return 1.0

        def slice_match_data(csv_dir: str, match_idx: int, start_frame: int, end_frame: int):
            required = ['left_xdata.csv', 'left_ydata.csv', 'right_xdata.csv', 'right_ydata.csv']
            frames: Dict[str, pd.DataFrame] = {}
            for name in required:
                path = os.path.join(csv_dir, name)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Required CSV not found: {path}")
                frames[name] = pd.read_csv(path)
            frame_column = 'Frame' if 'Frame' in frames['left_xdata.csv'].columns else None
            if frame_column:
                mask = (frames['left_xdata.csv'][frame_column] >= start_frame) & (frames['left_xdata.csv'][frame_column] <= end_frame)
                if not mask.any():
                    raise ValueError('No frames found in requested interval for match data')
                indices = frames['left_xdata.csv'][mask].index
            else:
                indices = frames['left_xdata.csv'].loc[start_frame:end_frame].index
            left_x_slice = frames['left_xdata.csv'].loc[indices].copy().reset_index(drop=True)
            left_y_slice = frames['left_ydata.csv'].loc[indices].copy().reset_index(drop=True)
            right_x_slice = frames['right_xdata.csv'].loc[indices].copy().reset_index(drop=True)
            right_y_slice = frames['right_ydata.csv'].loc[indices].copy().reset_index(drop=True)
            if frame_column:
                left_x_slice[frame_column] = left_x_slice[frame_column] - start_frame
                left_y_slice[frame_column] = left_y_slice[frame_column] - start_frame
                right_x_slice[frame_column] = right_x_slice[frame_column] - start_frame
                right_y_slice[frame_column] = right_y_slice[frame_column] - start_frame
            else:
                left_x_slice['Frame'] = np.arange(len(left_x_slice))
                left_y_slice['Frame'] = np.arange(len(left_y_slice))
                right_x_slice['Frame'] = np.arange(len(right_x_slice))
                right_y_slice['Frame'] = np.arange(len(right_y_slice))
            match_dir = os.path.join(result_dir_abs, 'match_data', f'match_{match_idx}')
            os.makedirs(match_dir, exist_ok=True)
            left_x_slice.to_csv(os.path.join(match_dir, 'left_xdata.csv'), index=False)
            left_y_slice.to_csv(os.path.join(match_dir, 'left_ydata.csv'), index=False)
            right_x_slice.to_csv(os.path.join(match_dir, 'right_xdata.csv'), index=False)
            right_y_slice.to_csv(os.path.join(match_dir, 'right_ydata.csv'), index=False)
            scale_factor = load_scaling_factor(csv_dir)
            return left_x_slice, left_y_slice, right_x_slice, right_y_slice, scale_factor

        def load_trim_settings(match_idx: int) -> Dict[str, Any]:
            trim_path = os.path.join(result_dir_abs, 'matches', f'match_{match_idx}', 'trim_settings.json')
            if os.path.exists(trim_path):
                try:
                    with open(trim_path, 'r', encoding='utf-8') as handle:
                        return json.load(handle)
                except Exception as exc:
                    logging.warning(f"Failed to read trim settings for match {match_idx}: {exc}")
            return {}

        def save_trim_settings(match_idx: int, payload: Dict[str, Any]) -> None:
            trim_path = os.path.join(result_dir_abs, 'matches', f'match_{match_idx}', 'trim_settings.json')
            ensure_parent(trim_path)
            with open(trim_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)

        # Define results_root_abs and _store_result_path before apply_manual_trim to fix closure issue
        results_root_abs = os.path.join(app.root_path, app.config['RESULT_FOLDER'])

        def _store_result_path(abs_path: str) -> str:
            rel = os.path.relpath(abs_path, app.root_path).replace(os.sep, '/')
            if rel.startswith('../'):
                rel = rel[3:]
            if rel.startswith(app.config['RESULT_FOLDER']):
                return rel
            inner_rel = os.path.relpath(abs_path, results_root_abs).replace(os.sep, '/')
            return f"{app.config['RESULT_FOLDER'].rstrip('/')}/{inner_rel}"

        def apply_manual_trim(bout_obj: Bout, user_start_sec: float, user_end_sec: float, apply_buffer: bool) -> None:
            source_rel = get_source_video_path(bout_obj)
            if not source_rel:
                raise FileNotFoundError('Source video path missing for bout')
            source_abs = absolute_path(source_rel)
            if not source_abs or not os.path.exists(source_abs):
                raise FileNotFoundError(f'Source video not found: {source_rel}')
            fps, total_frames, width, height = get_video_metadata(source_abs)
            max_duration = total_frames / fps if fps else 0.0
            user_start_sec = max(0.0, float(user_start_sec))
            user_end_sec = float(user_end_sec)
            if max_duration > 0:
                user_end_sec = min(user_end_sec, max_duration)
            if user_end_sec <= user_start_sec:
                raise ValueError('End time must be greater than start time.')

            video_start_frame = max(0, int(round(user_start_sec * fps)))
            video_end_frame = min(total_frames - 1, int(round(user_end_sec * fps)))
            if video_end_frame <= video_start_frame:
                video_end_frame = min(total_frames - 1, video_start_frame + max(1, int(round(0.5 * fps))))

            buffer_offset = 1.0 if apply_buffer else 0.0
            data_start_sec = user_start_sec + buffer_offset
            data_end_sec = user_end_sec - buffer_offset
            if max_duration > 0:
                data_start_sec = min(max_duration, max(0.0, data_start_sec))
                data_end_sec = min(max_duration, max(data_start_sec, data_end_sec))
            else:
                data_end_sec = max(data_start_sec, data_end_sec)

            data_start_frame = max(0, int(round(data_start_sec * fps)))
            data_end_frame = max(data_start_frame, int(round(data_end_sec * fps)))
            data_start_frame = min(data_start_frame, total_frames - 1)
            data_end_frame = min(data_end_frame, total_frames - 1)

            data_start_frame = min(max(data_start_frame, video_start_frame), video_end_frame)
            data_end_frame = min(max(data_end_frame, data_start_frame), video_end_frame)
            match_idx = bout_obj.match_idx
            match_dir_abs = os.path.join(result_dir_abs, 'matches', f'match_{match_idx}')
            os.makedirs(match_dir_abs, exist_ok=True)
            clip_path_abs = os.path.join(match_dir_abs, f'match_{match_idx}.mp4')
            extended_path_abs = os.path.join(match_dir_abs, f'match_{match_idx}_extended.mp4')
            write_video_segment(source_abs, video_start_frame, video_end_frame, clip_path_abs, fps, (width, height))
            ensure_web_playable_mp4(clip_path_abs)
            pad_frames = int(round(fps))
            extended_start = max(0, video_start_frame - pad_frames)
            extended_end = min(total_frames - 1, video_end_frame + pad_frames)
            if extended_end > extended_start:
                write_video_segment(source_abs, extended_start, extended_end, extended_path_abs, fps, (width, height))
                ensure_web_playable_mp4(extended_path_abs)
            else:
                if os.path.exists(extended_path_abs):
                    os.remove(extended_path_abs)
            csv_dir_abs = get_csv_dir(bout_obj)
            left_x_slice, left_y_slice, right_x_slice, right_y_slice, scale_factor = slice_match_data(csv_dir_abs, match_idx, data_start_frame, data_end_frame)
            keypoints_dir_abs = os.path.join(result_dir_abs, 'matches_with_keypoints')
            os.makedirs(keypoints_dir_abs, exist_ok=True)
            try:
                from your_scripts.match_separation import overlay_keypoints_on_clip
                overlay_keypoints_on_clip(
                    match_idx,
                    data_start_frame,
                    data_end_frame,
                    left_x_slice,
                    left_y_slice,
                    right_x_slice,
                    right_y_slice,
                    scale_factor,
                    source_abs,
                    keypoints_dir_abs,
                    fps
                )
            except Exception as exc:
                logging.warning(f'Failed to regenerate keypoint overlay for match {match_idx}: {exc}')
            clip_rel_path = _store_result_path(clip_path_abs)

            if os.path.exists(extended_path_abs):
                extended_rel_path = _store_result_path(extended_path_abs)
            else:
                extended_rel_path = None

            bout_obj.video_path = clip_rel_path
            if hasattr(bout_obj, 'extended_video_path'):
                bout_obj.extended_video_path = extended_rel_path
            bout_obj.start_frame = data_start_frame
            bout_obj.end_frame = data_end_frame
            save_trim_settings(match_idx, {
                'user_start_time': user_start_sec,
                'user_end_time': user_end_sec,
                'video_start_frame': video_start_frame,
                'video_end_frame': video_end_frame,
                'data_start_frame': data_start_frame,
                'data_end_frame': data_end_frame,
                'fps': fps
            })

        if request.method == 'POST':
            all_results_submitted = True
            adjustments_ok = True
            adjustments_made = False
            for bout in bouts:
                result_key = f'result_{bout.match_idx}'
                result = request.form.get(result_key)
                if result in ['left', 'right', 'skip']:
                    bout.result = result
                else:
                    all_results_submitted = False
                    if result not in (None, ''):
                        flash(f'Invalid result for bout {bout.match_idx}', 'error')

            for bout in bouts:
                start_key = f'start_time_{bout.match_idx}'
                end_key = f'end_time_{bout.match_idx}'
                confirm_key = f'confirm_{bout.match_idx}'
                original_start_key = f'original_start_{bout.match_idx}'
                original_end_key = f'original_end_{bout.match_idx}'

                if start_key not in request.form or end_key not in request.form:
                    continue

                confirm_selected = request.form.get(confirm_key) == '1'
                if not confirm_selected:
                    continue

                try:
                    start_value = float(request.form[start_key])
                    end_value = float(request.form[end_key])
                except (TypeError, ValueError):
                    adjustments_ok = False
                    flash(f'Invalid timing values supplied for bout {bout.match_idx}', 'error')
                    continue
                if end_value <= start_value:
                    adjustments_ok = False
                    flash(f'End time must be greater than start time for bout {bout.match_idx}', 'error')
                    continue

                try:
                    original_start_val = float(request.form.get(original_start_key, start_value))
                    original_end_val = float(request.form.get(original_end_key, end_value))
                except (TypeError, ValueError):
                    original_start_val = start_value
                    original_end_val = end_value

                has_change = abs(start_value - original_start_val) > 1e-3 or abs(end_value - original_end_val) > 1e-3
                if not has_change:
                    continue


                try:
                    apply_manual_trim(bout, start_value, end_value, apply_buffer=True)
                    adjustments_made = True
                except Exception as exc:
                    adjustments_ok = False
                    logging.error('Failed to apply manual trim for bout %s: %s', bout.match_idx, exc, exc_info=True)
                    flash(f'Failed to apply adjustments for bout {bout.match_idx}: {exc}', 'error')

            if adjustments_made and adjustments_ok:
                upload.bouts_analyzed = 0
                upload.cross_bout_analysis_path = None
                for dir_name in ('match_analysis', 'fencer_analysis'):
                    target_dir = os.path.join(result_dir_abs, dir_name)
                    if os.path.isdir(target_dir):
                        try:
                            shutil.rmtree(target_dir, ignore_errors=True)
                        except Exception as cleanup_exc:
                            logging.warning('Failed to clean %s after trim update: %s', target_dir, cleanup_exc)

                bout_id_list = [b.id for b in bouts]
                if bout_id_list:
                    BoutTag.query.filter(BoutTag.bout_id.in_(bout_id_list)).delete(synchronize_session=False)

                analysis_record = VideoAnalysis.query.filter_by(upload_id=upload.id).first()
                if analysis_record:
                    analysis_record.status = 'pending'
                    analysis_record.loss_analysis = None
                    analysis_record.win_analysis = None
                    analysis_record.left_overall_analysis = None
                    analysis_record.right_overall_analysis = None
                    analysis_record.left_category_analysis = None
                    analysis_record.right_category_analysis = None
                    analysis_record.detailed_analysis = None
                    analysis_record.error_message = None

                upload.status = 'awaiting_user_input'

            trigger_analysis = False
            if adjustments_ok and all_results_submitted and all(b.result in ['left', 'right', 'skip'] and b.result is not None for b in bouts):
                upload.status = 'results_submitted'
                trigger_analysis = True

            if adjustments_ok:
                try:
                    db.session.commit()
                except Exception as exc:
                    adjustments_ok = False
                    db.session.rollback()
                    logging.error('Failed to save bout updates: %s', exc, exc_info=True)
                    flash('Failed to save updates. Please try again.', 'error')
            else:
                db.session.rollback()

            if adjustments_ok and trigger_analysis:
                try:
                    from tasks import generate_analysis_task, regenerate_video_analysis_task
                    # Kick general analysis (per-bout) and also refresh video view AI analysis
                    generate_analysis_task.delay(upload_id)
                    regenerate_video_analysis_task.delay(upload_id)
                    logging.info(f"All bout results submitted for upload {upload_id}, triggered analysis and video view regeneration tasks")
                    return redirect(url_for('status', upload_id=upload_id))
                except Exception as exc:
                    db.session.rollback()
                    upload.status = 'awaiting_user_input'
                    db.session.commit()
                    logging.error('Failed to launch analysis task: %s', exc, exc_info=True)
                    flash('Results saved, but failed to start analysis task. Please try again.', 'error')
            if adjustments_ok:
                flash('Adjustments saved. Review all bouts before final submission.', 'success')
            return redirect(url_for('select_bout_winners', upload_id=upload_id))

        def _to_result_rel(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            candidate_abs = absolute_path(path)
            if candidate_abs and os.path.exists(candidate_abs):
                try:
                    rel = os.path.relpath(candidate_abs, results_root_abs).replace(os.sep, '/')
                    if not rel.startswith('..'):
                        return rel
                except Exception:
                    pass
            prefix = app.config['RESULT_FOLDER'].rstrip('/') + '/'
            if path.startswith(prefix):
                return path[len(prefix):]
            return path

        def _rel_from_result(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            abs_path = path if os.path.isabs(path) else os.path.join(app.root_path, path)
            if abs_path and os.path.exists(abs_path):
                try:
                    rel = os.path.relpath(abs_path, results_root_abs).replace(os.sep, '/')
                    if not rel.startswith('..'):
                        return rel
                except Exception:
                    pass
            prefix = app.config['RESULT_FOLDER'].rstrip('/') + '/'
            if path.startswith(prefix):
                return path[len(prefix):]
            return path

        bout_data = []
        for bout in bouts:
            source_path = get_source_video_path(bout)
            source_abs = absolute_path(source_path) if source_path else None
            has_video = bool(source_abs and os.path.exists(source_abs))

            source_fps = 30.0
            source_frames = 0
            if has_video:
                try:
                    source_fps, source_frames, _, _ = get_video_metadata(source_abs)
                except Exception as exc:
                    logging.warning(f"Failed to read metadata for source video {source_path}: {exc}")
            if source_fps <= 0:
                source_fps = 30.0
            source_duration = (source_frames / source_fps) if source_frames > 0 else 0.0

            trim_settings = load_trim_settings(bout.match_idx)
            if trim_settings:
                user_start = float(trim_settings.get('user_start_time', 0.0))
                user_end = float(trim_settings.get('user_end_time', source_duration or 0.0))
            else:
                analysis_start_sec = (bout.start_frame or 0) / source_fps
                analysis_end_sec = (bout.end_frame or 0) / source_fps
                user_start = max(0.0, analysis_start_sec - 1.0)
                user_end = analysis_end_sec + 1.0 if analysis_end_sec else analysis_start_sec + 5.0
                if source_duration:
                    user_end = min(user_end, source_duration)

            if source_duration and user_start > source_duration:
                user_start = max(0.0, source_duration - 1.0)
            if source_duration and user_end > source_duration:
                user_end = source_duration
            if user_end <= user_start:
                increment = max(1.0 / source_fps, 0.1)
                user_end = user_start + increment
                if source_duration and user_end > source_duration:
                    user_end = source_duration
                    user_start = max(0.0, user_end - increment)

            analysis_start = max(user_start + 1.0, user_start)
            analysis_end = max(analysis_start, user_end - 1.0)

            video_url = url_for('select_bout_winners_video_source', upload_id=upload_id, bout_idx=bout.match_idx) if has_video else None

            video_info = ""
            if upload.is_multi_video and bout.upload_video:
                video_info = f" (Video {bout.upload_video.sequence_order})"

            trimmed_rel = _to_result_rel(bout.video_path)

            bout_data.append({
                'idx': bout.match_idx,
                'video_url': video_url,
                'result': bout.result,
                'video_info': video_info,
                'has_video': has_video,
                'duration': round(source_duration, 3) if source_duration else 0.0,
                'user_start': round(user_start, 3),
                'user_end': round(user_end, 3),
                'analysis_start': round(analysis_start, 3),
                'analysis_end': round(analysis_end, 3),
                'fps': round(source_fps, 3),
                'trimmed_clip': trimmed_rel
            })

        return render_template('select_bout_winners.html', upload=upload, bouts=bout_data)

    @app.route('/results/<int:upload_id>', methods=['GET', 'POST'])
    @login_required
    def results(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403
        
        result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))

        # Check if bout results are pending
        bouts = Bout.query.filter_by(upload_id=upload_id).order_by(Bout.match_idx).all()
        if upload.status == 'awaiting_user_input' or any(bout.result is None for bout in bouts):
            return redirect(url_for('select_bout_winners', upload_id=upload_id))
        
        result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))
        logging.info(f"Accessing result directory for user {upload.user_id}, upload {upload_id}: {result_dir}")

        result_root_abs = os.path.join(app.root_path, app.config['RESULT_FOLDER'])

        def _rel_from_result(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            abs_path = path if os.path.isabs(path) else os.path.join(app.root_path, path)
            if abs_path and os.path.exists(abs_path):
                try:
                    rel = os.path.relpath(abs_path, result_root_abs).replace(os.sep, '/')
                    if not rel.startswith('..'):
                        return rel
                except Exception:
                    pass
            prefix = app.config['RESULT_FOLDER'].rstrip('/') + '/'
            if path.startswith(prefix):
                return path[len(prefix):]
            return path

        bout_data = []
        bout_summaries = []
        if upload.status == 'completed':
            summaries_path = os.path.join(result_dir, 'fencer_analysis', 'bout_summaries.json')
            try:
                with open(summaries_path, 'r', encoding='utf-8') as f:
                    bout_summaries = json.load(f).get('bouts', [])
            except FileNotFoundError:
                logging.error(f"Bout summaries file not found at {summaries_path}")
            except Exception as e:
                logging.error(f"Error loading bout summaries: {e}")

        for bout in bouts:
            bout_dir = os.path.join(result_dir, 'matches', f'match_{bout.match_idx}')
            summary = next((s for s in bout_summaries if s.get('match_idx') == bout.match_idx), {})
            # Load GPT analysis text from match_analysis JSON
            gpt_analysis_text = ''
            try:
                match_analysis_path = os.path.join(result_dir, 'match_analysis', f'match_{bout.match_idx}_analysis.json')
                if os.path.exists(match_analysis_path):
                    with open(match_analysis_path, 'r', encoding='utf-8') as f:
                        match_analysis = json.load(f)
                        gpt_analysis_text = match_analysis.get('gpt_analysis') or ''
            except Exception as e:
                logging.error(f"Error loading GPT analysis for upload {upload_id}, match {bout.match_idx}: {e}")
            
            # Get tags for this bout
            left_tags = []
            right_tags = []
            for bout_tag in bout.tags:
                if bout_tag.fencer_side == 'left':
                    left_tags.append(bout_tag.tag.name)
                elif bout_tag.fencer_side == 'right':
                    right_tags.append(bout_tag.tag.name)
            
            extended_rel_path = None
            if hasattr(bout, 'extended_video_path') and bout.extended_video_path:
                extended_rel_path = _rel_from_result(bout.extended_video_path)
                logging.debug(f"Upload {upload_id}, Bout {bout.match_idx}: Using extended_video_path -> {extended_rel_path}")
            if not extended_rel_path and bout.video_path:
                extended_rel_path = _rel_from_result(bout.video_path)
                logging.debug(f"Upload {upload_id}, Bout {bout.match_idx}: Using video_path -> {extended_rel_path}")

            if not extended_rel_path:
                logging.warning(f"Upload {upload_id}, Bout {bout.match_idx}: No video path available! extended={bout.extended_video_path if hasattr(bout, 'extended_video_path') else 'N/A'}, regular={bout.video_path}")

            bout_data.append({
                'idx': bout.match_idx,
                'video': extended_rel_path,
                'keypoints_video': None,
                'analysis': summary.get('individual_analysis'),
                'judgement': summary.get('judgement'),
                'result': bout.result,
                'left_tags': sorted(left_tags),
                'right_tags': sorted(right_tags),
                'gpt_analysis': gpt_analysis_text
            })

        if request.method == 'POST' and 'save_fencers' in request.form:
            left_fencer_id_str = request.form.get('left_fencer_id')
            right_fencer_id_str = request.form.get('right_fencer_id')
            prev_left_fencer_id = upload.left_fencer_id
            prev_right_fencer_id = upload.right_fencer_id
            upload.left_fencer_id = int(left_fencer_id_str) if left_fencer_id_str and left_fencer_id_str.isdigit() else None
            upload.right_fencer_id = int(right_fencer_id_str) if right_fencer_id_str and right_fencer_id_str.isdigit() else None
            try:
                db.session.commit()
                # Clean up analysis files for all affected fencers when associations change
                for fencer_id in {prev_left_fencer_id, prev_right_fencer_id, upload.left_fencer_id, upload.right_fencer_id}:
                    if fencer_id:
                        fencer_analysis_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), 'fencer', str(fencer_id))
                        
                        files_to_clean = [
                            os.path.join(fencer_analysis_dir, 'chat_history.json'),
                            os.path.join(fencer_analysis_dir, 'holistic_analysis.json'),
                            os.path.join(fencer_analysis_dir, f'fencer_{fencer_id}_radar_profile.png'),
                            os.path.join(fencer_analysis_dir, f'fencer_{fencer_id}_profile_analysis.png')
                        ]
                        
                        for file_path in files_to_clean:
                            if os.path.exists(file_path):
                                try:
                                    os.remove(file_path)
                                    logging.info(f"Cleaned up {os.path.basename(file_path)} for fencer {fencer_id} due to association change in upload {upload_id}")
                                except Exception as e:
                                    logging.error(f"Error cleaning up {os.path.basename(file_path)} for fencer {fencer_id}: {str(e)}")
                        
                        # Also clean up HolisticAnalysis database record
                        try:
                            holistic_analysis = HolisticAnalysis.query.filter_by(fencer_id=fencer_id).first()
                            if holistic_analysis:
                                db.session.delete(holistic_analysis)
                                logging.info(f"Removed holistic analysis database record for fencer {fencer_id}")
                        except Exception as e:
                            logging.error(f"Error removing holistic analysis database record for fencer {fencer_id}: {str(e)}")
            except Exception as e:
                db.session.rollback()
                fencers = Fencer.query.filter_by(user_id=current_user.id).all()
                return render_template('results.html', upload=upload, bouts=bout_data, fencers=fencers, error='Failed to save fencer association')
            return redirect(url_for('results', upload_id=upload_id))

        fencers = Fencer.query.filter_by(user_id=current_user.id).all()
        return render_template('results.html', upload=upload, bouts=bout_data, fencers=fencers)

    @app.route('/video_view/<int:upload_id>')
    @login_required
    def video_view(upload_id):
        """Display video view performance analysis for an upload."""
        from models import VideoAnalysis
        import json
        import re
        
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403

        force_refresh_requested = request.args.get('refresh') == '1'
        if force_refresh_requested:
            logging.info(f"Force refresh requested via query parameter for upload {upload_id}; redirecting to regeneration flow")
            return redirect(url_for('video_view_refresh_wait', upload_id=upload_id, start=1))

        try:
            # Load pre-generated AI analysis from database
            video_analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
            if video_analysis:
                logging.info(f"Found VideoAnalysis record for upload {upload_id}: status='{video_analysis.status}', generated_at={video_analysis.generated_at}")
            else:
                logging.info(f"No VideoAnalysis record found for upload {upload_id}")
            
            # Load basic video view data (non-AI parts) from the analysis functions
            from your_scripts.video_view_analysis import get_basic_video_data
            basic_data = get_basic_video_data(upload_id, current_user.id)
            
            if not basic_data['success']:
                flash(f"Error loading video data: {basic_data['error']}", 'error')
                return redirect(url_for('status', upload_id=upload_id))
            
            detailed_analysis = basic_data.get('detailed_analysis', {})
            category_chart_images = basic_data.get('category_chart_images', {})

            analysis_info = _extract_ai_video_analysis(video_analysis)
            loss_analysis = analysis_info['loss_analysis']
            win_analysis = analysis_info['win_analysis']
            category_performance_analysis = analysis_info['category_performance_analysis']
            overall_performance_analysis = analysis_info['overall_performance_analysis']
            loss_reason_reports = analysis_info['loss_reason_reports']
            win_reason_reports = analysis_info['win_reason_reports']
            reason_summary_bullets = analysis_info['reason_summary_bullets']
            analysis_status = analysis_info['analysis_status']
            analysis_generated_at = analysis_info['analysis_generated_at']
            analysis_ready = analysis_info['analysis_ready']
            analysis_complete = analysis_info['analysis_complete']
            ai_section_status = analysis_info['ai_section_status']
            missing_section_labels = analysis_info['missing_section_labels']

            if analysis_ready and not analysis_complete:
                logging.warning(
                    "Video analysis record for upload %s is marked completed but missing sections: %s",
                    upload_id,
                    ', '.join(missing_section_labels) if missing_section_labels else 'None'
                )
            elif not analysis_ready:
                logging.info(
                    "AI analysis for upload %s not ready (status=%s); serving cached basic data only",
                    upload_id,
                    analysis_status
                )

            tactical_summary = _build_tactical_summary(
                basic_data.get('bout_type_stats', {}),
                win_reason_reports,
                loss_reason_reports
            )

            # Ensure rapid adjustments are available even for cached analyses
            if isinstance(overall_performance_analysis, dict):
                for side_key, reason_key in (('left_fencer', 'left'), ('right_fencer', 'right')):
                    side_analysis = overall_performance_analysis.get(side_key)
                    if not isinstance(side_analysis, dict):
                        continue
                    rapid = side_analysis.get('rapid_adjustments')
                    if not rapid:
                        summary_source = (reason_summary_bullets or {}).get(reason_key)
                        if not summary_source:
                            summary_source = (reason_summary_bullets or {}).get(f"{reason_key}_fencer")
                        computed = _build_immediate_adjustments(summary_source or {})
                        if computed:
                            side_analysis['rapid_adjustments'] = computed

            return render_template('video_view.html', 
                                 upload=upload,
                                 radar_data=basic_data['radar_data'],
                                 bout_type_stats=basic_data['bout_type_stats'],
                                 total_touches=basic_data['total_touches'],
                                 detailed_analysis=detailed_analysis,
                                category_chart_images=category_chart_images,
                                loss_analysis=loss_analysis,
                                win_analysis=win_analysis,
                                loss_reason_reports=loss_reason_reports,
                                win_reason_reports=win_reason_reports,
                                reason_summary_bullets=reason_summary_bullets,
                                category_performance_analysis=category_performance_analysis,
                                overall_performance_analysis=overall_performance_analysis,
                                touch_summary=basic_data.get('touch_summary', []),
                                analysis_status=analysis_status,
                                analysis_ready=analysis_ready,
                                analysis_complete=analysis_complete,
                                analysis_generated_at=analysis_generated_at,
                                ai_section_status=ai_section_status,
                                missing_section_labels=missing_section_labels,
                                tactical_summary=tactical_summary)
                                 
        except Exception as e:
            logging.error(f"Error loading video view for upload {upload_id}: {e}")
            flash(f"Error loading performance analysis: {str(e)}", 'error')
            return redirect(url_for('status', upload_id=upload_id))

    @app.route('/video_view/<int:upload_id>/refresh_wait')
    @login_required
    def video_view_refresh_wait(upload_id):
        from models import VideoAnalysis
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403

        task_id = request.args.get('task_id')
        start_requested = request.args.get('start') == '1'

        if not task_id and not start_requested:
            return redirect(url_for('video_view_refresh_wait', upload_id=upload_id, start=1))

        analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()

        if start_requested:
            # Always dispatch a new regeneration task regardless of current status
            dispatch_task_id, ran_inline = _dispatch_video_view_regeneration(upload_id)
            task_id = dispatch_task_id or None
            if dispatch_task_id:
                logging.info(f"Triggered video analysis regeneration for upload {upload_id} (task {dispatch_task_id})")
                # Redirect once with task_id so the waiting page can poll Celery status
                return redirect(url_for('video_view_refresh_wait', upload_id=upload_id, task_id=dispatch_task_id))
            elif ran_inline:
                logging.info(f"Triggered inline video analysis regeneration for upload {upload_id}")
                # Fall through to render waiting page without redirect so it can poll DB status

        status_url = url_for('video_view_refresh_status', upload_id=upload_id)
        done_url = url_for('video_view', upload_id=upload_id)
        return render_template(
            'video_view_refresh.html',
            upload=upload,
            task_id=task_id,
            analysis_status=analysis.status if analysis else None,
            status_url=status_url,
            done_url=done_url
        )

    @app.route('/video_view_refresh_status/<int:upload_id>')
    @login_required
    def video_view_refresh_status(upload_id):
        from models import VideoAnalysis
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return jsonify({'status': 'unauthorized'}), 403

        task_id = request.args.get('task_id')
        analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()

        analysis_info = _extract_ai_video_analysis(analysis)

        base_status = analysis.status or 'none' if analysis else 'none'
        status = base_status
        if base_status == 'completed' and not analysis_info['analysis_complete']:
            status = 'processing'

        generated_at_dt = analysis_info['analysis_generated_at']
        generated_at = generated_at_dt.isoformat() if generated_at_dt else None

        celery_state = None
        if task_id:
            try:
                async_result = celery.AsyncResult(task_id)
                celery_state = async_result.state
            except Exception as e:
                logging.warning(f"Could not fetch Celery status for task {task_id}: {e}")

        if not analysis and celery_state in (None, 'PENDING', 'RECEIVED', 'STARTED', 'RETRY'):
            status = 'pending'
        elif not analysis and celery_state == 'FAILURE':
            status = 'error'
        elif analysis and base_status in (None, 'none') and celery_state in ('PENDING', 'RECEIVED', 'STARTED', 'RETRY'):
            status = 'pending'

        return jsonify({
            'status': status,
            'celery_state': celery_state,
            'generated_at': generated_at,
            'analysis_ready': analysis_info['analysis_ready'],
            'analysis_complete': analysis_info['analysis_complete'],
            'missing_sections': analysis_info['missing_section_labels'],
            'ai_section_status': analysis_info['ai_section_status'],
            'redirect_url': url_for('video_view', upload_id=upload_id)
        })

    @app.route('/status/<int:upload_id>')
    @login_required
    def status(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403
        return render_template('status.html', upload=upload)

    @app.route('/Uploads/<path:filename>')
    @login_required
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/results/<path:filename>')
    @login_required
    def result_file(filename):
        # Support HTTP Range requests for smoother streaming and seeking
        # Use conditional=True to enable Range handling across WSGI servers
        target_dir = app.config['RESULT_FOLDER']
        try:
            return send_from_directory(target_dir, filename, conditional=True)
        except TypeError:
            # Older Flask versions may not support conditional param on send_from_directory
            return send_from_directory(target_dir, filename)

    @app.route('/select_bout_winners/<int:upload_id>/video_source/<int:bout_idx>')
    @login_required
    def select_bout_winners_video_source(upload_id, bout_idx):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403

        bout = Bout.query.filter_by(upload_id=upload_id, match_idx=bout_idx).first()
        if not bout:
            abort(404)

        if upload.is_multi_video and bout.upload_video:
            source_path = bout.upload_video.video_path
        else:
            source_path = upload.video_path

        if source_path:
            source_abs = source_path if os.path.isabs(source_path) else os.path.join(app.root_path, source_path)
        else:
            source_abs = None

        if not source_abs or not os.path.exists(source_abs):
            abort(404)

        return send_file(source_abs, conditional=True)

    @app.route('/check_status/<int:upload_id>')
    @login_required
    def check_status(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        return jsonify({'status': upload.status})

    @app.route('/chat_view/<int:upload_id>')
    @login_required
    def chat_view(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload.id))
        chat_history_path = os.path.join(result_dir, 'fencer_analysis', f'chat_history_{upload_id}.json')
        conversation = []
        if os.path.exists(chat_history_path):
            try:
                with open(chat_history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversation = data.get('conversation', [])
            except Exception:
                logging.warning(f"Could not load chat history for upload {upload_id}")
        
        return render_template('chat.html', upload_id=upload_id, conversation=conversation)

    @app.route('/chat/<int:upload_id>', methods=['POST'])
    @login_required
    def interactive_chat(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user_input = request.json.get('user_input', '')
        result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload.id))
        summaries_path = os.path.join(result_dir, 'fencer_analysis', 'bout_summaries.json')
        cross_bout_path = os.path.join(result_dir, 'fencer_analysis', 'cross_bout_analysis.json')
        chat_history_path = os.path.join(result_dir, 'fencer_analysis', f'chat_history_{upload_id}.json')
        ai_reports_dir = os.path.join(result_dir, 'ai_reports')
        
        conversation = []
        if os.path.exists(chat_history_path):
            try:
                with open(chat_history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversation = data.get('conversation', [])
            except Exception as e:
                logging.warning(f"Could not load chat history for upload {upload_id}: {str(e)}")
        
        conversation_history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in conversation]) if conversation else "No conversation history."
        
        response_data = {'response': ''}
        
        if user_input:
            try:
                with open(summaries_path, 'r', encoding='utf-8') as f:
                    summaries_data = json.load(f)
                bouts = summaries_data.get('bouts', [])
                fps = summaries_data.get('fps', 30)
                if not bouts:
                    logging.error(f"No bouts found in summaries data for upload {upload_id}")
                    return jsonify({'error': 'No bout data available'}), 500
            except FileNotFoundError:
                logging.error(f"Bout summaries file not found at {summaries_path}")
                return jsonify({'error': 'Bout summaries file not found'}), 500
            except Exception as e:
                logging.error(f"Error loading bout summaries: {str(e)}\n{traceback.format_exc()}")
                return jsonify({'error': 'Error loading bout summaries'}), 500
            
            try:
                with open(cross_bout_path, 'r', encoding='utf-8') as f:
                    cross_bout_data = json.load(f)
                cross_bout_text = cross_bout_data.get('analysis', 'Analysis text not found.')
            except FileNotFoundError:
                logging.error(f"Cross-bout analysis file not found at {cross_bout_path}")
                cross_bout_text = 'Cross-bout analysis not found.'
            except Exception as e:
                logging.error(f"Error loading cross-bout analysis: {str(e)}\n{traceback.format_exc()}")
                cross_bout_text = 'Cross-bout analysis not found.'
            
            # Include weapon type for this upload
            weapon_label = upload.weapon_type.title() if getattr(upload, 'weapon_type', None) else 'Unknown'
            video_summary = f"""
        **Video Summary** (Video ID: {upload_id}, Weapon: {weapon_label}):
        - Number of matches: {len(bouts)}
        - Cross-bout analysis: {cross_bout_text}
        """

            # Enrich with Video-Type Analysis context (metrics + per-bout categories, and reasons if available)
            metrics_overview_lines = []
            bout_context_lines = []

            try:
                # Load deterministic metrics and category analyses (no AI calls)
                from your_scripts.video_view_analysis import get_basic_video_data
                basic_view = get_basic_video_data(upload_id, current_user.id)
                if basic_view.get('success'):
                    radar = basic_view.get('radar_data', {}) or {}
                    left_radar = radar.get('left_fencer') or {}
                    right_radar = radar.get('right_fencer') or {}
                    def _radar_pairs(r):
                        labels = r.get('labels') or []
                        values = r.get('values') or []
                        pairs = []
                        for i, label in enumerate(labels):
                            try:
                                val = values[i]
                            except Exception:
                                val = None
                            if val is not None:
                                pairs.append(f"{label}: {val}")
                        return pairs
                    left_pairs = _radar_pairs(left_radar)
                    right_pairs = _radar_pairs(right_radar)
                    if left_pairs:
                        metrics_overview_lines.append("Left Radar → " + "; ".join(left_pairs))
                    if right_pairs:
                        metrics_overview_lines.append("Right Radar → " + "; ".join(right_pairs))

                    # Add mirror-chart core stats (display_data) for inbox/attack/defense if present
                    detailed = basic_view.get('detailed_analysis') or {}
                    for cat_key, cat_label in (('in_box', 'In-Box'), ('attack', 'Attack'), ('defense', 'Defense')):
                        cat = detailed.get(cat_key) or {}
                        disp = cat.get('display_data') or {}
                        left_disp = disp.get('left_fencer') or {}
                        right_disp = disp.get('right_fencer') or {}
                        if left_disp:
                            metrics_overview_lines.append(f"Left {cat_label} → " + "; ".join([f"{k}: {v}" for k, v in list(left_disp.items())[:6]]))
                        if right_disp:
                            metrics_overview_lines.append(f"Right {cat_label} → " + "; ".join([f"{k}: {v}" for k, v in list(right_disp.items())[:6]]))
            except Exception as e:
                logging.warning(f"Video view basic data not available for chat context: {e}")

            # Determine per-fencer bout categories
            try:
                from your_scripts.bout_classification import classify_bout_categories
                for bout in bouts:
                    total_frames = bout.get('total_frames')
                    fps_local = bout.get('fps', fps)
                    left_data = bout.get('left_data', {}) or {}
                    right_data = bout.get('right_data', {}) or {}
                    left_cat, right_cat = classify_bout_categories(left_data, right_data, total_frames, fps_local)
                    winner_text = bout.get('result', 'unknown')
                    if winner_text == 'left':
                        winner_text = 'Left wins'
                    elif winner_text == 'right':
                        winner_text = 'Right wins'
                    elif winner_text == 'skip':
                        winner_text = 'Skipped'
                    bout_context_lines.append(
                        f"Match {bout.get('match_idx', '?')}: Left={left_cat}, Right={right_cat}, Result={winner_text}"
                    )
            except Exception as e:
                logging.warning(f"Could not classify bout categories for chat context: {e}")

            # Load precomputed win/loss reason reports if available
            try:
                reasons_lines = []
                win_path = os.path.join(ai_reports_dir, 'win_reason_reports.json')
                loss_path = os.path.join(ai_reports_dir, 'loss_reason_reports.json')
                if os.path.exists(win_path):
                    with open(win_path, 'r', encoding='utf-8') as f:
                        win_reports = json.load(f)
                    # Summarize by fencer and category
                    for side in ('left_fencer', 'right_fencer'):
                        for cat in ('in_box', 'attack', 'defense'):
                            entries = (win_reports.get(side, {}) or {}).get(cat, []) or []
                            if entries:
                                labels = [e.get('reason_label') or e.get('reason_key') for e in entries]
                                if labels:
                                    reasons_lines.append(f"{side.replace('_',' ').title()} {cat} wins → " + ", ".join(sorted(set([str(l) for l in labels if l]))))
                if os.path.exists(loss_path):
                    with open(loss_path, 'r', encoding='utf-8') as f:
                        loss_reports = json.load(f)
                    for side in ('left_fencer', 'right_fencer'):
                        for cat in ('in_box', 'attack', 'defense'):
                            entries = (loss_reports.get(side, {}) or {}).get(cat, []) or []
                            if entries:
                                labels = [e.get('reason_label') or e.get('reason_key') for e in entries]
                                if labels:
                                    reasons_lines.append(f"{side.replace('_',' ').title()} {cat} losses → " + ", ".join(sorted(set([str(l) for l in labels if l]))))
            except Exception as e:
                logging.warning(f"No precomputed reason reports for chat context: {e}")
            for bout in bouts:
                if 'left_data' not in bout or 'right_data' not in bout:
                    logging.warning(f"Bout {bout.get('match_idx', 'unknown')} missing left_data or right_data")
                    continue
                try:
                    bout_result = Bout.query.filter_by(upload_id=upload_id, match_idx=bout['match_idx']).first()
                    result_text = bout_result.result if bout_result and bout_result.result else "Not specified"
                    if result_text == 'skip':
                        result_text = "Skipped"
                    elif result_text == 'left':
                        result_text = "Left wins"
                    elif result_text == 'right':
                        result_text = "Right wins"
                    
                    # Define left_first_step and right_first_step here
                    left_first_step = bout['left_data'].get('first_step', {})
                    right_first_step = bout['right_data'].get('first_step', {})
                    
                    # Safely compute latest pause/retreat end times (seconds)
                    left_latest_end = bout['left_data'].get('latest_pause_retreat_end', bout['left_data'].get('latest_pause_end', -1))
                    right_latest_end = bout['right_data'].get('latest_pause_retreat_end', bout['right_data'].get('latest_pause_end', -1))
                    left_latest_end_sec = (left_latest_end / bout.get('fps', 30)) if (left_latest_end is not None and left_latest_end != -1) else None
                    right_latest_end_sec = (right_latest_end / bout.get('fps', 30)) if (right_latest_end is not None and right_latest_end != -1) else None

                    # Safe numeric helpers
                    def _to_float(val, default=0.0):
                        try:
                            return float(val)
                        except (TypeError, ValueError):
                            return default

                    # Frame range to seconds (guard None)
                    fps_local = bout.get('fps', fps)
                    fr = bout.get('frame_range') or [None, None]
                    fr_start_sec = None if fr[0] is None else (_to_float(fr[0]) / fps_local)
                    fr_end_sec = None if fr[1] is None else (_to_float(fr[1]) / fps_local)

                    # First step safe numbers
                    left_init_time = _to_float(left_first_step.get('init_time'))
                    left_init_velocity = _to_float(left_first_step.get('velocity'))
                    right_init_time = _to_float(right_first_step.get('init_time'))
                    right_init_velocity = _to_float(right_first_step.get('velocity'))

                    # Overall velocity/acceleration safe numbers
                    left_overall_v = _to_float(bout['left_data'].get('velocity'))
                    left_overall_a = _to_float(bout['left_data'].get('acceleration'))
                    right_overall_v = _to_float(bout['right_data'].get('velocity'))
                    right_overall_a = _to_float(bout['right_data'].get('acceleration'))

                    video_summary += f"""
        **Match {bout['match_idx']}**:
        - Frame range: {('N/A' if fr_start_sec is None else f"{fr_start_sec:.2f}")} to {('N/A' if fr_end_sec is None else f"{fr_end_sec:.2f}")} seconds
        - Type: {bout['type']} ({bout['total_frames']} frames)
        - Result: {result_text}
        - Left fencer data:
        - Start time: {left_init_time:.2f} seconds ({'Fast' if left_first_step.get('is_fast', False) else 'Slow'}), velocity: {left_init_velocity:.2f}
        - Advance intervals: {bout['left_data'].get('advance_sec', bout['left_data'].get('advance_intervals', 'N/A'))}
        - Pause/retreat intervals: {bout['left_data'].get('pause_sec', bout['left_data'].get('pause_intervals', 'N/A'))}
        - Arm extensions: {bout['left_data'].get('arm_extensions_sec', bout['left_data'].get('arm_extensions', 'N/A'))}, frequency: {bout['left_data'].get('arm_extension_freq', 0)}
        - Lunge: {'Yes' if bout['left_data'].get('has_launch', False) else 'No'}, {'N/A' if bout['left_data'].get('launch_frame') is None else f"{bout['left_data']['launch_frame'] / bout.get('fps', 30):.2f} seconds"}
        - Velocity/acceleration: {left_overall_v:.2f}/{left_overall_a:.2f}
        - Latest pause end: {('N/A' if left_latest_end_sec is None else f"{left_latest_end_sec:.2f} seconds")}
        - Lunge timing: {'N/A' if bout['left_data'].get('launch_promptness') is None or bout['left_data'].get('launch_promptness') == float('inf') else f"{bout['left_data']['launch_promptness']:.2f} seconds"}
        - Additional stats: Attack success rate {bout['left_data'].get('attack_success_rate', 0):.1%}, total attacks {bout['left_data'].get('total_attacks', 0)}
        - Right fencer data:
        - Start time: {right_init_time:.2f} seconds ({'Fast' if right_first_step.get('is_fast', False) else 'Slow'}), velocity: {right_init_velocity:.2f}
        - Advance intervals: {bout['right_data'].get('advance_sec', bout['right_data'].get('advance_intervals', 'N/A'))}
        - Pause/retreat intervals: {bout['right_data'].get('pause_sec', bout['right_data'].get('pause_intervals', 'N/A'))}
        - Arm extensions: {bout['right_data'].get('arm_extensions_sec', bout['right_data'].get('arm_extensions', 'N/A'))}, frequency: {bout['right_data'].get('arm_extension_freq', 0)}
        - Lunge: {'Yes' if bout['right_data'].get('has_launch', False) else 'No'}, {'N/A' if bout['right_data'].get('launch_frame') is None else f"{bout['right_data']['launch_frame'] / bout.get('fps', 30):.2f} seconds"}
        - Velocity/acceleration: {right_overall_v:.2f}/{right_overall_a:.2f}
        - Latest pause end: {('N/A' if right_latest_end_sec is None else f"{right_latest_end_sec:.2f} seconds")}
        - Lunge timing: {'N/A' if bout['right_data'].get('launch_promptness') is None or bout['right_data'].get('launch_promptness') == float('inf') else f"{bout['right_data']['launch_promptness']:.2f} seconds"}
        - Additional stats: Attack success rate {bout['right_data'].get('attack_success_rate', 0):.1%}, total attacks {bout['right_data'].get('total_attacks', 0)}
        """
                except Exception as e:
                    logging.error(f"Error processing bout {bout.get('match_idx', 'unknown')}: {str(e)}\n{traceback.format_exc()}")
                    continue
            
            chat_prompt = """
You are a fencing video analysis assistant. Engage in real-time conversation with users about videos containing multiple bouts. The goal is to analyze technical and tactical questions around "specific bouts" and provide meaningful suggestions. Avoid referee judgment language and win/loss determinations, focus on key points like "distance management, timing, rhythm, footwork, arm extension, lunging".

【Video Data】
{video_summary}

【Video-Type Analysis Context】
{metrics_overview}
{bout_context}
{reason_summaries}

【Conversation History】
{conversation_history}

【Guidance】
- If this is the first message: Greet and invite questions, for example: "Hello, I am your fencing analysis assistant. We can discuss specific bouts from video {upload_id}. Which fencer or which bout are you most interested in?"
- If user provides a question:
  - First confirm their focus (e.g., "You're interested in attacking strategy").
  - Provide key points and timestamps (precise to the second) based on video data.
  - Give actionable advice (e.g., "Against an opponent who pauses frequently, try initiating first at... seconds").
  - Emphasize technical terminology (distance management, timing, feints, footwork).
  - Answer in 150-300 words, concise and friendly with clear structure.
  - End with a follow-up question to continue deeper discussion (e.g., "Which coordination segment would you like to improve?").

【User Question】{user_input}
【Video ID】{upload_id}
            """
            
            try:
                chat_response = gemini_generate_text(
                    chat_prompt.format(
                        video_summary=video_summary,
                        metrics_overview=("\n".join(metrics_overview_lines) if metrics_overview_lines else "(metrics unavailable)"),
                        bout_context=("\n".join(bout_context_lines) if bout_context_lines else "(per-bout categories unavailable)"),
                        reason_summaries=("\n".join(reasons_lines) if 'reasons_lines' in locals() and reasons_lines else ""),
                        conversation_history=conversation_history,
                        user_input=user_input,
                        upload_id=upload_id,
                    ),
                    model=current_app.config.get('GEMINI_MODEL', DEFAULT_GEMINI_MODEL),
                    temperature=0.2,
                    top_k=1,
                    top_p=0.8,
                    max_output_tokens=2048,
                    timeout_seconds=45,
                    max_attempts=6,
                    response_mime_type='text/plain',
                )
                logging.info(f"Generated chat response for video upload {upload_id}: {chat_response[:200]}...")
            except Exception as e:
                logging.error(f"Error in GPT chat response: {str(e)}\n{traceback.format_exc()}")
                return jsonify({'error': 'Unable to generate response'}), 500

            conversation.append({
                'user': user_input,
                'assistant': chat_response
            })
            try:
                os.makedirs(os.path.dirname(chat_history_path), exist_ok=True)
                with open(chat_history_path, 'w', encoding='utf-8') as f:
                    json.dump({'conversation': conversation}, f, ensure_ascii=False, indent=4)
                logging.info(f"Updated chat history for video upload {upload_id}")
            except Exception as e:
                logging.error(f"Error saving chat history: {str(e)}\n{traceback.format_exc()}")

            response_data['response'] = chat_response
        else:
            response_data['response'] = f"Hello, I'm your fencing analysis assistant. We can discuss the content of video ID {upload_id}. Which fencer or bout in the video are you interested in?"

        return jsonify(response_data)

    @app.route('/holistic_chat/<int:fencer_id>', methods=['GET', 'POST'])
    @login_required
    def holistic_chat(fencer_id):
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403

        flash('Holistic analysis now lives inside the Professional Profile view.', 'info')
        return redirect(url_for('fencer_profile', fencer_id=fencer_id))

    @app.route('/delete_association/<int:upload_id>/<int:fencer_id>/<string:side>', methods=['POST'])
    @login_required
    def delete_association(upload_id, fencer_id, side):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        if side == 'left' and upload.left_fencer_id == fencer_id:
            upload.left_fencer_id = None
            db.session.commit()
        elif side == 'right' and upload.right_fencer_id == fencer_id:
            upload.right_fencer_id = None
            db.session.commit()
        else:
            return jsonify({'error': 'Invalid association'}), 400
        
        # Clean up fencer-specific analysis files when association is deleted
        fencer_analysis_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), 'fencer', str(fencer_id))
        
        files_to_clean = [
            os.path.join(fencer_analysis_dir, 'chat_history.json'),
            os.path.join(fencer_analysis_dir, 'holistic_analysis.json'),
            os.path.join(fencer_analysis_dir, f'fencer_{fencer_id}_radar_profile.png'),
            os.path.join(fencer_analysis_dir, f'fencer_{fencer_id}_profile_analysis.png')
        ]
        
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logging.info(f"Cleaned up {os.path.basename(file_path)} for fencer {fencer_id} due to association deletion in upload {upload_id}")
                except Exception as e:
                    logging.error(f"Error cleaning up {os.path.basename(file_path)} for fencer {fencer_id}: {str(e)}")
        
        # Also clean up HolisticAnalysis database record
        try:
            holistic_analysis = HolisticAnalysis.query.filter_by(fencer_id=fencer_id).first()
            if holistic_analysis:
                db.session.delete(holistic_analysis)
                db.session.commit()
                logging.info(f"Removed holistic analysis database record for fencer {fencer_id}")
        except Exception as e:
            logging.error(f"Error removing holistic analysis database record for fencer {fencer_id}: {str(e)}")
        
        return redirect(url_for('manage_fencers'))

    @app.route('/reanalyze/<int:upload_id>', methods=['POST'])
    @login_required
    def reanalyze_video(upload_id):
        upload = Upload.query.get_or_404(upload_id)
        if upload.user_id != current_user.id:
            return 'Unauthorized', 403
        
        try:
            # Reset status and counters to re-trigger analysis
            upload.status = 'results_submitted'
            upload.bouts_analyzed = 0
            
            # Clear existing tags for this upload to avoid duplicates
            bouts = Bout.query.filter_by(upload_id=upload.id).all()
            for bout in bouts:
                BoutTag.query.filter_by(bout_id=bout.id).delete()
            
            db.session.commit()
            
            from tasks import generate_analysis_task
            generate_analysis_task.delay(upload.id)
            
            flash('Re-analysis started. Page will refresh in a few seconds.', 'success')
            logging.info(f"Re-analysis triggered for upload {upload_id} by user {current_user.username}")

        except Exception as e:
            db.session.rollback()
            flash(f'Error starting re-analysis: {str(e)}', 'error')
            logging.error(f"Error triggering re-analysis for upload {upload_id}: {str(e)}")
            
        return redirect(url_for('results', upload_id=upload_id))

    def _get_fencer_profile_dir(user_id: int, fencer_id: int) -> str:
        return os.path.join('/workspace/Project/fencer_profiles', str(user_id), str(fencer_id))

    @app.route('/fencer/<int:fencer_id>/profile')
    @login_required
    def fencer_profile(fencer_id):
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return 'Unauthorized', 403

        if request.args.get('refreshed'):
            flash('Professional profile refreshed from the latest completed videos.', 'success')
        elif request.args.get('refresh_error'):
            error_message = request.args.get('error_msg') or 'Unable to refresh profile at this time.'
            flash(error_message, 'warning')

        uploads = (
            Upload.query.filter(
                (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
            )
            .filter_by(user_id=current_user.id, status='completed')
            .order_by(Upload.match_datetime.desc(), Upload.id.desc())
            .all()
        )

        profile_dir = _get_fencer_profile_dir(current_user.id, fencer_id)
        report_path = os.path.join(profile_dir, 'professional_profile.json')
        report_data = None
        report_error = None

        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir, exist_ok=True)

        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
            except Exception as exc:
                logging.error("Failed to load professional profile for fencer %s: %s", fencer_id, exc)
                report_data = None

        if report_data is None:
            try:
                generation_result = generate_professional_report(fencer_id, current_user.id, fencer.name)
                if generation_result.get('success') and os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                else:
                    report_error = generation_result.get('error') or 'Unable to build professional report for this fencer yet.'
            except Exception as exc:
                report_error = f'Professional report generation failed: {exc}'
                logging.error("Professional report generation failed for fencer %s: %s", fencer_id, exc, exc_info=True)

        uploads_context = [
            {
                'id': upload.id,
                'match_datetime': upload.match_datetime,
                'weapon_type': upload.weapon_type,
                'match_title': upload.match_title,
                'is_multi_video': upload.is_multi_video,
            }
            for upload in uploads
        ]

        report_json = json.dumps(report_data, ensure_ascii=False) if report_data else '{}'

        return render_template(
            'professional_profile.html',
            fencer=fencer,
            professional_report=report_data,
            professional_report_json=report_json,
            professional_report_error=report_error,
            uploads=uploads_context,
        )

    @app.route('/fencer/<int:fencer_id>/profile/refresh', methods=['POST'])
    @login_required
    def refresh_fencer_profile(fencer_id):
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return 'Unauthorized', 403

        from tasks import regenerate_fencer_profile_task

        try:
            async_result = regenerate_fencer_profile_task.apply_async(
                args=[fencer_id, current_user.id]
            )
        except Exception as exc:
            logging.error(
                "Failed to enqueue profile regeneration for fencer %s: %s",
                fencer_id,
                exc,
                exc_info=True,
            )
            flash(f'Unable to queue profile refresh: {exc}', 'error')
            return redirect(url_for('fencer_profile', fencer_id=fencer_id))

        return render_template(
            'profile_wait.html',
            fencer=fencer,
            task_id=async_result.id,
            profile_url=url_for('fencer_profile', fencer_id=fencer_id),
            status_url=url_for('profile_refresh_status', fencer_id=fencer_id, task_id=async_result.id),
        )

    @app.route('/fencer/<int:fencer_id>/profile/delete', methods=['POST'])
    @login_required
    def delete_fencer_profile(fencer_id):
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return 'Unauthorized', 403

        profile_dir = _get_fencer_profile_dir(current_user.id, fencer_id)
        removed_files = False
        try:
            if os.path.exists(profile_dir):
                shutil.rmtree(profile_dir)
                removed_files = True
        except Exception as exc:
            logging.error("Failed removing profile directory for fencer %s: %s", fencer_id, exc, exc_info=True)
            flash('Could not remove stored profile files. Nothing was changed.', 'error')
            return redirect(url_for('fencer_profile', fencer_id=fencer_id))

        try:
            holistic_analysis = HolisticAnalysis.query.filter_by(fencer_id=fencer_id).first()
            if holistic_analysis:
                db.session.delete(holistic_analysis)
                db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.error("Failed clearing holistic analysis cache for fencer %s: %s", fencer_id, exc, exc_info=True)
            flash('Profile files removed, but cached analysis data could not be cleared.', 'warning')
            return redirect(url_for('fencer_profile', fencer_id=fencer_id))

        if removed_files:
            flash('Saved professional profile removed. Generate a new one by refreshing.', 'success')
        else:
            flash('No saved professional profile was found to delete.', 'info')

        return redirect(url_for('fencer_profile', fencer_id=fencer_id))

    @app.route('/fencer/<int:fencer_id>/profile/refresh/status/<task_id>')
    @login_required
    def profile_refresh_status(fencer_id, task_id):
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return jsonify({'state': 'UNAUTHORIZED'}), 403

        async_result = AsyncResult(task_id, app=celery)
        state = async_result.state

        if state == 'PENDING':
            return jsonify({'state': 'PENDING'})
        if state == 'STARTED':
            return jsonify({'state': 'STARTED'})
        if state == 'SUCCESS':
            payload = async_result.result or {}
            success = bool(payload.get('success'))
            return jsonify({
                'state': 'SUCCESS',
                'success': success,
                'errors': payload.get('errors', []),
            })
        if state in ('FAILURE', 'REVOKED'):
            error_message = str(async_result.result) if async_result.result else 'Task failed.'
            return jsonify({
                'state': state,
                'success': False,
                'errors': [error_message],
            })

        return jsonify({'state': state})

    @app.route('/fencer/<int:fencer_id>/profile/<weapon_type>')
    @login_required
    def fencer_profile_weapon(fencer_id, weapon_type):
        return redirect(url_for('fencer_profile', fencer_id=fencer_id))

    @app.route('/fencer/<int:fencer_id>/tags')
    @login_required
    def fencer_tags(fencer_id):
        """Show all tags for a fencer with aggregated video counts"""
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return 'Unauthorized', 403

        # Get all uploads for this fencer
        uploads = Upload.query.filter(
            (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
        ).filter_by(user_id=current_user.id, status='completed').all()

        # Aggregate tags with video counts
        tag_stats = {}
        for upload in uploads:
            fencer_side = 'left' if upload.left_fencer_id == fencer_id else 'right'

            # Get all bout tags for this fencer in this upload
            bout_tags = db.session.query(BoutTag, Bout, Tag).join(
                Bout, BoutTag.bout_id == Bout.id
            ).join(
                Tag, BoutTag.tag_id == Tag.id
            ).filter(
                Bout.upload_id == upload.id,
                BoutTag.fencer_side == fencer_side
            ).all()

            for bout_tag, bout, tag in bout_tags:
                if tag.id not in tag_stats:
                    tag_stats[tag.id] = {
                        'tag': tag,
                        'count': 0,
                        'videos': []
                    }
                tag_stats[tag.id]['count'] += 1
                # Store unique upload info
                video_info = {
                    'upload_id': upload.id,
                    'bout_idx': bout.match_idx,
                    'side': fencer_side,
                    'result': bout.result,
                    'match_title': upload.match_title or f"Video {upload.id}",
                    'match_datetime': upload.match_datetime
                }
                # Avoid duplicates
                if video_info not in tag_stats[tag.id]['videos']:
                    tag_stats[tag.id]['videos'].append(video_info)

        # Sort tags by count (most common first)
        sorted_tags = sorted(tag_stats.values(), key=lambda x: x['count'], reverse=True)

        # Tag translations for better display
        TAG_TRANSLATIONS = {
            'launch': 'Launch',
            'no_launch': 'No Launch',
            'arm_extension': 'Arm Extension',
            'no_arm_extension': 'No Arm Extension',
            'over_extension': 'Over Extension',
            'simple_attack': 'Simple Attack',
            'compound_attack': 'Compound Attack',
            'holding_attack': 'Holding Attack',
            'preparation_attack': 'Preparation Attack',
            'simple_preparation': 'Simple Preparation',
            'no_attacks': 'No Attacks',
            'limited_attack_variety': 'Limited Attack Variety',
            'steady_tempo': 'Steady Tempo',
            'variable_tempo': 'Variable Tempo',
            'broken_tempo': 'Broken Tempo',
            'excessive_pausing': 'Excessive Pausing',
            'excessive_tempo_changes': 'Excessive Tempo Changes',
            'good_attack_distance': 'Good Attack Distance',
            'poor_attack_distance': 'Poor Attack Distance',
            'maintain_safe_distance': 'Maintain Safe Distance',
            'poor_distance_maintaining': 'Poor Distance Maintaining',
            'counter_opportunity_taken': 'Counter Opportunity Taken',
            'counter_opportunity_missed': 'Counter Opportunity Missed',
            'retreat_quality_good': 'Good Retreat Quality',
            'retreat_quality_poor': 'Poor Retreat Quality'
        }

        return render_template(
            'fencer_tags.html',
            fencer=fencer,
            tag_stats=sorted_tags,
            TAG_TRANSLATIONS=TAG_TRANSLATIONS
        )

    @app.route('/fencer/<int:fencer_id>/tag/<int:tag_id>')
    @login_required
    def tag_gallery(fencer_id, tag_id):
        fencer = Fencer.query.get_or_404(fencer_id)
        if fencer.user_id != current_user.id:
            return 'Unauthorized', 403
        
        tag = Tag.query.get_or_404(tag_id)
        
        # Get all bouts where this fencer has this tag
        bout_videos = []
        uploads = Upload.query.filter(
            (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
        ).filter_by(user_id=current_user.id, status='completed').all()
        
        for upload in uploads:
            fencer_side = 'left' if upload.left_fencer_id == fencer_id else 'right'
            
            # Get bouts for this upload that have the specified tag for this fencer
            bouts = db.session.query(Bout).join(BoutTag).filter(
                Bout.upload_id == upload.id,
                BoutTag.tag_id == tag_id,
                BoutTag.fencer_side == fencer_side
            ).all()
            
            for bout in bouts:
                if bout.video_path and os.path.exists(bout.video_path):
                    bout_videos.append({
                        'upload_id': upload.id,
                        'bout_idx': bout.match_idx,
                        'video_path': f'{upload.user_id}/{upload.id}/matches/match_{bout.match_idx}/match_{bout.match_idx}.mp4',
                        'result': bout.result
                    })
        
        # Tag translations for better display
        TAG_TRANSLATIONS = {
            'launch': 'Launch',
            'no_launch': 'No Launch',
            'arm_extension': 'Arm Extension',
            'no_arm_extension': 'No Arm Extension',
            'over_extension': 'Over Extension',
            'simple_attack': 'Simple Attack',
            'compound_attack': 'Compound Attack',
            'holding_attack': 'Holding Attack',
            'preparation_attack': 'Preparation Attack',
            'simple_preparation': 'Simple Preparation',
            'no_attacks': 'No Attacks',
            'limited_attack_variety': 'Limited Attack Variety',
            'steady_tempo': 'Steady Tempo',
            'variable_tempo': 'Variable Tempo',
            'broken_tempo': 'Broken Tempo',
            'excessive_pausing': 'Excessive Pausing',
            'excessive_tempo_changes': 'Excessive Tempo Changes',
            'good_attack_distance': 'Good Attack Distance',
            'poor_attack_distance': 'Poor Attack Distance',
            'maintain_safe_distance': 'Maintain Safe Distance',
            'poor_distance_maintaining': 'Poor Distance Maintaining',
            'poor_distance_maintenance': 'Poor Distance Maintenance',
            'consistent_spacing': 'Consistent Spacing',
            'inconsistent_spacing': 'Inconsistent Spacing',
            'good_defensive_quality': 'Good Defensive Quality',
            'poor_defensive_quality': 'Poor Defensive Quality',
            'missed_counter_opportunities': 'Missed Counter Opportunities',
            'failed_space_opening': 'Failed Space Opening',
            'failed_distance_pulls': 'Failed Distance Pulls',
            'low_speed': 'Low Speed',
            'poor_acceleration': 'Poor Acceleration',
            'fast_reaction_time': 'Fast Reaction Time',
        }
        
        return render_template('tag_gallery.html', 
                             fencer=fencer, 
                             tag=tag, 
                             bout_videos=bout_videos,
                             TAG_TRANSLATIONS=TAG_TRANSLATIONS)

    def get_safe_uploads_and_fencers(user_id):
        """Safely get uploads and fencers, handling database migration issues"""
        try:
            uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.match_datetime.desc(), Upload.id.desc()).all()
            fencers = Fencer.query.filter_by(user_id=user_id).all()
            return uploads, fencers, None
        except Exception as e:
            # Handle migration issues
            if 'is_multi_video' in str(e) or 'match_title' in str(e) or 'match_datetime' in str(e):
                logging.warning(f"Database schema issue detected: {e}")
                try:
                    from migrations.add_multi_video_support import run_migration
                    migration_performed = run_migration()
                    # Also ensure match_datetime column exists
                    try:
                        from migrations.add_match_datetime import run_migration as run_match_datetime_migration
                        migration_performed = run_match_datetime_migration() or migration_performed
                    except Exception as match_migration_error:
                        logging.error(f"Secondary migration failed: {match_migration_error}")
                    if migration_performed:
                        uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.match_datetime.desc(), Upload.id.desc()).all()
                        fencers = Fencer.query.filter_by(user_id=user_id).all()
                        return uploads, fencers, None
                except Exception as migration_error:
                    logging.error(f"Migration failed: {migration_error}")

            logging.error(f"Error loading uploads: {e}")
            try:
                fencers = Fencer.query.filter_by(user_id=user_id).all()
                return [], fencers, 'Database loading error, please refresh page and retry'
            except:
                return [], [], 'Database connection error'

    return app

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8384, debug=False)
