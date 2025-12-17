import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery_config import celery
import os
import json
import logging
import traceback
import google.generativeai as genai
from prompts import generate_holistic_gpt_prompt
from your_scripts.gemini_rest import generate_text as gemini_generate_text
from datetime import datetime
from typing import Any, Dict, List, Optional
from your_scripts.video_analysis import main as video_analysis_main, process_first_frame
from your_scripts.match_separation import main as match_separation_main
from your_scripts.bout_analysis import main as bout_analysis_main
from your_scripts.fencer_analysis import main as fencer_analysis_main
from your_scripts.tagging import extract_tags_from_bout_analysis, initialize_tags_in_database
from your_scripts.professional_report_generator import generate_professional_report
import shutil
import time
from sqlalchemy.exc import OperationalError


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/Project/celery.log'),
        logging.StreamHandler()
    ]
)
# Helper to commit with retry/backoff for transient SQLite locks
def commit_with_retry(session, retries=5, base_delay=0.5, cap=5.0):
    attempt = 0
    while True:
        try:
            session.commit()
            return True
        except OperationalError as exc:
            message = str(exc).lower()
            if "database is locked" in message or "database is busy" in message:
                if attempt >= retries:
                    raise
                session.rollback()
                delay = min(cap, base_delay * (2 ** attempt))
                delay = delay * (1.0 + 0.25)
                logging.warning("Commit locked; retry %s/%s after %.2fs", attempt + 1, retries, delay)
                time.sleep(delay)
                attempt += 1
                continue
            # Other operational errors are not retried here
            raise
        except Exception:
            # Non-OperationalError exceptions: bubble up after rollback
            session.rollback()
            raise

# Note: avoid configuring gRPC SDK here; use REST helper to prevent SRV DNS lookups

# GPT-based evaluation functions
def generate_gpt_attack_evaluation(kpis: Dict[str, Any], tags: Dict[str, float], side: str, weapon_type: str = "saber") -> List[str]:
    """Generate attack evaluation using Gemini based on performance data."""
    # Define variables outside try block to avoid UnboundLocalError
    attack_success = kpis.get('attack_success_rate', 0.0)
    launch_success = kpis.get('launch_success_rate', 0.0)
    velocity = kpis.get('avg_velocity', 0.0)
    acceleration = kpis.get('avg_acceleration', 0.0)
    good_distance = tags.get('good_attack_distance', 0.0)
    first_step = kpis.get('first_step_init', 0.0)
    advance_ratio = kpis.get('advance_ratio', 0.0)
    side_en = 'Left' if side == 'Left' else 'Right'

    try:

        prompt = f"""
        You are a professional {weapon_type} fencing analyst tasked with generating an attack ability assessment for the {side_en} fencer. Based on the following data, generate 3-5 specific, actionable assessment points in English:

        **Attack Performance Data:**
        - Attack Success Rate: {attack_success:.1%}
        - Launch Success Rate: {launch_success:.1%}
        - Average Velocity: {velocity:.2f} m/s
        - Average Acceleration: {acceleration:.2f} m/s²
        - Good Attack Distance Rate: {good_distance:.1%}
        - First Step Reaction Time: {first_step:.2f} seconds
        - Advance Ratio: {advance_ratio:.1%}

        **Requirements:**
        1. The first point must be an overall attack ability assessment (use **bold** title)
        2. Subsequent points should analyze specific attack technical details
        3. Each point should be specific, professional, and actionable
        4. Use professional fencing terminology
        5. Provide specific training recommendations
        6. Output format should be a list, one point per line
        7. All content must be in English

        Please output the assessment points directly without additional explanation:
        """

        system_prompt = "You are a professional fencing tactical analyst who specializes in generating specific, actionable training recommendations based on data."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Apply REST-based Gemini call (handles retries/backoff internally)
        analysis_text = gemini_generate_text(
            full_prompt,
            model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite'),
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=1024,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        
        # Split by lines and clean up
        evaluation_points = [line.strip() for line in analysis_text.split('\n') if line.strip() and not line.strip().startswith('-')]
        
        # Ensure we have 3-5 points
        if len(evaluation_points) > 5:
            evaluation_points = evaluation_points[:5]
        elif len(evaluation_points) < 3:
            # Fallback to ensure minimum content
            evaluation_points.extend([
                f"**Attack Ability** - {side_en} fencer's attack success rate is {attack_success:.1%}",
                f"Velocity Performance: Average velocity {velocity:.2f}m/s, reaction time {first_step:.2f} seconds",
                "Recommend focusing on improving attack consistency and timing selection"
            ])
            evaluation_points = evaluation_points[:5]
            
        return evaluation_points
        
    except Exception as e:
        logging.error(f"Error generating GPT attack evaluation: {e}")
        # Fallback to basic evaluation
        return [
            f"**Attack Assessment** - {side_en} fencer's attack success rate {attack_success:.1%}",
            f"Technical Metrics: Velocity {velocity:.2f}m/s, acceleration {acceleration:.2f}m/s²",
            f"Distance Control: Good attack distance {good_distance:.1%}",
            "Recommend adjusting training priorities based on specific performance"
        ]

def generate_gpt_defense_evaluation(kpis: Dict[str, Any], tags: Dict[str, float], side: str, weapon_type: str = "saber") -> List[str]:
    """Generate defense evaluation using Gemini based on performance data."""
    # Define variables outside try block to avoid UnboundLocalError
    def_quality = tags.get('good_defensive_quality', 0.0)
    safe_distance = tags.get('maintain_safe_distance', 0.0)
    consistent_spacing = tags.get('consistent_spacing', 0.0)
    pause_ratio = kpis.get('pause_ratio', 0.0)
    retreat_quality = tags.get('good_retreat_quality', 0.0)
    counter_opportunities = tags.get('counter_opportunities_used', 0.0)
    side_en = 'Left' if side == 'Left' else 'Right'

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        prompt = f"""
        You are a professional {weapon_type} fencing analyst tasked with generating a defense ability assessment for the {side_en} fencer. Based on the following data, generate 3-5 specific, actionable assessment points in English:

        **Defense Performance Data:**
        - Defensive Quality: {def_quality:.1%}
        - Maintain Safe Distance: {safe_distance:.1%}
        - Consistent Spacing: {consistent_spacing:.1%}
        - Pause Ratio: {pause_ratio:.1%}
        - Retreat Quality: {retreat_quality:.1%}
        - Counter Opportunities Used: {counter_opportunities:.1%}

        **Requirements:**
        1. The first point must be an overall defense ability assessment (use **bold** title)
        2. Subsequent points should analyze specific defensive technical details
        3. Each point should be specific, professional, and actionable
        4. Use professional fencing terminology
        5. Provide specific training recommendations
        6. Output format should be a list, one point per line
        7. All content must be in English

        Please output the assessment points directly without additional explanation:
        """

        system_prompt = "You are a professional fencing tactical analyst who specializes in generating specific, actionable training recommendations based on data."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Apply REST-based Gemini call (handles retries/backoff internally)
        analysis_text = gemini_generate_text(
            full_prompt,
            model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite'),
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=1024,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        
        # Split by lines and clean up
        evaluation_points = [line.strip() for line in analysis_text.split('\n') if line.strip() and not line.strip().startswith('-')]
        
        # Ensure we have 3-5 points
        if len(evaluation_points) > 5:
            evaluation_points = evaluation_points[:5]
        elif len(evaluation_points) < 3:
            # Fallback to ensure minimum content
            evaluation_points.extend([
                f"**Defense Ability** - {side_en} fencer's defensive quality is {def_quality:.1%}",
                f"Distance Management: Safe distance maintenance {safe_distance:.1%}, spacing consistency {consistent_spacing:.1%}",
                "Recommend focusing on improving defensive reaction speed and control ability"
            ])
            evaluation_points = evaluation_points[:5]
            
        return evaluation_points
        
    except Exception as e:
        logging.error(f"Error generating GPT defense evaluation: {e}")
        # Fallback to basic evaluation
        return [
            f"**Defense Assessment** - {side_en} fencer's defensive quality {def_quality:.1%}",
            f"Distance Control: Safe distance {safe_distance:.1%}, spacing consistency {consistent_spacing:.1%}",
            f"Activity Ratio: Pause ratio {pause_ratio:.1%}",
            "Recommend adjusting defensive training priorities based on specific performance"
        ]

def generate_gpt_recommendations(left_kpis: Dict[str, Any], right_kpis: Dict[str, Any], 
                                left_tags: Dict[str, float], right_tags: Dict[str, float], 
                                weapon_type: str = "saber") -> Dict[str, List[str]]:
    """Generate personalized recommendations using Gemini based on both fencers' performance."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Prepare comprehensive data summary
        data_summary = f"""
        **Left Fencer Performance:**
        - Attack Success Rate: {left_kpis.get('attack_success_rate', 0.0):.1%}
        - Launch Success Rate: {left_kpis.get('launch_success_rate', 0.0):.1%}
        - Average Velocity: {left_kpis.get('avg_velocity', 0.0):.2f}m/s
        - Defensive Quality: {left_tags.get('good_defensive_quality', 0.0):.1%}
        - Safe Distance: {left_tags.get('maintain_safe_distance', 0.0):.1%}
        - Pause Ratio: {left_kpis.get('pause_ratio', 0.0):.1%}

        **Right Fencer Performance:**
        - Attack Success Rate: {right_kpis.get('attack_success_rate', 0.0):.1%}
        - Launch Success Rate: {right_kpis.get('launch_success_rate', 0.0):.1%}
        - Average Velocity: {right_kpis.get('avg_velocity', 0.0):.2f}m/s
        - Defensive Quality: {right_tags.get('good_defensive_quality', 0.0):.1%}
        - Safe Distance: {right_tags.get('maintain_safe_distance', 0.0):.1%}
        - Pause Ratio: {right_kpis.get('pause_ratio', 0.0):.1%}
        """
        
        prompt = f"""
        You are a professional {weapon_type} fencing coach tasked with generating personalized training recommendations for two fencers.

        {data_summary}

        **Requirements:**
        Please generate 4-6 specific training recommendations for the left fencer and right fencer separately. The recommendations should:
        1. Target each fencer's specific weaknesses
        2. Leverage their existing strengths
        3. Provide specific executable training methods
        4. Use professional fencing terminology
        5. All content must be in English

        **Output Format:**
        Left Fencer Recommendations:
        1. [Specific recommendation]
        2. [Specific recommendation]
        ...

        Right Fencer Recommendations:
        1. [Specific recommendation]
        2. [Specific recommendation]
        ...

        Please output the recommendations directly without additional explanation:
        """
        
        system_instruction = "You are an experienced fencing coach and tactical analyst who specializes in creating personalized training plans for fencers based on data analysis."
        combined_prompt = f"{system_instruction}\n\n{prompt}"
        analysis_text = gemini_generate_text(
            combined_prompt,
            model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite'),
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=1024,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        
        # Parse the response to extract left and right recommendations
        lines = analysis_text.split('\n')
        left_recs = []
        right_recs = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if 'Left Fencer' in line:
                current_section = 'left'
                continue
            elif 'Right Fencer' in line:
                current_section = 'right'
                continue
            elif line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')) or line.startswith('-')):
                # Clean the line
                clean_line = line.lstrip('123456.- ').strip()
                if current_section == 'left' and len(left_recs) < 6:
                    left_recs.append(clean_line)
                elif current_section == 'right' and len(right_recs) < 6:
                    right_recs.append(clean_line)
        
        # Ensure we have minimum recommendations
        if len(left_recs) < 3:
            left_recs.extend([
                "Develop targeted training plan based on attack data",
                "Improve defensive reaction speed and accuracy",
                "Optimize distance control and timing selection"
            ])
        if len(right_recs) < 3:
            right_recs.extend([
                "Adjust training priorities based on defensive performance",
                "Strengthen attack launch consistency",
                "Improve match rhythm control ability"
            ])
            
        return {
            'left': left_recs[:6],
            'right': right_recs[:6]
        }
        
    except Exception as e:
        logging.error(f"Error generating GPT recommendations: {e}")
        # Fallback recommendations
        return {
            'left': [
                "Attack: Adjust attack strategy based on data analysis",
                "Defense: Improve defensive reaction and control ability",
                "Rhythm: Optimize match rhythm and timing selection",
                "Distance: Improve distance management and positioning"
            ],
            'right': [
                "Attack: Develop attack plan based on performance data",
                "Defense: Strengthen defensive techniques and counter-attack ability",
                "Rhythm: Improve rhythm variation and control ability",
                "Distance: Optimize distance judgment and movement"
            ]
        }


@celery.task(bind=True, max_retries=0, time_limit=600)
def generate_initial_detection_task(self, upload_id):
    """Generate first-frame detection asynchronously for a single-video upload."""
    from app import create_app, db
    from models import Upload

    app = create_app()
    with app.app_context():
        upload = Upload.query.get(upload_id)
        if not upload:
            logging.error(f"Upload {upload_id} not found for detection task")
            return {'status': 'missing'}

        if upload.is_multi_video:
            logging.info(
                "Upload %s is marked as multi-video; skipping shared detection task",
                upload_id
            )
            return {'status': 'skipped'}

        video_path = upload.video_path
        if not video_path or not os.path.exists(video_path):
            logging.error(f"Video path missing for upload {upload_id}: {video_path}")
            upload.status = 'error'
            upload.detection_image_path = None
            db.session.commit()
            return {'status': 'missing_video'}

        detection_dir = os.path.dirname(video_path) or app.config['UPLOAD_FOLDER']
        os.makedirs(detection_dir, exist_ok=True)

        try:
            detection_image_path, _ = process_first_frame(video_path, detection_dir)
            upload.detection_image_path = detection_image_path
            upload.status = 'awaiting_selection'
            db.session.commit()
            logging.info(
                "Initial detection completed for upload %s -> %s",
                upload_id,
                detection_image_path
            )
            return {
                'status': 'success',
                'detection_image_path': detection_image_path
            }
        except Exception as e:
            logging.error(
                "Failed to generate detection for upload %s: %s",
                upload_id,
                e,
                exc_info=True
            )
            db.session.rollback()
            upload = Upload.query.get(upload_id)
            if upload:
                try:
                    upload.status = 'error'
                    upload.detection_image_path = None
                    commit_with_retry(db.session, retries=6, base_delay=0.5, cap=6.0)
                except Exception as commit_error:
                    logging.error(
                        "Failed to record detection error for upload %s: %s",
                        upload_id,
                        commit_error,
                        exc_info=True
                    )
                    db.session.rollback()
            raise


@celery.task(bind=True, max_retries=0, time_limit=3600)
def analyze_video_task(self, upload_id):
    from app import create_app, db
    from models import Fencer, Upload, HolisticAnalysis, Bout
    app = create_app()
    with app.app_context():
        logging.debug(f"Starting analyze_video_task for upload_id: {upload_id}")
        upload = Upload.query.get(upload_id)
        if not upload:
            logging.error(f"Upload {upload_id} not found")
            return
        try:
            logging.info(f"Processing upload {upload_id}: {upload.video_path}")
            result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))
            csv_output_dir = os.path.join(result_dir, 'csv')
            os.makedirs(csv_output_dir, exist_ok=True)
            reid_model_path = os.path.join(os.getcwd(), "your_scripts", "checkpoints", "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
            selected_indexes = list(map(int, upload.selected_indexes.split()))

            # Video analysis
            logging.info("Calling video_analysis_main")
            left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle, final_output_path = video_analysis_main(
                upload.video_path, None, reid_model_path, csv_output_dir, selected_indexes
            )
            logging.info("video_analysis_main completed successfully")
            upload.csv_dir = csv_output_dir

            # Match separation
            matches_output_dir = os.path.join(result_dir, 'matches')
            match_data_dir = os.path.join(result_dir, 'match_data')
            keypoints_output_dir = os.path.join(result_dir, 'matches_with_keypoints')
            logging.info("Calling match_separation_main")
            matches, video_angle = match_separation_main(
                upload.video_path, csv_output_dir, matches_output_dir, match_data_dir, keypoints_output_dir
            )
            logging.info(f"match_separation_main completed, found {len(matches)} matches")
            upload.total_bouts = len(matches)

            # Create Bout entries
            for idx, (start_frame, end_frame) in enumerate(matches, 1):
                bout_dir = os.path.join(matches_output_dir, f'match_{idx}')
                video_path = os.path.join(bout_dir, f'match_{idx}.mp4')
                extended_video_path = os.path.join(bout_dir, f'match_{idx}_extended.mp4')
                
                # Create bout with backwards compatibility
                bout_kwargs = {
                    'upload_id': upload_id,
                    'match_idx': idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'video_path': video_path if os.path.exists(video_path) else None
                }
                
                # Only add extended_video_path if the column exists in the database
                if hasattr(Bout, 'extended_video_path'):
                    try:
                        bout_kwargs['extended_video_path'] = extended_video_path if os.path.exists(extended_video_path) else None
                    except Exception:
                        # Column might not exist in database yet
                        pass
                
                bout = Bout(**bout_kwargs)
                db.session.add(bout)
            
            upload.status = 'awaiting_user_input'
            try:
                commit_with_retry(db.session, retries=6, base_delay=0.5, cap=6.0)
                logging.info(f"Video processing completed for upload {upload_id}, awaiting user input")
            except Exception as commit_error:
                db.session.rollback()
                logging.error(f"Error committing bout data for upload {upload_id}: {str(commit_error)}")
                # Try to save without extended_video_path if column doesn't exist
                if "extended_video_path" in str(commit_error):
                    logging.info("Retrying without extended_video_path column...")
                    # Remove all bouts and re-add without extended_video_path
                    bouts_to_remove = Bout.query.filter_by(upload_id=upload_id).all()
                    for bout in bouts_to_remove:
                        db.session.delete(bout)
                    
                    for idx, (start_frame, end_frame) in enumerate(matches, 1):
                        bout_dir = os.path.join(matches_output_dir, f'match_{idx}')
                        video_path = os.path.join(bout_dir, f'match_{idx}.mp4')
                        bout = Bout(
                            upload_id=upload_id,
                            match_idx=idx,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            video_path=video_path if os.path.exists(video_path) else None
                        )
                        db.session.add(bout)
                    
                    upload.status = 'awaiting_user_input'
                    commit_with_retry(db.session, retries=6, base_delay=0.5, cap=6.0)
                    logging.info(f"Video processing completed for upload {upload_id} (without extended videos), awaiting user input")
                else:
                    raise commit_error
        except Exception as e:
            db.session.rollback()
            upload.status = 'error'
            try:
                commit_with_retry(db.session, retries=6, base_delay=0.5, cap=6.0)
            except Exception:
                # If even retry fails, fallback to single commit attempt after rollback
                try:
                    db.session.commit()
                except Exception:
                    pass
            logging.error(f"Error processing upload {upload_id}: {str(e)}\n{traceback.format_exc()}")
            return

@celery.task(bind=True, max_retries=0, time_limit=14400)
def analyze_multi_video_task(self, upload_id):
    """Process a multi-video upload by analyzing each UploadVideo sequentially and merging bouts."""
    from app import create_app, db
    from models import Upload, UploadVideo, Bout
    import shutil
    app = create_app()
    with app.app_context():
        logging.debug(f"Starting analyze_multi_video_task for upload_id: {upload_id}")
        upload = Upload.query.get(upload_id)
        if not upload:
            logging.error(f"Upload {upload_id} not found")
            return
        if not getattr(upload, 'is_multi_video', False):
            logging.warning(f"Upload {upload_id} is not multi-video; delegating to analyze_video_task")
            return analyze_video_task.apply_async((upload_id,))

        try:
            result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))
            os.makedirs(result_dir, exist_ok=True)

            # Combined output dirs
            combined_matches_dir = os.path.join(result_dir, 'matches')
            combined_match_data_dir = os.path.join(result_dir, 'match_data')
            combined_keypoints_dir = os.path.join(result_dir, 'matches_with_keypoints')
            os.makedirs(combined_matches_dir, exist_ok=True)
            os.makedirs(combined_match_data_dir, exist_ok=True)
            os.makedirs(combined_keypoints_dir, exist_ok=True)

            reid_model_path = os.path.join(os.getcwd(), "your_scripts", "checkpoints", "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")

            # Iterate videos in sequence order
            videos = UploadVideo.query.filter_by(upload_id=upload_id).order_by(UploadVideo.sequence_order).all()
            if not videos:
                logging.error(f"No UploadVideo records found for upload {upload_id}")
                upload.status = 'error'
                db.session.commit()
                return

            global_match_idx = 0
            last_csv_dir = None

            for video in videos:
                logging.info(f"Analyzing UploadVideo {video.id} (seq {video.sequence_order}) path={video.video_path}")
                if not video.video_path or not os.path.exists(video.video_path):
                    logging.error(f"Video file missing for UploadVideo {video.id}: {video.video_path}")
                    video.status = 'error'
                    db.session.commit()
                    continue

                # Parse selected indexes: prefer per-video, then upload-level
                raw_indexes = (video.selected_indexes or upload.selected_indexes or '').replace(',', ' ').strip()
                try:
                    selected_indexes = list(map(int, raw_indexes.split())) if raw_indexes else []
                except Exception:
                    selected_indexes = []

                # Per-video output dirs
                csv_output_dir = os.path.join(result_dir, f'csv_video_{video.sequence_order}')
                matches_output_dir_local = os.path.join(result_dir, f'matches_video_{video.sequence_order}')
                match_data_dir_local = os.path.join(result_dir, f'match_data_video_{video.sequence_order}')
                keypoints_output_dir_local = os.path.join(result_dir, f'matches_with_keypoints_video_{video.sequence_order}')
                for d in [csv_output_dir, matches_output_dir_local, match_data_dir_local, keypoints_output_dir_local]:
                    os.makedirs(d, exist_ok=True)

                # Run analysis for this video
                try:
                    _lx, _ly, _rx, _ry, _c, checker_list, video_angle, final_output_path = video_analysis_main(
                        video.video_path, None, reid_model_path, csv_output_dir, selected_indexes
                    )
                    last_csv_dir = csv_output_dir
                except Exception as e:
                    logging.error(f"video_analysis_main failed for UploadVideo {video.id}: {e}")
                    video.status = 'error'
                    db.session.commit()
                    continue

                try:
                    matches, video_angle = match_separation_main(
                        video.video_path, csv_output_dir, matches_output_dir_local, match_data_dir_local, keypoints_output_dir_local
                    )
                    video.total_bouts = len(matches)
                    video.status = 'processing'
                    db.session.commit()
                except Exception as e:
                    logging.error(f"match_separation_main failed for UploadVideo {video.id}: {e}")
                    video.status = 'error'
                    db.session.commit()
                    continue

                # Move/merge matches into combined dir and create Bout entries
                for idx, (start_frame, end_frame) in enumerate(matches, 1):
                    global_match_idx += 1
                    src_dir = os.path.join(matches_output_dir_local, f'match_{idx}')
                    dst_dir = os.path.join(combined_matches_dir, f'match_{global_match_idx}')
                    try:
                        if os.path.exists(dst_dir):
                            shutil.rmtree(dst_dir, ignore_errors=True)
                        if os.path.exists(src_dir):
                            shutil.move(src_dir, dst_dir)
                        else:
                            os.makedirs(dst_dir, exist_ok=True)
                    except Exception as move_e:
                        logging.warning(f"Failed moving match folder {src_dir} -> {dst_dir}: {move_e}")

                    video_path = os.path.join(dst_dir, f'match_{global_match_idx}.mp4')
                    # match_separation may have saved as match_{idx}.mp4; rename if needed
                    legacy_video_path = os.path.join(dst_dir, f'match_{idx}.mp4')
                    if os.path.exists(legacy_video_path) and not os.path.exists(video_path):
                        try:
                            os.rename(legacy_video_path, video_path)
                        except Exception:
                            video_path = legacy_video_path

                    extended_video_path = os.path.join(dst_dir, f'match_{global_match_idx}_extended.mp4')
                    legacy_ext_path = os.path.join(dst_dir, f'match_{idx}_extended.mp4')
                    if os.path.exists(legacy_ext_path) and not os.path.exists(extended_video_path):
                        try:
                            os.rename(legacy_ext_path, extended_video_path)
                        except Exception:
                            extended_video_path = legacy_ext_path if os.path.exists(legacy_ext_path) else None
                    elif not os.path.exists(extended_video_path):
                        extended_video_path = extended_video_path if os.path.exists(extended_video_path) else None

                    # Copy match_data for this bout into combined match_data directory
                    try:
                        src_match_data = os.path.join(match_data_dir_local, f'match_{idx}')
                        dst_match_data = os.path.join(combined_match_data_dir, f'match_{global_match_idx}')
                        if os.path.exists(dst_match_data):
                            shutil.rmtree(dst_match_data, ignore_errors=True)
                        if os.path.exists(src_match_data):
                            shutil.copytree(src_match_data, dst_match_data)
                        else:
                            logging.warning(f"Missing match_data source for video {video.id} match {idx}: {src_match_data}")
                    except Exception as copy_md_e:
                        logging.warning(f"Failed copying match_data {src_match_data} -> {dst_match_data}: {copy_md_e}")

                    # Copy matches_with_keypoints as well (optional but useful for UI)
                    try:
                        src_kp = os.path.join(keypoints_output_dir_local, f'match_{idx}')
                        dst_kp = os.path.join(combined_keypoints_dir, f'match_{global_match_idx}')
                        if os.path.exists(dst_kp):
                            shutil.rmtree(dst_kp, ignore_errors=True)
                        if os.path.exists(src_kp):
                            shutil.copytree(src_kp, dst_kp)
                    except Exception as copy_kp_e:
                        logging.warning(f"Failed copying keypoints {src_kp} -> {dst_kp}: {copy_kp_e}")

                    bout_kwargs = {
                        'upload_id': upload_id,
                        'upload_video_id': video.id,
                        'match_idx': global_match_idx,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'video_path': video_path if os.path.exists(video_path) else None
                    }
                    try:
                        if hasattr(Bout, 'extended_video_path'):
                            bout_kwargs['extended_video_path'] = extended_video_path
                    except Exception:
                        pass

                    bout = Bout(**bout_kwargs)
                    db.session.add(bout)

                # Mark video completed after creating bouts
                video.bouts_offset = global_match_idx - (video.total_bouts or 0)
                video.status = 'completed'
                db.session.commit()

            # Finalize upload
            upload.total_bouts = global_match_idx
            if last_csv_dir:
                upload.csv_dir = last_csv_dir  # use the last video's csv (provides meta.csv for downstream)
            upload.status = 'awaiting_user_input'
            db.session.commit()
            logging.info(f"Multi-video processing completed for upload {upload_id}: total bouts {global_match_idx}")

        except Exception as e:
            db.session.rollback()
            upload.status = 'error'
            db.session.commit()
            logging.error(f"Error processing multi-video upload {upload_id}: {str(e)}\n{traceback.format_exc()}")
            return

@celery.task(bind=True, max_retries=0, time_limit=3600)
def generate_analysis_task(self, upload_id):
    from app import create_app, db
    from models import Fencer, Upload, HolisticAnalysis, Bout, Tag, BoutTag, VideoAnalysis
    import pandas as pd
    app = create_app()
    with app.app_context():
        logging.debug(f"Starting generate_analysis_task for upload_id: {upload_id}")
        upload = Upload.query.get(upload_id)
        if not upload:
            logging.error(f"Upload {upload_id} not found")
            return
        try:
            result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))
            csv_output_dir = upload.csv_dir
            matches_output_dir = os.path.join(result_dir, 'matches')
            match_data_dir = os.path.join(result_dir, 'match_data')
            video_angle = pd.read_csv(os.path.join(csv_output_dir, 'meta.csv'))['video_angle'][0]

            # Bout analysis
            bouts = Bout.query.filter_by(upload_id=upload_id).order_by(Bout.match_idx).all()
            for bout in bouts:
                logging.info(f"Analyzing bout {bout.match_idx}: frames {bout.start_frame} to {bout.end_frame}")
                from your_scripts.bout_analysis import main as bout_analysis_main
                bout_analysis_main(bout.match_idx, bout.start_frame, bout.end_frame, match_data_dir, f"upload_{upload_id}", video_angle=video_angle, bout_result=bout.result)
                upload.bouts_analyzed += 1
                db.session.commit()
                logging.info(f"Bout {bout.match_idx} analyzed")

            # Fencer analysis
            if upload.bouts_analyzed == upload.total_bouts:
                logging.info("Calling fencer_analysis_main")
                fencer_analysis_main(
                    os.path.join(result_dir, 'match_analysis'),
                    match_data_dir,
                    os.path.join(result_dir, 'fencer_analysis')
                )
                upload.cross_bout_analysis_path = os.path.join(result_dir, 'fencer_analysis', 'cross_bout_analysis.json')
                
                # Generate tags for all bouts after analysis is complete
                logging.info("Starting tag generation for all bouts")
                _generate_tags_for_upload(upload_id, result_dir, db)
                
                # Generate AI analysis for video view
                logging.info("Starting AI analysis generation")
                _generate_ai_analysis_for_upload(upload_id, result_dir, db)
                
                upload.status = 'completed'
                db.session.commit()
                logging.info("fencer_analysis_main and tag generation completed")
                
                # Update fencer profiles after analysis completion using direct generator
                try:
                    logging.info("Updating fencer profiles after analysis completion")
                    sys.path.insert(0, '/workspace/Project/your_scripts')
                    from direct_profile_generator import generate_fencer_profile_directly
                    from models import Fencer
                    
                    # Refresh profiles for both fencers in this upload
                    fencers_to_update = []
                    if upload.left_fencer_id:
                        fencers_to_update.append(upload.left_fencer_id)
                    if upload.right_fencer_id:
                        fencers_to_update.append(upload.right_fencer_id)
                    
                    for fencer_id in fencers_to_update:
                        try:
                            # Get fencer name
                            fencer = Fencer.query.get(fencer_id)
                            if fencer:
                                result = generate_fencer_profile_directly(fencer_id, upload.user_id, fencer.name)
                                if result.get('success'):
                                    logging.info(f"Updated profile for {fencer.name} (ID: {fencer_id}): {result.get('total_bouts', 0)} bouts from {result.get('total_uploads', 0)} uploads")
                                else:
                                    logging.warning(f"Failed to update profile for {fencer.name}: {result.get('error', 'Unknown error')}")

                                try:
                                    report_result = generate_professional_report(fencer_id, upload.user_id, fencer.name)
                                    if report_result.get('success'):
                                        logging.info(
                                            "Generated professional report for %s (ID: %s) -> %s",
                                            fencer.name,
                                            fencer_id,
                                            report_result.get('report_path'),
                                        )
                                    else:
                                        logging.warning(
                                            "Professional report generation for %s failed: %s",
                                            fencer.name,
                                            report_result.get('error', 'Unknown error'),
                                        )
                                except Exception as report_exc:
                                    logging.error(
                                        "Error generating professional report for fencer %s: %s",
                                        fencer_id,
                                        report_exc,
                                    )
                            else:
                                logging.warning(f"Fencer {fencer_id} not found in database")
                        except Exception as e:
                            logging.error(f"Error updating profile for fencer {fencer_id}: {str(e)}")
                    
                    logging.info("Fencer profile updates completed")
                except Exception as e:
                    logging.error(f"Error during fencer profile updates: {str(e)}")
                    # Don't fail the whole process if profile updates fail

                # Generate concise report JSON for report view
                try:
                    _generate_report_for_upload(upload_id, app, db)
                    logging.info(f"Generated report.json for upload {upload_id}")
                except Exception as e:
                    logging.error(f"Error generating report.json for upload {upload_id}: {str(e)}\n{traceback.format_exc()}")
            else:
                upload.status = 'error'
                db.session.commit()
                logging.error(f"Not all bouts analyzed for upload {upload_id}")
        except Exception as e:
            upload.status = 'error'
            db.session.commit()
            logging.error(f"Error generating analysis for upload {upload_id}: {str(e)}\n{traceback.format_exc()}")
            return


@celery.task(bind=True)
def generate_holistic_report_task(self, fencer_id, user_id):
    from app import create_app, db
    from models import Fencer, Upload, HolisticAnalysis
    app = create_app()
    with app.app_context():
        try:
            fencer = Fencer.query.get(fencer_id)
            if not fencer:
                logging.error(f"Fencer with ID {fencer_id} not found.")
                return {'status': 'Failure', 'message': 'Fencer not found'}

            uploads = Upload.query.filter(
                (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
            ).all()

            if not uploads:
                logging.warning(f"No video data for fencer {fencer_id}.")
                return {'status': 'Failure', 'message': 'No video data available'}

            holistic_data = {}
            bout_data = []
            for upload in uploads:
                analysis_path = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), f'video_{upload.id}', 'analysis_results.json')
                if os.path.exists(analysis_path):
                    with open(analysis_path, 'r') as f:
                        analysis_results = json.load(f)
                        holistic_data[upload.id] = analysis_results.get('gpt_analysis_text', '')
                        bout_data.extend(analysis_results.get('bout_data', []))

            if not bout_data:
                logging.warning(f"No bout data for fencer {fencer_id}.")
                return {'status': 'Failure', 'message': 'No bout data available'}

            fencer_data = {
                'left': {'total_bouts': 0, 'avg_attacking_ratio': 0},
                'right': {'total_bouts': 0, 'avg_attacking_ratio': 0}
            }
            # Simplified data aggregation logic
            for bout in bout_data:
                if bout.get('left_fencer_id') == fencer_id:
                    fencer_data['left']['total_bouts'] += 1
                if bout.get('right_fencer_id') == fencer_id:
                    fencer_data['right']['total_bouts'] += 1

            # Determine weapon type - get most common weapon type from uploads
            weapon_types = [upload.weapon_type for upload in uploads if upload.weapon_type]
            if weapon_types:
                # Use most common weapon type
                weapon_type = max(set(weapon_types), key=weapon_types.count)
            else:
                weapon_type = 'saber'  # default
                
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            prompt = generate_holistic_gpt_prompt(bout_data, fencer_data, fencer.name, fencer_id, uploads, weapon_type=weapon_type)
            system_prompt = "你是一位击剑分析助手，请用中文进行详细分析。"
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Apply REST-based Gemini call (handles retries/backoff internally)
            analysis_text = gemini_generate_text(
                full_prompt,
                model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite'),
                temperature=0.2,
                top_k=1,
                top_p=0.8,
                max_output_tokens=1024,
                timeout_seconds=45,
                max_attempts=6,
                response_mime_type='text/plain',
            )
            
            holistic_report_content = analysis_text if analysis_text else (str(analysis_text) if analysis_text is not None else '')
            
            report_path = os.path.join(app.config['RESULT_FOLDER'], str(user_id), 'fencer', str(fencer_id), 'holistic_analysis.json')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({'analysis': holistic_report_content}, f, ensure_ascii=False, indent=4)
            
            # Save to database
            holistic_analysis = HolisticAnalysis.query.filter_by(fencer_id=fencer_id).first()
            if not holistic_analysis:
                holistic_analysis = HolisticAnalysis(fencer_id=fencer_id, user_id=user_id)
                db.session.add(holistic_analysis)

            holistic_analysis.report_path = report_path
            holistic_analysis.status = 'Completed'
            db.session.commit()

            logging.info(f"Holistic report for fencer {fencer_id} generated successfully.")
            return {'status': 'Success', 'report_path': report_path}

        except Exception as e:
            logging.error(f"Error in holistic report task for fencer {fencer_id}: {e}\n{traceback.format_exc()}")
            holistic_analysis = HolisticAnalysis.query.filter_by(fencer_id=fencer_id).first()
            if holistic_analysis:
                holistic_analysis.status = 'Failed'
                db.session.commit()
            return {'status': 'Failure', 'message': str(e)}


@celery.task(bind=True)
def regenerate_fencer_profile_task(self, fencer_id: int, user_id: int) -> Dict[str, Any]:
    """Regenerate fencer profile graphs and Gemini-backed report via Celery."""
    from app import create_app
    from models import Fencer
    from your_scripts.professional_report_generator import regenerate_fencer_profile_assets

    app = create_app()
    with app.app_context():
        fencer = Fencer.query.get(fencer_id)
        if not fencer or fencer.user_id != user_id:
            logging.error(
                "[Celery] Regeneration aborted: fencer %s not found for user %s",
                fencer_id,
                user_id,
            )
            return {
                'success': False,
                'error': 'Fencer not found for the current user.',
            }

        logging.info(
            "[Celery] Regenerating profile for fencer %s (%s)",
            fencer_id,
            fencer.name,
        )

        # Proceed with profile graphs and Gemini report only (no per-upload regen)
        outcome = regenerate_fencer_profile_assets(fencer_id, user_id, fencer.name)

        if outcome.get('success'):
            profile_meta = outcome.get('profile', {})
            logging.info(
                "[Celery] Regeneration completed for fencer %s -> %s bouts across %s uploads",
                fencer_id,
                profile_meta.get('total_bouts'),
                profile_meta.get('total_uploads'),
            )
        else:
            logging.warning(
                "[Celery] Regeneration for fencer %s finished with warnings: %s",
                fencer_id,
                '; '.join(outcome.get('errors') or []),
            )

        return outcome


@celery.task(bind=True, max_retries=0, time_limit=3600)
def regenerate_video_analysis_task(self, upload_id):
    """Rebuild the AI-driven video view analysis asynchronously."""
    from app import create_app, db
    from models import Upload, VideoAnalysis

    app = create_app()
    with app.app_context():
        upload = Upload.query.get(upload_id)
        if not upload:
            logging.error(f"Upload {upload_id} not found for regeneration")
            return {'status': 'error', 'message': 'upload_not_found'}

        result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))
        os.makedirs(result_dir, exist_ok=True)

        analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
        if not analysis:
            analysis = VideoAnalysis(upload_id=upload_id)
            db.session.add(analysis)

        analysis.status = 'pending'
        analysis.error_message = None
        analysis.generated_at = datetime.utcnow()
        db.session.commit()

        try:
            _generate_ai_analysis_for_upload(upload_id, result_dir, db, force=True)
            try:
                _generate_report_for_upload(upload_id, app, db)
                logging.info(f"Regenerated report.json for upload {upload_id}")
            except Exception as report_exc:
                logging.error(
                    f"Regeneration task for upload {upload_id} failed to update report.json: {report_exc}"
                )
        except Exception as exc:  # pragma: no cover - defensive, inner func already guards
            logging.error(f"Regeneration task for upload {upload_id} failed: {exc}\n{traceback.format_exc()}")
            failing = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
            if failing:
                failing.status = 'error'
                failing.error_message = str(exc)
                db.session.commit()
            return {'status': 'error', 'message': str(exc)}

        final = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
        status = final.status if final else 'completed'
        logging.info(f"Regeneration task for upload {upload_id} finished with status {status}")
        return {'status': status}


def _generate_tags_for_upload(upload_id, result_dir, db):
    """
    Generate and save tags for all bouts in an upload.
    
    Args:
        upload_id: ID of the upload to process
        result_dir: Directory containing analysis results
        db: Database session
    """
    from models import Bout, Tag, BoutTag
    
    try:
        # Initialize tags in database if not already done
        initialize_tags_in_database(db)
        
        # Get all bouts for this upload
        bouts = Bout.query.filter_by(upload_id=upload_id).order_by(Bout.match_idx).all()
        
        for bout in bouts:
            # Load the analysis JSON for this bout
            analysis_file_path = os.path.join(result_dir, 'match_analysis', f'match_{bout.match_idx}_analysis.json')
            
            if not os.path.exists(analysis_file_path):
                logging.warning(f"Analysis file not found for bout {bout.match_idx}: {analysis_file_path}")
                continue
                
            try:
                with open(analysis_file_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                # Extract tags for both fencers
                fencer_tags = extract_tags_from_bout_analysis(analysis_data)
                
                # Save tags to database
                for side in ['left', 'right']:
                    tags_for_side = fencer_tags.get(side, set())
                    
                    for tag_name in tags_for_side:
                        # Get or create the tag
                        tag = Tag.query.filter_by(name=tag_name).first()
                        if not tag:
                            logging.warning(f"Tag '{tag_name}' not found in database, skipping")
                            continue
                        
                        # Check if this bout-tag-side combination already exists
                        existing_bout_tag = BoutTag.query.filter_by(
                            bout_id=bout.id,
                            tag_id=tag.id,
                            fencer_side=side
                        ).first()
                        
                        if not existing_bout_tag:
                            # Create new bout tag
                            bout_tag = BoutTag(
                                bout_id=bout.id,
                                tag_id=tag.id,
                                fencer_side=side
                            )
                            db.session.add(bout_tag)
                            logging.debug(f"Added tag '{tag_name}' for {side} fencer in bout {bout.match_idx}")
                
                logging.info(f"Generated tags for bout {bout.match_idx}: Left={fencer_tags.get('left', set())}, Right={fencer_tags.get('right', set())}")
                
            except Exception as e:
                logging.error(f"Error processing analysis file for bout {bout.match_idx}: {str(e)}")
                continue
        
        # Commit all tag changes
        db.session.commit()
        logging.info(f"Successfully generated tags for all bouts in upload {upload_id}")
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error generating tags for upload {upload_id}: {str(e)}")
        raise


def _generate_ai_analysis_for_upload(upload_id, result_dir, db, force: bool = False):
    """
    Generate AI-powered analysis for video view and save to database.
    
    Args:
        upload_id: ID of the upload to process
        result_dir: Directory containing analysis results
        db: Database session
    """
    from models import VideoAnalysis
    import json
    import sys
    import os
    
    try:
        # Import analysis functions from video_view_analysis.py
        sys.path.insert(0, '/workspace/Project/your_scripts')
        from video_view_analysis import (
            analyze_overall_performance, 
            analyze_category_performance, 
            analyze_touch_outcomes,
            build_reason_reports,
            _derive_immediate_adjustments
        )
        
        logging.info(f"Starting AI analysis generation for upload {upload_id}")
        
        # Check if analysis already exists and is completed
        existing_analysis = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
        if existing_analysis and existing_analysis.status == 'completed' and not force:
            logging.info(f"AI analysis already completed for upload {upload_id}, skipping generation")
            return
        elif existing_analysis:
            logging.info(f"Updating existing analysis with status '{existing_analysis.status}' for upload {upload_id}")
            analysis_record = existing_analysis
        else:
            analysis_record = VideoAnalysis(upload_id=upload_id)
            db.session.add(analysis_record)

        # Get upload and determine user_id
        from models import Upload
        upload = Upload.query.get(upload_id)
        if not upload:
            logging.error(f"Upload {upload_id} not found for AI analysis")
            return
            
        user_id = upload.user_id
        
        # Load match data needed for AI analysis
        match_analysis_dir = os.path.join(result_dir, "match_analysis")
        match_data = []
        match_files = sorted(os.listdir(match_analysis_dir))
        for filename in match_files:
            if filename.endswith('_analysis.json'):
                filepath = os.path.join(match_analysis_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Inject filename and derived info
                        data['filename'] = filename
                        match_idx = None
                        import re
                        m = re.search(r'match_(\d+)_analysis\.json$', filename)
                        if m:
                            try:
                                match_idx = int(m.group(1))
                            except Exception:
                                match_idx = None
                        data['match_idx'] = match_idx
                        if match_idx is not None:
                            data['video_path'] = f"{user_id}/{upload_id}/matches/match_{match_idx}/match_{match_idx}.mp4"
                        else:
                            data['video_path'] = ''
                        data['touch_index'] = len(match_data)
                        match_data.append(data)
                except Exception as e:
                    logging.error(f"Error loading {filepath}: {e}")
                    continue
        
        # Load performance metrics for overall analysis
        from video_view_analysis import calculate_performance_metrics
        performance_data = calculate_performance_metrics(upload_id, user_id)
        
        # Generate outcome analysis FIRST (loss & win reasons)
        logging.info("Generating outcome analysis...")
        outcome_analysis = analyze_touch_outcomes(match_data, upload_id, user_id)
        loss_analysis = outcome_analysis['loss_grouped']
        win_analysis = outcome_analysis['win_grouped']
        
        # Generate reason-synthesis reports (win/loss patterns)
        logging.info("Generating detailed win/loss reason reports...")
        win_reason_reports, win_reason_briefs = build_reason_reports(match_data, win_analysis, 'win', upload_id, user_id)
        loss_reason_reports, loss_reason_briefs = build_reason_reports(match_data, loss_analysis, 'loss', upload_id, user_id)

        reason_summary_bullets = {
            'left': {
                'win': win_reason_briefs.get('left_fencer', {}),
                'loss': loss_reason_briefs.get('left_fencer', {})
            },
            'right': {
                'win': win_reason_briefs.get('right_fencer', {}),
                'loss': loss_reason_briefs.get('right_fencer', {})
            }
        }

        immediate_adjustments = {
            'left': _derive_immediate_adjustments(reason_summary_bullets.get('left', {})),
            'right': _derive_immediate_adjustments(reason_summary_bullets.get('right', {}))
        }

        # Generate overall performance analysis WITH loss analysis data
        logging.info("Generating overall performance analysis...")
        left_overall = analyze_overall_performance(match_data, 'left', performance_data['left_fencer_metrics'], upload_id, user_id, loss_analysis, win_analysis)
        right_overall = analyze_overall_performance(match_data, 'right', performance_data['right_fencer_metrics'], upload_id, user_id, loss_analysis, win_analysis)
        if isinstance(left_overall, dict):
            left_overall['rapid_adjustments'] = immediate_adjustments['left']
        if isinstance(right_overall, dict):
            right_overall['rapid_adjustments'] = immediate_adjustments['right']
        
        # Generate category-specific analysis WITH loss analysis data
        logging.info("Generating category-specific analysis...")
        categories = ['attack', 'defense', 'in_box']
        left_category_results = {}
        right_category_results = {}
        
        for category in categories:
            left_category_results[category] = analyze_category_performance(match_data, category, 'left', upload_id, user_id, loss_analysis, win_analysis)
            right_category_results[category] = analyze_category_performance(match_data, category, 'right', upload_id, user_id, loss_analysis, win_analysis)
        
        # Update or create VideoAnalysis record
        analysis_record.left_overall_analysis = json.dumps(left_overall) if left_overall else None
        analysis_record.right_overall_analysis = json.dumps(right_overall) if right_overall else None
        analysis_record.left_category_analysis = json.dumps(left_category_results) if left_category_results else None
        analysis_record.right_category_analysis = json.dumps(right_category_results) if right_category_results else None
        combined_outcomes = {
            'loss': loss_analysis,
            'win': win_analysis
        }
        analysis_record.loss_analysis = json.dumps(combined_outcomes, ensure_ascii=False) if combined_outcomes else None

        reason_cache = {
            'loss_reason_reports': loss_reason_reports,
            'win_reason_reports': win_reason_reports,
            'reason_summary_bullets': reason_summary_bullets
        }
        analysis_record.detailed_analysis = json.dumps(reason_cache, ensure_ascii=False)
        analysis_record.status = 'completed'
        analysis_record.error_message = None
        analysis_record.generated_at = datetime.utcnow()

        db.session.commit()

        logging.info(f"AI analysis generation completed for upload {upload_id}")
        
    except Exception as e:
        logging.error(f"Error generating AI analysis for upload {upload_id}: {str(e)}")
        
        analysis_record = VideoAnalysis.query.filter_by(upload_id=upload_id).first()
        if not analysis_record:
            analysis_record = VideoAnalysis(upload_id=upload_id)
            db.session.add(analysis_record)
        analysis_record.status = 'error'
        analysis_record.error_message = str(e)
        db.session.commit()

        # Don't re-raise the exception to avoid breaking the main task
        logging.warning(f"AI analysis failed for upload {upload_id}, but continuing with upload completion")


def _clamp(value: Optional[float], min_value: float = 0.0, max_value: float = 1.0) -> float:
    try:
        if value is None:
            return 0.0
        return max(min_value, min(max_value, float(value)))
    except Exception:
        return 0.0


def _safe_mean(values: List[Optional[float]]) -> float:
    numeric = [float(v) for v in values if v is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _normalize_kpis(kpis: Dict[str, Any]) -> Dict[str, float]:
    # Normalize KPIs into [0,1] for scoring
    return {
        'velocity': _clamp((kpis.get('avg_velocity') or 0.0) / 5.0),
        'acceleration': _clamp((kpis.get('avg_acceleration') or 0.0) / 3.0),
        'advance_ratio': _clamp(kpis.get('advance_ratio') or 0.0),
        'pause_ratio_inv': _clamp(1.0 - (kpis.get('pause_ratio') or 0.0)),
        'first_step_inv': _clamp(1.0 - _clamp((kpis.get('first_step_init') or 0.0) / 1.0)),
        'arm_extension_freq': _clamp((kpis.get('arm_extension_freq') or 0.0) / 20.0),
        'launch_success_rate': _clamp(kpis.get('launch_success_rate') or 0.0),
        'attack_success_rate': _clamp(kpis.get('attack_success_rate') or 0.0),
        'attacking_ratio': _clamp(kpis.get('attacking_ratio') or 0.0),
    }


def _score_subdomains(norm: Dict[str, float], tag_rates: Dict[str, float]) -> Dict[str, float]:
    # Offense, Defense, Tempo, Distance
    offense_vals = [norm['launch_success_rate'], norm['attack_success_rate'], norm['attacking_ratio']]
    offense = _safe_mean(offense_vals)

    defense_sources = [
        tag_rates.get('good_defensive_quality', 0.0),
        tag_rates.get('maintain_safe_distance', 0.0),
        tag_rates.get('consistent_spacing', 0.0),
    ]
    defense = _safe_mean(defense_sources)

    tempo_vals = [norm['pause_ratio_inv'], norm['first_step_inv']]
    tempo = _safe_mean(tempo_vals)

    distance_vals = [norm['advance_ratio'], tag_rates.get('good_attack_distance', 0.0), tag_rates.get('maintain_safe_distance', 0.0)]
    distance = _safe_mean(distance_vals)

    return {
        'offense': round(offense * 100),
        'defense': round(defense * 100),
        'tempo': round(tempo * 100),
        'distance': round(distance * 100),
    }


def _generate_report_for_upload(upload_id: int, app, db) -> None:
    """Build concise report.json for an upload based on existing analysis outputs and tags."""
    from models import Upload, Fencer, Bout, BoutTag, Tag

    upload = Upload.query.get(upload_id)
    if not upload:
        raise ValueError(f"Upload {upload_id} not found")

    result_dir = os.path.join(app.config['RESULT_FOLDER'], str(upload.user_id), str(upload_id))
    fencer_dir = os.path.join(result_dir, 'fencer_analysis')
    os.makedirs(fencer_dir, exist_ok=True)

    bout_summaries_path = os.path.join(fencer_dir, 'bout_summaries.json')
    cross_bout_path = os.path.join(fencer_dir, 'cross_bout_analysis.json')

    # Load bout data
    bouts_struct: List[Dict[str, Any]] = []
    fps = 30
    if os.path.exists(bout_summaries_path):
        try:
            with open(bout_summaries_path, 'r', encoding='utf-8') as f:
                bs = json.load(f)
            fps = bs.get('fps', 30)
            raw_bouts = bs.get('bouts', [])
            for b in raw_bouts:
                judgement = b.get('judgement') or {}
                result_val = (b.get('result') or '').lower()
                ai_winner = (judgement.get('winner') or '').lower() if isinstance(judgement, dict) else ''
                if result_val == 'skip' and ai_winner in ['left', 'right']:
                    result_display = ai_winner
                    result_source = 'ai'
                else:
                    result_display = result_val
                    result_source = 'user' if result_val else 'unknown'
                bouts_struct.append({
                    'match_idx': b.get('match_idx'),
                    'type': b.get('type'),
                    'result': result_val,
                    'result_display': result_display,
                    'result_source': result_source,
                    'judgement': judgement,
                    'frame_range': b.get('frame_range', [0, 0]),
                    'left': b.get('left_data', {}),
                    'right': b.get('right_data', {}),
                })
        except Exception as e:
            logging.warning(f"Failed to read bout_summaries.json for upload {upload_id}: {e}")

    # Fallback: derive from match_analysis files if needed
    if not bouts_struct:
        match_analysis_dir = os.path.join(result_dir, 'match_analysis')
        try:
            for fn in sorted(os.listdir(match_analysis_dir)) if os.path.exists(match_analysis_dir) else []:
                if not fn.endswith('_analysis.json'):
                    continue
                try:
                    with open(os.path.join(match_analysis_dir, fn), 'r', encoding='utf-8') as f:
                        md = json.load(f)
                    idx_part = fn.split('_')[1]
                    match_idx = int(idx_part)
                    bouts_struct.append({
                        'match_idx': match_idx,
                        'type': md.get('bout_type', 'Unknown'),
                        'result': md.get('result', 'unknown'),
                        'frame_range': md.get('frame_range', [0, 0]),
                        'left': md.get('fencer_left', {}),
                        'right': md.get('fencer_right', {}),
                    })
                except Exception as fe:
                    logging.warning(f"Error parsing {fn} for upload {upload_id}: {fe}")
        except Exception:
            pass

    # Build per-side KPI aggregates
    def _collect_kpis(side: str) -> Dict[str, Any]:
        velocities = []
        accelerations = []
        advance_ratios = []
        pause_ratios = []
        first_steps = []
        arm_ext_freqs = []
        arm_ext_durs = []
        has_launches = []
        launch_promptnesses = []
        is_attacking_flags = []
        attack_success_rates = []

        for b in bouts_struct:
            data = b.get(side, {})
            if not isinstance(data, dict):
                continue
            if data.get('velocity') is not None:
                velocities.append(float(data.get('velocity') or 0.0))
            if data.get('acceleration') is not None:
                accelerations.append(float(data.get('acceleration') or 0.0))
            if data.get('advance_ratio') is not None:
                advance_ratios.append(float(data.get('advance_ratio') or 0.0))
            if data.get('pause_ratio') is not None:
                pause_ratios.append(float(data.get('pause_ratio') or 0.0))
            fs = data.get('first_step', {})
            if isinstance(fs, dict) and fs.get('init_time') is not None:
                first_steps.append(float(fs.get('init_time') or 0.0))
            if data.get('arm_extension_freq') is not None:
                arm_ext_freqs.append(float(data.get('arm_extension_freq') or 0.0))
            if data.get('avg_arm_extension_duration') is not None:
                arm_ext_durs.append(float(data.get('avg_arm_extension_duration') or 0.0))
            if data.get('has_launch') is not None:
                has_launches.append(bool(data.get('has_launch')))
            if data.get('launch_promptness') is not None:
                launch_promptnesses.append(float(data.get('launch_promptness') or 0.0))
            if data.get('is_attacking') is not None:
                is_attacking_flags.append(bool(data.get('is_attacking')))
            if data.get('attack_success_rate') is not None:
                attack_success_rates.append(float(data.get('attack_success_rate') or 0.0))

        total_b = max(1, len(bouts_struct))
        launch_success_rate = (sum(1 for v in has_launches if v) / len(has_launches)) if has_launches else 0.0
        attacking_ratio = (sum(1 for v in is_attacking_flags if v) / len(is_attacking_flags)) if is_attacking_flags else 0.0
        avg_attack_success_rate = _safe_mean(attack_success_rates)

        return {
            'total_bouts': len(bouts_struct),
            'avg_velocity': _safe_mean(velocities),
            'avg_acceleration': _safe_mean(accelerations),
            'advance_ratio': _safe_mean(advance_ratios),
            'pause_ratio': _safe_mean(pause_ratios),
            'first_step_init': _safe_mean(first_steps),
            'arm_extension_freq': _safe_mean(arm_ext_freqs),
            'avg_arm_extension_duration': _safe_mean(arm_ext_durs),
            'launch_success_rate': launch_success_rate,
            'avg_launch_promptness': _safe_mean(launch_promptnesses),
            'attacking_ratio': attacking_ratio,
            'attack_success_rate': avg_attack_success_rate,
        }

    left_kpis = _collect_kpis('left')
    right_kpis = _collect_kpis('right')

    # Outcomes summary
    outcomes = {'left_wins': 0, 'right_wins': 0, 'skips': 0}
    for b in bouts_struct:
        r = (b.get('result') or '').lower()
        if r == 'left': outcomes['left_wins'] += 1
        elif r == 'right': outcomes['right_wins'] += 1
        elif r == 'skip': outcomes['skips'] += 1

    # Tag rates per side (counts per total bouts)
    bout_rows = Bout.query.filter_by(upload_id=upload_id).all()
    bout_ids = [br.id for br in bout_rows]
    # Collect all tag names dynamically across this upload
    tag_rates: Dict[str, Dict[str, float]] = {'left': {}, 'right': {}}
    all_tag_counts: Dict[str, Dict[str, int]] = {'left': {}, 'right': {}}
    if bout_ids:
        tags = BoutTag.query.filter(BoutTag.bout_id.in_(bout_ids)).all()
        for bt in tags:
            try:
                name = bt.tag.name if bt.tag else None
                side = bt.fencer_side
                if name:
                    all_tag_counts.setdefault(side, {})
                    all_tag_counts[side][name] = all_tag_counts[side].get(name, 0) + 1
            except Exception:
                continue
        for side in ['left', 'right']:
            tag_rates[side] = {}
            for name, count in all_tag_counts.get(side, {}).items():
                tag_rates[side][name] = (count / max(1, len(bouts_struct)))

    # Sub-scores and overall scores
    left_norm = _normalize_kpis(left_kpis)
    right_norm = _normalize_kpis(right_kpis)
    left_sub = _score_subdomains(left_norm, tag_rates['left'])
    right_sub = _score_subdomains(right_norm, tag_rates['right'])

    def _overall(sub: Dict[str, float]) -> int:
        offense = sub['offense'] / 100.0
        defense = sub['defense'] / 100.0
        tempo = sub['tempo'] / 100.0
        distance = sub['distance'] / 100.0
        overall = 0.30 * offense + 0.30 * defense + 0.20 * tempo + 0.20 * distance
        return round(overall * 100)

    left_overall = _overall(left_sub)
    right_overall = _overall(right_sub)

    # Highlights (concise)
    highlights: List[str] = []
    if left_overall >= right_overall:
        highlights.append(f"Left overall score leads ({left_overall} vs {right_overall})")
    else:
        highlights.append(f"Right overall score leads ({right_overall} vs {left_overall})")
    if left_kpis.get('launch_success_rate', 0) or right_kpis.get('launch_success_rate', 0):
        highlights.append(f"Launch success L/R: {round(left_kpis.get('launch_success_rate',0)*100)}% / {round(right_kpis.get('launch_success_rate',0)*100)}%")
    if left_kpis.get('pause_ratio', None) is not None and right_kpis.get('pause_ratio', None) is not None:
        highlights.append(f"Pause ratio L/R: {left_kpis['pause_ratio']:.2f} / {right_kpis['pause_ratio']:.2f}")

    # Graphs and insights (compact extraction)
    chart_insights: Dict[str, List[str]] = {}
    tactical_insights: Dict[str, List[str]] = {}
    if os.path.exists(cross_bout_path):
        try:
            with open(cross_bout_path, 'r', encoding='utf-8') as f:
                cb = json.load(f)
            ca = cb.get('chart_analysis', {}) or {}
            ga = cb.get('graph_analysis', {}) or {}
            def _top_bullets(text: Any, n: int = 3) -> List[str]:
                if not text:
                    return []
                if isinstance(text, dict):
                    # Collect all values
                    parts = []
                    for v in text.values():
                        if isinstance(v, str):
                            parts.extend([ln.strip() for ln in v.split('\n') if ln.strip()])
                    return parts[:n]
                if isinstance(text, str):
                    return [ln.strip() for ln in text.split('\n') if ln.strip()][:n]
                return []
            chart_insights['comparison_chart'] = _top_bullets(ca.get('comparison_chart'))
            chart_insights['radar_chart'] = _top_bullets(ca.get('radar_chart'))
            # Tactical keys
            keys = ['attack_type_analysis','tempo_type_analysis','attack_distance_analysis','counter_opportunities','retreat_quality','retreat_distance','defensive_quality','bout_outcome']
            for k in keys:
                tactical_insights[k] = _top_bullets(ga.get(k))
        except Exception as e:
            logging.warning(f"Failed to extract insights for upload {upload_id}: {e}")

    graphs = {
        'bar_comparison': {
            'image': 'fencer_analysis/plots/fencer_comparison.png',
            'insights': chart_insights.get('comparison_chart', [])
        },
        'radar': {
            'image': 'fencer_analysis/plots/fencer_radar_comparison.png',
            'insights': chart_insights.get('radar_chart', [])
        },
        'tactical': []
    }
    tactical_keys = ['attack_type_analysis','tempo_type_analysis','attack_distance_analysis','counter_opportunities','retreat_quality','retreat_distance','defensive_quality','bout_outcome']
    for key in tactical_keys:
        graphs['tactical'].append({
            'key': key,
            'left_image': f"fencer_analysis/advanced_plots/Fencer_Left/left_{key}.png",
            'right_image': f"fencer_analysis/advanced_plots/Fencer_Right/right_{key}.png",
            'insights': tactical_insights.get(key, [])
        })

    # Build structured analysis sections with concise bullets per side and comparisons
    def _pct(v: float) -> str:
        try:
            return f"{round(v * 100)}%"
        except Exception:
            return "—"
    
    def _get_chinese_evaluation(side: str, kpis: Dict[str, Any], tags: Dict[str, float]) -> List[str]:
        """Generate Chinese evaluation text for left/right fencer based on their performance data."""
        evaluations = []
        
        # Attack evaluation
        attack_success = kpis.get('attack_success_rate', 0.0)
        if attack_success >= 0.6:
            evaluations.append("攻击执行力强，成功率高")
        elif attack_success >= 0.4:
            evaluations.append("攻击效果中等，有改进空间")
        else:
            evaluations.append("攻击成功率偏低，需要重点改进")
            
        # Defense evaluation
        def_quality = tags.get('good_defensive_quality', 0.0)
        if def_quality >= 0.6:
            evaluations.append("防守质量优秀，定位准确")
        elif def_quality >= 0.4:
            evaluations.append("防守基础良好，需要细化")
        else:
            evaluations.append("防守质量有待提高")
            
        # Movement evaluation
        advance_ratio = kpis.get('advance_ratio', 0.0)
        if advance_ratio >= 0.6:
            evaluations.append("移动积极主动，前进意识强")
        elif advance_ratio >= 0.4:
            evaluations.append("移动节奏适中")
        else:
            evaluations.append("移动偏于保守，可增加进攻性")
            
        # Tempo evaluation
        pause_ratio = kpis.get('pause_ratio', 0.0)
        if pause_ratio <= 0.3:
            evaluations.append("节奏控制良好，连贯性强")
        else:
            evaluations.append("停顿较多，需要提高动作流畅度")
            
        return evaluations[:4]  # Return top 4 evaluations

    def _attack_bullets(kpis: Dict[str, Any], tags: Dict[str, float]) -> List[str]:
        bullets: List[str] = []
        bullets.append(f"速度 {kpis.get('avg_velocity', 0):.2f} m/s，加速度 {kpis.get('avg_acceleration', 0):.2f} m/s²")
        bullets.append(f"发起成功率 { _pct(kpis.get('launch_success_rate', 0.0)) }，首步时间 { kpis.get('first_step_init', 0.0):.2f}秒")
        # improvement hint
        if (kpis.get('pause_ratio') or 0) > 0.30:
            bullets.append("减少停顿，插入破碎节奏")
        elif (tags.get('good_attack_distance', 0.0) or 0) < 0.30:
            bullets.append("优化攻击距离设置")
        else:
            bullets.append("变化准备，提高攻击成功率")
        return bullets[:3]

    def _defense_bullets(kpis: Dict[str, Any], tags: Dict[str, float]) -> List[str]:
        bullets: List[str] = []
        bullets.append(f"防守质量 { _pct(tags.get('good_defensive_quality', 0.0)) }")
        bullets.append(f"安全距离 { _pct(tags.get('maintain_safe_distance', 0.0)) }，间距一致性 { _pct(tags.get('consistent_spacing', 0.0)) }")
        if (kpis.get('pause_ratio') or 0) > 0.30:
            bullets.append("在压力下缩短停顿时间")
        else:
            bullets.append("在快速交锋中保持间距")
        return bullets[:3]

    def _compare_bullets(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
        bullets: List[str] = []
        # First-step faster side
        try:
            lf = left.get('first_step_init') or 0.0
            rf = right.get('first_step_init') or 0.0
            if lf and rf:
                faster = '左侧' if lf < rf else '右侧'
                bullets.append(f"首步速度：{faster}更快 ({lf:.2f}秒 vs {rf:.2f}秒)")
        except Exception:
            pass
        # Launch success higher side
        try:
            ll = left.get('launch_success_rate') or 0.0
            rl = right.get('launch_success_rate') or 0.0
            if ll or rl:
                better = '左侧' if ll > rl else '右侧'
                bullets.append(f"发起成功率：{better}更高 ({_pct(max(ll,rl))})")
        except Exception:
            pass
        # Pause control
        try:
            lp = left.get('pause_ratio') or 0.0
            rp = right.get('pause_ratio') or 0.0
            if lp or rp:
                better = '左侧' if lp < rp else '右侧'
                bullets.append(f"停顿控制：{better}更稳定 ({min(lp,rp):.2f})")
        except Exception:
            pass
        return bullets[:3]

    # Map tactical keys to sections (attack vs defense) with detailed bullets per side
    def build_graph_bullets(key: str, lk: Dict[str, Any], rk: Dict[str, Any], lt: Dict[str, float], rt: Dict[str, float]) -> Dict[str, List[str]]:
        left_b: List[str] = []
        right_b: List[str] = []
        def pct(x: float) -> str: return f"{round((x or 0.0)*100)}%"
        
        if key == 'attack_type_analysis':
            # Left fencer attack analysis
            l_success = lk.get('attack_success_rate', 0.0)
            l_simple = lt.get('simple_attack', 0.0)
            l_compound = lt.get('compound_attack', 0.0)
            left_b.append(f"攻击成功率：{pct(l_success)}")
            left_b.append(f"攻击组合：{pct(l_simple)}简单攻击，{pct(l_compound)}复合攻击")
            if l_success < 0.4:
                left_b.append("专注于设置质量 - 训练发起前的剑刃准备")
                left_b.append("练习距离管理以提高时机准确性")
            else:
                left_b.append("攻击成功率强劲 - 保持当前准备风格")
                left_b.append("考虑在现有成功模式中加入欺骗动作")
            if l_compound < l_simple:
                left_b.append("增加复合攻击 - 添加佯攻和二次动作")
                left_b.append("练习破碎节奏为复合动作创造机会")
            else:
                left_b.append("复合攻击变化良好 - 专注于掩饰初始意图")
                left_b.append("变化复合元素之间的时机保持不可预测性")
                
            # Right fencer attack analysis  
            r_success = rk.get('attack_success_rate', 0.0)
            r_simple = rt.get('simple_attack', 0.0)
            r_compound = rt.get('compound_attack', 0.0)
            right_b.append(f"攻击成功率：{pct(r_success)}")
            right_b.append(f"攻击组合：{pct(r_simple)}简单攻击，{pct(r_compound)}复合攻击")
            if r_success < 0.4:
                right_b.append("改善攻击准备 - 专注于剑刃接触时机")
                right_b.append("练习距离控制以优化发起位置")
            else:
                right_b.append("攻击执行有效 - 保持当前方法")
                right_b.append("探索高级组合以扩展技能库")
            if r_compound < r_simple:
                right_b.append("发展复合攻击技能 - 练习佯攻-脱离序列")
                right_b.append("使用节奏变化为多动作攻击做准备")
            else:
                right_b.append("复合攻击强劲 - 专注于执行速度")
                right_b.append("保持复合攻击时机的变化避免可预测性")
            
        elif key == 'attack_distance_analysis':
            # Left fencer distance analysis
            l_good_dist = lt.get('good_attack_distance', 0.0)
            l_advance = lk.get('advance_ratio', 0.0)
            left_b.append(f"良好攻击距离：{pct(l_good_dist)}")
            left_b.append(f"前进比率：{l_advance:.2f}")
            if l_good_dist < 0.4:
                left_b.append("距离控制需要改进 - 练习半步进入")
                left_b.append("专注于准备步法以优化发起距离")
                left_b.append("避免从过远距离匆忙攻击")
            else:
                left_b.append("距离纪律扎实 - 保持当前定位")
                left_b.append("考虑变化攻击距离增加不可预测性")
            if l_advance < 0.3:
                left_b.append("增加前进压力 - 在前进中更加自信")
                left_b.append("更频繁使用前进-弓步组合")
            else:
                left_b.append("前进动作良好 - 与退却选项保持平衡")
                
            # Right fencer distance analysis
            r_good_dist = rt.get('good_attack_distance', 0.0)  
            r_advance = rk.get('advance_ratio', 0.0)
            right_b.append(f"良好攻击距离：{pct(r_good_dist)}")
            right_b.append(f"前进比率：{r_advance:.2f}")
            if r_good_dist < 0.4:
                right_b.append("改进距离判断 - 训练最佳发起范围")
                right_b.append("练习受控前进以设置合适距离")
                right_b.append("避免攻击尝试中过度伸展")
            else:
                right_b.append("距离控制出色 - 利用这一优势")
                right_b.append("使用距离掌握控制比赛节奏")
            if r_advance < 0.3:
                right_b.append("在前进动作中更加积极")
                right_b.append("将前进与攻击准备结合")
            else:
                right_b.append("前进压力强劲 - 保持战术平衡")
            
        elif key == 'tempo_type_analysis':
            # Left fencer tempo analysis
            l_steady = lt.get('steady_tempo', 0.0)
            l_variable = lt.get('variable_tempo', 0.0) 
            l_broken = lt.get('broken_tempo', 0.0)
            l_first = lk.get('first_step_init', 0.0)
            left_b.append(f"节奏分布：{pct(l_steady)}稳定，{pct(l_variable)}变化，{pct(l_broken)}破碎")
            left_b.append(f"首步时机：{l_first:.2f}秒")
            if l_broken < 0.2:
                left_b.append("增加破碎节奏 - 使用停顿打乱对手节奏")
                left_b.append("在攻击序列中练习节奏变化")
            else:
                left_b.append("节奏变化良好 - 继续利用时机变化")
            if l_first > 0.4:
                left_b.append("改善反应时间 - 练习爆发性首次动作")
                left_b.append("练习快速决策训练")
            else:
                left_b.append("快速初始反应 - 战术性利用速度优势")
            if l_variable < 0.3:
                left_b.append("增加节奏变化以减少可预测性")
                left_b.append("练习在动作中加速和减速")
            else:
                left_b.append("节奏控制强劲 - 保持不可预测性")
                
            # Right fencer tempo analysis
            r_steady = rt.get('steady_tempo', 0.0)
            r_variable = rt.get('variable_tempo', 0.0)
            r_broken = rt.get('broken_tempo', 0.0) 
            r_first = rk.get('first_step_init', 0.0)
            right_b.append(f"节奏分布：{pct(r_steady)}稳定，{pct(r_variable)}变化，{pct(r_broken)}破碎")
            right_b.append(f"首步时机：{r_first:.2f}秒")
            if r_broken < 0.2:
                right_b.append("融入更多破碎节奏以创造机会")
                right_b.append("使用战略性停顿控制比赛流程")
            else:
                right_b.append("节奏管理有效 - 继续当前方法")
            if r_first > 0.4:
                right_b.append("改进初始反应速度和爆发力")
                right_b.append("专注于从准备到行动的快速转换")
            else:
                right_b.append("反应时间出色 - 利用速度优势")
            if r_variable < 0.3:
                right_b.append("发展更多节奏变化避免可预测模式")
                right_b.append("在复合动作中练习节奏变化")
            else:
                right_b.append("节奏变化良好 - 保持战术灵活性")
            
        elif key == 'defensive_quality':
            # Left fencer defense analysis
            l_def_quality = lt.get('good_defensive_quality', 0.0)
            l_safe_dist = lt.get('maintain_safe_distance', 0.0)
            left_b.append(f"防守质量：{pct(l_def_quality)}")
            left_b.append(f"安全距离维持：{pct(l_safe_dist)}")
            if l_def_quality < 0.4:
                left_b.append("加强格挡技术 - 专注于剑刃控制")
                left_b.append("练习反击时机以利用防守动作")
                left_b.append("在防守序列中改进距离判断")
            else:
                left_b.append("防守基础扎实 - 在当前技能基础上发展")
                left_b.append("在防守动作中增加反击变化")
            if l_safe_dist < 0.4:
                left_b.append("改善距离意识 - 保持安全间距")
                left_b.append("在压力下练习受控退却")
            else:
                left_b.append("防守中距离纪律良好")
                left_b.append("使用距离控制设置反击")
                
            # Right fencer defense analysis
            r_def_quality = rt.get('good_defensive_quality', 0.0)
            r_safe_dist = rt.get('maintain_safe_distance', 0.0)
            right_b.append(f"防守质量：{pct(r_def_quality)}")
            right_b.append(f"安全距离维持：{pct(r_safe_dist)}")
            if r_def_quality < 0.4:
                right_b.append("提高格挡精度和反击执行")
                right_b.append("专注于防守动作与对手攻击的时机配合")
                right_b.append("在防守中发展更强的剑刃接触")
            else:
                right_b.append("防守技能强劲 - 利用战术优势")
                right_b.append("探索高级反击选项")
            if r_safe_dist < 0.4:
                right_b.append("致力于保持最佳防守距离")
                right_b.append("练习战略性退却控制交锋范围")
            else:
                right_b.append("防守中距离管理出色")
                right_b.append("使用距离掌握控制比赛节奏")
            
        elif key == 'retreat_quality':
            # Left fencer retreat analysis
            l_spacing = lt.get('consistent_spacing', 0.0)
            l_pause = lk.get('pause_ratio', 0.0)
            left_b.append(f"间距一致性：{pct(l_spacing)}")
            left_b.append(f"停顿比率：{l_pause:.2f}")
            if l_spacing < 0.4:
                left_b.append("改善退却一致性 - 保持均匀间距")
                left_b.append("在压力下练习受控后退动作")
                left_b.append("专注于退却序列中的平衡")
            else:
                left_b.append("退却纪律良好 - 保持当前技术")
                left_b.append("使用一致间距设置反击机会")
            if l_pause > 0.4:
                left_b.append("减少过度停顿 - 保持比赛流畅")
                left_b.append("练习连续动作避免拖延")
            else:
                left_b.append("活动水平良好 - 平衡行动与战术停顿")
                left_b.append("使用战略性停顿控制节奏")
                
            # Right fencer retreat analysis
            r_spacing = rt.get('consistent_spacing', 0.0)
            r_pause = rk.get('pause_ratio', 0.0)
            right_b.append(f"间距一致性：{pct(r_spacing)}")
            right_b.append(f"停顿比率：{r_pause:.2f}")
            if r_spacing < 0.4:
                right_b.append("改进退却质量 - 专注于受控动作")
                right_b.append("练习在退却中保持适当距离")
                right_b.append("在后退动作中发展更好的平衡和恢复")
            else:
                right_b.append("退却技术扎实 - 利用防守优势")
                right_b.append("使用间距控制创造反击机会")
            if r_pause > 0.4:
                right_b.append("增加活动 - 避免过度犹豫")
                right_b.append("练习在动作间更流畅转换")
            else:
                right_b.append("活动水平适当 - 保持当前节奏")
                right_b.append("继续有效使用战术停顿")
            
        elif key == 'retreat_distance':
            # Left fencer retreat distance analysis
            l_advance = lk.get('advance_ratio', 0.0)
            l_pause = lk.get('pause_ratio', 0.0)
            left_b.append(f"动作平衡：{l_advance:.2f}前进 vs {l_pause:.2f}停顿")
            left_b.append("专注于战术定位的最佳退却长度")
            if l_advance < 0.3:
                left_b.append("增加前进压力 - 更加自信")
                left_b.append("练习前进-攻击组合")
                left_b.append("通过前进动作控制剑道中心")
            else:
                left_b.append("前进侵略性良好 - 保持战术平衡")
                left_b.append("继续利用前进压力优势")
            left_b.append("练习从退却中轻快恢复")
            left_b.append("致力于从退却到攻击的流畅转换")
            
            # Right fencer retreat distance analysis
            r_advance = rk.get('advance_ratio', 0.0)
            r_pause = rk.get('pause_ratio', 0.0)
            right_b.append(f"动作平衡：{r_advance:.2f}前进 vs {r_pause:.2f}停顿")
            right_b.append("优化退却距离以创造反击机会")
            if r_advance < 0.3:
                right_b.append("发展更多前进动作 - 主动出击")
                right_b.append("练习积极的前进模式")
                right_b.append("使用前进压力控制比赛动态")
            else:
                right_b.append("前进动作强劲 - 保持战术优势")
                right_b.append("继续有效使用前进压力")
            right_b.append("专注于高效退却机制")
            right_b.append("练习从防守到攻击的快速转换")
            
        elif key == 'counter_opportunities':
            # Left fencer counter-attack analysis
            l_extensions = lk.get('arm_extension_freq', 0.0)
            left_b.append(f"手臂伸展频率：{l_extensions:.1f}")
            left_b.append("发展反击意识和时机")
            if l_extensions < 2.0:
                left_b.append("增加手臂伸展使用 - 寻找阻击机会")
                left_b.append("练习识别对手准备阶段")
            else:
                left_b.append("伸展频率良好 - 专注于时机准确性")
                left_b.append("致力于变化伸展时机增加不可预测性")
            left_b.append("在对手破碎节奏后练习反时机")
            left_b.append("发展距离陷阱创造反击机会")
            left_b.append("致力于成功格挡后立即反击")
            
            # Right fencer counter-attack analysis
            r_extensions = rk.get('arm_extension_freq', 0.0)
            right_b.append(f"手臂伸展频率：{r_extensions:.1f}")
            right_b.append("专注于最大化反击机会")
            if r_extensions < 2.0:
                right_b.append("增加反击尝试 - 使用更多伸展")
                right_b.append("练习解读对手攻击准备")
            else:
                right_b.append("伸展游戏强劲 - 完善时机精度")
                right_b.append("继续利用反击机会")
            right_b.append("发展格挡-反击组合用于早期伸展")
            right_b.append("练习利用对手节奏变化反制")
            right_b.append("致力于距离控制设置反弓步")
            
        else:
            # Fallback generic
            left_b = _attack_bullets(lk, lt)
            right_b = _defense_bullets(rk, rt)
            
        return {'left': left_b[:6], 'right': right_b[:6]}
    attack_keys = ['attack_type_analysis', 'attack_distance_analysis', 'tempo_type_analysis']
    defense_keys = ['defensive_quality', 'retreat_quality', 'retreat_distance', 'counter_opportunities']
    tactical_by_key = {g['key']: g for g in graphs['tactical']}

    sections = {
        'attack': [],
        'defense': [],
        'overall': {
            'bar_comparison': {
                'image': graphs['bar_comparison']['image'],
                'left_bullets': [f"攻击 {left_sub['offense']}", f"节奏 {left_sub['tempo']}", f"距离 {left_sub['distance']}"][:3],
                'right_bullets': [f"攻击 {right_sub['offense']}", f"节奏 {right_sub['tempo']}", f"距离 {right_sub['distance']}"][:3],
                'compare': _compare_bullets(left_kpis, right_kpis)
            },
            'radar': {
                'image': graphs['radar']['image'],
                'left_bullets': [f"速度 {left_kpis.get('avg_velocity',0):.2f}", f"加速度 {left_kpis.get('avg_acceleration',0):.2f}", f"前进 {left_kpis.get('advance_ratio',0):.2f}"][:3],
                'right_bullets': [f"速度 {right_kpis.get('avg_velocity',0):.2f}", f"加速度 {right_kpis.get('avg_acceleration',0):.2f}", f"前进 {right_kpis.get('advance_ratio',0):.2f}"][:3],
                'compare': _compare_bullets(left_kpis, right_kpis)
            },
            'bout_outcome': {
                'left_bullets': [f"获胜 {outcomes['left_wins']}", f"跳过 {outcomes['skips']}"][:2],
                'right_bullets': [f"获胜 {outcomes['right_wins']}", f"跳过 {outcomes['skips']}"][:2],
                'compare': [f"胜负：左 {outcomes['left_wins']} vs 右 {outcomes['right_wins']}"]
            }
        }
    }

    for k in attack_keys:
        g = tactical_by_key.get(k, {})
        bullets = build_graph_bullets(k, left_kpis, right_kpis, tag_rates['left'], tag_rates['right'])
        sections['attack'].append({
            'key': k,
            'left_image': g.get('left_image', ''),
            'right_image': g.get('right_image', ''),
            'left_bullets': bullets['left'],
            'right_bullets': bullets['right']
        })

    for k in defense_keys:
        g = tactical_by_key.get(k, {})
        bullets = build_graph_bullets(k, left_kpis, right_kpis, tag_rates['left'], tag_rates['right'])
        sections['defense'].append({
            'key': k,
            'left_image': g.get('left_image', ''),
            'right_image': g.get('right_image', ''),
            'left_bullets': bullets['left'],
            'right_bullets': bullets['right']
        })

    # Add overall evaluations to sections using GPT
    weapon_type = upload.weapon_type or 'saber'
    sections['attack_evaluation'] = {
        'left': generate_gpt_attack_evaluation(left_kpis, tag_rates['left'], 'Left', weapon_type),
        'right': generate_gpt_attack_evaluation(right_kpis, tag_rates['right'], 'Right', weapon_type)
    }
    
    sections['defense_evaluation'] = {
        'left': generate_gpt_defense_evaluation(left_kpis, tag_rates['left'], 'Left', weapon_type), 
        'right': generate_gpt_defense_evaluation(right_kpis, tag_rates['right'], 'Right', weapon_type)
    }

    # Tags summary (top 5 per side)
    top_left: Dict[str, int] = {}
    top_right: Dict[str, int] = {}
    tags = BoutTag.query.filter(BoutTag.bout_id.in_(bout_ids)).all() if bout_ids else []
    for bt in tags:
        name = bt.tag.name if bt.tag else None
        if not name:
            continue
        if bt.fencer_side == 'left':
            top_left[name] = top_left.get(name, 0) + 1
        elif bt.fencer_side == 'right':
            top_right[name] = top_right.get(name, 0) + 1
    left_top_sorted = sorted(top_left.items(), key=lambda x: x[1], reverse=True)[:5]
    right_top_sorted = sorted(top_right.items(), key=lambda x: x[1], reverse=True)[:5]

    # Generate GPT-based recommendations
    gpt_recommendations = generate_gpt_recommendations(left_kpis, right_kpis, tag_rates['left'], tag_rates['right'], weapon_type)
    rec_left = gpt_recommendations['left']
    rec_right = gpt_recommendations['right']
    
    # Generate general recommendations for both (simplified categorized version for template compatibility)
    rec_both: Dict[str, List[str]] = {'offense': [], 'defense': [], 'tempo': [], 'distance': []}
    
    # Extract some general recommendations based on performance patterns
    if left_kpis.get('launch_success_rate', 0.0) < 0.5 or right_kpis.get('launch_success_rate', 0.0) < 0.5:
        rec_both['offense'].append('训练复合攻击时机；在发起前增加佯攻')
    if left_kpis.get('attack_success_rate', 0.0) < 0.5 or right_kpis.get('attack_success_rate', 0.0) < 0.5:
        rec_both['offense'].append('增加准备变化；针对获胜中有效的设置')
    if left_kpis.get('pause_ratio', 0.0) > 0.3 or right_kpis.get('pause_ratio', 0.0) > 0.3:
        rec_both['tempo'].append('减少长时间停顿；在交锋后插入破碎节奏')
    if tag_rates['left'].get('maintain_safe_distance', 0.0) < 0.3 or tag_rates['right'].get('maintain_safe_distance', 0.0) < 0.3:
        rec_both['distance'].append('强化距离训练；在压力下保持安全距离')
    if tag_rates['left'].get('good_defensive_quality', 0.0) < 0.3 or tag_rates['right'].get('good_defensive_quality', 0.0) < 0.3:
        rec_both['defense'].append('练习快速格挡-反击；提高对快速启动的后退质量')

    # Build meta
    left_fencer = Fencer.query.get(upload.left_fencer_id) if upload.left_fencer_id else None
    right_fencer = Fencer.query.get(upload.right_fencer_id) if upload.right_fencer_id else None

    report: Dict[str, Any] = {
        'meta': {
            'upload_id': upload.id,
            'user_id': upload.user_id,
            'video_name': os.path.basename(upload.video_path) if upload.video_path else '',
            'weapon': upload.weapon_type or 'saber',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'total_bouts': len(bouts_struct),
            'left_fencer': {'id': left_fencer.id, 'name': left_fencer.name} if left_fencer else None,
            'right_fencer': {'id': right_fencer.id, 'name': right_fencer.name} if right_fencer else None,
            'fps': fps
        },
        'scores': {
            'left': {'overall': left_overall, **left_sub},
            'right': {'overall': right_overall, **right_sub}
        },
        'kpis': {
            'left': left_kpis,
            'right': right_kpis
        },
        'outcomes': outcomes,
        'highlights': highlights[:5],
        'graphs': graphs,
        'sections': sections,
        'per_bout': bouts_struct,
        'tags_summary': {
            'left_top': [{'name': n, 'count': c} for n, c in left_top_sorted],
            'right_top': [{'name': n, 'count': c} for n, c in right_top_sorted]
        },
        'recommendations': rec_both,
        'recommendations_left': rec_left[:8],
        'recommendations_right': rec_right[:8],
        'metric_notes': {
            'velocity': 'Attacking velocity (averaged over launch intervals)',
            'acceleration': 'Attacking acceleration (averaged over launch intervals)'
        },
        'downloads': {
            'csv': [
                'csv/left_xdata.csv', 'csv/left_ydata.csv', 'csv/right_xdata.csv', 'csv/right_ydata.csv', 'csv/meta.csv'
            ]
        }
    }

    # Persist report.json
    report_path = os.path.join(fencer_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
