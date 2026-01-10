#!/usr/bin/env python3
"""Professional fencer profile report generator.

This module aggregates chronological bout data for a fencer, synthesises tactical
insights, and produces the JSON payload that powers the professional fencer
profile UI.

It follows the structure defined in PROFESSIONAL_REPORT_IMPLEMENTATION_PLAN.md.
The generator is designed to run inside a Flask app context so that SQLAlchemy
objects are available. It can be invoked from Celery tasks or command-line
utilities.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from your_scripts.gemini_rest import generate_text

BASE_DIR = os.getenv('FENCER_BASE_DIR', '/workspace/Project')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FENCER_PROFILE_DIR = os.path.join(BASE_DIR, 'fencer_profiles')

if not os.getenv('GEMINI_API_KEY'):
    os.environ['GEMINI_API_KEY'] = 'AIzaSyCAKZxJCnt7BKfsBH1ImvunKuaui-2L_9U'

GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')
GEMINI_MIN_INTERVAL_S = float(os.getenv('GEMINI_MIN_INTERVAL_S', '2.5'))
GEMINI_BACKOFF_BASE_S = float(os.getenv('GEMINI_BACKOFF_BASE_S', '2.0'))
GEMINI_BACKOFF_MAX_S = float(os.getenv('GEMINI_BACKOFF_MAX_S', '12'))

_POSITIVE_TAGS = {
    'good_attack_distance',
    'good_defensive_quality',
    'maintain_safe_distance',
    'consistent_spacing',
    'fast_reaction_time',
    'variable_tempo',
    'good_retreat_quality',
    'counter_opportunities_used',
    'superior_timing',
}

_NEGATIVE_TAGS = {
    'poor_attack_distance',
    'poor_defensive_quality',
    'poor_distance_maintaining',
    'poor_distance_maintenance',
    'missed_counter_opportunities',
    'failed_space_opening',
    'failed_distance_pulls',
    'low_speed',
    'poor_acceleration',
    'no_launch',
    'no_arm_extension',
    'over_extension',
    'excessive_pausing',
    'excessive_tempo_changes',
    'collapsed_distance',
    'slow_reaction',
}

_TAG_DISPLAY_OVERRIDES = {
    'no_attacks': 'No Attacks',
    'no_launch': 'No Launch',
    'no_arm_extension': 'No Arm Extension',
    'good_attack_distance': 'Good Attack Distance',
    'poor_attack_distance': 'Poor Attack Distance',
    'fast_reaction_time': 'Fast Reaction Time',
    'poor_defensive_quality': 'Poor Defensive Quality',
    'good_defensive_quality': 'Good Defensive Quality',
    'low_speed': 'Low Speed',
    'poor_acceleration': 'Poor Acceleration',
    'variable_tempo': 'Variable Tempo',
    'steady_tempo': 'Steady Tempo',
    'excessive_tempo_changes': 'Excessive Tempo Changes',
    'excessive_pausing': 'Excessive Pausing',
    'maintain_safe_distance': 'Maintain Safe Distance',
    'consistent_spacing': 'Consistent Spacing',
    'inconsistent_spacing': 'Inconsistent Spacing',
    'launch': 'Launch',
    'over_extension': 'Over Extension',
    'good_retreat_quality': 'Good Retreat Quality',
    'missed_counter_opportunities': 'Missed Counter Opportunities',
    'failed_space_opening': 'Failed Space Opening',
    'failed_distance_pulls': 'Failed Distance Pulls',
    'collapsed_distance': 'Collapsed Distance',
}

_logger = logging.getLogger(__name__)

# Configure Gemini once at import-time to avoid repeated configuration in Celery workers
_GEMINI_NEXT_AVAILABLE: float = 0.0


def _sleep_until(timestamp: float) -> None:
    to_wait = timestamp - time.time()
    if to_wait > 0:
        time.sleep(to_wait)


def _call_gemini(prompt: str, *, max_tokens: int = 768, temperature: float = 0.35) -> str:
    """Call Gemini REST helper with pacing/backoff, returning plain text."""
    global _GEMINI_NEXT_AVAILABLE

    _sleep_until(_GEMINI_NEXT_AVAILABLE)

    backoff = GEMINI_BACKOFF_BASE_S
    for attempt in range(3):
        try:
            text = generate_text(
                prompt,
                model=GEMINI_MODEL_ID,
                temperature=temperature,
                top_k=1,
                top_p=0.8,
                max_output_tokens=max_tokens,
                timeout_seconds=45,
                max_attempts=6,
                response_mime_type='text/plain',
            )
            _GEMINI_NEXT_AVAILABLE = time.time() + GEMINI_MIN_INTERVAL_S
            if text:
                return text.strip()
        except Exception as exc:  # pragma: no cover - API/network failure path
            error_text = str(exc)
            _logger.error("Gemini generation failed (attempt %s): %s", attempt + 1, error_text)

            if any(marker in error_text.lower() for marker in (
                'dns',
                'name or service not known',
                'temporary failure in name resolution',
                'could not contact dns servers',
            )):
                _logger.warning("Gemini request aborted early due to DNS resolution failure.")
                break

        backoff = min(backoff * 2, GEMINI_BACKOFF_MAX_S)
        _GEMINI_NEXT_AVAILABLE = time.time() + backoff
        _sleep_until(_GEMINI_NEXT_AVAILABLE)

    return ''


_TAG_CANONICAL_PATTERN = re.compile(r'[^a-z0-9]+')


def _canonical_tag_name(tag: Any) -> str:
    """Normalise tag identifiers so aggregates treat stylistic variants equally."""
    if tag is None:
        return ''
    try:
        text = str(tag).strip().lower()
    except Exception:
        return ''
    if not text:
        return ''
    canonical = _TAG_CANONICAL_PATTERN.sub('_', text)
    canonical = re.sub(r'_+', '_', canonical).strip('_')
    return canonical


def _prettify_tag_label(tag_key: str) -> str:
    if not tag_key:
        return 'Unlabelled'
    if tag_key in _TAG_DISPLAY_OVERRIDES:
        return _TAG_DISPLAY_OVERRIDES[tag_key]
    words = [word for word in tag_key.replace('_', ' ').split(' ') if word]
    if not words:
        return tag_key
    return ' '.join(word.capitalize() for word in words)


def _tag_counter_to_entries(counter: Counter, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = counter.most_common(limit) if limit else counter.most_common()
    return [
        {
            'tag': tag_key,
            'label': _prettify_tag_label(tag_key),
            'count': count,
        }
        for tag_key, count in items
    ]


def _reason_counter_to_entries(counter: Counter, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = counter.most_common(limit) if limit else counter.most_common()
    return [
        {
            'reason': reason,
            'count': count,
        }
        for reason, count in items
    ]


def _category_display_name(category: str) -> str:
    mapping = {
        'attack': 'Attack',
        'defense': 'Defense',
        'in_box': 'In-Box',
    }
    return mapping.get(category, category.replace('_', ' ').title())


def _best_entry(entries: List[Dict[str, Any]], label_key: str) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if entry.get('count'):
            return entry
    return entries[0] if entries else None


def _build_tactical_analysis(category: str,
                             metrics: Dict[str, Any],
                             win_reasons: List[Dict[str, Any]],
                             loss_reasons: List[Dict[str, Any]],
                             positive_tags: List[Dict[str, Any]],
                             negative_tags: List[Dict[str, Any]]) -> Dict[str, str]:
    category_label = _category_display_name(category)
    situation_parts: List[str] = []
    improvement_parts: List[str] = []

    win_rate = metrics.get('win_rate')
    bout_count = metrics.get('bout_count')
    wins = metrics.get('wins')

    if bout_count:
        situation_parts.append(
            f"{category_label} produced a {win_rate:.1f}% win rate across {bout_count} decided touches." if win_rate is not None else
            f"{category_label} logged {bout_count} decided touches."
        )

    top_win = _best_entry(win_reasons, 'reason')
    if top_win and top_win.get('reason'):
        situation_parts.append(
            f"Most successful outcome driver: {top_win['reason']} ({top_win['count']} touches)."
        )

    top_positive = _best_entry(positive_tags, 'label')
    if top_positive and top_positive.get('label'):
        situation_parts.append(
            f"Positive trait showing up most often: {top_positive['label']} ({top_positive['count']} occurrences)."
        )

    top_loss = _best_entry(loss_reasons, 'reason')
    if top_loss and top_loss.get('reason'):
        improvement_parts.append(
            f"Primary loss driver: {top_loss['reason']} ({top_loss['count']} touches)."
        )

    top_negative = _best_entry(negative_tags, 'label')
    if top_negative and top_negative.get('label'):
        improvement_parts.append(
            f"Key risk indicator: {top_negative['label']} ({top_negative['count']} occurrences)."
        )

    if wins and wins > 0 and not situation_parts:
        situation_parts.append(
            f"{category_label} recorded {wins} scoring touches, but lacks clear distinguishing patterns."
        )

    if not improvement_parts:
        improvement_parts.append(
            f"Focus on gathering more {category_label.lower()} sequences to surface consistent improvement targets."
        )

    situation_text = ' '.join(situation_parts) if situation_parts else (
        f"{category_label} data is limited so far; continue analysing matches to expose reliable winning patterns."
    )
    improvement_text = ' '.join(improvement_parts)

    return {
        'situation': situation_text,
        'improvement': improvement_text,
    }


def _generate_category_ai_summary(
    category: str,
    metrics: Dict[str, Any],
    win_reasons: List[Dict[str, Any]],
    loss_reasons: List[Dict[str, Any]],
    positive_tags: List[Dict[str, Any]],
    negative_tags: List[Dict[str, Any]],
    narrative_reference: List[str],
) -> Dict[str, Any]:
    category_label = _category_display_name(category)
    payload = {
        'category': category_label,
        'metrics': metrics,
        'win_reasons': win_reasons,
        'loss_reasons': loss_reasons,
        'positive_tags': positive_tags,
        'negative_tags': negative_tags,
        'insight_snippets': narrative_reference[:6],
    }

    prompt = (
        "You are a world-class fencing performance analyst. Summarize the {category} "
        "phase across multiple matches. Use the structured JSON format below. "
        "Base every statement on the provided data."
        "\n\nData: {data}\n\n"
        "Return strict JSON with keys:"
        "\n- performance_summary (string, 2 sentences highlighting win rate context)"
        "\n- strengths (array of 2-4 short bullet strings)"
        "\n- weaknesses (array of 2-4 short bullet strings)"
        "\n- tactical_focus (string, 2 sentences linking patterns to tactical adjustments)"
        "\n- training_suggestions (array of 3 concrete drill or focus items)"
        "\n- narrative (optional string with additional nuance)"
        "\nDo not include markdown or additional commentary."
    ).format(
        category=category_label,
        data=json.dumps(payload, ensure_ascii=False)
    )

    raw = _call_gemini(prompt, max_tokens=420, temperature=0.25)
    if not raw:
        return {}

    try:
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('`')
            cleaned = cleaned.replace('json\n', '', 1)
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        summary = json.loads(cleaned)
        if isinstance(summary, dict):
            performance_summary = str(summary.get('performance_summary', '')).strip()
            strengths = summary.get('strengths') or []
            weaknesses = summary.get('weaknesses') or []
            tactical_focus = str(summary.get('tactical_focus', '')).strip()
            training = summary.get('training_suggestions') or summary.get('training_plan') or []
            narrative = str(summary.get('narrative', '')).strip()

            def _coerce_list(values: Any) -> List[str]:
                if isinstance(values, list):
                    coerced = []
                    for item in values:
                        if item is None:
                            continue
                        coerced.append(str(item).strip())
                    return [item for item in coerced if item]
                if values:
                    return [str(values).strip()]
                return []

            sanitized = {
                'performance_summary': performance_summary,
                'strengths': _coerce_list(strengths),
                'weaknesses': _coerce_list(weaknesses),
                'tactical_focus': tactical_focus,
                'training_suggestions': _coerce_list(training),
            }
            if narrative:
                sanitized['narrative'] = narrative

            win_rate = metrics.get('win_rate')
            bout_count = metrics.get('bout_count')
            wins = metrics.get('wins')
            decided_text = ''
            if isinstance(bout_count, (int, float)):
                decided_text = f" across {int(bout_count)} decided touches"
            if isinstance(wins, (int, float)) and isinstance(bout_count, (int, float)):
                decided_text += f" ({int(wins)} wins)"
            deterministic_summary = (
                f"{category_label} win rate {float(win_rate):.1f}%{decided_text}."
                if isinstance(win_rate, (int, float))
                else f"{category_label} analysis across recent bouts{decided_text}."
            )
            if performance_summary:
                sanitized['performance_summary'] = f"{deterministic_summary} {performance_summary}"
            else:
                sanitized['performance_summary'] = deterministic_summary
            return sanitized
    except Exception:
        _logger.warning("Failed to parse category AI summary for %s: %s", category, raw[:200])

    return {}


def _safe_json_loads(raw: Optional[str]) -> Optional[Any]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _ensure_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return None


@dataclass
class BoutRecord:
    match_idx: int
    category: Optional[str]
    result: Optional[str]
    tags: List[str]
    positive_tags: List[str]
    negative_tags: List[str]


@dataclass
class MatchRecord:
    upload_id: int
    fencer_side: str
    match_datetime: Optional[datetime]
    match_label: str
    match_title: Optional[str]
    weapon_type: Optional[str]
    is_multi_video: bool
    bouts: List[BoutRecord]
    total_wins: int
    total_losses: int
    total_skips: int
    category_stats: Dict[str, Dict[str, float]]
    video_analysis: Dict[str, Any]
    detailed_analysis: Dict[str, Any]
    reason_summaries: Dict[str, Any]


def _load_bout_summaries(user_id: int, upload_id: int) -> Dict[int, Dict[str, Any]]:
    summaries_path = os.path.join(
        RESULTS_DIR,
        str(user_id),
        str(upload_id),
        'fencer_analysis',
        'bout_summaries.json',
    )
    if not os.path.exists(summaries_path):
        return {}
    try:
        with open(summaries_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = {}
        for bout in data.get('bouts', []):
            idx = bout.get('match_idx')
            if idx is not None:
                result[int(idx)] = bout
        return result
    except Exception as exc:  # pragma: no cover - disk corruption edge case
        _logger.error("Failed reading bout summaries for upload %s: %s", upload_id, exc)
        return {}


def _categorise_bout(summary: Optional[Dict[str, Any]], fencer_side: str) -> Optional[str]:
    if not summary:
        return None
    
    # Prefer touch_classification data which comes from bout_classification.py (authoritative source)
    touch_classification = summary.get('touch_classification')
    if touch_classification:
        category_key = f'{fencer_side}_category'
        category = touch_classification.get(category_key)
        if category and category != 'unknown':
            return category
    
    # Fallback to old logic for backwards compatibility
    bout_type = (summary.get('type') or '').lower()
    if 'in-box' in bout_type or 'inbox' in bout_type or bout_type == 'in_box':
        return 'in_box'
    
    # Check legacy data structure
    side_key = f'{fencer_side}_fencer'
    side_data = summary.get(side_key) or {}
    
    # Also check new data structure (left_data/right_data)
    if not side_data:
        side_data = summary.get(f'{fencer_side}_data') or {}
    
    if side_data.get('is_attacking') is True:
        return 'attack'
    if side_data.get('is_attacking') is False:
        return 'defense'
    
    # fall back to bout type heuristics
    if 'attack' in bout_type:
        return 'attack'
    if 'defense' in bout_type:
        return 'defense'
    return None


def _classify_tags(tag_names: List[str]) -> Tuple[List[str], List[str]]:
    positives, negatives = [], []
    for tag in tag_names:
        canonical = _canonical_tag_name(tag)
        if not canonical:
            continue
        if canonical in _POSITIVE_TAGS:
            positives.append(canonical)
        if canonical in _NEGATIVE_TAGS:
            negatives.append(canonical)
    return positives, negatives


def get_fencer_data(fencer_id: int, user_id: int) -> List[MatchRecord]:
    """Fetch chronological match data for the fencer."""
    from models import Upload, VideoAnalysis, Bout

    uploads = (
        Upload.query.filter(
            (Upload.left_fencer_id == fencer_id) | (Upload.right_fencer_id == fencer_id)
        )
        .filter_by(user_id=user_id, status='completed')
        .outerjoin(VideoAnalysis)
        .order_by(Upload.match_datetime.asc(), Upload.id.asc())
        .all()
    )

    matches: List[MatchRecord] = []

    for upload in uploads:
        fencer_side = 'left' if upload.left_fencer_id == fencer_id else 'right'
        match_dt = upload.match_datetime
        if not match_dt and getattr(upload, 'video_analysis', None):
            match_dt = _ensure_datetime(upload.video_analysis.generated_at)

        label = match_dt.strftime('%Y-%m-%d %H:%M') if match_dt else f'Upload {upload.id}'
        if upload.match_title:
            label = f"{upload.match_title} ({label})"

        summaries = _load_bout_summaries(user_id, upload.id)

        video_analysis = {}
        detailed_analysis = {}
        reason_summaries = {}
        if upload.video_analysis:
            va = upload.video_analysis
            video_analysis = {
                'overall_left': _safe_json_loads(va.left_overall_analysis) or {},
                'overall_right': _safe_json_loads(va.right_overall_analysis) or {},
                'category_left': _safe_json_loads(va.left_category_analysis) or {},
                'category_right': _safe_json_loads(va.right_category_analysis) or {},
                'loss_breakdown': _safe_json_loads(va.loss_analysis) or {},
            }
            detailed_analysis = _safe_json_loads(va.detailed_analysis) or {}
            summary_block = detailed_analysis.get('reason_summary_bullets') or {}
            reason_summaries = summary_block.get(fencer_side, {}) if isinstance(summary_block, dict) else {}

        bouts: List[BoutRecord] = []
        category_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {'wins': 0.0, 'losses': 0.0, 'total': 0.0, 'win_rate': 0.0})
        total_wins = total_losses = total_skips = 0

        # Prefer aggregated stats from cross-bout analysis when available
        cross_summary = None
        cross_bout_path = os.path.join(
            RESULTS_DIR,
            str(user_id),
            str(upload.id),
            'fencer_analysis',
            'cross_bout_analysis.json',
        )
        if os.path.exists(cross_bout_path):
            try:
                with open(cross_bout_path, 'r', encoding='utf-8') as f:
                    cross_data = json.load(f)
                stats_block = cross_data.get('touch_category_analysis', {}).get('statistics', {})
                cross_summary = stats_block.get(f'{fencer_side}_fencer') if stats_block else None
            except Exception as exc:
                _logger.warning("Failed to parse cross-bout stats for upload %s: %s", upload.id, exc)
                cross_summary = None

        if cross_summary:
            for category in ('attack', 'defense', 'in_box'):
                cat_data = cross_summary.get(category, {}) or {}
                wins = float(cat_data.get('wins', 0) or 0)
                losses = float(cat_data.get('losses', 0) or 0)
                count = float(cat_data.get('count', wins + losses))
                stats = category_stats[category]
                stats['wins'] = wins
                stats['losses'] = losses
                stats['total'] = count
                stats['win_rate'] = float(cat_data.get('win_rate', (wins / count * 100 if count else 0)))
                total_wins += int(wins)
                total_losses += int(losses)

            total_bouts_reported = float(cross_summary.get('total_bouts', total_wins + total_losses))
            decided = total_wins + total_losses
            total_skips = max(int(round(total_bouts_reported - decided)), 0)

        bout_rows = (
            Bout.query.filter_by(upload_id=upload.id)
            .order_by(Bout.match_idx.asc())
            .all()
        )

        for bout in bout_rows:
            summary = summaries.get(bout.match_idx)
            category = _categorise_bout(summary, fencer_side)

            # Gather tags for this bout and this fencer
            tag_names: List[str] = []
            for bout_tag in bout.tags:
                if bout_tag.fencer_side == fencer_side and bout_tag.tag:
                    canonical_tag = _canonical_tag_name(bout_tag.tag.name)
                    if canonical_tag:
                        tag_names.append(canonical_tag)
            positive_tags, negative_tags = _classify_tags(tag_names)

            result = None
            if bout.result == 'skip':
                if not cross_summary:
                    total_skips += 1
                result = 'skip'
            elif bout.result in ('left', 'right'):
                if (bout.result == 'left' and fencer_side == 'left') or (
                    bout.result == 'right' and fencer_side == 'right'
                ):
                    if not cross_summary:
                        total_wins += 1
                    result = 'win'
                elif bout.result:
                    if not cross_summary:
                        total_losses += 1
                    result = 'loss'

            if category and not cross_summary:
                stats = category_stats[category]
                if result == 'win':
                    stats['wins'] += 1
                elif result == 'loss':
                    stats['losses'] += 1
                if result in ('win', 'loss'):
                    stats['total'] += 1

            bouts.append(
                BoutRecord(
                    match_idx=bout.match_idx,
                    category=category,
                    result=result,
                    tags=tag_names,
                    positive_tags=positive_tags,
                    negative_tags=negative_tags,
                )
            )

        if not cross_summary:
            for stats in category_stats.values():
                total = stats['total']
                wins = stats['wins']
                stats['win_rate'] = (wins / total * 100) if total else 0.0

        matches.append(
            MatchRecord(
                upload_id=upload.id,
                fencer_side=fencer_side,
                match_datetime=match_dt,
                match_label=label,
                match_title=upload.match_title,
                weapon_type=upload.weapon_type,
                is_multi_video=bool(upload.is_multi_video),
                bouts=bouts,
                total_wins=total_wins,
                total_losses=total_losses,
                total_skips=total_skips,
                category_stats=category_stats,
                video_analysis=video_analysis,
                detailed_analysis=detailed_analysis,
                reason_summaries=reason_summaries,
            )
        )

    return matches


def _compute_overall_totals(matches: List[MatchRecord]) -> Tuple[int, int, int]:
    wins = losses = skips = 0
    for match in matches:
        wins += match.total_wins
        losses += match.total_losses
        skips += match.total_skips
    return wins, losses, skips


def process_executive_summary(
    matches: List[MatchRecord],
    fencer_name: str,
) -> Dict[str, Any]:
    wins, losses, skips = _compute_overall_totals(matches)
    total_decided = wins + losses
    win_rate = (wins / total_decided) * 100 if total_decided else 0.0

    synopsis = ''
    recent_summaries: List[str] = []
    for match in reversed(matches[-3:]):
        analysis = match.video_analysis
        side_key = 'overall_left' if match.fencer_side == 'left' else 'overall_right'
        overall = analysis.get(side_key) or {}
        if overall:
            recent_summaries.append(json.dumps(overall, ensure_ascii=False))

    if recent_summaries:
        prompt = (
            "You are a world-class fencing analyst summarizing a fencer's recent "
            "performance for a professional report. Based only on the following "
            "AI-generated summaries from their last three matches, write a 4-sentence "
            "executive synopsis. Cover their overall tactical style, their most "
            "prominent strength, their most significant weakness, and conclude with "
            "their recent performance trend. Fencer's name: "
            f"{fencer_name}. Summaries: \n" + "\n".join(recent_summaries)
        )
        synopsis = _call_gemini(prompt, max_tokens=320)

    if not synopsis:
        synopsis = (
            f"{fencer_name} has competed in {len(matches)} tracked matches with a "
            f"current win rate of {win_rate:.1f}%. Key strengths and development "
            "areas will appear once more AI summaries are available."
        )

    return {
        'synopsis': synopsis,
        'overall_score': int(round(win_rate)),
        'wins': wins,
        'losses': losses,
        'skips': skips,
    }


def _aggregate_tags(bouts: List[BoutRecord]) -> Tuple[Counter, Counter, Counter]:
    all_tags = Counter()
    positive = Counter()
    negative = Counter()
    for bout in bouts:
        all_tags.update(bout.tags)
        positive.update(bout.positive_tags)
        negative.update(bout.negative_tags)
    return all_tags, positive, negative


def process_performance_dashboard(matches: List[MatchRecord], bouts: List[BoutRecord]) -> Dict[str, Any]:
    category_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {'wins': 0.0, 'losses': 0.0, 'total': 0.0})

    for match in matches:
        for category, stats in match.category_stats.items():
            aggregate = category_totals[category]
            aggregate['wins'] += stats.get('wins', 0.0)
            aggregate['losses'] += stats.get('losses', 0.0)
            aggregate['total'] += stats.get('total', stats.get('wins', 0.0) + stats.get('losses', 0.0))

    all_tags, positive_tags, negative_tags = _aggregate_tags(bouts)

    style_profile = []
    for category in ('attack', 'defense', 'in_box'):
        totals = category_totals.get(category, {'wins': 0.0, 'losses': 0.0, 'total': 0.0})
        wins = totals['wins']
        losses = totals['losses']
        total = totals['total'] if totals['total'] else wins + losses
        win_rate = (wins / total) * 100 if total else 0.0
        style_profile.append(
            {
                'label': category,
                'count': int(round(total)),
                'win_rate': round(win_rate, 1),
                'wins': int(round(wins)),
                'losses': int(round(losses)),
            }
        )

    top_category = max(style_profile, key=lambda item: (item['count'] >= 5, item['win_rate'], item['count']), default=None)
    weakest_category = min(style_profile, key=lambda item: (item['count'] >= 5, item['win_rate'], -item['count']), default=None)

    return {
        'style_profile': style_profile,
        'key_strengths': {
            'top_category': top_category,
            'positive_tags': _tag_counter_to_entries(positive_tags, limit=3),
            'volume_tags': _tag_counter_to_entries(all_tags, limit=3),
        },
        'key_weaknesses': {
            'lowest_category': weakest_category,
            'negative_tags': _tag_counter_to_entries(negative_tags, limit=3),
        },
        'tag_counts': {
            'positive': _tag_counter_to_entries(positive_tags, limit=8),
            'negative': _tag_counter_to_entries(negative_tags, limit=8),
        },
    }


def process_temporal_analysis(matches: List[MatchRecord]) -> Dict[str, Any]:
    timeline = []
    rolling_wins = rolling_losses = 0
    narrative_prompt_rows = []

    for match in matches:
        total = match.total_wins + match.total_losses
        win_rate = (match.total_wins / total) * 100 if total else 0.0
        cat_rates = {}
        for category, stats in match.category_stats.items():
            decided = stats.get('total', 0.0)
            if 'win_rate' in stats:
                cat_rates[category] = float(stats.get('win_rate', 0.0))
            else:
                wins = stats.get('wins', 0.0)
                cat_rates[category] = (wins / decided) * 100 if decided else 0.0

        point = {
            'match_id': match.upload_id,
            'label': match.match_label,
            'match_datetime': match.match_datetime.isoformat() if match.match_datetime else None,
            'win_rate': round(win_rate, 1),
            'wins': match.total_wins,
            'losses': match.total_losses,
            'skips': match.total_skips,
            'category_win_rates': {k: round(v, 1) for k, v in cat_rates.items()},
        }
        timeline.append(point)
        narrative_prompt_rows.append(point)

        rolling_wins += match.total_wins
        rolling_losses += match.total_losses

    narrative = ''
    if narrative_prompt_rows:
        prompt = (
            "You are writing the temporal trend section for a professional fencing report. "
            "Review the chronological match data below (each entry contains match label, "
            "win rate, wins, losses, and category win rates). Compose an insightful "
            "paragraph (4-5 sentences) explaining performance trends over time, "
            "highlighting improvements, regressions, or streaks that stand out. "
            "Data: " + json.dumps(narrative_prompt_rows, ensure_ascii=False)
        )
        narrative = _call_gemini(prompt, max_tokens=360)

    if not narrative:
        if timeline:
            latest = timeline[-1]
            narrative = (
                "Recent matches show a win rate of "
                f"{latest['win_rate']:.1f}% with continued opportunities to stabilise performance."
            )
        else:
            narrative = "Temporal analysis will populate after additional completed matches."

    return {
        'timeline': timeline,
        'narrative': narrative,
    }


def _extract_reason_counts(report_section: Dict[str, Any], category: str, reason_type: str = 'loss') -> List[Dict[str, Any]]:
    items = report_section.get(category) or []
    counter = Counter()
    for entry in items:
        key = entry.get('reason_label') or entry.get('reason_key') or 'Unspecified'
        if entry.get('reason_type', reason_type) == reason_type:
            counter[key] += 1
    return [{'reason': label, 'count': count} for label, count in counter.most_common(5)]


def _extract_loss_breakdown(loss_breakdown: Dict[str, Any], fencer_side: str, category: str) -> List[Dict[str, Any]]:
    side = loss_breakdown.get(f'{fencer_side}_fencer') or {}
    category_group = side.get(category) or {}
    result = []
    for reason, info in category_group.items():
        count = info.get('count', 0)
        if count:
            result.append({'reason': reason, 'count': count})
    result.sort(key=lambda item: item['count'], reverse=True)
    return result[:5]


def process_tactical_analysis(
    matches: List[MatchRecord],
) -> Dict[str, Any]:
    categories = ['attack', 'defense', 'in_box']
    combined_bouts: Dict[str, List[BoutRecord]] = defaultdict(list)
    for match in matches:
        for bout in match.bouts:
            if bout.category in categories:
                combined_bouts[bout.category].append(bout)

    results: Dict[str, Any] = {}

    # Aggregate category stats from matches (uses same source as performance_dashboard)
    category_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {'wins': 0.0, 'losses': 0.0, 'total': 0.0})
    for match in matches:
        for cat, stats in match.category_stats.items():
            if cat in categories:
                aggregate = category_totals[cat]
                aggregate['wins'] += stats.get('wins', 0.0)
                aggregate['losses'] += stats.get('losses', 0.0)
                aggregate['total'] += stats.get('total', stats.get('wins', 0.0) + stats.get('losses', 0.0))

    for category in categories:
        bouts = combined_bouts.get(category, [])
        
        # Use aggregated stats from category_totals for consistency with performance_dashboard
        totals = category_totals.get(category, {'wins': 0.0, 'losses': 0.0, 'total': 0.0})
        wins = totals['wins']
        count = totals['total'] if totals['total'] else wins + totals['losses']
        win_rate = (wins / count) * 100 if count else 0.0
        
        all_tags, positive_tags, negative_tags = _aggregate_tags(bouts)

        # Collect reason data from analysis payloads
        win_reasons_counter = Counter()
        loss_reasons_counter = Counter()
        narrative_reference = []

        for match in matches:
            win_reports = (
                match.detailed_analysis.get('win_reason_reports', {})
                if match.detailed_analysis else {}
            )
            loss_reports = (
                match.detailed_analysis.get('loss_reason_reports', {})
                if match.detailed_analysis else {}
            )
            side_win = win_reports.get(f"{match.fencer_side}_fencer") or {}
            side_loss = loss_reports.get(f"{match.fencer_side}_fencer") or {}

            for item in _extract_reason_counts(side_win, category, reason_type='win'):
                win_reasons_counter[item['reason']] += item['count']
            for item in _extract_reason_counts(side_loss, category, reason_type='loss'):
                loss_reasons_counter[item['reason']] += item['count']

            loss_breakdown = match.video_analysis.get('loss_breakdown') or {}
            for item in _extract_loss_breakdown(loss_breakdown, match.fencer_side, category):
                loss_reasons_counter[item['reason']] += item['count']

            if match.reason_summaries:
                for outcome_key in ('win', 'loss'):
                    side_summary = match.reason_summaries.get(outcome_key, {})
                    if isinstance(side_summary, dict):
                        narrative_reference.extend(side_summary.get(category, []))

        prompt_blob = {
            'category': category,
            'win_rate': round(win_rate, 1),
            'bout_count': count,
            'top_positive_tags': _tag_counter_to_entries(positive_tags, limit=3),
            'top_negative_tags': _tag_counter_to_entries(negative_tags, limit=3),
            'win_reasons': _reason_counter_to_entries(win_reasons_counter, limit=5),
            'loss_reasons': _reason_counter_to_entries(loss_reasons_counter, limit=5),
        }

        summary_prompt = ''
        if bouts:
            prompt = (
                "Write a concise tactical narrative (3-4 sentences) analysing the "
                f"fencer's {category} performance. Reference the win rate, tag "
                "patterns, and the most common win/loss reasons. Data: "
                + json.dumps(prompt_blob, ensure_ascii=False)
            )
            summary_prompt = _call_gemini(prompt, max_tokens=260)

        if not summary_prompt:
            summary_prompt = (
                f"{category.title()} touches: {count} analysed, win rate {win_rate:.1f}%."
            )

        positive_entries = _tag_counter_to_entries(positive_tags, limit=8)
        negative_entries = _tag_counter_to_entries(negative_tags, limit=8)
        win_entries = _reason_counter_to_entries(win_reasons_counter, limit=8)
        loss_entries = _reason_counter_to_entries(loss_reasons_counter, limit=8)

        metrics = {
            'bout_count': int(round(count)),
            'win_rate': round(win_rate, 1),
            'wins': int(round(wins)),
            'losses': int(round(count - wins)),
        }

        tag_totals = {
            'positive': int(sum(item['count'] for item in positive_entries)),
            'negative': int(sum(item['count'] for item in negative_entries)),
        }

        ai_summary = _generate_category_ai_summary(
            category,
            metrics,
            win_entries,
            loss_entries,
            positive_entries,
            negative_entries,
            narrative_reference,
        )

        results[category] = {
            'narrative': summary_prompt,
            'analysis': _build_tactical_analysis(
                category,
                metrics,
                win_entries,
                loss_entries,
                positive_entries,
                negative_entries,
            ),
            'summary': ai_summary,
            'key_metrics': metrics,
            'reason_breakdown': {
                'win': win_entries,
                'loss': loss_entries,
            },
            'tag_counts': {
                'positive': positive_entries,
                'negative': negative_entries,
            },
            'tag_highlights': {  # backward-compatibility for existing consumers
                'positive': positive_entries[:4],
                'negative': negative_entries[:4],
            },
            'tag_totals': tag_totals,
        }

    return results


def process_signature_patterns(matches: List[MatchRecord]) -> Dict[str, Any]:
    strengths: Counter = Counter()
    risks: Counter = Counter()

    for match in matches:
        loss_breakdown = match.video_analysis.get('loss_breakdown') or {}
        for item in _extract_loss_breakdown(loss_breakdown, match.fencer_side, 'attack'):
            risks[item['reason']] += item['count']
        for item in _extract_loss_breakdown(loss_breakdown, match.fencer_side, 'defense'):
            risks[item['reason']] += item['count']
        for item in _extract_loss_breakdown(loss_breakdown, match.fencer_side, 'in_box'):
            risks[item['reason']] += item['count']

        summaries = match.reason_summaries
        if isinstance(summaries, dict):
            for bullet_list in (summaries.get('win') or {}).values():
                for bullet in bullet_list:
                    strengths[bullet] += 1
            for bullet_list in (summaries.get('loss') or {}).values():
                for bullet in bullet_list:
                    risks[bullet] += 1

    strength_highlights = [
        {'text': text, 'count': count}
        for text, count in strengths.most_common(5)
    ]
    risk_highlights = [
        {'text': text, 'count': count}
        for text, count in risks.most_common(5)
    ]

    return {
        'strengths': strength_highlights,
        'risks': risk_highlights,
    }


def process_actionable_recommendations(
    dashboard: Dict[str, Any],
    tactical: Dict[str, Any],
    signature: Dict[str, Any],
    fencer_name: str,
) -> Dict[str, Any]:
    prompt_payload = {
        'style_profile': dashboard.get('style_profile', []),
        'key_strengths': dashboard.get('key_strengths', {}),
        'key_weaknesses': dashboard.get('key_weaknesses', {}),
        'tactical_overview': tactical,
        'signature_patterns': signature,
    }
    prompt = (
        "You are a professional fencing coach preparing a short action plan for "
        f"{fencer_name}. Based on the aggregated data below, write 3-4 bullet "
        "points with **bold** headers and specific, actionable training focus areas. "
        "Each bullet should include a concrete recommendation referencing the "
        "underlying data trend. Use concise professional tone. Data: "
        + json.dumps(prompt_payload, ensure_ascii=False)
    )
    markdown = _call_gemini(prompt, max_tokens=360)
    if not markdown:
        markdown = (
            "- **Stabilise Tactical Execution**: Focus on balanced drills to maintain win rate.\n"
            "- **Video Review**: Add more completed matches to unlock richer AI guidance."
        )
    return {
        'markdown': markdown,
    }


def generate_professional_report(
    fencer_id: int,
    user_id: int,
    fencer_name: str,
    *,
    base_dir: str = BASE_DIR,
) -> Dict[str, Any]:
    """Generate the professional report JSON and persist it to disk."""
    matches = get_fencer_data(fencer_id, user_id)
    if not matches:
        return {
            'success': False,
            'error': 'No completed matches found for this fencer.',
        }

    all_bouts = [bout for match in matches for bout in match.bouts]

    executive_summary = process_executive_summary(matches, fencer_name)
    dashboard = process_performance_dashboard(matches, all_bouts)
    temporal = process_temporal_analysis(matches)
    tactical = process_tactical_analysis(matches)
    signature = process_signature_patterns(matches)
    recommendations = process_actionable_recommendations(
        dashboard,
        tactical,
        signature,
        fencer_name,
    )

    report_payload = {
        'generated_at': datetime.utcnow().isoformat(),
        'fencer': {
            'id': fencer_id,
            'name': fencer_name,
            'user_id': user_id,
            'match_count': len(matches),
        },
        'executive_summary': executive_summary,
        'performance_dashboard': dashboard,
        'performance_over_time': temporal,
        'tactical_analysis': tactical,
        'signature_patterns': signature,
        'actionable_recommendations': recommendations,
    }

    profile_dir = os.path.join(FENCER_PROFILE_DIR, str(user_id), str(fencer_id))
    os.makedirs(profile_dir, exist_ok=True)
    report_path = os.path.join(profile_dir, 'professional_profile.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    return {
        'success': True,
        'report_path': report_path,
        'matches_processed': len(matches),
    }


def regenerate_fencer_profile_assets(
    fencer_id: int,
    user_id: int,
    fencer_name: str,
) -> Dict[str, Any]:
    """Regenerate plot assets and professional report for a fencer in one step."""
    from flask import has_app_context
    from your_scripts.direct_profile_generator import generate_fencer_profile_directly

    def _execute() -> Dict[str, Any]:
        profile_result: Dict[str, Any] = {}
        report_result: Dict[str, Any] = {}
        errors: List[str] = []

        try:
            profile_result = generate_fencer_profile_directly(fencer_id, user_id, fencer_name)
        except Exception as exc:  # pragma: no cover - safety net for CLI execution
            _logger.error(
                "Direct profile regeneration failed for fencer %s: %s",
                fencer_id,
                exc,
                exc_info=True,
            )
            errors.append(f"Graph regeneration failed: {exc}")

        if profile_result.get('success'):
            try:
                report_result = generate_professional_report(fencer_id, user_id, fencer_name)
                if not report_result.get('success'):
                    errors.append(report_result.get('error') or 'Professional report generation failed.')
            except Exception as exc:  # pragma: no cover - API failure path
                _logger.error(
                    "Professional report regeneration failed for fencer %s: %s",
                    fencer_id,
                    exc,
                    exc_info=True,
                )
                errors.append(f"Professional report generation failed: {exc}")
                report_result = {
                    'success': False,
                    'error': str(exc),
                }
        elif profile_result:
            errors.append(profile_result.get('error') or 'Profile graph regeneration failed.')

        profile_success = bool(profile_result.get('success'))
        report_success = bool(report_result.get('success'))
        success = profile_success and report_success

        return {
            'success': success,
            'profile': profile_result,
            'report': report_result,
            'errors': errors,
        }

    if has_app_context():
        return _execute()

    import sys
    from importlib import import_module

    # Ensure we import the Flask app module from project root, not your_scripts.app
    existing_app_module = sys.modules.get('app')
    if existing_app_module and hasattr(existing_app_module, '__file__'):
        module_path = os.path.abspath(existing_app_module.__file__ or '')
        if module_path.startswith(os.path.join(BASE_DIR, 'your_scripts')):
            sys.modules.pop('app', None)

    original_sys_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.join(BASE_DIR, 'your_scripts')]
        create_app = import_module('app').create_app
    finally:
        sys.path = original_sys_path

    app = create_app()
    with app.app_context():
        return _execute()


def main():  # pragma: no cover - CLI helper
    import argparse
    from models import db
    from app import create_app

    parser = argparse.ArgumentParser(description='Generate professional fencer profile report.')
    parser.add_argument('fencer_id', type=int)
    parser.add_argument('user_id', type=int)
    parser.add_argument('fencer_name', type=str)
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        result = generate_professional_report(args.fencer_id, args.user_id, args.fencer_name)
        if result.get('success'):
            print(f"Report generated: {result['report_path']}")
        else:
            print(f"Failed: {result.get('error')}")


if __name__ == '__main__':  # pragma: no cover
    main()
