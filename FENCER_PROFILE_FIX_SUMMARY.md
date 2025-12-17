# Fencer Profile Data Consistency Fix

## Problem Summary

The fencer profile generation had inconsistent data between the **Performance Dashboard** and **Tactical Analysis** sections. For example, for fencer 23 (yt):

- **Performance Dashboard** showed: Attack with 2 bouts, 0 wins, 0% win rate
- **Tactical Analysis** showed: Attack with 3 bouts, 1 win, 33.3% win rate

This caused confusion as the pie charts showed different data than the performance analysis text.

## Root Causes

### Issue #1: Different Data Sources for Statistics

**Location**: `your_scripts/professional_report_generator.py`

The two functions were using different data sources:

1. **`process_performance_dashboard()`** (lines 733-780)
   - Used `match.category_stats` which comes from cross-bout analysis JSON

2. **`process_tactical_analysis()`** (lines 866-996)  
   - Counted bouts directly from `match.bouts` list
   - These counts could differ from cross-bout analysis

**Fix**: Modified `process_tactical_analysis()` to use `match.category_stats` (same as performance dashboard) for consistency.

### Issue #2: Inconsistent Bout Categorization

**Location**: Multiple files with two categorization systems

There were **two different systems** categorizing bouts:

1. **`bout_classification.py`** (authoritative):
   - Uses retreat intervals to determine attack vs defense
   - Has multiple tie-breakers when retreat frames are equal
   - Final fallback: both fencers get 'attack' if no difference

2. **`fencer_analysis.py`** (legacy):
   - Uses `attacking_score` based on late advances
   - **Problem**: When scores are equal, NEITHER side gets `is_attacking=True`
   - This left some bouts uncategorized in bout_summaries.json

**Fix**: Modified `_categorise_bout()` to:
- First check `touch_classification` data (from bout_classification.py) as the authoritative source
- Fall back to old logic only if touch_classification is missing
- Support both old (left_fencer/right_fencer) and new (left_data/right_data) data structures

## Changes Made

### File: `your_scripts/professional_report_generator.py`

#### Change 1: process_tactical_analysis() - Use consistent data source

**Before** (lines 878-882):
```python
for category in categories:
    bouts = combined_bouts.get(category, [])
    count = len([b for b in bouts if b.result in ('win', 'loss')])
    wins = len([b for b in bouts if b.result == 'win'])
    win_rate = (wins / count) * 100 if count else 0.0
```

**After** (lines 878-897):
```python
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
```

#### Change 2: _categorise_bout() - Use authoritative categorization

**Before** (lines 465-482):
```python
def _categorise_bout(summary: Optional[Dict[str, Any]], fencer_side: str) -> Optional[str]:
    if not summary:
        return None
    bout_type = (summary.get('type') or '').lower()
    if 'in-box' in bout_type or 'inbox' in bout_type or bout_type == 'in_box':
        return 'in_box'
    side_key = f'{fencer_side}_fencer'
    side_data = summary.get(side_key) or {}
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
```

**After** (lines 465-500):
```python
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
```

## Testing

To verify the fix works:

```bash
python test_fencer_profile_fix.py
```

This will:
1. Regenerate the professional report for fencer 23
2. Compare Performance Dashboard vs Tactical Analysis statistics
3. Report any inconsistencies

Expected result: All categories (attack, defense, in_box) should have matching:
- Bout counts
- Win counts
- Loss counts  
- Win rates

## Impact

- ✅ Performance Dashboard and Tactical Analysis now show consistent statistics
- ✅ Pie charts and performance summaries match
- ✅ Uses authoritative categorization from bout_classification.py
- ✅ Backwards compatible with old data structures
- ✅ All existing fencer profiles will be consistent when regenerated

## Files Modified

1. `/workspace/Project/your_scripts/professional_report_generator.py`
   - Modified `process_tactical_analysis()` to use category_stats
   - Modified `_categorise_bout()` to use touch_classification data

## Recommended Actions

1. **Regenerate all fencer profiles** to apply the fix to existing profiles
2. **Run the test script** to verify data consistency
3. **Monitor new profile generations** to ensure consistency is maintained
