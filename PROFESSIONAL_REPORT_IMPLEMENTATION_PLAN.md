# Professional Fencer Profile Report: Implementation Plan (v2)

This document outlines the detailed plan for creating a new, professional, data-driven fencer profile report. This version incorporates requirements for temporal analysis and provides hyper-specific details on data processing, aggregation, and display.

---

## Phase 0: Core Infrastructure Modifications

**Objective:** Modify the database and application to capture match dates, enabling temporal analysis.

### 0.1. Database Model Update
-   **File:** `models.py`
-   **Action:** Add a `match_date` column to the `Upload` model.
    -   **Field:** `match_date = db.Column(db.Date, nullable=True)`
    -   **Details:** It will be a `Date` field, not `DateTime`, as only the day is relevant. It's `nullable=True` to ensure backward compatibility with existing uploads that don't have this information.

### 0.2. Frontend Upload Form Update
-   **File:** `templates/upload.html`
-   **Action:** Add a date input field to the upload form.
    -   **Implementation:** Add `<input type="date" name="match_date" class="form-control">` within the main upload form. Make this a required field for new uploads. The label should be "Match Date".

### 0.3. Backend Route Update
-   **File:** `app.py`
-   **Route:** `def handle_single_video_upload()` and `def handle_multi_video_upload()`
-   **Action:** Process the new `match_date` field from the form.
    -   **Logic:**
        1.  Retrieve the date: `match_date_str = request.form.get('match_date')`.
        2.  Validate and convert to a date object: `match_date = datetime.strptime(match_date_str, '%Y-%m-%d').date() if match_date_str else None`.
        3.  If no date is provided, default to the current date.
        4.  Pass `match_date=match_date` when creating the new `Upload` object in the database.

---

## Phase 1: Backend Development - Data Aggregation & Processing

**Objective:** Create a new Python script that gathers all necessary data (chronologically) and processes it into a single, structured JSON file to power the report.

### 1.1. New Report Generator Script
-   **File:** `your_scripts/professional_report_generator.py`
-   **Purpose:** The core engine for the new report. It will be designed to be stateless and callable from a Celery task.
-   **Key Functions:** `generate_professional_report`, `get_fencer_data`, and specific processing functions for each report section.

### 1.2. Chronological Data Retrieval
-   **Function:** `get_fencer_data(fencer_id, user_id)`
-   **Action:** Query and return a comprehensive, chronologically sorted dataset for the fencer.
    -   **Logic:**
        1.  Perform a database query that joins `Upload` and `VideoAnalysis` tables.
        2.  Filter by `fencer_id`, `user_id`, and `status='completed'`.
        3.  **Crucially, order the results by `Upload.match_date.asc()`.**
        4.  For each result, pre-load the related `Bouts` and their associated `Tags`.
        5.  Return a list of rich "match" objects, each containing the `Upload` record, the full `VideoAnalysis` JSON data, and all associated bout/tag information.

### 1.3. Hyper-Specific Data Processing
Each function below will process its section and return a structured dictionary.

#### 1.3.1. Executive Summary (`process_executive_summary`)
-   **Input:** List of all chronological "match" objects.
-   **Processing:**
    1.  **Synopsis Generation:**
        -   Extract the `overall_performance_analysis` JSON from the **most recent 3 matches**.
        -   **Gemini Prompt:** "You are a world-class fencing analyst summarizing a fencer's recent performance for a professional report. Based *only* on the following AI-generated summaries from their last three matches, write a 4-sentence executive synopsis. Cover their overall tactical style, their most prominent strength, their most significant weakness, and conclude with their recent performance trend (e.g., 'showing consistent improvement in defensive actions,' 'struggling to convert attacks lately,' 'maintaining a stable performance level'). Fencer's name: [Fencer Name]. Summaries: [Paste JSON summaries here]."
    2.  **Overall Score Calculation:**
        -   Iterate through every bout for the fencer across all matches.
        -   For each bout, determine if it was a win, loss, or skip for the fencer.
        -   Calculate a simple `win_rate = total_wins / (total_wins + total_losses)`.
        -   The final score will be this win rate scaled to 100 (e.g., `int(win_rate * 100)`).
-   **Output:** `{"synopsis": "...", "overall_score": 78}`.

#### 1.3.2. Performance Dashboard (`process_performance_dashboard`)
-   **Input:** All aggregated bout data.
-   **Processing:**
    1.  **Style Profile:** Tally the counts for 'attack', 'defense', and 'in_box' categories for the fencer across all bouts. Calculate the percentage of each.
    2.  **Key Strengths:**
        -   Find the tactical category ('attack', 'defense', 'in_box') with the highest win rate (min. 5 bouts).
        -   Find the 2 most frequent *positive* tags (e.g., 'good_attack_distance', 'variable_tempo').
    3.  **Key Weaknesses:**
        -   Find the tactical category with the lowest win rate.
        -   Find the 2 most frequent *negative* tags (e.g., 'poor_distance_maintaining', 'excessive_pausing').
-   **Display:**
    -   **Graph:** A donut chart (via Chart.js) for the Style Profile.
    -   **Content:** Bulleted lists for Strengths and Weaknesses, e.g., "Strength: High success rate in Attack (65% win rate)".

#### 1.3.3. Temporal Analysis (`process_temporal_analysis`) - NEW
-   **Input:** The chronologically sorted list of "match" objects.
-   **Processing:**
    1.  Create a data structure: `[{'match_date': date, 'win_rate': float, 'attack_win_rate': float, 'defense_win_rate': float, 'inbox_win_rate': float}, ...]`.
    2.  Iterate through each match, calculate the required win rates for that specific match, and populate the structure.
    3.  **Gemini Prompt:** "Analyze the following performance trends for fencer [Fencer Name] over their last [N] matches. The data shows their overall win rate and tactical win rates per match. Identify any clear upward or downward trends, points of significant change, and summarize their performance trajectory in 2-3 professional sentences. Data: [Paste the data structure as a string]."
-   **Display:**
    -   **Graph 1:** A line chart titled "Overall Win Rate Over Time". X-axis: Match Dates. Y-axis: Win Rate (%).
    -   **Graph 2:** A multi-line chart titled "Tactical Performance Over Time" showing three lines (Attack, Defense, In-Box) for their respective win rates across matches.
    -   **Content:** The Gemini-generated narrative summary of their trends.

#### 1.3.4. Detailed Tactical Analysis (`process_tactical_analysis`)
-   **Structure:** A main dictionary with keys 'offense', 'defense', 'in_box'. Each key holds the processed data for that category.
-   **Processing per category (e.g., Offense):**
    1.  **Narrative:** Extract the relevant category summary (e.g., `left_category_analysis['attack']`) from the `VideoAnalysis` JSON of the **single most recent match**.
    2.  **Key Metrics:** Calculate aggregate metrics for this category across *all* matches (e.g., Overall Attack Win Rate, Avg. Attack Velocity).
    3.  **Win/Loss Patterns:** Aggregate `win_reason_reports` and `loss_reason_reports` from *all* matches. Create a frequency count for each unique `reason_key` within this specific category.
-   **Display:**
    -   **Content:** The AI narrative, followed by a display of the key metrics.
    -   **Graph:** A horizontal grouped bar chart for "Top 3 Win/Loss Patterns". One group for wins, one for losses. Bars represent the frequency of each reason.

#### 1.3.5. Actionable Recommendations (`process_actionable_recommendations`)
-   **Input:** The aggregated top loss patterns and negative habits from previous steps.
-   **Processing:**
    1.  **Priority Focus:** Identify the single loss reason with the highest frequency across all categories. This becomes the priority.
    2.  **Gemini Prompt:** "You are an elite fencing coach writing a report for [Fencer Name]. Their primary area for improvement is '[Priority Focus]'. Their other key issues are [List top 2 other loss patterns/negative habits]. Generate 3 primary recommendations to address these issues. For each recommendation, provide 2 specific, actionable drills with clear instructions, sets, and reps where applicable. Format the response in Markdown with '### Recommendation X' and '#### Suggested Drills' headings."
-   **Display:** The structured Markdown response from Gemini, rendered as formatted text.

### 1.4. Final JSON Structure
The output `professional_profile.json` will be expanded:
```json
{
  // ... executive_summary, performance_dashboard ...
  "temporal_analysis": {
    "narrative": "...",
    "win_rate_over_time_data": [...], // Data for the chart
    "tactical_rates_over_time_data": [...] // Data for the chart
  },
  "tactical_analysis": {
    "offense": {
      "narrative": "...",
      "key_metrics": {"win_rate": 65.5, "avg_velocity": 3.2},
      "win_loss_patterns_data": { // Data for the bar chart
        "win_reasons": [{"reason": "...", "count": 8}],
        "loss_reasons": [{"reason": "...", "count": 12}]
      }
    },
    // ... defense, in_box ...
  },
  // ... signature_patterns, actionable_recommendations ...
}
```

---

## Phase 2 & 3: Backend Task and Frontend Visualization

These phases remain structurally similar but will be adapted to the new data.

### 2.1. Celery Task
-   The task `generate_professional_report_task` will now call the new, more detailed generator script.

### 3.1. Frontend Template (`templates/professional_profile.html`)
-   **New Section:** A dedicated "Performance Over Time" section will be added to display the two new line charts (rendered with Chart.js) and the temporal narrative.
-   **Updated Sections:**
    -   The "Tactical Analysis" tabs will now render the new bar charts for win/loss reasons.
    -   The "Recommendations" section will be designed to parse and display the Markdown-formatted response from Gemini cleanly.
-   **Layout:** The report will be a single-page view, with clear cards/sections for each phase of the analysis, starting with the high-level summary and progressively getting more detailed.

---

## Phase 4: Integration and Testing

-   This phase remains the same, but testing will now need to cover the new temporal analysis features and ensure that the chronological data processing is correct. A test case with a fencer who has competed in matches out of chronological upload order will be essential.