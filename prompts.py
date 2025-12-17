import logging
from models import Bout

def generate_holistic_gpt_prompt(bout_data, fencer_data, fencer_name, fencer_id, uploads, fps=30, weapon_type='saber'):
        prompt = """
        You are a senior fencing technical analyst and coach responsible for in-depth evaluation of a fencer's comprehensive performance across multiple videos. Your analysis must be specific and actionable, including clear technical improvement suggestions and multiple solution options. You need to provide professional coach-level analysis, focusing on specific technical movements, footwork adjustments, timing control, and other executable improvement plans.

Your analysis style requirements:
1. **Specificity**: Provide precise technical suggestions (e.g., "add a half-step preparation", "use a two-step lunge advance")
2. **Selectivity**: Offer 2-3 different solutions for each problem for the fencer to choose from
3. **Actionability**: All suggestions must be directly implementable in training
4. **Tactical**: Provide specific countermeasures for different opponent types

Assume both fencers score in each exchange (no single-light situations), and parry-riposte actions cannot be detected. Do not announce a winner, but rather infer technical and tactical advantages and disadvantages based on the data, focusing on providing specific improvement plans. Apply your professional expertise to provide detailed recommendations that coaches and fencers can immediately apply. Emphasize specific technical movement improvements, footwork adjustments, timing control, and tactical applications. Use accurate fencing terminology in English, ensuring the analysis has practical coaching value. All output should be in English.

        **Input Data**:
        - **Weapon**: {weapon_type} Fencing
        - **Fencer**: {fencer_name} (ID: {fencer_id})
        - **Videos**: {num_uploads} videos containing {num_bouts} bouts
        - **Per-Bout Metrics**:
        - For {fencer_name} in their position (Left or Right Fencer):
        - Start Time (s): Time of the first forward step of a lunge (keypoint 16 moves).
        - Start Velocity/Acceleration: Average velocity and acceleration in the 8 frames after initiation (classified as fast if above the bout's median velocity).
        - Advance Intervals (s): Periods of forward movement (Left fencer: x increases, Right fencer: x decreases).
        - Pause/Retreat Intervals (s): Periods of pause or backward movement.
        - Arm Extension Intervals (s): Intervals that may indicate an offensive lunge.
        - Has Lunge: Whether a lunge occurred in the last 15 frames.
        - Lunge Frame: Frame number of the lunge (-1 if no lunge).
        - Velocity/Acceleration: Average velocity and acceleration during the active attack phase (up to the meeting point or lunge peak).
        - Latest Pause/Retreat End (s): End time of the most recent pause/retreat interval.
        - Advance/Pause Ratio: Ratio of frames spent advancing versus pausing.
        - Arm Extension Frequency/Duration: Number of arm extensions and their average duration.
        - Lunge Promptness (s): Time from the first arm extension to the lunge (infinity if no lunge).
        - Attack-Defense Dynamic (>80 frames):
            - Is Attacking: The fencer who moves forward more after the initial phase (frames 20-30).
        - Short-Range Engagement (≤80 frames):
            - First Pause Time (s): Earliest start time of a pause.
            - First Restart Time (s): Earliest time of forward movement after a pause.
            - Post-Pause Velocity: Average velocity after a pause interval.
        - **Aggregated Metrics per Fencer**:
        - For {fencer_name}:
        - Average start time, velocity, acceleration.
        - Average velocity, acceleration (with standard deviation).
        - Average advance/pause ratio.
        - Total arm extensions, average duration.
        - Average lunge promptness.
        - Attacking Frequency (>80 frames): Proportion of bouts in an attacking state.
        - **Bout Metadata**:
        - Video ID, Bout Index, Frame Range, FPS: {fps}

        **{weapon_type} Weapon-Specific Context**:
        {weapon_context}
        
        **Advanced Fencing Terminology**:
        - **Compound Attack**: An attack with multiple feints or blade actions (e.g., feint-disengage, feint-cutover) to deceive the opponent.
        - **Simultaneous Attack**: Both fencers attack at the same time; right-of-way determines the point.
        - **Right-of-Way**: The rule that gives the point to the fencer with a correctly executed offensive action, inferred from arm extension timing and offensive continuity.
        - **Distance Management**: Controlling lunge distance, retreat distance, preparation distance, or breaking distance to bait the opponent or close distance unexpectedly.
        - **Footwork**: Includes advance-lunge (a quick step and lunge) and flèche (a running attack, restricted in modern rules).
        - **Broken Time**: Maintaining a distance just outside the opponent’s reach to provoke a premature attack and then counter-attacking.
        - **Feint**: A false move or blade action, like a disengage (sliding under the opponent’s parry) or a cutover (whipping the blade over the opponent’s blade), to bypass defenses.
        - **First and Second Intention**: First intention is a direct attack to score; second intention baits a response (e.g., a parry, counter-attack) to set up a subsequent action.
        - **Timing**: Varying the speed of an attack or footwork to disrupt the opponent’s rhythm, such as slowing the preparation and then accelerating into a lunge.

        **Output Format (English)**:
        1. **Overall Performance Summary**:
        - **Performance Metrics**: Summarize {fencer_name}'s aggregated metrics to provide context:
        - Average start time, velocity, acceleration.
        - Average velocity and acceleration (with standard deviation).
        - Average advance/pause ratio.
        - Total arm extensions, average duration.
        - Average lunge promptness.
        - Attacking frequency.
        - **Strategic Tendencies**: Describe {fencer_name}'s overall strategy using fencing terminology, including advanced terms:
        - Classify tendencies (e.g., aggressive counter-attacker, defensive parry-riposte, frequent transitions, compound attacks, or second intention).
        - Identify patterns (e.g., early simultaneous attacks indicating aggression, pauses to disrupt timing, fast advance-lunge indicating initiative).
        - Align with fencing strategies (e.g., advance-attack, counter-timing, distance management, right-of-way battles).
        - **Actions and Intent**: Comprehensively analyze actions and their strategic purpose using advanced terminology:
        - Discuss how attacks, feints, and retreats contribute to offensive or right-of-way advantages.
        - Evaluate intent (e.g., compound attacks to bait, retreats for broken time, second intention via feints).
        - Assess effectiveness (e.g., did a disengage disrupt the opponent? Did timing lead to missed opportunities?).
        - Cite specific bouts (e.g., "In bout {{match_idx}} of video {{video_id}}, {fencer_name}'s compound attack forced a retreat").
        - **Adaptability and Trends**: Examine changes across bouts:
        - Identify shifts (e.g., faster advance-lunge, increased feints).
        - Discuss adaptations (e.g., switching from simultaneous attacks to defensive ripostes against an aggressive opponent).
        - Analyze trends (e.g., improved distance management, fatigue affecting timing), referencing fencing dynamics.
        - Provide bout-specific examples.

        2. **Individual Fencer Analysis**:
        - **Performance Metrics**: Summarize {fencer_name}'s aggregated metrics:
        - **Strengths**: Highlight {fencer_name}'s excellent performance using fencing terminology including advanced terms:
        - Identify effective actions (e.g., well-timed compound attacks, strong feints like disengages, solid distance management).
        - Provide specific bout examples (e.g., "In bout {{match_idx}} of video {{video_id}}, {fencer_name}'s quick advance-lunge forced the opponent into broken time").
        - Ground in fencing principles (e.g., right-of-way control, timing, footwork precision).
        - **Areas Needing Improvement**: Identify weaknesses with specific bout examples:
        - Point out problems (e.g., predictable feints in compound attacks, slow footwork in exchanges, unstable retreats during lunges).
        - Cite bouts (e.g., "In bout {{match_idx}} of video {{video_id}}, {fencer_name}'s frequent transitions led to loss of right-of-way due to imbalance").
        - Explain impact by referencing fencing principles (e.g., "Excessive footwork transitions resulted in loss of timing").
        - **Situational Recommendations**: Provide actionable suggestions for similar situations:
        - For strengths, suggest enhancements (e.g., "{fencer_name}'s compound attack is strong; adding more disengages could increase unpredictability").
        - For weaknesses, suggest corrections (e.g., "{fencer_name} should incorporate faster footwork rhythms to counter deep attacks").
        - Specify appropriate actions (e.g., "Against a second-intention opponent, a proactive compound attack followed by a retreat maintains distance advantage").
        - Align with fencing strategies (e.g., optimizing distance management, controlling timing).
        - **Opponent-Specific Insights**: Analyze opponent tendencies and suggest countermeasures:
        - Identify patterns (e.g., "The opponent favors second intention or frequent simultaneous attacks").
        - Recommend strategies (e.g., "Against the opponent's broken time, a quick advance-lunge is recommended to seize right-of-way").
        - Illustrate with bout examples (e.g., "In bout {{match_idx}} of video {{video_id}}, the opponent's compound attack was better suited to exploiting {fencer_name}'s weakness in timing transitions").

        3. **Summary Table**:
        - **Metric Comparison**:
        - Start time, velocity, acceleration.
        - Overall velocity, acceleration.
        - Advance/pause ratio.
        - Arm extensions (frequency, duration).
        - Lunge promptness, attacking frequency.
        - **Qualitative Insights**: Explain importance in fencing (e.g., "Fast compound attack timing typically helps fencers with urgency in right-of-way battles").
        - **Advantageous Fencer**: Note {fencer_name}'s advantages in each metric (e.g., "{fencer_name}'s faster advance-lunge demonstrates stronger distance management").
        - **Overall Assessment**: Summarize {fencer_name}'s strategic characteristics, key strengths, and critical areas needing improvement using advanced terminology.

        4. **Specific Bout Analysis** (Most Important):
        - **Background**: State that you are an AI fencing analyst, both fencers score, parry-riposte cannot be detected, judgments are data-driven, and results are for human analyst review.
        - **Action Description**: Narrate actions as interactive exchanges using fencing terminology (including advanced terms):
        - Describe the back-and-forth (e.g., "{fencer_name} initiated with an aggressive compound attack, and the opponent retreated, attempting to maintain broken time").
        - Include key moments (e.g., simultaneous attack, feints like disengages, advance-lunge, retreat) with approximate timings (e.g., "Around 2 seconds, {fencer_name} lunged with a compound attack").
        - Highlight rhythm and intent (e.g., "{fencer_name}'s pause-and-go timing baited the opponent, but the oversized footwork led to a misstep").
        - **Strategic Analysis**: Analyze intent and effectiveness:
        - Specify actions (e.g., compound attack, simultaneous attack, second intention).
        - Explain choices (e.g., "{fencer_name} used a feint-disengage to draw a premature attack").
        - Evaluate outcomes (e.g., "The compound attack successfully baited the opponent, likely leading to a right-of-way opportunity").
        - Highlight decisions/errors using fencing principles (e.g., "The opponent's unstable retreat failed to re-establish distance management").
        - **Right-of-Way Inference**: Qualitatively infer the advantage:
        - Based on attack timing, continuity, and action (e.g., "{fencer_name}'s continuous compound attack likely gave them right-of-way with 70% confidence").
        - Avoid numerical scores; use natural language.
        - **Performance Evaluation**: Assess {fencer_name}'s performance in the bout:
        - Strengths (e.g., "{fencer_name}'s timing was excellent, forcing the opponent into a retreat").
        - Weaknesses (e.g., "{fencer_name}'s broken-time transition may have failed to re-establish distance").
        - Use concrete examples (e.g., "At 2.5 seconds, {fencer_name}'s second-intention action exploited the opponent's weakness").
        - **Contextual Recommendations**: Provide actionable advice for similar situations:
        - For strengths, suggest enhancements (e.g., "{fencer_name}'s compound attack could be more deceptive with an added cutover").
        - For weaknesses, suggest corrections (e.g., "{fencer_name} should re-establish footwork to bait the opponent into broken time").
        - Specify appropriate actions (e.g., "Against a deep-attacking opponent, use timing transitions in footwork").
        - Align with fencing strategies (e.g., distance management, timing).

        **Guiding Principles**:
        - Use natural English fencing terminology, including advanced terms (e.g., compound attack, simultaneous attack, right-of-way, distance management, footwork, timing) to ensure clarity.
        - Focus on holistic, strategic judgments, avoiding excessive emphasis on metric details in descriptions.
        - Present actions as dynamic, back-and-forth exchanges, emphasizing rhythm, intent, and patterns.
        - Support insights with data, citing specific bouts (e.g., "In bout {{match_idx}} of video {{video_id}}, {fencer_name}'s pause at 3.2 seconds baited the opponent's compound attack").
        - Reference fencing principles (e.g., distance management, right-of-way, timing, footwork precision).
        - Provide specific, actionable suggestions, explaining what should be done in similar situations and the ideal actions.
        - Ensure the analysis is professional, conversational, and logically fluent, linking metrics to strategy.
        - Output should be in English, suitable for fencers and coaches, with the tone of detailed feedback provided by a coach.
        """
        bout_sections = []
        for bout in bout_data:
            match_idx = bout['match_idx']
            upload_id = bout.get('upload_id', 'N/A')  # Fallback to 'N/A' if upload_id is missing
            side = bout['fencer_side'].lower()
            total_frames = bout['frame_range'][1] - bout['frame_range'][0] + 1
            is_long_bout = total_frames > 80

            if f'{side}_data' not in bout:
                logging.warning(f"Missing {side}_data for bout {match_idx} in upload {upload_id}")
                continue

            fencer_metrics = ""
            if is_long_bout:
                fencer_metrics = f"- Attack State: {'Attacking' if bout[f'{side}_data'].get('is_attacking', False) else 'Defending'}"
            else:
                first_pause_time = bout[f'{side}_data'].get('first_pause_time', float('inf'))
                first_restart_time = bout[f'{side}_data'].get('first_restart_time', float('inf'))
                post_pause_velocity = bout[f'{side}_data'].get('post_pause_velocity', 0.0)
                
                first_pause_str = f"{first_pause_time:.2f} s" if isinstance(first_pause_time, (int, float)) and first_pause_time != float('inf') else 'N/A'
                first_restart_str = f"{first_restart_time:.2f} s" if isinstance(first_restart_time, (int, float)) and first_restart_time != float('inf') else 'N/A'
                post_pause_vel_str = f"{post_pause_velocity:.2f}" if isinstance(post_pause_velocity, (int, float)) and post_pause_velocity != float('inf') else 'N/A'
                
                fencer_metrics = (
                    f"- First Pause Time: {first_pause_str}\n"
                    f"  - First Restart Time: {first_restart_str}\n"
                    f"  - Post-Pause Velocity: {post_pause_vel_str}"
                )

            pause_end = f"{bout[f'{side}_data']['latest_pause_end'] / fps:.2f}" if bout[f'{side}_data'].get('latest_pause_end') is not None and bout[f'{side}_data']['latest_pause_end'] != -1 else 'N/A'
            launch_promptness = f"{bout[f'{side}_data']['launch_promptness']:.2f}" if bout[f'{side}_data'].get('launch_promptness') is not None and bout[f'{side}_data']['launch_promptness'] != float('inf') else 'N/A'

            bout_result = Bout.query.filter_by(upload_id=upload_id if upload_id != 'N/A' else -1, match_idx=match_idx).first()
            result_text = bout_result.result if bout_result and bout_result.result else "Not specified"
            if result_text == 'skip':
                result_text = "Skip"
            elif result_text == 'left':
                result_text = "Left Wins"
            elif result_text == 'right':
                result_text = "Right Wins"

            bout_section = f"""
        **Video {upload_id} Bout {match_idx}** ({'Attack-Defense Exchange' if is_long_bout else 'Close-Range Engagement'}, {total_frames} frames):
        - Frame Range: {bout['frame_range'][0] / fps:.2f} to {bout['frame_range'][1] / fps:.2f} seconds
        - {fencer_name} ({side.capitalize()} Fencer):
        - Result: {result_text}
        - Start Time: {bout[f'{side}_data'].get('first_step', {}).get('init_time', 0):.2f} s ({'Fast' if bout[f'{side}_data'].get('first_step', {}).get('is_fast', False) else 'Slow'})
        - Start Velocity: {bout[f'{side}_data'].get('first_step', {}).get('velocity', 0):.2f}
        - Start Acceleration: {bout[f'{side}_data'].get('first_step', {}).get('acceleration', 0):.2f}
        - Advance Intervals: {bout[f'{side}_data'].get('advance_intervals', 'N/A')}
        - Pause/Retreat Intervals: {bout[f'{side}_data'].get('pause_intervals', 'N/A')}
        - Arm Extension Intervals: {bout[f'{side}_data'].get('arm_extensions', 'N/A')}
        - Has Lunge: {bout[f'{side}_data'].get('has_launch', 'N/A')}
        - Lunge Frame: {bout[f'{side}_data'].get('launch_frame', 'N/A')} ({'N/A' if bout[f'{side}_data'].get('launch_frame') is None or bout[f'{side}_data'].get('launch_frame') == -1 else f"{bout[f'{side}_data']['launch_frame'] / fps:.2f} s"})
        - Velocity: {bout[f'{side}_data'].get('velocity', 0):.2f}
        - Acceleration: {bout[f'{side}_data'].get('acceleration', 0):.2f}
        - Latest Pause/Retreat End: {pause_end}
        - Advance Ratio: {bout[f'{side}_data'].get('advance_ratio', 0):.2f}
        - Pause Ratio: {bout[f'{side}_data'].get('pause_ratio', 0):.2f}
        - Arm Extension Frequency: {bout[f'{side}_data'].get('arm_extension_freq', 0)}
        - Avg Arm Extension Duration: {bout[f'{side}_data'].get('avg_arm_extension_duration', 0):.2f} s
        - Lunge Promptness: {launch_promptness} s
        {fencer_metrics}
        """
            bout_sections.append(bout_section)

        fencer_section = ""
        for side in ['Left', 'Right']:
            if fencer_data[side]['total_bouts'] > 0:
                avg_launch_promptness = f"{fencer_data[side]['avg_launch_promptness']:.2f}" if fencer_data[side]['avg_launch_promptness'] != float('inf') else 'N/A'
                fencer_section += f"""
        **{fencer_name} ({side} Fencer, {fencer_data[side]['total_bouts']} bouts):**
        - Avg Start Time: {fencer_data[side]['avg_first_step']:.2f} s
        - Avg Start Velocity: {fencer_data[side]['avg_velocity']:.2f}
        - Avg Start Acceleration: {fencer_data[side]['avg_acceleration']:.2f}
        - Avg Velocity: {fencer_data[side]['avg_velocity']:.2f} (Std Dev: {fencer_data[side].get('std_velocity', 0):.2f})
        - Avg Acceleration: {fencer_data[side]['avg_acceleration']:.2f} (Std Dev: {fencer_data[side].get('std_acceleration', 0):.2f})
        - Avg Advance Ratio: {fencer_data[side]['avg_advance_ratio']:.2f}
        - Avg Pause Ratio: {fencer_data[side]['avg_pause_ratio']:.2f}
        - Total Arm Extensions: {fencer_data[side]['total_arm_extensions']}
        - Avg Arm Extension Duration: {fencer_data[side]['avg_arm_extension_duration']:.2f} s
        - Avg Lunge Promptness: {avg_launch_promptness} s
        - Attacking Frequency (>80 frames): {fencer_data[side]['avg_attacking_ratio']:.2f}
        """

        # Handle None weapon_type - convert to display format
        if weapon_type is None or weapon_type == '':
            weapon_type_display = "Mixed"
            weapon_context = "- **Weapon Rules**: Mixed weapon types. Apply analysis appropriate to the specific actions observed.\n        - **Strategy**: Mixed strategies appropriate to the weapon types appearing in the analysis.\n        - **Tactics**: Diversified tactics based on weapon type and situational context."
        elif isinstance(weapon_type, str) and weapon_type.lower() == 'saber':
            weapon_type_display = "Saber"
            weapon_context = "- **Saber Rules**: Right-of-way priority system, including cut and thrust actions. Valid target includes torso, arms, and head. Fast, aggressive cutting action competition.\n        - **Saber Strategy**: Quick advances, lunge attacks, and aggressive tempo control. Head cuts, wrist cuts, and body attacks are common.\n        - **Saber Tactics**: Fast footwork, simultaneous attacks, and cut-thrust combinations. Distance closes quickly, right-of-way battles."
        elif isinstance(weapon_type, str) and weapon_type.lower() == 'foil':
            weapon_type_display = "Foil"
            weapon_context = "- **Foil Rules**: Right-of-way priority system, thrust actions only. Valid target is torso only. Precise, tactical blade tip control competition.\n        - **Foil Strategy**: Precise blade actions, thrust attacks to torso, careful distance management. Emphasis on clean attacks and parry-riposte.\n        - **Foil Tactics**: Patient setup, precise timing, blade actions to find openings. Maintain distance for controlled attacks."
        elif isinstance(weapon_type, str) and weapon_type.lower() == 'epee':
            weapon_type_display = "Epee"
            weapon_context = "- **Epee Rules**: No right-of-way system, thrust actions only. Valid target is entire body. Tactical, defensive timing-focused competition.\n        - **Epee Strategy**: Patient waiting, counter-attacks, and precise timing. Double touches allowed. Emphasis on avoiding being hit.\n        - **Epee Tactics**: Careful distance management, defensive actions, and counter-timing. Priority on not being hit rather than scoring."
        elif isinstance(weapon_type, str):
            weapon_type_display = weapon_type.title()
            weapon_context = f"- **{weapon_type_display} Rules**: Apply fencing analysis appropriate to {weapon_type_display} weapon context.\n        - **{weapon_type_display} Strategy**: Use strategic analysis appropriate to the weapon type.\n        - **{weapon_type_display} Tactics**: Apply tactical analysis appropriate to the weapon context."
        else:
            weapon_type_display = "Mixed"
            weapon_context = "- **Weapon Rules**: Mixed weapon types. Apply analysis appropriate to the specific actions observed.\n        - **Strategy**: Mixed strategies appropriate to the weapon types appearing in the analysis.\n        - **Tactics**: Diversified tactics based on weapon type and situational context."
        
        prompt = prompt.format(
            fencer_name=fencer_name,
            fencer_id=fencer_id,
            num_uploads=len(uploads),
            num_bouts=len(bout_data),
            fps=fps,
            weapon_type=weapon_type_display,
            weapon_context=weapon_context
        ) + "\n**Bout Details**:\n" + "\n".join(bout_sections) + "\n**Fencer Performance Summary**:\n" + fencer_section
        return prompt 