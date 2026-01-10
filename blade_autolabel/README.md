Blade Auto-Labeling

This tool auto-labels blade tip keypoints using a single click per blade on
an init frame. It uses SAM to get initial masks, XMem to track them, and then
extracts the tip from each mask per frame.

Example:
  python /workspace/Project/blade_autolabel/blade_autolabel.py \
    --video /path/to/video.mp4 \
    --output-dir /path/to/output \
    --left-tip 320,220 \
    --right-tip 960,240 \
    --output-video /path/to/output/blade_overlay.mp4

If you omit --left-tip or --right-tip, the script will open the init frame
and let you click left then right blade tips.

Output:
  /path/to/output/blade_tip_keypoints.csv
