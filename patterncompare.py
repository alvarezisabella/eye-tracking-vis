import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import numpy as np
from collections import Counter

# AOI mapping with full instrument names
AOI_NAMES = {
    'A': 'No AOI',
    'B': 'Altitude/VSI',
    'C': 'Attitude Indicator',
    'D': 'Turn/Heading',
    'E': 'Speed/Slip Indicator',
    'F': 'Airspeed Indicator',
    'G': 'RPM',
    'H': 'Window (Outside)'
}

print("Loading flight pattern data...")
print("=" * 80)

# Read CSV files
success_df = pd.read_csv('successpatterns.csv', encoding='utf-8')
fail_df = pd.read_csv('failpatterns.csv', encoding='utf-8')

print(f"✓ Successful pilots: {len(success_df)} patterns loaded")
print(f"✓ Unsuccessful pilots: {len(fail_df)} patterns loaded")

# Get column names
pattern_col = 'Pattern String' if 'Pattern String' in success_df.columns else success_df.columns[0]
freq_col = 'Frequency' if 'Frequency' in success_df.columns else success_df.columns[1]
avg_freq_col = 'Average Pattern Frequency' if 'Average Pattern Frequency' in success_df.columns else success_df.columns[
    3]


# Analysis functions
def count_aoi_occurrences(df, pattern_column, freq_column):
    """Count weighted AOI occurrences"""
    aoi_counts = {aoi: 0 for aoi in AOI_NAMES.keys()}
    for _, row in df.iterrows():
        pattern = str(row[pattern_column])
        freq = row[freq_column]
        for char in pattern:
            if char in aoi_counts:
                aoi_counts[char] += freq
    return aoi_counts


def analyze_pattern_characteristics(df, pattern_column, freq_column):
    """Analyze pattern characteristics"""
    patterns = df[pattern_column].values
    frequencies = df[freq_column].values

    # Repetitive patterns (same AOI repeated)
    repetitive = sum(freq for pattern, freq in zip(patterns, frequencies)
                     if len(set(str(pattern))) == 1)

    # Back-and-forth patterns (ABAB, BCBC, etc.)
    back_forth = sum(freq for pattern, freq in zip(patterns, frequencies)
                     if len(str(pattern)) >= 4 and
                     str(pattern)[0] == str(pattern)[2] and
                     str(pattern)[1] == str(pattern)[3])

    # Systematic scans (increasing diversity)
    systematic = sum(freq for pattern, freq in zip(patterns, frequencies)
                     if len(set(str(pattern))) >= 3)

    total = sum(frequencies)
    return {
        'repetitive': (repetitive / total) * 100,
        'back_forth': (back_forth / total) * 100,
        'systematic': (systematic / total) * 100
    }


def extract_transitions(df, pattern_column, freq_column):
    """Extract AOI-to-AOI transitions"""
    transitions = Counter()
    for _, row in df.iterrows():
        pattern = str(row[pattern_column])
        freq = row[freq_column]
        for i in range(len(pattern) - 1):
            transitions[(pattern[i], pattern[i + 1])] += freq
    return transitions


# Perform analysis
success_aoi = count_aoi_occurrences(success_df, pattern_col, freq_col)
fail_aoi = count_aoi_occurrences(fail_df, pattern_col, freq_col)
success_behaviors = analyze_pattern_characteristics(success_df, pattern_col, freq_col)
fail_behaviors = analyze_pattern_characteristics(fail_df, pattern_col, freq_col)
success_trans = extract_transitions(success_df, pattern_col, freq_col)
fail_trans = extract_transitions(fail_df, pattern_col, freq_col)

# Create comprehensive figure
fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=(
        '<b>Q1: Gaze Behaviors - Successful Pilots</b>',
        '<b>Q1: Gaze Behaviors - Unsuccessful Pilots</b>',
        '<b>Q2: Attention Distribution - Side by Side</b>',
        '<b>Q2: Top Instrument Transitions (Successful)</b>',
        '<b>Q2: Attention Distribution Comparison</b>',
        '<b>Q2: Key Differences in AOI Focus</b>',
        '<b>Q3: Dominant Patterns - Successful</b>',
        None,
        '<b>Q3: Dominant Patterns - Unsuccessful</b>',
        None
    ),
    specs=[
        [{'type': 'bar'}, {'type': 'bar'}],
        [{'type': 'bar'}, {'type': 'bar'}],
        [{'type': 'bar'}, {'type': 'bar'}],
        [{'type': 'bar', 'colspan': 2}, None],
        [{'type': 'bar', 'colspan': 2}, None]
    ],
    row_heights=[0.16, 0.16, 0.18, 0.25, 0.25],
    vertical_spacing=0.10,
    horizontal_spacing=0.12
)

# Q1: GAZE BEHAVIORS
behavior_categories = ['Repetitive<br>Fixations', 'Back-and-Forth<br>Scanning', 'Systematic<br>Multi-AOI Scans']
success_behavior_values = [success_behaviors['repetitive'], success_behaviors['back_forth'],
                           success_behaviors['systematic']]
fail_behavior_values = [fail_behaviors['repetitive'], fail_behaviors['back_forth'], fail_behaviors['systematic']]

fig.add_trace(
    go.Bar(
        x=behavior_categories,
        y=success_behavior_values,
        marker_color=['#27ae60', '#2ecc71', '#52be80'],
        text=[f'{v:.1f}%' for v in success_behavior_values],
        textposition='outside',
        textfont=dict(size=14, color='black'),
        showlegend=False,
        hovertemplate='%{x}<br>%{y:.1f}% of patterns<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=behavior_categories,
        y=fail_behavior_values,
        marker_color=['#c0392b', '#e74c3c', '#ec7063'],
        text=[f'{v:.1f}%' for v in fail_behavior_values],
        textposition='outside',
        textfont=dict(size=14, color='black'),
        showlegend=False,
        hovertemplate='%{x}<br>%{y:.1f}% of patterns<extra></extra>'
    ),
    row=1, col=2
)

# Q2: WHERE DID THEY LOOK?
aoi_labels = list(AOI_NAMES.keys())
success_totals = [success_aoi[aoi] for aoi in aoi_labels]
fail_totals = [fail_aoi[aoi] for aoi in aoi_labels]

success_total = sum(success_totals)
fail_total = sum(fail_totals)
success_pcts = [(x / success_total) * 100 for x in success_totals]
fail_pcts = [(x / fail_total) * 100 for x in fail_totals]

aoi_labels_full = [AOI_NAMES[k] for k in aoi_labels]

# Sort by successful pilot attention
sorted_indices = sorted(range(len(success_pcts)), key=lambda i: success_pcts[i], reverse=True)
sorted_labels = [aoi_labels_full[i] for i in sorted_indices]
sorted_success = [success_pcts[i] for i in sorted_indices]
sorted_fail = [fail_pcts[i] for i in sorted_indices]

fig.add_trace(
    go.Bar(
        y=sorted_labels[::-1],
        x=sorted_success[::-1],
        orientation='h',
        marker_color='#2ecc71',
        text=[f'{v:.1f}%' for v in sorted_success[::-1]],
        textposition='outside',
        textfont=dict(size=12),
        name='Successful',
        hovertemplate='<b>%{y}</b><br>%{x:.1f}% of gaze time<extra></extra>'
    ),
    row=2, col=1
)

# Top instrument transitions
top_trans = success_trans.most_common(10)
trans_labels = [f"{AOI_NAMES[t[0][0]]} → {AOI_NAMES[t[0][1]]}" for t in top_trans]
trans_values = [t[1] for t in top_trans]
trans_total = sum(trans_values)
trans_pcts = [(v / trans_total) * 100 for v in trans_values]

fig.add_trace(
    go.Bar(
        y=trans_labels[::-1],
        x=trans_pcts[::-1],
        orientation='h',
        marker_color='#3498db',
        text=[f'{v:.1f}%' for v in trans_pcts[::-1]],
        textposition='outside',
        textfont=dict(size=11),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>%{x:.1f}% of transitions<extra></extra>'
    ),
    row=2, col=2
)

# Q2 Continued: Attention distribution comparison
fig.add_trace(
    go.Bar(
        x=aoi_labels_full,
        y=success_pcts,
        name='Successful',
        marker_color='#2ecc71',
        text=[f'{v:.1f}%' for v in success_pcts],
        textposition='outside',
        textfont=dict(size=10)
    ),
    row=3, col=1
)

fig.add_trace(
    go.Bar(
        x=aoi_labels_full,
        y=fail_pcts,
        name='Unsuccessful',
        marker_color='#e74c3c',
        text=[f'{v:.1f}%' for v in fail_pcts],
        textposition='outside',
        textfont=dict(size=10)
    ),
    row=3, col=1
)

# Key differences
differences = [s - f for s, f in zip(success_pcts, fail_pcts)]
diff_colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in differences]

fig.add_trace(
    go.Bar(
        x=aoi_labels_full,
        y=differences,
        marker_color=diff_colors,
        text=[f'{d:+.1f}%' for d in differences],
        textposition='outside',
        textfont=dict(size=11, color='black'),
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>Difference: %{y:+.1f}%<br>(Positive = more in successful)<extra></extra>'
    ),
    row=3, col=2
)

# Add zero reference line
fig.add_shape(
    type="line",
    x0=0, x1=1, xref="x6 domain",
    y0=0, y1=0, yref="y6",
    line=dict(dash="dash", color="black", width=1),
    opacity=0.5
)

# Q3: DOMINANT PATTERNS
top15_success = success_df.nlargest(15, freq_col).copy()
top15_fail = fail_df.nlargest(15, freq_col).copy()


def pattern_to_readable(pattern):
    return ' → '.join([AOI_NAMES.get(c, c) for c in str(pattern)])


success_patterns_readable = [pattern_to_readable(p) for p in top15_success[pattern_col]]
fail_patterns_readable = [pattern_to_readable(p) for p in top15_fail[pattern_col]]

# Successful patterns (row 4, col 1 - spans both columns)
fig.add_trace(
    go.Bar(
        y=success_patterns_readable[::-1],
        x=top15_success[freq_col].values[::-1],
        orientation='h',
        marker_color='#2ecc71',
        text=top15_success[freq_col].values[::-1],
        textposition='outside',
        textfont=dict(size=11),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
    ),
    row=4, col=1
)

# Unsuccessful patterns (row 5, col 1 - spans both columns)
fig.add_trace(
    go.Bar(
        y=fail_patterns_readable[::-1],
        x=top15_fail[freq_col].values[::-1],
        orientation='h',
        marker_color='#e74c3c',
        text=top15_fail[freq_col].values[::-1],
        textposition='outside',
        textfont=dict(size=11),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
    ),
    row=5, col=1
)

# Update layout
fig.update_layout(
    height=2600,
    title_text="<b>ILS APPROACH ANALYSIS: Successful vs Unsuccessful Pilots</b><br><sup>Answering Key Research Questions</sup>",
    title_font_size=24,
    title_x=0.5,
    showlegend=True,
    font=dict(size=10),
    legend=dict(x=0.4, y=0.63)
)

# Update axes
fig.update_yaxes(title_text="% of Total Patterns", row=1, col=1)
fig.update_yaxes(title_text="% of Total Patterns", row=1, col=2)
fig.update_xaxes(title_text="% of Gaze Time", row=2, col=1)
fig.update_xaxes(title_text="% of Transitions", row=2, col=2)
fig.update_yaxes(title_text="% of Gaze Time", row=3, col=1)
fig.update_xaxes(tickangle=45, row=3, col=1)
fig.update_yaxes(title_text="Difference (%)", row=3, col=2)
fig.update_xaxes(title_text="Success Looks More ←  → Fail Looks More", tickangle=45, row=3, col=2)
fig.update_xaxes(title_text="Frequency", row=4, col=1)
fig.update_xaxes(title_text="Frequency", row=5, col=1)

# Save
output_file = 'ils_research_questions.html'
fig.write_html(output_file)

# Generate detailed report
print("\n" + "=" * 80)
print("RESEARCH QUESTIONS ANSWERED")
print("=" * 80)

print("\n📊 Q1: WHAT GAZE BEHAVIORS CHARACTERIZE SUCCESSFUL APPROACHES?")
print("-" * 80)
print(f"Successful pilots show:")
print(f"  • Repetitive Fixations (same AOI): {success_behaviors['repetitive']:.1f}%")
print(f"  • Back-and-Forth Scanning: {success_behaviors['back_forth']:.1f}%")
print(f"  • Systematic Multi-AOI Scans: {success_behaviors['systematic']:.1f}%")
print(f"\nUnsuccessful pilots show:")
print(f"  • Repetitive Fixations (same AOI): {fail_behaviors['repetitive']:.1f}%")
print(f"  • Back-and-Forth Scanning: {fail_behaviors['back_forth']:.1f}%")
print(f"  • Systematic Multi-AOI Scans: {fail_behaviors['systematic']:.1f}%")

print("\n" + "=" * 80)
print("🎯 Q2: WHERE DID SUCCESSFUL PILOTS LOOK COMPARED TO UNSUCCESSFUL?")
print("-" * 80)
print(f"{'Instrument':<25} {'Successful':>12} {'Unsuccessful':>14} {'Difference':>12}")
print("-" * 80)
for i, aoi in enumerate(aoi_labels):
    name = AOI_NAMES[aoi]
    diff = success_pcts[i] - fail_pcts[i]
    arrow = "✓ MORE" if diff > 0 else "✗ LESS"
    print(f"{name:<25} {success_pcts[i]:>10.1f}% {fail_pcts[i]:>12.1f}% {diff:>8.1f}% {arrow}")

print("\n" + "=" * 80)
print("🔍 Q3: WHICH DOMINANT PATTERNS EMERGED?")
print("-" * 80)
print("\nTop 5 SUCCESSFUL patterns:")
for i, (idx, row) in enumerate(top15_success.head(5).iterrows(), 1):
    pattern = str(row[pattern_col])
    readable = pattern_to_readable(pattern)
    print(f"  {i}. {pattern:8s} ({readable})")
    print(f"     Frequency: {row[freq_col]:.0f}")

print("\nTop 5 UNSUCCESSFUL patterns:")
for i, (idx, row) in enumerate(top15_fail.head(5).iterrows(), 1):
    pattern = str(row[pattern_col])
    readable = pattern_to_readable(pattern)
    print(f"  {i}. {pattern:8s} ({readable})")
    print(f"     Frequency: {row[freq_col]:.0f}")

print("\n" + "=" * 80)
print("💡 KEY INSIGHTS:")
print("-" * 80)

# Find biggest differences
max_diff_idx = differences.index(max(differences, key=abs))
max_diff_aoi = AOI_NAMES[aoi_labels[max_diff_idx]]
print(f"• Biggest attention difference: {max_diff_aoi} ({differences[max_diff_idx]:+.1f}%)")

# Behavior difference
behavior_diff = success_behaviors['systematic'] - fail_behaviors['systematic']
print(f"• Systematic scanning difference: {behavior_diff:+.1f}% (successful pilots)")

print("\n" + "=" * 80)
print(f"✓ Visualization saved to: {output_file}")
print("Opening in web browser...")
print("=" * 80 + "\n")

webbrowser.open('file://' + os.path.realpath(output_file))