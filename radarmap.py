import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from collections import Counter
from plotly.subplots import make_subplots

# Read directly from your CSV file
df = pd.read_csv('AOI_DGMs.csv')

# AOI mapping with full instrument names (from patterncompare.py)
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


# Analysis functions (from patterncompare.py)
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


def pattern_to_readable(pattern):
    return ' → '.join([AOI_NAMES.get(c, c) for c in str(pattern)])


# Read pattern data (from patterncompare.py)
def load_pattern_data():
    """Load pattern data and perform analysis"""
    print("Loading flight pattern data...")
    try:
        success_df = pd.read_csv('successpatterns.csv', encoding='utf-8')
        fail_df = pd.read_csv('failpatterns.csv', encoding='utf-8')
        pattern_data_available = True
        print(f"✓ Successful pilots: {len(success_df)} patterns loaded")
        print(f"✓ Unsuccessful pilots: {len(fail_df)} patterns loaded")

        # Get column names for pattern data
        pattern_col = 'Pattern String' if 'Pattern String' in success_df.columns else success_df.columns[0]
        freq_col = 'Frequency' if 'Frequency' in success_df.columns else success_df.columns[1]

        # Perform pattern analysis
        success_aoi = count_aoi_occurrences(success_df, pattern_col, freq_col)
        fail_aoi = count_aoi_occurrences(fail_df, pattern_col, freq_col)
        success_behaviors = analyze_pattern_characteristics(success_df, pattern_col, freq_col)
        fail_behaviors = analyze_pattern_characteristics(fail_df, pattern_col, freq_col)
        success_trans = extract_transitions(success_df, pattern_col, freq_col)
        fail_trans = extract_transitions(fail_df, pattern_col, freq_col)

        return {
            'available': True,
            'success_df': success_df,
            'fail_df': fail_df,
            'pattern_col': pattern_col,
            'freq_col': freq_col,
            'success_aoi': success_aoi,
            'fail_aoi': fail_aoi,
            'success_behaviors': success_behaviors,
            'fail_behaviors': fail_behaviors,
            'success_trans': success_trans,
            'fail_trans': fail_trans
        }

    except FileNotFoundError:
        print("⚠ Pattern data files not found. Pattern comparison will be disabled.")
        return {'available': False}


# Load pattern data
pattern_data = load_pattern_data()

numeric_cols = [
    # main metrics
    'Mean_fixation_duration_s', 'mean_saccade_length', 'Average_Peak_Saccade_Velocity',
    'stationary_entropy', 'transition_entropy', 'Average_Blink_Rate_per_Minute',
    'fixation_to_saccade_ratio', 'Total_Number_of_Fixations',
    'Sum_of_all_fixation_duration_s', 'total_number_of_saccades',

    # AOI bar columns
    'Window_Mean_fixation_duration_s', 'AI_Mean_fixation_duration_s',
    'Alt_VSI_Mean_fixation_duration_s', 'ASI_Mean_fixation_duration_s',
    'TI_HSI_Mean_fixation_duration_s', 'SSI_Mean_fixation_duration_s',
    'RPM_Mean_fixation_duration_s', 'NoAOI_Mean_fixation_duration_s',

    # Used in parallel coordinates
    'Approach_Score'
]

# Handles null values in file
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


def create_enhanced_radar_dashboard(df):
    """
    Create an enhanced 3D radar chart dashboard with data-driven success/failure analysis
    """

    # Available metrics with descriptions
    metrics_config = {
        'Mean_fixation_duration_s': {
            'name': 'Fixation Duration (s)',
            'description': 'Average time eyes remain stationary. Shorter = faster information processing'
        },
        'mean_saccade_length': {
            'name': 'Saccade Length (px)',
            'description': 'Average distance between fixations. Moderate = efficient scanning'
        },
        'Average_Peak_Saccade_Velocity': {
            'name': 'Saccade Velocity (deg/s)',
            'description': 'Speed of eye movements. Higher = quicker scanning'
        },
        'stationary_entropy': {
            'name': 'Fixation Entropy',
            'description': 'Randomness of fixation locations. Lower = more systematic viewing'
        },
        'transition_entropy': {
            'name': 'Transition Entropy',
            'description': 'Predictability of scanpaths. Lower = more consistent patterns'
        },
        'Average_Blink_Rate_per_Minute': {
            'name': 'Blink Rate (/min)',
            'description': 'Blinks per minute. Moderate = optimal cognitive load'
        },
        'fixation_to_saccade_ratio': {
            'name': 'Fixation/Saccade Ratio',
            'description': 'Balance between processing and searching. Balanced = efficient'
        },
        'Total_Number_of_Fixations': {
            'name': 'Total Fixations',
            'description': 'Total number of fixations. Moderate = efficient information gathering'
        },
        'Sum_of_all_fixation_duration_s': {
            'name': 'Total Fixation Time (s)',
            'description': 'Cumulative fixation duration'
        },
        'total_number_of_saccades': {
            'name': 'Total Saccades',
            'description': 'Total number of eye movements between fixations'
        }
    }

    # Available metrics for bar chart
    bar_config = {
        'Window_Mean_fixation_duration_s': {
            'name': 'Window',
            'description': 'Average time spent looking at window',
        },
        'AI_Mean_fixation_duration_s': {
            'name': 'AI',
            'description': 'Average time spent looking at AI',
        },
        'Alt_VSI_Mean_fixation_duration_s': {
            'name': 'Alt VSI',
            'description': 'Average time spent looking at ALT',
        },
        'ASI_Mean_fixation_duration_s': {
            'name': 'ASI',
            'description': 'Average time spent looking at ASI',
        },
        'TI_HSI_Mean_fixation_duration_s': {
            'name': 'TI HSI',
            'description': 'Average time spent looking at TI',
        },
        'SSI_Mean_fixation_duration_s': {
            'name': 'SSI',
            'description': 'Average time spent looking at SSI',
        },
        'RPM_Mean_fixation_duration_s': {
            'name': 'RPM',
            'description': 'Average time spent looking at RPM',
        },
        'NoAOI_Mean_fixation_duration_s': {
            'name': 'NoAOI',
            'description': 'Average time spent looking at NoAOI',
        }
    }

    # Calculate benchmark statistics from the dataset
    successful_df = df[df['pilot_success'] == 'Successful']
    unsuccessful_df = df[df['pilot_success'] == 'Unsuccessful']

    # Success/Failure analysis configurations with data-driven benchmarks
    success_analysis_config = {
        'success_patterns': {
            'title': '✅ SUCCESSFUL PATTERN',
            'indicators': [
                'Moderate to long fixation duration (thorough processing)',
                'Low fixation and transition entropy (systematic scanning)',
                'Balanced fixation/saccade ratio (efficient visual strategy)',
                'Moderate saccade length (optimal area coverage)',
                'Stable blink rate (managed cognitive load)'
            ]
        },
        'failure_patterns': {
            'title': '❌ FAILED PATTERN',
            'indicators': [
                'Extremely short or long fixation duration (rushed or stuck)',
                'High entropy values (random, unorganized scanning)',
                'Imbalanced fixation/saccade ratio (over-fixating or over-scanning)',
                'Extreme saccade lengths (too narrow or too broad focus)',
                'High blink rate (cognitive overload or fatigue)'
            ]
        }
    }

    # Default metrics selection
    default_metrics = [
        'Mean_fixation_duration_s', 'mean_saccade_length', 'Average_Peak_Saccade_Velocity',
        'stationary_entropy', 'transition_entropy', 'Average_Blink_Rate_per_Minute',
        'fixation_to_saccade_ratio'
    ]

    # Initialize Dash app with custom CSS for dropdowns
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # Custom CSS for dropdown styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* Custom dropdown styles */
                .Select-control, .Select-multi-value-wrapper, .Select-input {
                    background-color: #2c3e50 !important;
                    color: white !important;
                    border: 1px solid #34495e !important;
                }

                .Select-menu-outer {
                    background-color: #2c3e50 !important;
                    color: white !important;
                    border: 1px solid #34495e !important;
                }

                .Select-option {
                    background-color: #2c3e50 !important;
                    color: white !important;
                    border-bottom: 1px solid #34495e !important;
                }

                .Select-option.is-focused {
                    background-color: #34495e !important;
                    color: white !important;
                }

                .Select-value-label {
                    color: white !important;
                }

                .Select-placeholder {
                    color: #95a5a6 !important;
                }

                /* Radio items styling */
                .dash-radio-items label {
                    color: white !important;
                    margin-right: 15px;
                }

                /* Success/Failure status styling */
                .success-status {
                    background: linear-gradient(135deg, #2e7d32, #4caf50);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    margin-bottom: 10px;
                }

                .failure-status {
                    background: linear-gradient(135deg, #c62828, #f44336);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    margin-bottom: 10px;
                }

                /* Stat highlight styling */
                .stat-highlight {
                    background: rgba(255,255,255,0.1);
                    padding: 5px;
                    border-radius: 3px;
                    margin: 2px 0;
                    font-family: monospace;
                }

                .good-stat {
                    border-left: 3px solid #4caf50;
                }

                .bad-stat {
                    border-left: 3px solid #f44336;
                }

                .neutral-stat {
                    border-left: 3px solid #ff9800;
                }

                .metric-item {
                    background: rgba(255,255,255,0.05);
                    padding: 8px;
                    margin: 5px 0;
                    border-radius: 4px;
                    border-left: 4px solid #666;
                }

                .metric-good {
                    border-left-color: #4caf50;
                }

                .metric-bad {
                    border-left-color: #f44336;
                }

                .metric-neutral {
                    border-left-color: #ff9800;
                }

                .range-text {
                    font-size: 0.85em;
                    color: #cccccc;
                }

                .iqr-text {
                    font-size: 0.85em;
                    font-weight: bold;
                }

                .iqr-good {
                    color: #4caf50;
                }

                .iqr-bad {
                    color: #f44336;
                }

                .approach-score {
                    background: linear-gradient(135deg, #1565c0, #42a5f5);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    margin: 10px 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 10px;
                }

                .score-value {
                    font-size: 24px;
                    font-weight: bold;
                }

                .score-status {
                    font-size: 16px;
                }

                .threshold-info {
                    background: rgba(255,255,255,0.1);
                    padding: 8px;
                    border-radius: 4px;
                    border-left: 4px solid #ff9800;
                    margin: 5px 0;
                }

                .metric-description {
                    font-size: 0.75em;
                    color: #aaaaaa;
                    font-style: italic;
                    margin-top: 2px;
                }

                .iqr-explanation {
                    background: rgba(255,255,255,0.05);
                    padding: 8px;
                    border-radius: 4px;
                    border-left: 4px solid #42a5f5;
                    margin: 10px 0;
                    font-size: 0.8em;
                }

                /* Behavior Characterization Styles */
                .behavior-pattern {
                    background: rgba(255,255,255,0.05);
                    padding: 10px;
                    border-radius: 5px;
                    margin: 8px 0;
                    border-left: 4px solid #666;
                }

                .success-pattern {
                    border-left-color: #4caf50;
                }

                .failure-pattern {
                    border-left-color: #f44336;
                }

                .pattern-header {
                    font-weight: bold;
                    margin-bottom: 5px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .pattern-metric {
                    background: rgba(255,255,255,0.1);
                    padding: 5px 8px;
                    border-radius: 3px;
                    margin: 3px 0;
                    font-family: monospace;
                    font-size: 0.9em;
                }

                .metric-good {
                    color: #4caf50;
                    border-left: 2px solid #4caf50;
                    padding-left: 5px;
                }

                .metric-bad {
                    color: #f44336;
                    border-left: 2px solid #f44336;
                    padding-left: 5px;
                }

                .stat-difference {
                    font-weight: bold;
                    background: rgba(255,255,255,0.1);
                    padding: 2px 6px;
                    border-radius: 3px;
                    margin: 0 5px;
                }

                .difference-positive {
                    color: #4caf50;
                }

                .difference-negative {
                    color: #f44336;
                }

                .key-insight {
                    background: linear-gradient(135deg, #1565c0, #42a5f5);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                    border-left: 4px solid #42a5f5;
                }

                .insight-header {
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 5px;
                }

                /* Hidden elements */
                .hidden {
                    display: none !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Pilot Gaze Behavior Analysis Dashboard",
                        className="text-center mb-4",
                        style={'color': '#ffffff'}),
                html.P("Interactive 3D Radar Chart Comparing Successful vs Unsuccessful ILS Approaches",
                       className="text-center text-light mb-4")
            ])
        ]),

        # Controls Row
        dbc.Row([
            # Left Column - Filters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Chart Controls", className="h5"),
                    dbc.CardBody([
                        # Mode Selection Switch
                        html.Label("Analysis Mode:", className="fw-bold text-light"),
                        dbc.Switch(
                            id="mode-switch",
                            label="Patterns Mode",
                            value=False,  # Default to AOI DGM mode
                            className="mb-4"
                        ),

                        # AOI DGM Controls (visible by default)
                        html.Div([
                            # Metric Selection
                            html.Label("Select Metrics (3-8 recommended):", className="fw-bold text-light"),
                            dcc.Dropdown(
                                id='metrics-dropdown',
                                options=[{'label': metrics_config[metric]['name'], 'value': metric}
                                         for metric in metrics_config.keys()],
                                value=default_metrics,
                                multi=True,
                                className="mb-3"
                            ),

                            # Normalization Toggle
                            dbc.Switch(
                                id="normalization-toggle",
                                label="Normalize Metrics (0-1 scale)",
                                value=True,
                                className="mb-3"
                            ),

                            # Group Selection
                            html.Label("Compare Groups:", className="fw-bold text-light"),
                            dbc.Checklist(
                                options=[
                                    {"label": " Successful Pilots", "value": "Successful"},
                                    {"label": " Unsuccessful Pilots", "value": "Unsuccessful"},
                                    {"label": " All Pilots Average", "value": "All"}
                                ],
                                value=["Successful", "Unsuccessful"],
                                id="group-checklist",
                                switch=True,
                                className="mb-3"
                            ),

                            # Individual Pilot Selection
                            html.Label("Add Individual Pilot:", className="fw-bold text-light"),
                            dcc.Dropdown(
                                id='pilot-dropdown',
                                options=[{'label': f'Pilot {pid}', 'value': pid}
                                         for pid in df['PID'].unique()],
                                placeholder="Select individual pilot...",
                                className="mb-3"
                            ),

                            # Visualization Type - Main Charts
                            html.Label("Visualization Type:", className="fw-bold text-light"),
                            dcc.RadioItems(
                                id='visualization-type',
                                options=[
                                    {'label': ' Radar', 'value': 'radar'},
                                    {'label': ' Parallel Coordinates', 'value': 'parallel'},
                                    {'label': ' AOI Bar Chart', 'value': 'bar'}
                                ],
                                value='radar',
                                className="mb-3"
                            ),
                        ], id='aoi-dgm-controls'),

                        # Pattern Comparison Controls (hidden by default)
                        html.Div([
                            html.Label("Pattern Analysis Charts:", className="fw-bold text-light"),
                            dcc.RadioItems(
                                id='pattern-chart-type',
                                options=[
                                    {'label': ' Attention Distribution Comparison',
                                     'value': 'attention_distribution'},
                                    {'label': ' Key Differences in AOI Focus', 'value': 'aoi_differences'},
                                    {'label': ' Dominant Patterns - Successful', 'value': 'success_patterns'},
                                    {'label': ' Dominant Patterns - Unsuccessful', 'value': 'fail_patterns'}
                                ],
                                value='attention_distribution',
                                className="mb-3"
                            ),
                        ], id='pattern-controls', className='hidden'),

                        # Action Buttons - Only Reset button remains
                        dbc.Row([
                            dbc.Col(dbc.Button("Reset Filters", id="reset-btn", color="secondary"), width=12)
                        ])
                    ])
                ], className="h-100")
            ], width=3),

            # Right Column - Visualizations
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Main Visualization", className="h5"),
                            dbc.CardBody([
                                dcc.Graph(id='main-visualization', style={'height': '600px'})
                            ])
                        ])
                    ])
                ]),

                dbc.Row([
                    # Behavior Characterization Card (Replaced Key Statistics)
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Behavior Characterization", className="h5"),
                            dbc.CardBody(id='behavior-characterization')
                        ])
                    ], width=6),

                    # Success Analysis Card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Success Analysis", className="h5"),
                            dbc.CardBody(id='success-analysis')
                        ])
                    ], width=6)
                ], className="mt-3")
            ], width=9)
        ]),

        # Hidden div for storing intermediate values
        dcc.Store(id='current-data')
    ], fluid=True, style={'backgroundColor': '#1a1a1a'})

    def calculate_benchmarks():
        """Calculate statistical benchmarks from the dataset"""
        benchmarks = {}
        for metric in metrics_config.keys():
            if metric in df.columns:
                successful_vals = successful_df[metric]
                unsuccessful_vals = unsuccessful_df[metric]

                benchmarks[metric] = {
                    'success_iqr': (successful_vals.quantile(0.25), successful_vals.quantile(0.75)),
                    'success_mean': successful_vals.mean(),
                    'success_median': successful_vals.median(),
                    'unsuccess_mean': unsuccessful_vals.mean(),
                    'unsuccess_median': unsuccessful_vals.median(),
                    'all_range': (df[metric].min(), df[metric].max()),
                    'difference': successful_vals.mean() - unsuccessful_vals.mean(),
                    'difference_pct': ((
                                               successful_vals.mean() - unsuccessful_vals.mean()) / unsuccessful_vals.mean() * 100) if unsuccessful_vals.mean() != 0 else 0
                }

        # Add Approach Score benchmarks separately
        if 'Approach_Score' in df.columns:
            successful_scores = successful_df['Approach_Score']
            unsuccessful_scores = unsuccessful_df['Approach_Score']

            benchmarks['Approach_Score'] = {
                'success_iqr': (successful_scores.quantile(0.25), successful_scores.quantile(0.75)),
                'success_mean': successful_scores.mean(),
                'success_median': successful_scores.median(),
                'unsuccess_mean': unsuccessful_scores.mean(),
                'unsuccess_median': unsuccessful_scores.median(),
                'all_range': (df['Approach_Score'].min(), df['Approach_Score'].max()),
                'difference': successful_scores.mean() - unsuccessful_scores.mean(),
                'difference_pct': (
                        (successful_scores.mean() - unsuccessful_scores.mean()) / unsuccessful_scores.mean() * 100),
                'threshold': 0.7  # Success threshold
            }

        return benchmarks

    # Pre-calculate benchmarks
    benchmarks = calculate_benchmarks()

    def characterize_behavior(selected_metrics, stats_data):
        """Characterize what ALL successful pilots did differently compared to ALL unsuccessful pilots"""

        characterization_content = []

        characterization_content.append(html.H5("Group Behavior Patterns",
                                                className="text-info mb-3"))

        if 'Successful' in stats_data and 'Unsuccessful' in stats_data:
            successful_data = stats_data['Successful']
            unsuccessful_data = stats_data['Unsuccessful']

            # Overall performance summary
            characterization_content.append(html.Div([
                html.Div("📊 OVERALL PERFORMANCE SUMMARY", className="pattern-header"),
                html.P(
                    f"Analysis based on {len(successful_df)} successful vs {len(unsuccessful_df)} unsuccessful pilots",
                    className="text-light small mb-2")
            ], className="key-insight"))

            # Key behavioral differences
            characterization_content.append(html.Div([
                html.Div("🎯 KEY BEHAVIORAL DIFFERENCES", className="pattern-header"),
                html.P("What successful pilots did differently:", className="text-light small mb-2")
            ], className="behavior-pattern success-pattern"))

            key_differences = []

            for metric in selected_metrics:
                if metric in successful_data and metric in unsuccessful_data and metric in benchmarks:
                    success_val = successful_data[metric]
                    unsuccess_val = unsuccessful_data[metric]
                    bench = benchmarks[metric]
                    difference = bench['difference']
                    difference_pct = bench['difference_pct']

                    # Only show significant differences (>10% difference)
                    if abs(difference_pct) > 10:
                        direction = "higher" if difference > 0 else "lower"

                        # Determine interpretation based on metric type
                        if metric in ['stationary_entropy', 'transition_entropy']:
                            # For entropy, lower is better
                            if difference < 0:
                                interpretation = "More systematic scanning patterns"
                                diff_class = "difference-positive"
                            else:
                                interpretation = "Less systematic scanning"
                                diff_class = "difference-negative"
                        elif metric == 'Average_Blink_Rate_per_Minute':
                            # Moderate blink rate is best - small differences are better
                            if abs(difference) < 5:
                                interpretation = "Better cognitive load management"
                                diff_class = "difference-positive"
                            else:
                                interpretation = "Different stress/cognitive load levels"
                                diff_class = "difference-negative"
                        elif metric in ['Mean_fixation_duration_s', 'fixation_to_saccade_ratio']:
                            # Moderate values are typically better
                            if abs(difference_pct) < 25:
                                interpretation = "More balanced visual processing"
                                diff_class = "difference-positive"
                            else:
                                interpretation = "Different information processing strategy"
                                diff_class = "difference-negative"
                        else:
                            interpretation = "More efficient visual behavior" if difference > 0 else "Less efficient visual behavior"
                            diff_class = "difference-positive" if difference > 0 else "difference-negative"

                        key_differences.append(
                            html.Div([
                                html.Strong(f"{metrics_config[metric]['name']}:", className="text-light"),
                                html.Span(f" {direction} by ", className="text-light"),
                                html.Span(f"{abs(difference_pct):.1f}%", className=f"stat-difference {diff_class}"),
                                html.Span(f" → {interpretation}", className="text-light small")
                            ], className="pattern-metric")
                        )

            if key_differences:
                characterization_content.extend(key_differences)
            else:
                characterization_content.append(html.P("No significant differences found in selected metrics.",
                                                       className="text-light small"))

            # Statistical Significance Note
            characterization_content.append(html.Div([
                html.Div("📈 STATISTICAL NOTE", className="pattern-header"),
                html.P("Differences shown are based on group averages with >10% magnitude. ",
                       className="text-light small mb-0"),
                html.P("IQR analysis confirms consistent pattern differences between groups.",
                       className="text-light small mb-0")
            ], className="behavior-pattern"))

        return characterization_content

    def analyze_success_patterns(selected_metrics, stats_data, selected_pilot=None):
        """Analyze gaze patterns to determine success/failure reasons with statistical evidence"""

        analysis_content = []

        if selected_pilot:
            # Individual pilot analysis
            pilot_key = f'Pilot {selected_pilot}'
            if pilot_key in stats_data:
                pilot_data = stats_data[pilot_key]
                pilot_success = df[df['PID'] == selected_pilot]['pilot_success'].iloc[0]

                # Status header
                status_class = "success-status" if pilot_success == 'Successful' else "failure-status"
                status_text = f"✅ PILOT {selected_pilot} - SUCCESSFUL" if pilot_success == 'Successful' else f"❌ PILOT {selected_pilot} - FAILED"

                analysis_content.append(html.Div(status_text, className=status_class))

                # Add IQR explanation
                iqr_explanation = html.Div([
                    html.Strong("📊 About IQR (Interquartile Range):", className="text-light"),
                    html.Br(),
                    html.Span("IQR shows the middle 50% of successful pilots (25th to 75th percentile). ",
                              className="text-light small"),
                    html.Span("Values within IQR represent typical expert performance.", className="text-light small")
                ], className="iqr-explanation")

                analysis_content.append(iqr_explanation)

                # Add Approach Score section
                if 'Approach_Score' in df.columns and 'Approach_Score' in benchmarks:
                    pilot_score = df[df['PID'] == selected_pilot]['Approach_Score'].iloc[0]
                    bench = benchmarks['Approach_Score']
                    success_iqr_low, success_iqr_high = bench['success_iqr']
                    threshold = bench['threshold']

                    # Determine score status
                    if pilot_success == 'Successful':
                        if pilot_score >= 0.9:
                            score_status = "Excellent"
                            score_color = "text-success"
                        elif pilot_score >= 0.8:
                            score_status = "Very Good"
                            score_color = "text-success"
                        elif pilot_score >= threshold:
                            score_status = "Good"
                            score_color = "text-success"
                        else:
                            score_status = "Marginal"
                            score_color = "text-warning"
                    else:
                        if pilot_score >= threshold:
                            score_status = "Near Success"
                            score_color = "text-warning"
                        elif pilot_score >= 0.5:
                            score_status = "Below Standard"
                            score_color = "text-warning"
                        else:
                            score_status = "Poor"
                            score_color = "text-danger"

                    approach_score_display = html.Div([
                        html.H6("Approach Score", className="text-center text-light"),
                        html.Div([
                            html.Span(f"{pilot_score:.3f}", className="score-value"),
                            html.Span(f"{score_status}", className=f"score-status {score_color}")
                        ], className="approach-score"),
                        html.Div([
                            html.Strong("Success Threshold: ≥ 0.7", className="text-light"),
                            html.Br(),
                            html.Span(f"Successful IQR: {success_iqr_low:.3f} - {success_iqr_high:.3f}",
                                      className="iqr-text iqr-good")
                        ], className="threshold-info")
                    ])

                    analysis_content.append(approach_score_display)

                # Show ALL metrics with detailed statistical analysis
                analysis_content.append(html.H6("Gaze Metrics Analysis:", className="text-info mt-3"))

                for metric in selected_metrics:
                    if metric in pilot_data and metric in benchmarks:
                        value = pilot_data[metric]
                        bench = benchmarks[metric]
                        success_iqr_low, success_iqr_high = bench['success_iqr']

                        # Determine status and reasoning based only on success IQR
                        if success_iqr_low <= value <= success_iqr_high:
                            status_class = "metric-good"
                            reasoning = f"Within typical successful range (IQR)"
                            iqr_color_class = "iqr-good"
                        elif value < success_iqr_low:
                            if 'entropy' in metric:
                                status_class = "metric-good"
                                reasoning = f"Better than typical - more systematic"
                                iqr_color_class = "iqr-good"
                            else:
                                status_class = "metric-bad"
                                reasoning = f"Below typical successful range"
                                iqr_color_class = "iqr-bad"
                        else:
                            if 'entropy' in metric:
                                status_class = "metric-bad"
                                reasoning = f"Above typical successful range - less systematic"
                                iqr_color_class = "iqr-bad"
                            else:
                                status_class = "metric-bad"
                                reasoning = f"Above typical successful range"
                                iqr_color_class = "iqr-bad"

                        # Create metric item with description and IQR data
                        metric_item = html.Div([
                            html.Strong(f"{metrics_config[metric]['name']}: {value:.3f}", className="text-light"),
                            html.Br(),
                            html.Span(f"{metrics_config[metric]['description']}", className="metric-description"),
                            html.Br(),
                            html.Span(f"Status: {reasoning}", className="text-light small"),
                            html.Br(),
                            html.Span(f"Successful IQR: {success_iqr_low:.3f} - {success_iqr_high:.3f}",
                                      className=f"iqr-text {iqr_color_class}")
                        ], className=f"metric-item {status_class}")

                        analysis_content.append(metric_item)

                # Summary section
                analysis_content.append(html.H6("Performance Summary:", className="text-warning mt-3"))

                if pilot_success == 'Successful':
                    analysis_content.append(html.P(
                        "This pilot demonstrates strong expert gaze patterns with metrics predominantly within typical successful ranges.",
                        className="text-light small"))
                else:
                    analysis_content.append(html.P(
                        "This pilot shows suboptimal gaze behaviors with several metrics outside typical successful patterns.",
                        className="text-light small"))

        else:
            # Group comparison analysis
            analysis_content.append(html.H5("Group Performance Analysis", className="text-info mb-3"))

            # Add IQR explanation
            iqr_explanation = html.Div([
                html.Strong("📊 About IQR (Interquartile Range):", className="text-light"),
                html.Br(),
                html.Span("IQR shows the middle 50% of successful pilots (25th to 75th percentile). ",
                          className="text-light small"),
                html.Span("Values within IQR represent typical expert performance.", className="text-light small")
            ], className="iqr-explanation")

            analysis_content.append(iqr_explanation)

            # Add Approach Score comparison
            if 'Approach_Score' in df.columns and 'Approach_Score' in benchmarks:
                bench = benchmarks['Approach_Score']
                threshold = bench['threshold']
                success_iqr_low, success_iqr_high = bench['success_iqr']

                analysis_content.append(html.H6("Approach Score Comparison:", className="text-warning"))

                approach_comparison = html.Div([
                    html.Strong("Success Threshold: ≥ 0.7", className="text-light"),
                    html.Br(),
                    html.Span(
                        f"Successful IQR: {success_iqr_low:.3f} - {success_iqr_high:.3f} (avg: {bench['success_mean']:.3f})",
                        className="iqr-text iqr-good")
                ], className="stat-highlight good-stat")

                analysis_content.append(approach_comparison)
                analysis_content.append(html.Hr(className="my-3"))

            if 'Successful' in stats_data and 'Unsuccessful' in stats_data:
                successful_data = stats_data['Successful']
                unsuccessful_data = stats_data['Unsuccessful']

                # Show ALL metrics comparison with only success IQR
                analysis_content.append(html.H6("Gaze Metrics Comparison:", className="text-warning"))

                for metric in selected_metrics:
                    if metric in successful_data and metric in unsuccessful_data and metric in benchmarks:
                        success_val = successful_data[metric]
                        unsuccess_val = unsuccessful_data[metric]
                        bench = benchmarks[metric]
                        success_iqr_low, success_iqr_high = bench['success_iqr']

                        # Simple comparison - check if unsuccessful average is within success IQR
                        if success_iqr_low <= unsuccess_val <= success_iqr_high:
                            status_class = "neutral-stat"
                            comparison = f"Unsuccessful avg within typical successful range"
                        else:
                            status_class = "good-stat"
                            comparison = f"Clear difference from typical successful pattern"

                        metric_item = html.Div([
                            html.Strong(f"{metrics_config[metric]['name']}: ", className="text-light"),
                            html.Br(),
                            html.Span(f"{metrics_config[metric]['description']}", className="metric-description"),
                            html.Br(),
                            html.Span(
                                f"Successful IQR: {success_iqr_low:.3f} - {success_iqr_high:.3f} (avg: {success_val:.3f})",
                                className="iqr-text iqr-good"),
                            html.Br(),
                            html.Span(f"Unsuccessful average: {unsuccess_val:.3f}",
                                      className="text-danger"),
                            html.Br(),
                            html.Span(f"Comparison: {comparison}",
                                      className="text-light")
                        ], className=f"stat-highlight {status_class}")

                        analysis_content.append(metric_item)

            # Add benchmark summary
            analysis_content.append(html.Hr(className="my-3"))
            analysis_content.append(html.H6("Dataset Benchmarks:", className="text-info"))
            analysis_content.append(
                html.P(f"Successful pilots: {len(successful_df)} samples (≥ 0.7)", className="text-light small"))
            analysis_content.append(
                html.P(f"Unsuccessful pilots: {len(unsuccessful_df)} samples (< 0.7)", className="text-light small"))
            analysis_content.append(
                html.P(f"Total gaze metrics analyzed: {len(selected_metrics)}", className="text-light small"))

            # Add general patterns with statistical basis
            analysis_content.append(html.H6("Successful Pilot Pattern:", className="text-success mt-3"))
            for indicator in success_analysis_config['success_patterns']['indicators']:
                analysis_content.append(html.P(f"✓ {indicator}", className="text-light small mb-1"))

            analysis_content.append(html.H6("Failed Pilot Pattern:", className="text-danger mt-3"))
            for indicator in success_analysis_config['failure_patterns']['indicators']:
                analysis_content.append(html.P(f"✗ {indicator}", className="text-light small mb-1"))

        return analysis_content

    def create_attention_distribution_figure():
        """Create Q2: Attention Distribution Comparison graph"""
        if not pattern_data['available']:
            return create_error_figure("Pattern data not available")

        aoi_labels_full = [AOI_NAMES[k] for k in AOI_NAMES.keys()]
        success_totals = [pattern_data['success_aoi'][aoi] for aoi in AOI_NAMES.keys()]
        fail_totals = [pattern_data['fail_aoi'][aoi] for aoi in AOI_NAMES.keys()]

        success_total = sum(success_totals)
        fail_total = sum(fail_totals)
        success_pcts = [(x / success_total) * 100 for x in success_totals]
        fail_pcts = [(x / fail_total) * 100 for x in fail_totals]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=aoi_labels_full,
            y=success_pcts,
            name='Successful',
            marker_color='#2ecc71',
            text=[f'{v:.1f}%' for v in success_pcts],
            textposition='outside',
            textfont=dict(size=10, color='white')
        ))

        fig.add_trace(go.Bar(
            x=aoi_labels_full,
            y=fail_pcts,
            name='Unsuccessful',
            marker_color='#e74c3c',
            text=[f'{v:.1f}%' for v in fail_pcts],
            textposition='outside',
            textfont=dict(size=10, color='white')
        ))

        fig.update_layout(
            title="<b>Q2: Attention Distribution Comparison</b>",
            xaxis_title="Areas of Interest (AOI)",
            yaxis_title="% of Gaze Time",
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                font=dict(color='white')
            )
        )

        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

        return fig

    def create_aoi_differences_figure():
        """Create Q2: Key Differences in AOI Focus graph"""
        if not pattern_data['available']:
            return create_error_figure("Pattern data not available")

        aoi_labels_full = [AOI_NAMES[k] for k in AOI_NAMES.keys()]
        success_totals = [pattern_data['success_aoi'][aoi] for aoi in AOI_NAMES.keys()]
        fail_totals = [pattern_data['fail_aoi'][aoi] for aoi in AOI_NAMES.keys()]

        success_total = sum(success_totals)
        fail_total = sum(fail_totals)
        success_pcts = [(x / success_total) * 100 for x in success_totals]
        fail_pcts = [(x / fail_total) * 100 for x in fail_totals]

        differences = [s - f for s, f in zip(success_pcts, fail_pcts)]
        diff_colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in differences]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=aoi_labels_full,
            y=differences,
            marker_color=diff_colors,
            text=[f'{d:+.1f}%' for d in differences],
            textposition='outside',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{x}</b><br>Difference: %{y:+.1f}%<br>(Positive = more in successful)<extra></extra>'
        ))

        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(aoi_labels_full) - 0.5,
            y0=0, y1=0,
            line=dict(dash="dash", color="white", width=1),
            opacity=0.5
        )

        fig.update_layout(
            title="<b>Q2: Key Differences in AOI Focus</b>",
            xaxis_title="Areas of Interest (AOI)",
            yaxis_title="Difference in % Gaze Time (Success - Fail)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550,
            showlegend=False
        )

        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

        return fig

    def create_success_patterns_figure():
        """Create Q3: Dominant Patterns - Successful graph"""
        if not pattern_data['available']:
            return create_error_figure("Pattern data not available")

        top15_success = pattern_data['success_df'].nlargest(15, pattern_data['freq_col']).copy()
        success_patterns_readable = [pattern_to_readable(p) for p in top15_success[pattern_data['pattern_col']]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=success_patterns_readable[::-1],
            x=top15_success[pattern_data['freq_col']].values[::-1],
            orientation='h',
            marker_color='#2ecc71',
            text=top15_success[pattern_data['freq_col']].values[::-1],
            textposition='outside',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
        ))

        fig.update_layout(
            title="<b>Q3: Dominant Patterns - Successful</b>",
            xaxis_title="Frequency",
            yaxis_title="Pattern Sequence",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550,
            showlegend=False
        )

        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

        return fig

    def create_fail_patterns_figure():
        """Create Q3: Dominant Patterns - Unsuccessful graph"""
        if not pattern_data['available']:
            return create_error_figure("Pattern data not available")

        top15_fail = pattern_data['fail_df'].nlargest(15, pattern_data['freq_col']).copy()
        fail_patterns_readable = [pattern_to_readable(p) for p in top15_fail[pattern_data['pattern_col']]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=fail_patterns_readable[::-1],
            x=top15_fail[pattern_data['freq_col']].values[::-1],
            orientation='h',
            marker_color='#e74c3c',
            text=top15_fail[pattern_data['freq_col']].values[::-1],
            textposition='outside',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
        ))

        fig.update_layout(
            title="<b>Q3: Dominant Patterns - Unsuccessful</b>",
            xaxis_title="Frequency",
            yaxis_title="Pattern Sequence",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550,
            showlegend=False
        )

        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

        return fig

    def create_error_figure(message):
        """Create an error figure when pattern data is not available"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550
        )
        return fig

    # Callbacks for showing/hiding controls based on mode
    @app.callback(
        [Output('aoi-dgm-controls', 'className'),
         Output('pattern-controls', 'className')],
        [Input('mode-switch', 'value')]
    )
    def toggle_controls(mode_switch_value):
        if mode_switch_value:  # Patterns mode
            return 'hidden', ''
        else:  # AOI DGM mode
            return '', 'hidden'

    # Main callback
    @app.callback(
        [Output('main-visualization', 'figure'),
         Output('behavior-characterization', 'children'),
         Output('success-analysis', 'children'),
         Output('current-data', 'data')],
        [Input('metrics-dropdown', 'value'),
         Input('normalization-toggle', 'value'),
         Input('group-checklist', 'value'),
         Input('pilot-dropdown', 'value'),
         Input('visualization-type', 'value'),
         Input('pattern-chart-type', 'value'),
         Input('mode-switch', 'value'),
         Input('reset-btn', 'n_clicks')],
        [State('metrics-dropdown', 'options')]
    )
    def update_dashboard(selected_metrics, normalize, selected_groups, selected_pilot, visualization_type,
                         pattern_chart_type, mode_switch_value, reset_clicks, metric_options):
        # Handle reset button
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-btn.n_clicks':
            selected_metrics = default_metrics
            normalize = True
            selected_groups = ["Successful", "Unsuccessful"]
            selected_pilot = None
            visualization_type = 'radar'
            pattern_chart_type = 'attention_distribution'
            mode_switch_value = False

        # Handle pattern chart types when in Patterns mode
        if mode_switch_value:
            if pattern_chart_type == 'attention_distribution':
                fig = create_attention_distribution_figure()
            elif pattern_chart_type == 'aoi_differences':
                fig = create_aoi_differences_figure()
            elif pattern_chart_type == 'success_patterns':
                fig = create_success_patterns_figure()
            elif pattern_chart_type == 'fail_patterns':
                fig = create_fail_patterns_figure()
            else:
                fig = create_attention_distribution_figure()

            # For pattern charts, we don't need the other analysis panels
            behavior_characterization = []
            success_analysis = []
            stats_data = {}

            # Add pattern insights if data is available
            if pattern_data['available']:
                behavior_characterization.append(html.H5("Pattern Analysis", className="text-info mb-3"))

                # Add some basic insights
                aoi_labels = list(AOI_NAMES.keys())
                success_pcts = [(pattern_data['success_aoi'][aoi] / sum(pattern_data['success_aoi'].values())) * 100 for
                                aoi in aoi_labels]
                fail_pcts = [(pattern_data['fail_aoi'][aoi] / sum(pattern_data['fail_aoi'].values())) * 100 for aoi in
                             aoi_labels]
                differences = [s - f for s, f in zip(success_pcts, fail_pcts)]
                max_diff_idx = differences.index(max(differences, key=abs))
                max_diff_aoi = AOI_NAMES[aoi_labels[max_diff_idx]]

                behavior_characterization.append(html.Div([
                    html.Strong("Key Insight:", className="text-light"),
                    html.Br(),
                    html.Span(f"Largest attention difference: {max_diff_aoi} ({differences[max_diff_idx]:+.1f}%)",
                              className="text-light small")
                ], className="pattern-metric"))

            return fig, behavior_characterization, success_analysis, stats_data

        # Original visualization logic for AOI DGM mode
        if not selected_metrics or len(selected_metrics) < 3:
            selected_metrics = default_metrics[:3]

        # Prepare data for other visualization types
        stats_data = {}

        # Normalization function
        def normalize_series(series):
            range_val = series.max() - series.min()
            if range_val == 0:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - series.min()) / range_val

        # Add group data to stats_data
        if "Successful" in selected_groups:
            stats_data['Successful'] = df[df['pilot_success'] == 'Successful'][selected_metrics].mean().to_dict()

        if "Unsuccessful" in selected_groups:
            stats_data['Unsuccessful'] = df[df['pilot_success'] == 'Unsuccessful'][selected_metrics].mean().to_dict()

        if "All" in selected_groups:
            stats_data['All'] = df[selected_metrics].mean().to_dict()

        # Add individual pilot data if selected
        if selected_pilot:
            stats_data[f'Pilot {selected_pilot}'] = df[df['PID'] == selected_pilot][selected_metrics].iloc[0].to_dict()

        # Create visualization based on selected type
        if visualization_type == 'bar':
            # Bar chart visualization
            bar_metrics = [k for k in bar_config.keys() if k in df.columns]
            bar_traces = []
            individual_bar = [bar_config[m]['name'] for m in bar_metrics]

            if bar_metrics:
                # Add group traces (bar)
                if "Successful" in selected_groups:
                    bar_success_raw = df[df['pilot_success'] == 'Successful'][bar_metrics].mean()
                    bar_success = normalize_series(bar_success_raw) if normalize else bar_success_raw

                    bar_traces.append(go.Bar(
                        x=individual_bar,
                        y=bar_success.values,
                        name='Successful Pilots',
                        marker=dict(color='rgb(76, 175, 80)')
                    ))

                if "Unsuccessful" in selected_groups:
                    bar_unsuccess_raw = df[df['pilot_success'] == 'Unsuccessful'][bar_metrics].mean()
                    bar_unsuccess = normalize_series(bar_unsuccess_raw) if normalize else bar_unsuccess_raw

                    bar_traces.append(go.Bar(
                        x=individual_bar,
                        y=bar_unsuccess.values,
                        name='Unsuccessful Pilots',
                        marker=dict(color='rgb(244, 67, 54)')
                    ))

            label_y = 'Mean Proportion Fixation Duration' if normalize else 'Metric Value'

            fig = go.Figure(data=bar_traces)
            fig.update_layout(
                barmode='group',
                xaxis=dict(
                    title=dict(
                        text='AOI',
                        font=dict(color='white')
                    ),
                    tickfont=dict(color='white')
                ),
                yaxis=dict(
                    title=dict(
                        text=label_y,
                        font=dict(color='white')
                    ),
                    tickfont=dict(color='white')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(
                    bgcolor='rgba(0,0,0,0.5)',
                    font=dict(color='white')
                ),
                title=dict(
                    text='Proportion of Fixation Duration per AOI (Successful vs Unsuccessful)',
                    font=dict(color='white', size=16),
                    x=0.5
                ),
                height=550
            )

        elif visualization_type == 'parallel':
            # Parallel coordinates visualization
            fig = go.Figure(data=go.Parcoords(
                line=dict(color=df['Approach_Score'],
                          colorscale='Viridis',
                          showscale=True,
                          cmin=df['Approach_Score'].min(),
                          cmax=df['Approach_Score'].max()),
                dimensions=[dict(range=[df[col].min(), df[col].max()],
                                 label=metrics_config[col]['name'], values=df[col])
                            for col in selected_metrics]
            ))
            fig.update_layout(
                title="Parallel Coordinates Plot",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=550
            )
        else:
            # Radar chart visualization (default)
            traces = []

            # Add group traces
            if "Successful" in selected_groups:
                successful_data = df[df['pilot_success'] == 'Successful'][selected_metrics].mean()
                if normalize:
                    successful_data = normalize_series(successful_data)

                traces.append(go.Scatterpolar(
                    r=np.append(successful_data.values, successful_data.values[0]),
                    theta=np.append([metrics_config[m]['name'] for m in selected_metrics],
                                    [metrics_config[selected_metrics[0]]['name']]),
                    fill='toself',
                    fillcolor='rgba(76, 175, 80, 0.4)',
                    line=dict(color='rgb(76, 175, 80)', width=3),
                    name='Successful Pilots (Avg)',
                    hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
                ))

            if "Unsuccessful" in selected_groups:
                unsuccessful_data = df[df['pilot_success'] == 'Unsuccessful'][selected_metrics].mean()
                if normalize:
                    unsuccessful_data = normalize_series(unsuccessful_data)

                traces.append(go.Scatterpolar(
                    r=np.append(unsuccessful_data.values, unsuccessful_data.values[0]),
                    theta=np.append([metrics_config[m]['name'] for m in selected_metrics],
                                    [metrics_config[selected_metrics[0]]['name']]),
                    fill='toself',
                    fillcolor='rgba(244, 67, 54, 0.4)',
                    line=dict(color='rgb(244, 67, 54)', width=3),
                    name='Unsuccessful Pilots (Avg)',
                    hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
                ))

            if "All" in selected_groups:
                all_data = df[selected_metrics].mean()
                if normalize:
                    all_data = normalize_series(all_data)

                traces.append(go.Scatterpolar(
                    r=np.append(all_data.values, all_data.values[0]),
                    theta=np.append([metrics_config[m]['name'] for m in selected_metrics],
                                    [metrics_config[selected_metrics[0]]['name']]),
                    fill='toself',
                    fillcolor='rgba(33, 150, 243, 0.4)',
                    line=dict(color='rgb(33, 150, 243)', width=3),
                    name='All Pilots (Avg)',
                    hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
                ))

            # Add individual pilot trace if selected
            if selected_pilot:
                pilot_data = df[df['PID'] == selected_pilot][selected_metrics].iloc[0]
                if normalize:
                    pilot_data = normalize_series(pilot_data)

                pilot_success = df[df['PID'] == selected_pilot]['pilot_success'].iloc[0]
                # Use blue color for individual pilots
                color = 'rgb(33, 150, 243)'  # Blue color for individual pilots

                traces.append(go.Scatterpolar(
                    r=np.append(pilot_data.values, pilot_data.values[0]),
                    theta=np.append([metrics_config[m]['name'] for m in selected_metrics],
                                    [metrics_config[selected_metrics[0]]['name']]),
                    fill=None,  # No fill for individual pilots
                    line=dict(color=color, width=4, dash='dash'),
                    name=f'Pilot {selected_pilot} ({pilot_success})',
                    hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
                ))

            fig = go.Figure(data=traces)
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1] if normalize else [df[selected_metrics].min().min(),
                                                        df[selected_metrics].max().max()],
                        gridcolor='rgba(255,255,255,0.3)',
                        tickfont=dict(color='white')
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='white', size=12),
                        gridcolor='rgba(255,255,255,0.3)',
                        rotation=90
                    ),
                    bgcolor='rgba(0,0,0,0.2)'
                ),
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(0,0,0,0.5)',
                    font=dict(color='white')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=550,
                title=dict(
                    text=f"Gaze Behavior Profile Comparison<br><sub>{'Normalized ' if normalize else ''}Metrics: {len(selected_metrics)} selected</sub>",
                    font=dict(color='white', size=16),
                    x=0.5
                )
            )

        # Create behavior characterization (only for group comparisons)
        behavior_characterization = characterize_behavior(selected_metrics, stats_data)

        # Create success analysis (unchanged)
        success_analysis = analyze_success_patterns(selected_metrics, stats_data, selected_pilot)

        return fig, behavior_characterization, success_analysis, stats_data

    return app


# Create and run the dashboard
print("Starting Enhanced Radar Chart Dashboard...")
print("Loading data from AOI_DGMs.csv...")
print(f"Loaded {len(df)} pilots")
print(f"Successful: {len(df[df['pilot_success'] == 'Successful'])}")
print(f"Unsuccessful: {len(df[df['pilot_success'] == 'Unsuccessful'])}")
if pattern_data['available']:
    print("✓ Pattern comparison data loaded successfully")
else:
    print("⚠ Pattern comparison data not available")
print("\nDashboard will open in your web browser...")

# Create the dashboard app
app = create_enhanced_radar_dashboard(df)

if __name__ == '__main__':
    app.run(debug=True, port=8050)