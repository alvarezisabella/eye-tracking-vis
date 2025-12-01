import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from collections import Counter
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate
import dash_daq as daq

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
        'back_and_forth': (back_forth / total) * 100,
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
    Create a dashboard with linear scale visualization showing metrics as dots along a line
    """

    # Available metrics with descriptions
    metrics_config = {
        'Mean_fixation_duration_s': {
            'name': 'Fixation Duration (s)',
            'description': 'Average time eyes remain stationary. Shorter = faster information processing',
            'min': 0.1,
            'max': 1.0,
            'default': 0.3
        },
        'mean_saccade_length': {
            'name': 'Saccade Length (px)',
            'description': 'Average distance between fixations. Moderate = efficient scanning',
            'min': 50,
            'max': 500,
            'default': 200
        },
        'Average_Peak_Saccade_Velocity': {
            'name': 'Saccade Velocity (deg/s)',
            'description': 'Speed of eye movements. Higher = quicker scanning',
            'min': 100,
            'max': 800,
            'default': 350
        },
        'stationary_entropy': {
            'name': 'Fixation Entropy',
            'description': 'Randomness of fixation locations. Lower = more systematic viewing',
            'min': 0.5,
            'max': 3.0,
            'default': 1.5
        },
        'transition_entropy': {
            'name': 'Transition Entropy',
            'description': 'Predictability of scanpaths. Lower = more consistent patterns',
            'min': 0.5,
            'max': 3.0,
            'default': 1.5
        },
        'Average_Blink_Rate_per_Minute': {
            'name': 'Blink Rate (/min)',
            'description': 'Blinks per minute. Moderate = optimal cognitive load',
            'min': 5,
            'max': 50,
            'default': 20
        },
        'fixation_to_saccade_ratio': {
            'name': 'Fixation/Saccade Ratio',
            'description': 'Balance between processing and searching. Balanced = efficient',
            'min': 0.5,
            'max': 3.0,
            'default': 1.5
        },
        'Total_Number_of_Fixations': {
            'name': 'Total Fixations',
            'description': 'Total number of fixations. Moderate = efficient information gathering',
            'min': 50,
            'max': 500,
            'default': 200
        },
        'Sum_of_all_fixation_duration_s': {
            'name': 'Total Fixation Time (s)',
            'description': 'Cumulative fixation duration',
            'min': 30,
            'max': 300,
            'default': 120
        },
        'total_number_of_saccades': {
            'name': 'Total Saccades',
            'description': 'Total number of eye movements between fixations',
            'min': 40,
            'max': 400,
            'default': 150
        },
        'Approach_Score': {
            'name': 'Approach Score',
            'description': 'Overall approach performance score (0-1 scale)',
            'min': 0.0,
            'max': 1.0,
            'default': 0.7
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
        'fixation_to_saccade_ratio', 'Approach_Score'
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

                /* Pattern explanation styles */
                .pattern-explanation {
                    background: rgba(255,255,255,0.08);
                    padding: 8px;
                    border-radius: 4px;
                    margin: 5px 0;
                    border-left: 3px solid #666;
                }

                .pattern-key {
                    font-weight: bold;
                    color: #42a5f5;
                    margin-right: 5px;
                }

                .pattern-meaning {
                    font-size: 0.85em;
                    color: #cccccc;
                }

                .pattern-list {
                    background: rgba(255,255,255,0.05);
                    padding: 8px;
                    border-radius: 4px;
                    margin: 5px 0;
                }

                .pattern-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 4px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }

                .pattern-item:last-child {
                    border-bottom: none;
                }

                .pattern-number {
                    font-weight: bold;
                    color: #42a5f5;
                    min-width: 30px;
                }

                .pattern-description {
                    flex-grow: 1;
                    font-size: 0.9em;
                }

                .pattern-frequency {
                    font-family: monospace;
                    color: #cccccc;
                    min-width: 60px;
                    text-align: right;
                }

                /* Hidden elements */
                .hidden {
                    display: none !important;
                }

                /* Custom input styles */
                .custom-numeric-input {
                    background-color: #2c3e50 !important;
                    color: white !important;
                    border: 1px solid #34495e !important;
                    border-radius: 4px !important;
                }

                .custom-numeric-input input {
                    color: white !important;
                    background-color: #2c3e50 !important;
                }

                /* Switch styling */
                .custom-switch .form-check-input:checked {
                    background-color: #4caf50 !important;
                    border-color: #4caf50 !important;
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
                html.P("Linear Scale Visualization Showing Metrics as Dots",
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
                            value=False,
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

                            # Custom Values Switch (NEW - replaces dropdown option)
                            html.Div([
                                html.Label("Enable Custom Values:", className="fw-bold text-light mb-2"),
                                dbc.Switch(
                                    id="custom-values-switch",
                                    label="Add Custom Values to Graph",
                                    value=False,
                                    className="mb-3 custom-switch"
                                ),
                            ], className="mb-3 p-3 bg-dark rounded"),

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
                                options=[
                                    {'label': 'All Pilots', 'value': 'All'},
                                    *[{'label': f'Pilot {pid}', 'value': pid}
                                      for pid in df['PID'].unique()]
                                ],
                                value='All',
                                placeholder="Select individual pilot...",
                                className="mb-3"
                            ),

                            # Visualization Type - Main Charts
                            html.Label("Visualization Type:", className="fw-bold text-light"),
                            dcc.RadioItems(
                                id='visualization-type',
                                options=[
                                    {'label': ' Linear Scale', 'value': 'linear'},
                                    {'label': ' Parallel Coordinates', 'value': 'parallel'}
                                ],
                                value='linear',
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

                        # Action Buttons
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
                    # Behavior Characterization Card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Behavior Characterization", className="h5"),
                            dbc.CardBody(id='behavior-characterization')
                        ])
                    ], width=6, id='behavior-col'),

                    # Success Analysis Card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Success Analysis", className="h5"),
                            dbc.CardBody(id='success-analysis')
                        ])
                    ], width=6, id='success-col')
                ], className="mt-3", id='analysis-row')
            ], width=9)
        ]),

        # Custom Input Modal (will pop up when switch is turned ON)
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Enter Custom Metric Values")),
            dbc.ModalBody([
                html.P("Enter values for your custom metrics. These will appear as a 'Custom' line in the graph.",
                       className="text-light mb-4"),

                # Dynamic input fields for selected metrics
                html.Div(id='custom-input-fields', className="mb-3"),

                html.Div([
                    html.Strong("Hint:", className="text-warning me-2"),
                    html.Span(
                        "Use the sliders or type values directly. Values outside typical ranges will be shown in red.",
                        className="text-light small")
                ], className="mb-4 p-2 bg-dark rounded"),

                # Option to use defaults
                dbc.Checklist(
                    options=[
                        {"label": " Fill with successful pilot averages", "value": "success_avg"},
                        {"label": " Fill with unsuccessful pilot averages", "value": "unsuccess_avg"}
                    ],
                    value=[],
                    id="fill-defaults-checklist",
                    switch=True,
                    className="mb-3"
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-custom-btn", className="me-2", color="secondary"),
                dbc.Button("Clear All", id="clear-custom-btn", className="me-2", color="warning"),
                dbc.Button("Apply Custom Values", id="apply-custom-btn", color="primary")
            ])
        ], id="custom-input-modal", size="lg", is_open=False, backdrop="static"),

        # Hidden divs for storing custom values and switch state
        dcc.Store(id='custom-values-store', data={}),
        dcc.Store(id='custom-switch-state', data=False),
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

    def create_linear_scale_figure(selected_metrics, selected_groups, selected_pilot, custom_values=None,
                                   custom_enabled=False):
        """Create a linear scale visualization with dots for metrics"""

        # Filter out any invalid metrics
        plot_metrics = [m for m in selected_metrics if m in metrics_config]

        # If no valid metrics, use defaults
        if not plot_metrics:
            plot_metrics = default_metrics[:3]

        # Prepare data
        metric_names = [metrics_config[m]['name'] for m in plot_metrics]

        # Create figure FIRST - before any condition checks
        fig = go.Figure()

        # Add metric lines and labels
        for i, metric_name in enumerate(metric_names):
            fig.add_shape(
                type="line",
                x0=0, x1=1, y0=i, y1=i,
                line=dict(color="rgba(255,255,255,0.3)", width=1),
            )

            # Add metric name labels
            fig.add_annotation(
                x=-0.00, y=i,
                text=metric_name,
                showarrow=False,
                xref="paper", yref="y",
                xanchor="right",
                font=dict(color="white", size=10)
            )

        # Add group data
        group_colors = {
            'Successful': '#2ecc71',
            'Unsuccessful': '#e74c3c',
            'All': '#3498db',
            'Custom': '#f39c12'  # Orange for custom values
        }

        group_symbols = {
            'Successful': 'circle',
            'Unsuccessful': 'square',
            'All': 'diamond',
            'Custom': 'star'  # Star for custom values
        }

        # Check if we have any data to plot
        has_data = False

        # Add dots for each group
        for group in selected_groups:
            if group == 'Successful':
                group_data = successful_df[plot_metrics].mean()
            elif group == 'Unsuccessful':
                group_data = unsuccessful_df[plot_metrics].mean()
            elif group == 'All':
                group_data = df[plot_metrics].mean()
            elif group == 'Custom' and custom_enabled and custom_values:
                # Use custom values
                group_data = pd.Series({m: custom_values.get(m, metrics_config[m]['default'])
                                        for m in plot_metrics})
            else:
                continue

            # If we get here, we have data to plot
            has_data = True

            x_values = []
            for metric in plot_metrics:
                # Scale to 0-1 for visualization
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val == min_val:
                    x_values.append(0.5)
                else:
                    value = group_data[metric]
                    # Clip value to reasonable range for scaling
                    clipped_value = max(min_val, min(max_val, value))
                    x_values.append((clipped_value - min_val) / (max_val - min_val))

            fig.add_trace(go.Scatter(
                x=x_values,
                y=list(range(len(plot_metrics))),
                mode='markers',
                marker=dict(
                    color=group_colors[group],
                    size=12,
                    symbol=group_symbols[group],
                    line=dict(color='white', width=1)
                ),
                name=f'{group} {"(Custom)" if group == "Custom" else ""}',
                hovertemplate='<b>%{text}</b><br>Value: %{customdata:.3f}<extra></extra>',
                text=[metrics_config[m]['name'] for m in plot_metrics],
                customdata=[group_data[m] for m in plot_metrics]
            ))

        # Add individual pilot if selected and not "All"
        if selected_pilot and selected_pilot != "All":
            try:
                pilot_data = df[df['PID'] == selected_pilot][plot_metrics].iloc[0]
                pilot_success = df[df['PID'] == selected_pilot]['pilot_success'].iloc[0]
                has_data = True

                x_values = []
                for metric in plot_metrics:
                    min_val = df[metric].min()
                    max_val = df[metric].max()
                    if max_val == min_val:
                        x_values.append(0.5)
                    else:
                        x_values.append((pilot_data[metric] - min_val) / (max_val - min_val))

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=list(range(len(plot_metrics))),
                    mode='markers',
                    marker=dict(
                        color='#9b59b6',  # Purple for individual pilot
                        size=16,
                        symbol='cross',
                        line=dict(color='white', width=2)
                    ),
                    name=f'Pilot {selected_pilot} ({pilot_success})',
                    hovertemplate='<b>%{text}</b><br>Value: %{customdata:.3f}<extra></extra>',
                    text=[metrics_config[m]['name'] for m in plot_metrics],
                    customdata=[pilot_data[m] for m in plot_metrics]
                ))
            except (IndexError, KeyError):
                # Pilot not found or data missing
                pass

        # If no data was added, add a placeholder message
        if not has_data:
            fig.add_annotation(
                text="No data to display. Please select groups or a pilot.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white")
            )

        # Update layout with legend positioned above the plot, moved to the left
        title_text = "Gaze Metrics Linear Scale Visualization"
        if custom_enabled and custom_values:
            title_text += ""

        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(color='white', size=16),
                x=0.5
            ),
            xaxis=dict(
                title=dict(
                    text='Scaled Value (0-1)',
                    font=dict(color='white')
                ),
                range=[-0.05, 1.05],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                range=[-0.5, len(plot_metrics) - 0.5],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                font=dict(color='white', size=10),
                yanchor="top",
                y=1.15,  # Position above the plot (greater than 1)
                xanchor="left",  # Changed from "center" to "left"
                x=-0.2,  # Changed from 0.5 to 0.02 (moved to left)
                orientation="h",
                borderwidth=1,
                bordercolor="rgba(255,255,255,0.3)"
            ),
            margin=dict(l=130, r=50, t=100, b=50)  # Increased top margin for legend
        )

        # Add an invisible trace to push the legend to the top
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=0),
            showlegend=False
        ))

        return fig

    def create_attention_distribution_figure():
        """Create Attention Distribution Comparison graph"""
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
            title="<b>Attention Distribution Comparison</b>",
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
        """Create Key Differences in AOI Focus graph"""
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
            title="<b>Key Differences in AOI Focus</b>",
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
        """Create Dominant Patterns - Successful graph"""
        if not pattern_data['available']:
            return create_error_figure("Pattern data not available"), None

        top15_success = pattern_data['success_df'].nlargest(15, pattern_data['freq_col']).copy()

        # Use numbers for y-axis labels instead of pattern strings
        pattern_numbers = [f"Pattern {i + 1}" for i in range(len(top15_success))]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=pattern_numbers[::-1],  # Reverse to show Pattern 1 at top
            x=top15_success[pattern_data['freq_col']].values[::-1],
            orientation='h',
            marker_color='#2ecc71',
            text=top15_success[pattern_data['freq_col']].values[::-1],
            textposition='outside',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
        ))

        fig.update_layout(
            title="<b>Dominant Patterns - Successful</b>",
            xaxis_title="Frequency",
            yaxis_title="Pattern Sequence",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550,
            showlegend=False,
            margin=dict(l=120, r=50, t=80, b=50)
        )

        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(
            tickfont=dict(color='white', size=10),
            gridcolor='rgba(255,255,255,0.1)',
            automargin=True
        )

        return fig, top15_success

    def create_fail_patterns_figure():
        """Create Dominant Patterns - Unsuccessful graph"""
        if not pattern_data['available']:
            return create_error_figure("Pattern data not available"), None

        top15_fail = pattern_data['fail_df'].nlargest(15, pattern_data['freq_col']).copy()

        # Use numbers for y-axis labels instead of pattern strings
        pattern_numbers = [f"Pattern {i + 1}" for i in range(len(top15_fail))]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=pattern_numbers[::-1],  # Reverse to show Pattern 1 at top
            x=top15_fail[pattern_data['freq_col']].values[::-1],
            orientation='h',
            marker_color='#e74c3c',
            text=top15_fail[pattern_data['freq_col']].values[::-1],
            textposition='outside',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
        ))

        fig.update_layout(
            title="<b>Dominant Patterns - Unsuccessful</b>",
            xaxis_title="Frequency",
            yaxis_title="Pattern Sequence",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=550,
            showlegend=False,
            margin=dict(l=100, r=50, t=80, b=50)
        )

        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(
            tickfont=dict(color='white', size=10),
            gridcolor='rgba(255,255,255,0.1)',
            automargin=True
        )

        return fig, top15_fail

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
        return fig, None

    def create_parallel_coordinates_figure(selected_metrics, selected_pilot=None, custom_values=None,
                                           custom_enabled=False):
        """Create parallel coordinates plot with pilot filtering, highlighting, and custom values"""

        # Filter out any invalid metrics
        valid_metrics = [m for m in selected_metrics if m in metrics_config]

        if not valid_metrics:
            return go.Figure()

        # Create a copy of the dataframe for all pilots
        plot_df = df.copy()

        # Prepare dimensions list for all selected metrics
        dimensions = []
        for col in valid_metrics:
            dimensions.append(dict(
                range=[plot_df[col].min(), plot_df[col].max()],
                label=metrics_config[col]['name'],
                values=plot_df[col]
            ))

        # Handle different scenarios
        if selected_pilot and selected_pilot != "All":
            # Highlight selected pilot in orange, show custom values in red
            colors = []
            pilot_indices = []

            # Identify which rows belong to the selected pilot
            for i in range(len(plot_df)):
                if plot_df.iloc[i]['PID'] == selected_pilot:
                    colors.append(0.8)  # High value for selected pilot (will be orange)
                    pilot_indices.append(i)
                else:
                    colors.append(0.2)  # Low value for others (will be faded blue)

            # Add custom values row if available
            all_values = []
            if custom_enabled and custom_values:
                # First add all original data
                for i in range(len(plot_df)):
                    row_values = [plot_df.iloc[i][metric] for metric in valid_metrics]
                    all_values.append(row_values)

                # Create custom values row
                custom_row = []
                for metric in valid_metrics:
                    if metric in custom_values:
                        custom_row.append(custom_values[metric])
                    else:
                        # Use default if custom value not provided
                        custom_row.append(metrics_config[metric]['default'])

                # Then add custom values row
                all_values.append(custom_row)
                colors.append(1.0)  # Highest value for custom values (will be red)

            # Create colorscale with normalized values (0 to 1)
            colorscale = [
                [0.0, 'rgba(30, 136, 229, 0.2)'],  # Faded blue for others (0.0-0.3)
                [0.3, 'rgba(30, 136, 229, 0.5)'],  # Medium blue
                [0.6, 'rgba(30, 136, 229, 0.8)'],  # Strong blue
                [0.8, '#FF9800'],  # Orange for selected pilot
                [0.9, '#FF9800'],  # Orange continuation
                [1.0, '#FF0000']  # Red for custom values
            ]

            # Create the figure
            if custom_enabled and custom_values:
                # Use combined data with custom values
                combined_dimensions = []
                for idx, metric in enumerate(valid_metrics):
                    metric_values = [row[idx] for row in all_values]
                    combined_dimensions.append(dict(
                        range=[min(metric_values), max(metric_values)],
                        label=metrics_config[metric]['name'],
                        values=metric_values
                    ))

                fig = go.Figure(data=go.Parcoords(
                    line=dict(
                        color=colors,
                        colorscale=colorscale,
                        showscale=True,
                        cmin=0,
                        cmax=1.0,
                        colorbar=dict(
                            title="Line Type",
                            tickvals=[0.1, 0.8, 1.0],
                            ticktext=['Other Pilots', f'Pilot {selected_pilot}', 'Custom Values'],
                            len=0.5
                        )
                    ),
                    dimensions=combined_dimensions
                ))
            else:
                # Regular plot without custom values
                fig = go.Figure(data=go.Parcoords(
                    line=dict(
                        color=colors,
                        colorscale=[
                            [0, 'rgba(30, 136, 229, 0.2)'],
                            [0.5, 'rgba(30, 136, 229, 0.5)'],
                            [1, '#FF9800']
                        ],
                        showscale=True,
                        cmin=0,
                        cmax=1.0,
                        colorbar=dict(
                            title="Line Type",
                            tickvals=[0.25, 0.9],
                            ticktext=['Other Pilots', f'Pilot {selected_pilot}'],
                            len=0.5
                        )
                    ),
                    dimensions=dimensions
                ))

            # Update layout
            title_text = f"Parallel Coordinates Plot - Pilot {selected_pilot} Highlighted"
            if custom_enabled and custom_values:
                title_text += " with Custom Values"

            fig.update_layout(
                title=dict(
                    text=title_text,
                    font=dict(color='white', size=16),
                    x=0.5
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=550,
                margin=dict(l=80, r=80, t=80, b=50)
            )

            # Add annotation explaining the visualization
            annotation_text = f"Orange line: Pilot {selected_pilot} | Faded lines: All other pilots"
            if custom_enabled and custom_values:
                annotation_text += " | Red line: Custom Values"

            fig.add_annotation(
                x=0.5,
                y=-0.1,
                xref="paper",
                yref="paper",
                text=annotation_text,
                showarrow=False,
                font=dict(color='white', size=12),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            )

        else:
            # Show all pilots with Approach Score coloring, plus custom values in red
            if custom_enabled and custom_values:
                # We need to create a combined dataset with custom values
                all_values = []
                colors = []

                # Add all pilot data with Approach Score coloring
                for i in range(len(plot_df)):
                    row_values = [plot_df.iloc[i][metric] for metric in valid_metrics]
                    all_values.append(row_values)
                    colors.append(plot_df.iloc[i]['Approach_Score'] if 'Approach_Score' in plot_df.columns else 0.5)

                # Create custom values row
                custom_row = []
                for metric in valid_metrics:
                    if metric in custom_values:
                        custom_row.append(custom_values[metric])
                    else:
                        # Use default if custom value not provided
                        custom_row.append(metrics_config[metric]['default'])

                # Add custom values row
                all_values.append(custom_row)
                colors.append(1.0)  # Special value for custom values (red)

                # Create dimensions with combined data
                dimensions_with_custom = []
                for idx, metric in enumerate(valid_metrics):
                    metric_values = [row[idx] for row in all_values]
                    dimensions_with_custom.append(dict(
                        range=[min(metric_values), max(metric_values)],
                        label=metrics_config[metric]['name'],
                        values=metric_values
                    ))

                # Normalize Approach Score values to 0-0.9 range, keep 1.0 for custom
                max_app_score = max(c for c in colors if c < 1.0)  # Max excluding custom
                normalized_colors = []
                for c in colors:
                    if c == 1.0:  # Custom value
                        normalized_colors.append(1.0)
                    else:  # Pilot values
                        normalized_colors.append(c / max_app_score * 0.9)  # Scale to 0-0.9

                # Create custom colorscale that includes red for custom values
                fig = go.Figure(data=go.Parcoords(
                    line=dict(
                        color=normalized_colors,
                        colorscale=[
                            [0.0, '#440154'],  # Dark purple (low Approach Score)
                            [0.25, '#3b528b'],  # Blue
                            [0.5, '#21918c'],  # Teal
                            [0.75, '#5ec962'],  # Green
                            [0.9, '#fde725'],  # Yellow (high Approach Score)
                            [1.0, '#FF0000']  # Red for custom values
                        ],
                        showscale=True,
                        cmin=0,
                        cmax=1.0,
                        colorbar=dict(
                            title="Approach Score / Custom",
                            tickvals=[0, 0.25, 0.5, 0.75, 0.9, 1.0],
                            ticktext=['0.0', '0.25', '0.5', '0.75', '1.0', 'Custom'],
                            len=0.5
                        )
                    ),
                    dimensions=dimensions_with_custom
                ))

                title_text = "Parallel Coordinates Plot - All Pilots with Custom Values (Red)"
            else:
                # Regular plot without custom values
                fig = go.Figure(data=go.Parcoords(
                    line=dict(
                        color=plot_df['Approach_Score'] if 'Approach_Score' in plot_df.columns else [0.5] * len(
                            plot_df),
                        colorscale='Viridis',
                        showscale=True,
                        cmin=plot_df['Approach_Score'].min() if 'Approach_Score' in plot_df.columns else 0,
                        cmax=plot_df['Approach_Score'].max() if 'Approach_Score' in plot_df.columns else 1
                    ),
                    dimensions=dimensions
                ))

                title_text = "Parallel Coordinates Plot - All Pilots"

            # Update layout
            fig.update_layout(
                title=dict(
                    text=title_text,
                    font=dict(color='white', size=16),
                    x=0.5
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=550,
                margin=dict(l=80, r=80, t=80, b=50)
            )

            # Add annotation if custom values are shown
            if custom_enabled and custom_values:
                fig.add_annotation(
                    x=0.5,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    text="Red line: Custom Values | Colored lines: Pilots (color = Approach Score)",
                    showarrow=False,
                    font=dict(color='white', size=12),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )

        return fig

    def get_pattern_explanations():
        """Return explanations for common pattern types"""
        explanations = {
            "Repetitive Patterns": "Single instrument focus - may indicate fixation or tunnel vision",
            "Back-and-Forth Patterns": "Systematic switching between 2-3 key instruments - shows good cross-checking",
            "Complex Patterns": "Multi-instrument scanning - indicates comprehensive situational awareness",
            "Window-Focused": "High external focus - may indicate visual flight reliance",
            "Instrument-Focused": "High panel focus - shows systematic instrument scanning",
            "Att→Alt→Turn": "Core flight parameter monitoring - fundamental attitude and navigation tracking",
            "Speed→RPM→ASI": "Power and performance monitoring - important for approach management"
        }
        return explanations

    def create_pattern_list_content(pattern_df, title, success=True):
        """Create pattern list content for the Behavior Characterization box"""
        content = []

        if pattern_df is not None and len(pattern_df) > 0:
            # Get the actual column names from the pattern data
            pattern_col = pattern_data['pattern_col']
            freq_col = pattern_data['freq_col']

            content.append(html.Div([
                html.Div(f"📋 {title}", className="pattern-header"),
                html.Div([
                    html.Div([
                        html.Span(f"Pattern {i + 1}: ", className="pattern-number"),
                        html.Span(pattern_to_readable(row[pattern_col]), className="pattern-description"),
                        html.Span(f"Freq: {row[freq_col]}", className="pattern-frequency")
                    ], className="pattern-item")
                    for i, (_, row) in enumerate(pattern_df.iterrows())
                ], className="pattern-list")
            ], className="behavior-pattern success-pattern" if success else "behavior-pattern failure-pattern"))

        return content

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

        return characterization_content

    def analyze_success_patterns(selected_metrics, stats_data, selected_pilot=None, custom_values=None,
                                 custom_enabled=False):
        """Analyze gaze patterns to determine success/failure reasons with statistical evidence"""

        analysis_content = []

        # Check if custom values are being used
        if custom_enabled and custom_values and 'Custom' in stats_data:
            analysis_content.append(html.H5("Custom Values Analysis", className="text-info mb-3"))

            # Status header for custom
            status_text = "🎯 CUSTOM VALUES ANALYSIS"
            analysis_content.append(html.Div(status_text, className="success-status"))

            # Add benchmark comparison
            analysis_content.append(html.H6("Comparison with Group Averages:", className="text-warning mt-3"))

            if 'Successful' in stats_data and 'Unsuccessful' in stats_data:
                custom_data = stats_data['Custom']
                success_avg = stats_data['Successful']
                unsuccess_avg = stats_data['Unsuccessful']

                for metric in selected_metrics:
                    if metric in custom_data:
                        custom_val = custom_data[metric]
                        success_val = success_avg.get(metric, 0)
                        unsuccess_val = unsuccess_avg.get(metric, 0)

                        # Calculate percentage differences
                        if success_val != 0:
                            diff_pct_success = ((custom_val - success_val) / success_val) * 100
                        else:
                            diff_pct_success = 0

                        if unsuccess_val != 0:
                            diff_pct_unsuccess = ((custom_val - unsuccess_val) / unsuccess_val) * 100
                        else:
                            diff_pct_unsuccess = 0

                        # Determine if closer to successful or unsuccessful
                        dist_from_success = abs(diff_pct_success)
                        dist_from_unsuccess = abs(diff_pct_unsuccess)

                        if dist_from_success < dist_from_unsuccess:
                            status_class = "metric-good"
                            comparison = f"Closer to successful average ({dist_from_success:.1f}% difference)"
                        else:
                            status_class = "metric-bad"
                            comparison = f"Closer to unsuccessful average ({dist_from_unsuccess:.1f}% difference)"

                        # Add Approach Score threshold check
                        if metric == 'Approach_Score':
                            threshold = benchmarks.get('Approach_Score', {}).get('threshold', 0.7)
                            if custom_val >= threshold:
                                score_status = "Above success threshold"
                                score_class = "text-success"
                            else:
                                score_status = "Below success threshold"
                                score_class = "text-danger"

                            metric_item = html.Div([
                                html.Strong(f"{metrics_config[metric]['name']}: {custom_val:.3f}",
                                            className="text-light"),
                                html.Br(),
                                html.Span(f"Threshold check: {score_status}", className=f"{score_class}"),
                                html.Br(),
                                html.Span(f"{comparison}", className="text-light small"),
                                html.Br(),
                                html.Span(f"Successful average: {success_val:.3f}", className="text-success"),
                                html.Br(),
                                html.Span(f"Unsuccessful average: {unsuccess_val:.3f}", className="text-danger")
                            ], className=f"metric-item {status_class}")
                        else:
                            metric_item = html.Div([
                                html.Strong(f"{metrics_config[metric]['name']}: {custom_val:.3f}",
                                            className="text-light"),
                                html.Br(),
                                html.Span(f"{metrics_config[metric]['description']}", className="metric-description"),
                                html.Br(),
                                html.Span(f"{comparison}", className="text-light small"),
                                html.Br(),
                                html.Span(f"Successful average: {success_val:.3f}", className="text-success"),
                                html.Br(),
                                html.Span(f"Unsuccessful average: {unsuccess_val:.3f}", className="text-danger")
                            ], className=f"metric-item {status_class}")

                        analysis_content.append(metric_item)

                # Overall assessment
                analysis_content.append(html.H6("Overall Assessment:", className="text-warning mt-3"))

                # Count metrics closer to success vs unsuccessful
                success_count = 0
                unsuccess_count = 0

                for metric in selected_metrics:
                    if metric in custom_data:
                        custom_val = custom_data[metric]
                        success_val = success_avg.get(metric, 0)
                        unsuccess_val = unsuccess_avg.get(metric, 0)

                        if success_val != 0 and unsuccess_val != 0:
                            dist_from_success = abs((custom_val - success_val) / success_val) * 100
                            dist_from_unsuccess = abs((custom_val - unsuccess_val) / unsuccess_val) * 100

                            if dist_from_success < dist_from_unsuccess:
                                success_count += 1
                            else:
                                unsuccess_count += 1

                total = success_count + unsuccess_count
                if total > 0:
                    success_percentage = (success_count / total) * 100

                    if success_percentage >= 70:
                        assessment = "STRONG correlation with successful pilot patterns"
                        assessment_class = "text-success"
                    elif success_percentage >= 50:
                        assessment = "MODERATE correlation with successful pilot patterns"
                        assessment_class = "text-warning"
                    else:
                        assessment = "WEAK correlation with successful pilot patterns"
                        assessment_class = "text-danger"

                    analysis_content.append(html.P(
                        f"{success_count} of {total} metrics ({success_percentage:.0f}%) align more closely with successful pilots.",
                        className="text-light small"
                    ))
                    analysis_content.append(html.P(
                        f"Assessment: {assessment}",
                        className=f"{assessment_class} font-weight-bold"
                    ))

            return analysis_content

        # Check if selected_pilot is "All" or None
        if selected_pilot and selected_pilot != "All":
            # Individual pilot analysis
            pilot_key = f'Pilot {selected_pilot}'
            if pilot_key in stats_data:
                pilot_data = stats_data[pilot_key]
                pilot_success = df[df['PID'] == selected_pilot]['pilot_success'].iloc[0]

                # Status header
                status_class = "success-status" if pilot_success == 'Successful' else "failure-status"
                status_text = f"✅ PILOT {selected_pilot} - SUCCESSFUL" if pilot_success == 'Successful' else f"❌ PILOT {selected_pilot} - FAILED"

                analysis_content.append(html.Div(status_text, className=status_class))

                # Add Approach Score section
                if 'Approach_Score' in df.columns and 'Approach_Score' in benchmarks:
                    pilot_score = df[df['PID'] == selected_pilot]['Approach_Score'].iloc[0]
                    bench = benchmarks['Approach_Score']
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
                            html.Span(f"Successful average: {bench['success_mean']:.3f}",
                                      className="text-success")
                        ], className="threshold-info")
                    ])

                    analysis_content.append(approach_score_display)

                # Show ALL metrics with detailed statistical analysis
                analysis_content.append(html.H6("Gaze Metrics Analysis:", className="text-info mt-3"))

                # Get group averages for comparison
                success_avg = successful_df[selected_metrics].mean().to_dict()
                unsuccess_avg = unsuccessful_df[selected_metrics].mean().to_dict()

                for metric in selected_metrics:
                    if metric in pilot_data:
                        # SKIP Approach Score - it has its own special box
                        if metric == 'Approach_Score':
                            continue  # Skip this metric in the list

                        value = pilot_data[metric]
                        success_value = success_avg.get(metric, 0)
                        unsuccess_value = unsuccess_avg.get(metric, 0)

                        # Calculate percentage difference from successful average
                        if success_value != 0:
                            diff_pct_success = ((value - success_value) / success_value) * 100
                        else:
                            diff_pct_success = 0

                        # Calculate percentage difference from unsuccessful average
                        if unsuccess_value != 0:
                            diff_pct_unsuccess = ((value - unsuccess_value) / unsuccess_value) * 100
                        else:
                            diff_pct_unsuccess = 0

                        # Determine good/bad based on proximity to successful vs unsuccessful averages
                        dist_from_success = abs(diff_pct_success)
                        dist_from_unsuccess = abs(diff_pct_unsuccess)

                        # Check which average the pilot is closer to
                        is_closer_to_success = dist_from_success < dist_from_unsuccess
                        is_closer_to_unsuccess = dist_from_unsuccess < dist_from_success

                        # For entropy metrics (lower is better)
                        if metric in ['stationary_entropy', 'transition_entropy']:
                            # ENTROPY METRICS: Lower is better (more systematic)
                            # If closer to unsuccessful average, mark as bad
                            if is_closer_to_unsuccess and dist_from_unsuccess < 20:
                                status_class = "metric-bad"
                                reasoning = f"Closer to unsuccessful average ({unsuccess_value:.3f}) - needs more systematic scanning"
                            elif diff_pct_success <= 10:  # Within 10% of success average or lower
                                status_class = "metric-good"
                                if diff_pct_success <= 0:
                                    reasoning = f"Better or equal to successful average ({success_value:.3f})"
                                else:
                                    reasoning = f"Close to successful average ({success_value:.3f})"
                            else:  # More than 10% different from success average
                                status_class = "metric-bad"
                                reasoning = f"Not close to successful average ({success_value:.3f})"

                        # For blink rate (moderate is best)
                        elif metric == 'Average_Blink_Rate_per_Minute':
                            # BLINK RATE: Moderate is best
                            # If closer to unsuccessful average, mark as bad
                            if is_closer_to_unsuccess and dist_from_unsuccess < 25:
                                status_class = "metric-bad"
                                reasoning = f"Closer to unsuccessful average ({unsuccess_value:.3f}) - may indicate stress"
                            elif abs(diff_pct_success) <= 25:
                                status_class = "metric-good"
                                reasoning = f"Within normal blink rate range ({success_value:.3f})"
                            elif diff_pct_success > 25:
                                status_class = "metric-bad"
                                reasoning = f"Too high - {abs(diff_pct_success):.0f}% above successful average ({success_value:.3f})"
                            else:
                                status_class = "metric-bad"
                                reasoning = f"Too low - {abs(diff_pct_success):.0f}% below successful average ({success_value:.3f})"

                        # For other metrics
                        else:
                            # GENERAL RULE: If closer to unsuccessful average (within 15%), mark as bad
                            if is_closer_to_unsuccess and dist_from_unsuccess < 15:
                                status_class = "metric-bad"
                                reasoning = f"Closer to unsuccessful average ({unsuccess_value:.3f}) - needs improvement"
                            elif abs(diff_pct_success) <= 15:
                                status_class = "metric-good"
                                reasoning = f"Close to successful average ({success_value:.3f})"
                            elif diff_pct_success > 15:
                                status_class = "metric-bad"
                                reasoning = f"Too high - {abs(diff_pct_success):.0f}% above successful average ({success_value:.3f})"
                            else:
                                status_class = "metric-bad"
                                reasoning = f"Too low - {abs(diff_pct_success):.0f}% below successful average ({success_value:.3f})"

                        # Create metric item with description and comparison data
                        metric_item = html.Div([
                            html.Strong(f"{metrics_config[metric]['name']}: {value:.3f}", className="text-light"),
                            html.Span(f" ({diff_pct_success:+.0f}% vs successful avg)",
                                      className="text-warning" if abs(diff_pct_success) <= 20 else "text-danger"),
                            html.Br(),
                            html.Span(f"{metrics_config[metric]['description']}", className="metric-description"),
                            html.Br(),
                            html.Span(f"Status: {reasoning}", className="text-light small"),
                            html.Br(),
                            html.Span(f"Successful average: {success_value:.3f}", className="text-success"),
                            html.Br(),
                            html.Span(f"Unsuccessful average: {unsuccess_value:.3f}", className="text-danger")
                        ], className=f"metric-item {status_class}")

                        analysis_content.append(metric_item)

                # Summary section
                analysis_content.append(html.H6("Performance Summary:", className="text-warning mt-3"))

                if selected_pilot:
                    # Analyze metrics by deviation from success average
                    close_to_success = []
                    close_to_unsuccess = []
                    other_metrics = []

                    for metric in selected_metrics:
                        if metric in pilot_data and metric != 'Approach_Score':
                            value = pilot_data[metric]
                            success_val = success_avg.get(metric, 0)
                            unsuccess_val = unsuccess_avg.get(metric, 0)

                            # Calculate distances
                            if success_val != 0:
                                dist_from_success = abs((value - success_val) / success_val) * 100
                            else:
                                dist_from_success = 0

                            if unsuccess_val != 0:
                                dist_from_unsuccess = abs((value - unsuccess_val) / unsuccess_val) * 100
                            else:
                                dist_from_unsuccess = 0

                            # Categorize
                            if dist_from_unsuccess < 15 and dist_from_unsuccess < dist_from_success:
                                close_to_unsuccess.append(metric)
                            elif dist_from_success <= 15:
                                close_to_success.append(metric)
                            else:
                                other_metrics.append(metric)

                    total = len(close_to_success) + len(close_to_unsuccess) + len(other_metrics)

                    if total > 0:
                        summary_text = f"Metrics compared to group averages:"
                        analysis_content.append(html.P(summary_text, className="text-light small"))

                        if close_to_success:
                            analysis_content.append(html.P(
                                f"✓ {len(close_to_success)} metrics close to successful average",
                                className="text-success small mb-1"))

                        if close_to_unsuccess:
                            close_names = [metrics_config[m]['name'] for m in close_to_unsuccess[:5]]
                            analysis_content.append(html.P(
                                f"✗ {len(close_to_unsuccess)} metrics close to unsuccessful average: {', '.join(close_names)}",
                                className="text-danger small mb-1"))

                        if other_metrics:
                            analysis_content.append(html.P(
                                f"⚠ {len(other_metrics)} metrics not close to either group average",
                                className="text-warning small mb-1"))
                    else:
                        # For unsuccessful pilots
                        analysis_content.append(html.P(
                            f"This pilot shows gaze patterns that differ significantly from successful group averages.",
                            className="text-light small"))
                else:
                    # Group comparison summary
                    analysis_content.append(html.P(
                        f"Comparing {len(successful_df)} successful vs {len(unsuccessful_df)} unsuccessful pilots.",
                        className="text-light small"))
                    analysis_content.append(html.P(
                        "Analysis based on group averages shown in the graph above.",
                        className="text-light small"))

        else:
            # Group comparison analysis when "All" is selected or no pilot is selected
            analysis_content.append(html.H5("Group Performance Analysis", className="text-info mb-3"))

            # Add Approach Score comparison
            if 'Approach_Score' in df.columns and 'Approach_Score' in benchmarks:
                bench = benchmarks['Approach_Score']
                threshold = bench['threshold']

                analysis_content.append(html.H6("Approach Score Comparison:", className="text-warning"))

                approach_comparison = html.Div([
                    html.Strong("Success Threshold: ≥ 0.7", className="text-light"),
                    html.Br(),
                    html.Span(
                        f"Successful average: {bench['success_mean']:.3f}",
                        className="text-success")
                ], className="stat-highlight good-stat")

                analysis_content.append(approach_comparison)
                analysis_content.append(html.Hr(className="my-3"))

            if 'Successful' in stats_data and 'Unsuccessful' in stats_data:
                successful_data = stats_data['Successful']
                unsuccessful_data = stats_data['Unsuccessful']

                # Show ALL metrics comparison with averages
                analysis_content.append(html.H6("Gaze Metrics Comparison:", className="text-warning"))

                for metric in selected_metrics:
                    if metric in successful_data and metric in unsuccessful_data:
                        success_val = successful_data[metric]
                        unsuccess_val = unsuccessful_data[metric]
                        diff_pct = ((success_val - unsuccess_val) / unsuccess_val * 100) if unsuccess_val != 0 else 0

                        # Determine if difference is significant
                        if abs(diff_pct) > 15:  # More than 15% difference
                            status_class = "good-stat"
                            comparison = f"Significant difference ({diff_pct:+.1f}%)"
                        else:
                            status_class = "neutral-stat"
                            comparison = f"Similar values ({diff_pct:+.1f}%)"

                        metric_item = html.Div([
                            html.Strong(f"{metrics_config[metric]['name']}: ", className="text-light"),
                            html.Br(),
                            html.Span(f"{metrics_config[metric]['description']}", className="metric-description"),
                            html.Br(),
                            html.Span(f"Successful average: {success_val:.3f}", className="text-success"),
                            html.Br(),
                            html.Span(f"Unsuccessful average: {unsuccess_val:.3f}", className="text-danger"),
                            html.Br(),
                            html.Span(f"Comparison: {comparison}", className="text-light")
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

    # Callback for layout changes in pattern mode
    @app.callback(
        [Output('behavior-col', 'width'),
         Output('success-col', 'className')],
        [Input('mode-switch', 'value')]
    )
    def update_layout_for_pattern_mode(mode_switch_value):
        if mode_switch_value:  # Patterns mode
            # Expand behavior characterization to full width, hide success analysis
            return 12, 'hidden'
        else:  # AOI DGM mode
            # Return to normal layout (6 columns each, both visible)
            return 6, ''

    # Callback to open custom input modal when switch is turned ON
    @app.callback(
        [Output('custom-input-modal', 'is_open'),
         Output('custom-switch-state', 'data')],
        [Input('custom-values-switch', 'value'),
         Input('cancel-custom-btn', 'n_clicks'),
         Input('apply-custom-btn', 'n_clicks'),
         Input('clear-custom-btn', 'n_clicks')],
        [State('custom-input-modal', 'is_open'),
         State('custom-switch-state', 'data')],
        prevent_initial_call=True
    )
    def toggle_custom_modal(switch_value, cancel_clicks, apply_clicks, clear_clicks, is_open, switch_state):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # If switch is turned ON, open modal
        if trigger_id == 'custom-values-switch' and switch_value and not switch_state:
            return True, True  # Open modal, store switch state as ON

        # If switch is turned OFF, close modal
        elif trigger_id == 'custom-values-switch' and not switch_value and switch_state:
            return False, False  # Close modal, store switch state as OFF

        # If cancel, apply, or clear buttons are clicked, close modal but keep switch ON
        elif trigger_id in ['cancel-custom-btn', 'apply-custom-btn', 'clear-custom-btn']:
            return False, switch_state  # Close modal, keep switch state

        return is_open, switch_state

    # Callback to generate custom input fields based on selected metrics
    @app.callback(
        Output('custom-input-fields', 'children'),
        [Input('custom-input-modal', 'is_open'),
         Input('fill-defaults-checklist', 'value'),
         Input('clear-custom-btn', 'n_clicks')],
        [State('metrics-dropdown', 'value'),
         State('custom-values-store', 'data')],
        prevent_initial_call=True
    )
    def generate_custom_fields(is_open, fill_defaults, clear_clicks, selected_metrics, stored_values):
        # Only generate fields when modal is open
        if not is_open:
            raise PreventUpdate

        # Clear all values if clear button was clicked
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-custom-btn.n_clicks':
            stored_values = {}

        # Filter out any invalid metrics
        plot_metrics = [m for m in selected_metrics if m in metrics_config]

        if not plot_metrics:
            return html.Div("Please select at least one metric in the dropdown above.", className="text-warning")

        fields = []
        for metric in plot_metrics:
            config = metrics_config[metric]

            # Determine initial value
            current_value = stored_values.get(metric, config['default'])

            # Override with success/unsuccess averages if requested
            if fill_defaults:
                if 'success_avg' in fill_defaults and 'Successful' in successful_df.columns and metric in successful_df.columns:
                    current_value = successful_df[metric].mean()
                elif 'unsuccess_avg' in fill_defaults and 'Unsuccessful' in unsuccessful_df.columns and metric in unsuccessful_df.columns:
                    current_value = unsuccessful_df[metric].mean()

            # Create a row with label, description, range, and input
            field = dbc.Row([
                dbc.Col([
                    html.Label(f"{config['name']}", className="fw-bold text-light"),
                    html.P(f"{config['description']}", className="text-light small mb-1"),
                    html.Div([
                        html.Span(f"Typical range: {config['min']:.2f} - {config['max']:.2f}",
                                  className="text-light small"),
                        html.Br(),
                        html.Span(f"Dataset range: {df[metric].min():.2f} - {df[metric].max():.2f}",
                                  className="text-light small text-muted")
                    ])
                ], width=6),
                dbc.Col([
                    daq.NumericInput(
                        id={'type': 'custom-input', 'index': metric},
                        value=current_value,
                        min=config['min'],
                        max=config['max'],
                        size=120,
                        className="custom-numeric-input"
                    ),
                    html.Div(id={'type': 'range-warning', 'index': metric}, className="mt-1")
                ], width=6)
            ], className="mb-3 p-3 bg-dark rounded", id={'type': 'metric-row', 'index': metric})

            fields.append(field)

        return fields

    # Callback to show warnings for values outside typical range
    @app.callback(
        [Output({'type': 'range-warning', 'index': ALL}, 'children'),
         Output({'type': 'metric-row', 'index': ALL}, 'className')],
        [Input({'type': 'custom-input', 'index': ALL}, 'value')],
        [State({'type': 'custom-input', 'index': ALL}, 'id')]
    )
    def show_range_warnings(values, ids):
        warnings = []
        classes = []

        for value, input_id in zip(values, ids):
            if input_id and 'index' in input_id:
                metric = input_id['index']
                config = metrics_config[metric]

                if value is not None:
                    # Check if value is outside typical range
                    if value < config['min'] or value > config['max']:
                        warning_text = f"⚠ Value outside typical range ({config['min']:.2f}-{config['max']:.2f})"
                        warnings.append(html.Span(warning_text, className="text-warning small"))
                        classes.append("mb-3 p-3 bg-dark rounded border border-warning")
                    else:
                        warnings.append(html.Span("✓ Within typical range", className="text-success small"))
                        classes.append("mb-3 p-3 bg-dark rounded")
                else:
                    warnings.append(html.Span("", className="small"))
                    classes.append("mb-3 p-3 bg-dark rounded")

        return warnings, classes

    # Callback to store custom values when Apply is clicked
    @app.callback(
        Output('custom-values-store', 'data'),
        [Input('apply-custom-btn', 'n_clicks'),
         Input('clear-custom-btn', 'n_clicks')],
        [State('metrics-dropdown', 'value'),
         State({'type': 'custom-input', 'index': ALL}, 'id'),
         State({'type': 'custom-input', 'index': ALL}, 'value')],
        prevent_initial_call=True
    )
    def store_custom_values(apply_clicks, clear_clicks, selected_metrics, input_ids, input_values):
        ctx = callback_context

        # Clear all values if clear button was clicked
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-custom-btn.n_clicks':
            return {}

        # Store values if apply button was clicked
        custom_data = {}
        for input_id, value in zip(input_ids, input_values):
            if input_id and 'index' in input_id and value is not None:
                metric = input_id['index']
                custom_data[metric] = value

        return custom_data

    # Callback to update group checklist when custom values are added
    @app.callback(
        Output('group-checklist', 'value'),
        [Input('custom-values-store', 'data'),
         Input('custom-values-switch', 'value')],
        [State('group-checklist', 'value')]
    )
    def update_group_checklist(custom_data, switch_value, current_values):
        # If switch is ON and we have custom data, add 'Custom' to groups
        if switch_value and custom_data and 'Custom' not in current_values:
            return current_values + ['Custom']
        # If switch is OFF or no custom data, remove 'Custom' from groups
        elif (not switch_value or not custom_data) and 'Custom' in current_values:
            return [v for v in current_values if v != 'Custom']
        return current_values

    # Callback to update the switch appearance based on whether we have custom values
    @app.callback(
        Output('custom-values-switch', 'label'),
        [Input('custom-values-store', 'data'),
         Input('custom-values-switch', 'value')]
    )
    def update_switch_label(custom_data, switch_value):
        if custom_data and switch_value:
            return "Custom Values Added ✓"
        elif switch_value:
            return "Add Custom Values to Graph"
        else:
            return "Add Custom Values to Graph"

    # Main callback
    @app.callback(
        [Output('main-visualization', 'figure'),
         Output('behavior-characterization', 'children'),
         Output('success-analysis', 'children'),
         Output('current-data', 'data')],
        [Input('metrics-dropdown', 'value'),
         Input('group-checklist', 'value'),
         Input('pilot-dropdown', 'value'),
         Input('visualization-type', 'value'),
         Input('pattern-chart-type', 'value'),
         Input('mode-switch', 'value'),
         Input('reset-btn', 'n_clicks'),
         Input('custom-values-store', 'data'),
         Input('custom-values-switch', 'value')],
        [State('metrics-dropdown', 'options')]
    )
    def update_dashboard(selected_metrics, selected_groups, selected_pilot, visualization_type,
                         pattern_chart_type, mode_switch_value, reset_clicks, custom_values,
                         custom_switch_value, metric_options):
        # Handle reset button
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-btn.n_clicks':
            selected_metrics = default_metrics
            selected_groups = ["Successful", "Unsuccessful"]
            selected_pilot = 'All'
            visualization_type = 'linear'
            pattern_chart_type = 'attention_distribution'
            mode_switch_value = False
            custom_values = {}
            custom_switch_value = False

        # Handle pattern chart types when in Patterns mode
        if mode_switch_value:
            pattern_data_for_display = None

            if pattern_chart_type == 'attention_distribution':
                fig = create_attention_distribution_figure()
            elif pattern_chart_type == 'aoi_differences':
                fig = create_aoi_differences_figure()
            elif pattern_chart_type == 'success_patterns':
                fig, pattern_data_for_display = create_success_patterns_figure()
            elif pattern_chart_type == 'fail_patterns':
                fig, pattern_data_for_display = create_fail_patterns_figure()
            else:
                fig = create_attention_distribution_figure()

            # For pattern charts, enhance the behavior characterization with pattern explanations
            behavior_characterization = []
            success_analysis = []
            stats_data = {}

            # Add pattern insights if data is available
            if pattern_data['available']:
                behavior_characterization.append(html.H5("Pattern Analysis Insights", className="text-info mb-3"))

                # Only show Pattern Key for attention_distribution and aoi_differences, not for success/fail patterns
                if pattern_chart_type in ['attention_distribution', 'aoi_differences']:
                    behavior_characterization.append(html.Div([
                        html.Div("🔑 PATTERN KEY", className="pattern-header"),
                        html.Div([
                            html.Span("A", className="pattern-key"),
                            html.Span(" = No AOI", className="pattern-meaning"),
                            html.Br(),
                            html.Span("B", className="pattern-key"),
                            html.Span(" = Altitude/VSI", className="pattern-meaning"),
                            html.Br(),
                            html.Span("C", className="pattern-key"),
                            html.Span(" = Attitude Indicator", className="pattern-meaning"),
                            html.Br(),
                            html.Span("D", className="pattern-key"),
                            html.Span(" = Turn/Heading", className="pattern-meaning"),
                            html.Br(),
                            html.Span("E", className="pattern-key"),
                            html.Span(" = Speed/Slip Indicator", className="pattern-meaning"),
                            html.Br(),
                            html.Span("F", className="pattern-key"),
                            html.Span(" = Airspeed Indicator", className="pattern-meaning"),
                            html.Br(),
                            html.Span("G", className="pattern-key"), html.Span(" = RPM", className="pattern-meaning"),
                            html.Br(),
                            html.Span("H", className="pattern-key"),
                            html.Span(" = Window (Outside)", className="pattern-meaning")
                        ], className="pattern-explanation")
                    ], className="behavior-pattern"))

                # Add pattern list for success/fail patterns
                if pattern_chart_type == 'success_patterns' and pattern_data_for_display is not None:
                    behavior_characterization.extend(
                        create_pattern_list_content(pattern_data_for_display, "SUCCESSFUL PATTERNS DETAIL",
                                                    success=True)
                    )
                elif pattern_chart_type == 'fail_patterns' and pattern_data_for_display is not None:
                    behavior_characterization.extend(
                        create_pattern_list_content(pattern_data_for_display, "UNSUCCESSFUL PATTERNS DETAIL",
                                                    success=False)
                    )

                # Only show Pattern Interpretation for attention_distribution and aoi_differences, not for success/fail patterns
                if pattern_chart_type in ['attention_distribution', 'aoi_differences']:
                    explanations = get_pattern_explanations()
                    behavior_characterization.append(html.Div([
                        html.Div("📋 PATTERN INTERPRETATIONS", className="pattern-header"),
                        *[html.Div([
                            html.Strong(f"{pattern_type}:", className="text-light"),
                            html.Span(f" {explanation}", className="pattern-meaning")
                        ], className="pattern-explanation") for pattern_type, explanation in explanations.items()]
                    ], className="behavior-pattern"))

                # Add some basic insights for attention distribution and differences charts
                if pattern_chart_type in ['attention_distribution', 'aoi_differences']:
                    aoi_labels = list(AOI_NAMES.keys())
                    success_pcts = [(pattern_data['success_aoi'][aoi] / sum(pattern_data['success_aoi'].values())) * 100
                                    for aoi in aoi_labels]
                    fail_pcts = [(pattern_data['fail_aoi'][aoi] / sum(pattern_data['fail_aoi'].values())) * 100 for aoi
                                 in aoi_labels]
                    differences = [s - f for s, f in zip(success_pcts, fail_pcts)]
                    max_diff_idx = differences.index(max(differences, key=abs))
                    max_diff_aoi = AOI_NAMES[aoi_labels[max_diff_idx]]

                    behavior_characterization.append(html.Div([
                        html.Div("🎯 KEY INSIGHT", className="pattern-header"),
                        html.Div([
                            html.Strong("Largest attention difference:", className="text-light"),
                            html.Span(f" {max_diff_aoi} ({differences[max_diff_idx]:+.1f}%)",
                                      className="stat-difference difference-positive" if differences[
                                                                                             max_diff_idx] > 0 else "stat-difference difference-negative")
                        ], className="pattern-metric")
                    ], className="key-insight"))

            return fig, behavior_characterization, success_analysis, stats_data

        # Original visualization logic for AOI DGM mode
        if not selected_metrics or len([m for m in selected_metrics if m in metrics_config]) < 3:
            selected_metrics = default_metrics[:3]

        # Prepare data for other visualization types
        stats_data = {}

        # Add group data to stats_data
        if "Successful" in selected_groups:
            valid_metrics = [m for m in selected_metrics if m in metrics_config]
            stats_data['Successful'] = df[df['pilot_success'] == 'Successful'][
                valid_metrics
            ].mean().to_dict()

        if "Unsuccessful" in selected_groups:
            valid_metrics = [m for m in selected_metrics if m in metrics_config]
            stats_data['Unsuccessful'] = df[df['pilot_success'] == 'Unsuccessful'][
                valid_metrics
            ].mean().to_dict()

        if "All" in selected_groups:
            valid_metrics = [m for m in selected_metrics if m in metrics_config]
            stats_data['All'] = df[valid_metrics].mean().to_dict()

        # Add custom values if selected and switch is ON
        if "Custom" in selected_groups and custom_switch_value and custom_values:
            stats_data['Custom'] = custom_values

        # Add individual pilot data if selected (and not "All")
        if selected_pilot and selected_pilot != "All":
            valid_metrics = [m for m in selected_metrics if m in metrics_config]
            stats_data[f'Pilot {selected_pilot}'] = df[df['PID'] == selected_pilot][
                valid_metrics
            ].iloc[0].to_dict()

        # Create visualization based on selected type
        if visualization_type == 'parallel':
            # For parallel coordinates, now support custom values
            fig = create_parallel_coordinates_figure(
                [m for m in selected_metrics if m in metrics_config],
                selected_pilot,
                custom_values,
                custom_switch_value
            )
        else:
            # Linear scale visualization with custom values
            fig = create_linear_scale_figure(
                selected_metrics,
                selected_groups,
                selected_pilot,
                custom_values,
                custom_switch_value
            )

        # Create behavior characterization
        valid_selected_metrics = [m for m in selected_metrics if m in metrics_config]
        behavior_characterization = characterize_behavior(
            valid_selected_metrics,
            stats_data
        )

        # Create success analysis with custom values support
        success_analysis = analyze_success_patterns(
            valid_selected_metrics,
            stats_data,
            selected_pilot,
            custom_values,
            custom_switch_value
        )

        return fig, behavior_characterization, success_analysis, stats_data

    return app


# Create and run the dashboard
print("Starting Enhanced Dashboard with Linear Scale Visualization...")
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