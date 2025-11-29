import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Read directly from your CSV file
df = pd.read_csv('AOI_DGMs.csv')


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

                        # Chart Style
                        html.Label("Chart Style:", className="fw-bold text-light"),
                        dcc.RadioItems(
                            id='chart-style',
                            options=[
                                {'label': ' Radar', 'value': '3d'},
                                {'label': ' Parallel Coordinates', 'value': 'parallel'}
                            ],
                            value='3d',
                            className="mb-3"
                        ),

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
                                dcc.Graph(id='radar-chart', style={'height': '600px'})
                            ])
                        ])
                    ])
                ]),

                dbc.Row([
                    # Statistics Card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Key Statistics", className="h5"),
                            dbc.CardBody(id='stats-card')
                        ])
                    ], width=6),

                    # Success Analysis Card (Replaced Metric Descriptions)
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

                benchmarks[metric] = {
                    'success_iqr': (successful_vals.quantile(0.25), successful_vals.quantile(0.75)),
                    'success_mean': successful_vals.mean(),
                    'success_median': successful_vals.median(),
                    'all_range': (df[metric].min(), df[metric].max())
                }

        # Add Approach Score benchmarks separately
        if 'Approach_Score' in df.columns:
            successful_scores = successful_df['Approach_Score']

            benchmarks['Approach_Score'] = {
                'success_iqr': (successful_scores.quantile(0.25), successful_scores.quantile(0.75)),
                'success_mean': successful_scores.mean(),
                'success_median': successful_scores.median(),
                'all_range': (df['Approach_Score'].min(), df['Approach_Score'].max()),
                'threshold': 0.7  # Success threshold
            }

        return benchmarks

    # Pre-calculate benchmarks
    benchmarks = calculate_benchmarks()

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

    # Callbacks
    @app.callback(
        [Output('radar-chart', 'figure'),
         Output('stats-card', 'children'),
         Output('success-analysis', 'children'),
         Output('current-data', 'data')],
        [Input('metrics-dropdown', 'value'),
         Input('normalization-toggle', 'value'),
         Input('group-checklist', 'value'),
         Input('pilot-dropdown', 'value'),
         Input('chart-style', 'value'),
         Input('reset-btn', 'n_clicks')],
        [State('metrics-dropdown', 'options')]
    )
    def update_dashboard(selected_metrics, normalize, selected_groups, selected_pilot, chart_style, reset_clicks,
                         metric_options):
        # Handle reset button
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-btn.n_clicks':
            selected_metrics = default_metrics
            normalize = True
            selected_groups = ["Successful", "Unsuccessful"]
            selected_pilot = None
            chart_style = '3d'

        if not selected_metrics or len(selected_metrics) < 3:
            selected_metrics = default_metrics[:3]

        # Prepare data
        traces = []
        stats_data = {}

        # Normalization function
        def normalize_series(series):
            range_val = series.max() - series.min()
            if range_val == 0:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - series.min()) / range_val

        # Add group traces
        if "Successful" in selected_groups:
            successful_data = df[df['pilot_success'] == 'Successful'][selected_metrics].mean()
            if normalize:
                successful_data = normalize_series(successful_data)
            stats_data['Successful'] = df[df['pilot_success'] == 'Successful'][selected_metrics].mean().to_dict()

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
            stats_data['Unsuccessful'] = df[df['pilot_success'] == 'Unsuccessful'][selected_metrics].mean().to_dict()

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
            stats_data['All'] = df[selected_metrics].mean().to_dict()

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
            stats_data[f'Pilot {selected_pilot}'] = df[df['PID'] == selected_pilot][selected_metrics].iloc[0].to_dict()

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

        # Create figure based on chart style
        if chart_style == 'parallel':
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
                font=dict(color='white')
            )
        else:
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

        # Create statistics card
        stats_card = []
        for group, values in stats_data.items():
            stats_card.append(html.H6(f"{group}:", className="text-info"))
            for metric, value in list(values.items())[:3]:  # Show first 3 metrics
                stats_card.append(html.P(f"{metrics_config[metric]['name']}: {value:.3f}",
                                         className="text-light small mb-1"))
            stats_card.append(html.Hr(className="my-2"))

        # Create success analysis
        success_analysis = analyze_success_patterns(selected_metrics, stats_data, selected_pilot)

        return fig, stats_card, success_analysis, stats_data

    return app


# Create and run the dashboard
print("Starting Enhanced Radar Chart Dashboard...")
print("Loading data from AOI_DGMs.csv...")
print(f"Loaded {len(df)} pilots")
print(f"Successful: {len(df[df['pilot_success'] == 'Successful'])}")
print(f"Unsuccessful: {len(df[df['pilot_success'] == 'Unsuccessful'])}")
print("\nDashboard will open in your web browser...")

# Create the dashboard app
app = create_enhanced_radar_dashboard(df)

if __name__ == '__main__':
    app.run(debug=True, port=8050)