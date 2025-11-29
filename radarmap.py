import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Read directly from your CSV file
df = pd.read_csv('AOI_DGMs.csv')

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
    Create an enhanced 3D radar chart dashboard with comprehensive interactive features
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
                                {'label': ' 3D Radar', 'value': '3d'},
                                {'label': ' 2D Radar', 'value': '2d'},
                                {'label': ' Parallel Coordinates', 'value': 'parallel'}
                            ],
                            value='3d',
                            className="mb-3"
                        ),

                        # Action Buttons
                        dbc.Row([
                            dbc.Col(dbc.Button("Export Chart", id="export-btn", color="primary", className="me-2")),
                            dbc.Col(dbc.Button("Reset Filters", id="reset-btn", color="secondary"))
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

                    # Metric Descriptions Card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Metric Descriptions", className="h5"),
                            dbc.CardBody(id='metric-descriptions')
                        ])
                    ], width=6)
                ], className="mt-3")
            ], width=9),

            # Second visualization Card
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Second Visualization", className="h5"),
                            dbc.CardBody([
                                dcc.Graph(id='bar-chart', style={'height': '600px'})
                            ])
                        ])
                    ])
                ])
            ])
        ]),

        # Hidden div for storing intermediate values
        dcc.Store(id='current-data'),

        # Download component
        dcc.Download(id="download-chart")
    ], fluid=True, style={'backgroundColor': '#1a1a1a'})

    # Callbacks
    @app.callback(
        [Output('radar-chart', 'figure'),
         Output('stats-card', 'children'),
         Output('metric-descriptions', 'children'),
         Output('current-data', 'data'),
         Output('bar-chart', 'figure')],
        [Input('metrics-dropdown', 'value'),
         Input('normalization-toggle', 'value'),
         Input('group-checklist', 'value'),
         Input('pilot-dropdown', 'value'),
         Input('chart-style', 'value'),
         Input('reset-btn', 'n_clicks')],
        [State('metrics-dropdown', 'options')]
    )
    def update_dashboard(selected_metrics, normalize, selected_groups, selected_pilot,
                         chart_style, reset_clicks, metric_options):

        # Handle reset button
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-btn.n_clicks':
            selected_metrics = default_metrics
            normalize = True
            selected_groups = ["Successful", "Unsuccessful"]
            selected_pilot = None
            chart_style = '3d'

        # Ensure we have some selected metrics
        if not selected_metrics:
            selected_metrics = default_metrics

        # Only use metrics that exist
        radar_metrics = [
            m for m in selected_metrics
            if m in metrics_config and m in df.columns
        ]

        # Fallback if fewer than 3 valid metrics are selected
        if len(radar_metrics) < 3:
            radar_metrics = [m for m in default_metrics if m in df.columns][:3]

        # Prepare data
        traces = []
        stats_data = {}

        # Normalization function
        def normalize_series(series):
            range_val = series.max() - series.min()
            if range_val == 0:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - series.min()) / range_val

        # Add group traces (radar)
        if "Successful" in selected_groups and len(radar_metrics) > 0:
            successful_data_raw = df[df['pilot_success'] == 'Successful'][radar_metrics].mean()
            successful_data = normalize_series(successful_data_raw) if normalize else successful_data_raw
            stats_data['Successful'] = successful_data_raw.to_dict()

            traces.append(go.Scatterpolar(
                r=np.append(successful_data.values, successful_data.values[0]),
                theta=np.append([metrics_config[m]['name'] for m in radar_metrics],
                                [metrics_config[radar_metrics[0]]['name']]),
                fill='toself',
                fillcolor='rgba(76, 175, 80, 0.4)',
                line=dict(color='rgb(76, 175, 80)', width=3),
                name='Successful Pilots (Avg)',
                hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
            ))

        if "Unsuccessful" in selected_groups and len(radar_metrics) > 0:
            unsuccessful_data_raw = df[df['pilot_success'] == 'Unsuccessful'][radar_metrics].mean()
            unsuccessful_data = normalize_series(unsuccessful_data_raw) if normalize else unsuccessful_data_raw
            stats_data['Unsuccessful'] = unsuccessful_data_raw.to_dict()

            traces.append(go.Scatterpolar(
                r=np.append(unsuccessful_data.values, unsuccessful_data.values[0]),
                theta=np.append([metrics_config[m]['name'] for m in radar_metrics],
                                [metrics_config[radar_metrics[0]]['name']]),
                fill='toself',
                fillcolor='rgba(244, 67, 54, 0.4)',
                line=dict(color='rgb(244, 67, 54)', width=3),
                name='Unsuccessful Pilots (Avg)',
                hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
            ))

        if "All" in selected_groups and len(radar_metrics) > 0:
            all_data_raw = df[radar_metrics].mean()
            all_data = normalize_series(all_data_raw) if normalize else all_data_raw
            stats_data['All'] = all_data_raw.to_dict()

            traces.append(go.Scatterpolar(
                r=np.append(all_data.values, all_data.values[0]),
                theta=np.append([metrics_config[m]['name'] for m in radar_metrics],
                                [metrics_config[radar_metrics[0]]['name']]),
                fill='toself',
                fillcolor='rgba(33, 150, 243, 0.4)',
                line=dict(color='rgb(33, 150, 243)', width=3),
                name='All Pilots (Avg)',
                hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
            ))

        # Add individual pilot trace if selected
        if selected_pilot and len(radar_metrics) > 0:
            pilot_row = df[df['PID'] == selected_pilot]
            if not pilot_row.empty:
                pilot_data_raw = pilot_row[radar_metrics].iloc[0]
                pilot_data = normalize_series(pilot_data_raw) if normalize else pilot_data_raw
                stats_data[f'Pilot {selected_pilot}'] = pilot_data_raw.to_dict()

                pilot_success = pilot_row['pilot_success'].iloc[0]
                color = 'rgb(76, 175, 80)' if pilot_success == 'Successful' else 'rgb(244, 67, 54)'

                traces.append(go.Scatterpolar(
                    r=np.append(pilot_data.values, pilot_data.values[0]),
                    theta=np.append([metrics_config[m]['name'] for m in radar_metrics],
                                    [metrics_config[radar_metrics[0]]['name']]),
                    line=dict(color=color, width=4, dash='dash'),
                    name=f'Pilot {selected_pilot} ({pilot_success})',
                    hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<extra></extra>'
                ))

        # Create figure based on chart style
        if chart_style == 'parallel' and len(radar_metrics) > 0:
            fig = go.Figure(data=go.Parcoords(
                line=dict(
                    color=df['Approach_Score'] if 'Approach_Score' in df.columns else np.zeros(len(df)),
                    colorscale='Viridis',
                    showscale=True,
                    cmin=df['Approach_Score'].min() if 'Approach_Score' in df.columns else 0,
                    cmax=df['Approach_Score'].max() if 'Approach_Score' in df.columns else 1
                ),
                dimensions=[
                    dict(
                        range=[df[col].min(), df[col].max()],
                        label=metrics_config[col]['name'],
                        values=df[col]
                    )
                    for col in radar_metrics
                ]
            ))
            fig.update_layout(
                title="Parallel Coordinates Plot",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        else:
            fig = go.Figure(data=traces)
            if len(radar_metrics) > 0:
                radial_min = df[radar_metrics].min().min()
                radial_max = df[radar_metrics].max().max()
            else:
                radial_min, radial_max = 0, 1

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1] if normalize else [radial_min, radial_max],
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
                    text=f"Gaze Behavior Profile Comparison<br><sub>{'Normalized ' if normalize else ''}Metrics: {len(radar_metrics)} selected</sub>",
                    font=dict(color='white', size=16),
                    x=0.5
                )
            )

        # Create statistics card
        stats_card = []
        for group, values in stats_data.items():
            stats_card.append(html.H6(f"{group}:", className="text-info"))
            # Show first 3 metrics
            for metric, value in list(values.items())[:3]:
                if metric in metrics_config:
                    label = metrics_config[metric]['name']
                else:
                    label = metric
                stats_card.append(
                    html.P(
                        f"{label}: {value:.3f}",
                        className="text-light small mb-1"
                    )
                )
            stats_card.append(html.Hr(className="my-2"))

        # Create metric descriptions (radar metrics)
        metric_descriptions = []
        for metric in radar_metrics:
            metric_descriptions.append(
                html.Div([
                    html.Strong(metrics_config[metric]['name'], className="text-warning"),
                    html.P(metrics_config[metric]['description'], className="text-light small mb-2")
                ])
            )

        # Bar chart
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

        bar_fig = go.Figure(data=bar_traces)
        bar_fig.update_layout(
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
            )
        )

        return fig, stats_card, metric_descriptions, stats_data, bar_fig

    @app.callback(
        Output("download-chart", "data"),
        Input("export-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_chart(n_clicks):
        return dcc.send_file("./radar_chart_export.html")

    return app


# Create and run the dashboard
print("Starting Enhanced Chart Dashboard...")
print("Loading data from AOI_DGMs.csv...")
print(f"Loaded {len(df)} pilots")
print(f"Successful: {len(df[df['pilot_success'] == 'Successful'])}")
print(f"Unsuccessful: {len(df[df['pilot_success'] == 'Unsuccessful'])}")
print("\nDashboard will open in your web browser...")

# Create the dashboard app
app = create_enhanced_radar_dashboard(df)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
