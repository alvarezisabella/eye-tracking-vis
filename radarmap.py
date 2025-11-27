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
            ], width=9)
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
            color = 'rgb(76, 175, 80)' if pilot_success == 'Successful' else 'rgb(244, 67, 54)'

            traces.append(go.Scatterpolar(
                r=np.append(pilot_data.values, pilot_data.values[0]),
                theta=np.append([metrics_config[m]['name'] for m in selected_metrics],
                                [metrics_config[selected_metrics[0]]['name']]),
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

        # Create metric descriptions
        metric_descriptions = []
        for metric in selected_metrics:
            metric_descriptions.append(
                html.Div([
                    html.Strong(metrics_config[metric]['name'], className="text-warning"),
                    html.P(metrics_config[metric]['description'], className="text-light small mb-2")
                ])
            )

        return fig, stats_card, metric_descriptions, stats_data

    @app.callback(
        Output("download-chart", "data"),
        Input("export-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_chart(n_clicks):
        return dcc.send_file("./radar_chart_export.html")

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