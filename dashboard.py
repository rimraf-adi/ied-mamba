"""
Interactive Plotly Dash Dashboard for TUH EEG Event Corpus Dataset Loader (Beginner Friendly + Fast Caching).

Features:
1. Multi-channel EDF Signal & ACNS TCP Montage Visualizer with Color-Coded Event Overlays & Banners.
2. Annotation Inspector (.rec second-precision vs .lab microsecond-precision).
3. HTK Differential Energy Feature Heatmap & Line Inspector.
4. PyTorch EEGEventDataset & DataLoader Live Simulator.
5. Interactive Help Guide & Clinical Terminology Documentation.
"""

import os
import glob
import functools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table

from dataset_loader import (
    parse_rec_file, parse_lab_file, parse_htk_file, read_edf_file,
    build_tcp_montage, EEGEventDataset, LABEL_MAP, REVERSE_LABEL_MAP,
    TCP_MONTAGE_DEFINITIONS, REC_CODE_MAP
)

CORPUS_ROOT = r"d:\ied\TU-v2.0.1"

# LRU Cache wrappers for fast sub-millisecond slider updates
@functools.lru_cache(maxsize=64)
def cached_read_edf(edf_path: str):
    return read_edf_file(edf_path)

@functools.lru_cache(maxsize=64)
def cached_parse_rec(rec_path: str):
    return parse_rec_file(rec_path)

# Dash App Setup with Light Theme Styling
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'
    ]
)
app.title = "TUH EEG Event Corpus - Interactive Dataset Dashboard"

# Custom Index HTML string for clean CSS overrides (Light Theme)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #f8fafc !important;
                color: #0f172a !important;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            }
            .rc-slider-mark-text {
                color: #475569 !important;
                font-weight: 600 !important;
                font-size: 12px !important;
            }
            .rc-slider-mark-text-active {
                color: #0284c7 !important;
            }
            .rc-slider-track {
                background-color: #0284c7 !important;
            }
            .rc-slider-handle {
                border-color: #0284c7 !important;
                background-color: #ffffff !important;
            }
            .rc-slider-tooltip-inner {
                background-color: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
                box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
                font-weight: bold !important;
            }
            .Select-control {
                background-color: #ffffff !important;
                border-color: #cbd5e1 !important;
                color: #0f172a !important;
            }
            .Select-value-label, .Select-placeholder {
                color: #0f172a !important;
            }
            .Select-menu-outer {
                background-color: #ffffff !important;
                color: #0f172a !important;
                border-color: #cbd5e1 !important;
            }
            .Select-option {
                background-color: #ffffff !important;
                color: #0f172a !important;
            }
            .Select-option.is-focused {
                background-color: #e0f2fe !important;
                color: #0369a1 !important;
            }
            .nav-tabs .nav-link {
                color: #475569 !important;
                font-weight: 600 !important;
            }
            .nav-tabs .nav-link.active {
                color: #0284c7 !important;
                font-weight: 700 !important;
                border-bottom: 3px solid #0284c7 !important;
            }
            .help-card {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .event-badge {
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 4px;
                margin-right: 8px;
                font-size: 13px;
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

# High-Visibility Color palette for event classes
CLASS_COLORS = {
    'bckg': 'rgba(148, 163, 184, 0.25)',  # Slate Gray
    'spsw': 'rgba(239, 68, 68, 0.45)',    # Bright Red (Spike Wave / Seizure)
    'gped': 'rgba(249, 115, 22, 0.45)',   # Orange (Generalized Discharge)
    'pled': 'rgba(234, 179, 8, 0.45)',    # Gold (Periodic Discharge)
    'eyem': 'rgba(16, 185, 129, 0.45)',   # Emerald (Eye Movement)
    'artf': 'rgba(59, 130, 246, 0.45)'    # Blue (Artifact)
}

CLASS_HEX = {
    'bckg': '#94a3b8',
    'spsw': '#ef4444',
    'gped': '#f97316',
    'pled': '#eab308',
    'eyem': '#10b981',
    'artf': '#3b82f6'
}

def get_session_files():
    files_map = {'train': [], 'eval': []}
    for split in ['train', 'eval']:
        split_dir = os.path.join(CORPUS_ROOT, 'edf', split)
        if os.path.exists(split_dir):
            edfs = glob.glob(os.path.join(split_dir, '**', '*.edf'), recursive=True)
            files_map[split] = sorted([os.path.relpath(p, CORPUS_ROOT) for p in edfs])
    return files_map

SESSION_FILES = get_session_files()

# App Layout - High-Contrast Light Theme
app.layout = html.Div(style={'backgroundColor': '#f8fafc', 'color': '#0f172a', 'minHeight': '100vh', 'padding': '24px'}, children=[

    # Header Title Banner
    html.Div(className='d-flex justify-content-between align-items-center mb-4 p-4 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
        html.Div([
            html.H2("🧠 TUH EEG Event Corpus Dataset Loader Dashboard", className='m-0 font-weight-bold', style={'color': '#0284c7'}),
            html.P("Interactive signal visualizer, TCP montage calculator, HTK feature inspector, & PyTorch DataLoader simulator", className='m-0 text-muted', style={'fontSize': '14px', 'marginTop': '4px'})
        ]),
        html.Span("TU-v2.0.1", className="badge bg-primary fs-6 px-3 py-2")
    ]),

    # Main Grid Layout (Sidebar + Content)
    html.Div(className='row g-4', children=[

        # Left Control Sidebar
        html.Div(className='col-md-3', children=[
            html.Div(className='p-4 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                html.H5("⚙️ Dataset Controls", style={'color': '#0f172a', 'fontWeight': '700'}),
                html.Hr(style={'borderColor': '#cbd5e1', 'margin': '12px 0 20px 0'}),

                # Split Selector
                html.Label("Dataset Split:", className='fw-bold mb-2', style={'fontSize': '14px', 'color': '#1e293b'}),
                dcc.Dropdown(
                    id='split-select',
                    options=[{'label': 'Train Set', 'value': 'train'}, {'label': 'Eval Set', 'value': 'eval'}],
                    value='eval',
                    clearable=False,
                    style={'color': '#0f172a', 'marginBottom': '18px'}
                ),

                # Session EDF Dropdown
                html.Label("Select Session / EDF File:", className='fw-bold mb-2', style={'fontSize': '14px', 'color': '#1e293b'}),
                dcc.Dropdown(
                    id='edf-file-select',
                    clearable=False,
                    style={'color': '#0f172a', 'marginBottom': '18px'}
                ),

                # View Mode: Raw vs TCP Montage
                html.Label("Channel Montage Mode:", className='fw-bold mb-2', style={'fontSize': '14px', 'color': '#1e293b'}),
                dcc.RadioItems(
                    id='montage-mode',
                    options=[
                        {'label': ' ACNS TCP Montage (22 Channels)', 'value': 'tcp'},
                        {'label': ' Raw Scalp Reference Channels', 'value': 'raw'}
                    ],
                    value='tcp',
                    inputStyle={'marginRight': '6px'},
                    labelStyle={'display': 'block', 'marginBottom': '8px', 'color': '#1e293b', 'fontSize': '14px', 'fontWeight': '500'}
                ),

                html.Hr(style={'borderColor': '#cbd5e1', 'margin': '20px 0'}),
                html.H5("⚡ PyTorch Loader Config", style={'color': '#0f172a', 'fontWeight': '700'}),

                # Window Size Slider (updatemode='mouseup' for fast responsiveness)
                html.Label("Window Size (sec):", className='fw-bold mb-2', style={'fontSize': '14px', 'color': '#1e293b'}),
                dcc.Slider(id='win-size-slider', min=0.5, max=5.0, step=0.5, value=2.0,
                           updatemode='mouseup',
                           marks={0.5: '0.5s', 2.0: '2s', 5.0: '5s'}, tooltip={'always_visible': True}),

                # Stride Slider (updatemode='mouseup')
                html.Label("Stride (sec):", className='fw-bold mb-2 mt-4', style={'fontSize': '14px', 'color': '#1e293b'}),
                dcc.Slider(id='stride-slider', min=0.2, max=2.0, step=0.2, value=1.0,
                           updatemode='mouseup',
                           marks={0.2: '0.2s', 1.0: '1s', 2.0: '2s'}, tooltip={'always_visible': True}),

                # Target FS Dropdown
                html.Label("Target Sampling Frequency (Hz):", className='fw-bold mb-2 mt-4', style={'fontSize': '14px', 'color': '#1e293b'}),
                dcc.Dropdown(
                    id='target-fs-select',
                    options=[{'label': f'{fs} Hz', 'value': fs} for fs in [100, 200, 250, 400, 500]],
                    value=250,
                    clearable=False,
                    style={'color': '#0f172a', 'marginBottom': '18px'}
                ),

                # Batch Size Dropdown
                html.Label("PyTorch Batch Size:", className='fw-bold mb-2', style={'fontSize': '14px', 'color': '#1e293b'}),
                dcc.Dropdown(
                    id='batch-size-select',
                    options=[{'label': f'Batch Size = {b}', 'value': b} for b in [8, 16, 32, 64]],
                    value=16,
                    clearable=False,
                    style={'color': '#0f172a'}
                )
            ])
        ]),

        # Right Visualization Tabs
        html.Div(className='col-md-9', children=[
            dcc.Tabs(id='main-tabs', value='tab-signals', colors={'border': '#cbd5e1', 'primary': '#0284c7', 'background': '#f1f5f9'}, children=[

                # TAB 1: EDF & TCP Montage Signals
                dcc.Tab(label='📈 EDF Signals & TCP Montage', value='tab-signals', className='p-3', style={'backgroundColor': '#ffffff', 'color': '#334155'}, selected_style={'backgroundColor': '#ffffff', 'color': '#0284c7', 'fontWeight': 'bold'}, children=[

                    # Beginner Guide Explanatory Card
                    html.Div(className='p-3 mb-3 rounded shadow-sm', style={'backgroundColor': '#e0f2fe', 'border': '1px solid #7dd3fc'}, children=[
                        html.H5("💡 What Am I Looking At? (Quick Guide for Beginners)", style={'color': '#0369a1', 'fontWeight': '700', 'marginBottom': '8px'}),
                        html.P(children=[
                            html.B("Blue Waves: "), "Electrical signals recorded from brain scalp electrodes (e.g. FP1-F7 = Frontal to Temporal). ",
                            html.Br(),
                            html.B("Colored Vertical Overlays: "), "Clinical event annotations automatically plotted on the exact channels where they occur:"
                        ], style={'margin': 0, 'fontSize': '14px', 'color': '#0c4a6e'}),
                        html.Div(className='mt-2 d-flex flex-wrap gap-2', children=[
                            html.Span("🔴 spsw: Spike & Wave (Seizure)", className="badge bg-danger"),
                            html.Span("🟧 gped: Generalized Periodic", className="badge bg-warning text-dark"),
                            html.Span("🟨 pled: Periodic Discharge", className="badge bg-warning text-dark"),
                            html.Span("🟩 eyem: Eye Blink", className="badge bg-success"),
                            html.Span("🟦 artf: Noise Artifact", className="badge bg-primary"),
                            html.Span("⬜ bckg: Baseline Rhythm", className="badge bg-secondary")
                        ])
                    ]),

                    # Control Bar & Time Slider
                    html.Div(className='p-3 mb-3 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                        html.Div(className='d-flex align-items-center justify-content-between', children=[
                            html.Span("Display Time Range (Seconds):", className='fw-bold', style={'color': '#1e293b'}),
                            html.Div(style={'width': '65%'}, children=[
                                dcc.RangeSlider(id='time-range-slider', min=0, max=60, step=1, value=[0, 15],
                                               updatemode='mouseup', tooltip={'always_visible': True})
                            ])
                        ])
                    ]),

                    # Main EEG Signal Plot
                    html.Div(className='p-2 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                        dcc.Graph(id='eeg-signal-graph', style={'height': '780px'})
                    ])
                ]),

                # TAB 2: HTK Differential Energy Features
                dcc.Tab(label='📊 HTK Feature Inspector', value='tab-htk', className='p-3', style={'backgroundColor': '#ffffff', 'color': '#334155'}, selected_style={'backgroundColor': '#ffffff', 'color': '#0284c7', 'fontWeight': 'bold'}, children=[
                    html.Div(className='p-3 mb-3 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                        html.Div(className='row align-items-center', children=[
                            html.Div(className='col-md-5', children=[
                                html.Label("Select HTK Channel File:", className='fw-bold mb-1', style={'color': '#1e293b'}),
                                dcc.Dropdown(id='htk-ch-select', style={'color': '#0f172a'})
                            ])
                        ])
                    ]),
                    html.Div(className='p-3 mb-3 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                        dcc.Graph(id='htk-heatmap-graph', style={'height': '360px'})
                    ]),
                    html.Div(className='p-3 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                        dcc.Graph(id='htk-line-graph', style={'height': '360px'})
                    ])
                ]),

                # TAB 3: .rec vs .lab Annotations Inspector
                dcc.Tab(label='🏷️ Event Annotations (.rec / .lab)', value='tab-annotations', className='p-3', style={'backgroundColor': '#ffffff', 'color': '#334155'}, selected_style={'backgroundColor': '#ffffff', 'color': '#0284c7', 'fontWeight': 'bold'}, children=[
                    html.Div(className='row g-4', children=[
                        html.Div(className='col-md-6', children=[
                            html.Div(className='p-3 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                                html.H5("⏱️ .rec Recording Annotations (Seconds)", style={'color': '#0284c7', 'marginBottom': '16px'}),
                                dash_table.DataTable(
                                    id='rec-table',
                                    columns=[
                                        {'name': 'Channel', 'id': 'channel_idx'},
                                        {'name': 'Start (s)', 'id': 'start_sec'},
                                        {'name': 'Stop (s)', 'id': 'stop_sec'},
                                        {'name': 'Duration (s)', 'id': 'duration_sec'},
                                        {'name': 'Label', 'id': 'label_str'}
                                    ],
                                    style_header={'backgroundColor': '#f1f5f9', 'color': '#0284c7', 'fontWeight': 'bold', 'border': '1px solid #e2e8f0'},
                                    style_cell={'backgroundColor': '#ffffff', 'color': '#0f172a', 'textAlign': 'center', 'padding': '10px', 'border': '1px solid #e2e8f0'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}],
                                    page_size=10
                                )
                            ])
                        ]),
                        html.Div(className='col-md-6', children=[
                            html.Div(className='p-3 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                                html.H5("🔬 .lab Channel Annotations (10-us Resolution)", style={'color': '#0284c7', 'marginBottom': '16px'}),
                                dash_table.DataTable(
                                    id='lab-table',
                                    columns=[
                                        {'name': 'Channel', 'id': 'channel_idx'},
                                        {'name': 'Start (s)', 'id': 'start_sec'},
                                        {'name': 'Stop (s)', 'id': 'stop_sec'},
                                        {'name': 'Duration (s)', 'id': 'duration_sec'},
                                        {'name': 'Label', 'id': 'label_str'}
                                    ],
                                    style_header={'backgroundColor': '#f1f5f9', 'color': '#0284c7', 'fontWeight': 'bold', 'border': '1px solid #e2e8f0'},
                                    style_cell={'backgroundColor': '#ffffff', 'color': '#0f172a', 'textAlign': 'center', 'padding': '10px', 'border': '1px solid #e2e8f0'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}],
                                    page_size=10
                                )
                            ])
                        ])
                    ])
                ]),

                # TAB 4: PyTorch DataLoader Live Simulator
                dcc.Tab(label='🔥 PyTorch DataLoader Simulator', value='tab-pytorch', className='p-3', style={'backgroundColor': '#ffffff', 'color': '#334155'}, selected_style={'backgroundColor': '#ffffff', 'color': '#0284c7', 'fontWeight': 'bold'}, children=[
                    html.Div(className='row g-3 mb-4', children=[
                        html.Div(className='col-md-4', children=[
                            html.Div(className='p-3 rounded text-center shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #0284c7'}, children=[
                                html.H6("Tensor Output Shape (X)", style={'color': '#64748b', 'marginBottom': '4px'}),
                                html.H4(id='tensor-shape-x', style={'color': '#0284c7', 'fontWeight': 'bold'})
                            ])
                        ]),
                        html.Div(className='col-md-4', children=[
                            html.Div(className='p-3 rounded text-center shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #10b981'}, children=[
                                html.H6("Labels Output Shape (Y)", style={'color': '#64748b', 'marginBottom': '4px'}),
                                html.H4(id='tensor-shape-y', style={'color': '#10b981', 'fontWeight': 'bold'})
                            ])
                        ]),
                        html.Div(className='col-md-4', children=[
                            html.Div(className='p-3 rounded text-center shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #f59e0b'}, children=[
                                html.H6("Extracted Windows Count", style={'color': '#64748b', 'marginBottom': '4px'}),
                                html.H4(id='windows-count', style={'color': '#f59e0b', 'fontWeight': 'bold'})
                            ])
                        ])
                    ]),
                    html.Div(className='p-3 rounded shadow-sm', style={'backgroundColor': '#ffffff', 'border': '1px solid #e2e8f0'}, children=[
                        dcc.Graph(id='pytorch-batch-graph', style={'height': '460px'})
                    ])
                ]),

                # TAB 5: Interactive Help Guide & Terminology
                dcc.Tab(label='📖 Help Guide & Terms', value='tab-help', className='p-3', style={'backgroundColor': '#ffffff', 'color': '#334155'}, selected_style={'backgroundColor': '#ffffff', 'color': '#0284c7', 'fontWeight': 'bold'}, children=[
                    html.Div(className='help-card', children=[
                        html.H4("🧠 Clinical Event Definitions (6 Classes)", style={'color': '#0284c7', 'marginBottom': '16px'}),
                        html.Table(className='table table-hover table-bordered align-middle', children=[
                            html.Thead(className='table-light', children=[
                                html.Tr([
                                    html.Th("Class Code"),
                                    html.Th("Clinical Name"),
                                    html.Th("Medical Description"),
                                    html.Th("Diagnostic Significance")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Span("spsw", className="badge bg-danger fs-6")),
                                    html.Td(html.B("Spike and Slow Wave")),
                                    html.Td("Sharp transient (<70 ms) followed by a slow wave (200-500 ms)"),
                                    html.Td("Focal or generalized interictal epileptiform discharges (IEDs)")
                                ]),
                                html.Tr([
                                    html.Td(html.Span("gped", className="badge bg-warning text-dark fs-6")),
                                    html.Td(html.B("Generalized Periodic Discharge")),
                                    html.Td("Periodic waveforms appearing synchronously over both hemispheres"),
                                    html.Td("Acute encephalopathy, status epilepticus, anoxia")
                                ]),
                                html.Tr([
                                    html.Td(html.Span("pled", className="badge bg-warning text-dark fs-6")),
                                    html.Td(html.B("Periodic Lateralized Discharge")),
                                    html.Td("Repetitive sharp or spike wave complexes localized to one hemisphere"),
                                    html.Td("Focal brain lesions, acute stroke, viral encephalitis")
                                ]),
                                html.Tr([
                                    html.Td(html.Span("eyem", className="badge bg-success fs-6")),
                                    html.Td(html.B("Eye Movement")),
                                    html.Td("Low-frequency (<4 Hz) frontal signal deflections from eye blinks"),
                                    html.Td("Non-epileptic biological artifact")
                                ]),
                                html.Tr([
                                    html.Td(html.Span("artf", className="badge bg-primary fs-6")),
                                    html.Td(html.B("Artifact")),
                                    html.Td("High-amplitude muscle (EMG), electrode pop, or movement noise"),
                                    html.Td("Non-cerebral electrical interference")
                                ]),
                                html.Tr([
                                    html.Td(html.Span("bckg", className="badge bg-secondary fs-6")),
                                    html.Td(html.B("Background")),
                                    html.Td("Normal baseline cerebral rhythms (Alpha, Beta, Theta, Delta)"),
                                    html.Td("Normal resting cerebral activity")
                                ])
                            ])
                        ])
                    ]),

                    html.Div(className='help-card', children=[
                        html.H4("⚡ 22-Channel ACNS TCP Montage Standard", style={'color': '#0284c7', 'marginBottom': '16px'}),
                        html.P("To eliminate common reference noise, signals are computed as differential voltage pairs: V_ch = V_pos - V_neg"),
                        html.Div(className='row', children=[
                            html.Div(className='col-md-6', children=[
                                html.Ul(children=[
                                    html.Li([html.B("Left Temporal Chain: "), "FP1-F7, F7-T3, T3-T5, T5-O1"]),
                                    html.Li([html.B("Right Temporal Chain: "), "FP2-F8, F8-T4, T4-T6, T6-O2"]),
                                    html.Li([html.B("Parasagittal & Center Chain: "), "A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2"])
                                ])
                            ]),
                            html.Div(className='col-md-6', children=[
                                html.Ul(children=[
                                    html.Li([html.B("Left Parasagittal Chain: "), "FP1-F3, F3-C3, C3-P3, P3-O1"]),
                                    html.Li([html.B("Right Parasagittal Chain: "), "FP2-F4, F4-C4, C4-P4, P4-O2"])
                                ])
                            ])
                        ])
                    ]),

                    html.Div(className='help-card', children=[
                        html.H4("📊 Signal Processing Metrics & Formulas", style={'color': '#0284c7', 'marginBottom': '16px'}),
                        html.Ul(children=[
                            html.Li([html.B("Welch Power Spectral Density (PSD): "), "Estimates signal power across Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz), and Gamma (30-50Hz) bands via trapezoidal integration."]),
                            html.Li([html.B("Hjorth Activity: "), "Total signal power / variance Var(x)."]),
                            html.Li([html.B("Hjorth Mobility: "), "Square root of variance ratio of first derivative Var(dx/dt) / Var(x). Represents mean frequency."]),
                            html.Li([html.B("Hjorth Complexity: "), "Ratio of mobility of first derivative to mobility of signal. Represents frequency deviation."]),
                            html.Li([html.B("Z-Score Normalization: "), "(x - mean) / std. Standardizes channel voltage amplitudes."]),
                            html.Li([html.B("Robust Scaling: "), "(x - median) / IQR. Resilient against extreme noise artifacts."])
                        ])
                    ])
                ])
            ])
        ])
    ])
])

# Callbacks with Fast Caching
@app.callback(
    Output('edf-file-select', 'options'),
    Output('edf-file-select', 'value'),
    Input('split-select', 'value')
)
def update_edf_options(split):
    files = SESSION_FILES.get(split, [])
    options = [{'label': f, 'value': f} for f in files]
    default_val = files[0] if files else None
    return options, default_val


@app.callback(
    Output('htk-ch-select', 'options'),
    Output('htk-ch-select', 'value'),
    Input('edf-file-select', 'value')
)
def update_htk_options(rel_edf_path):
    if not rel_edf_path:
        return [], None
    abs_edf = os.path.join(CORPUS_ROOT, rel_edf_path)
    base_dir = os.path.dirname(abs_edf)
    base_name = os.path.splitext(os.path.basename(abs_edf))[0]

    htk_files = glob.glob(os.path.join(base_dir, f"{base_name}_ch*.htk"))
    options = [{'label': os.path.basename(p), 'value': p} for p in sorted(htk_files)]
    default_val = options[0]['value'] if options else None
    return options, default_val


@app.callback(
    Output('eeg-signal-graph', 'figure'),
    Output('time-range-slider', 'max'),
    Input('edf-file-select', 'value'),
    Input('montage-mode', 'value'),
    Input('time-range-slider', 'value')
)
def update_eeg_signals(rel_edf_path, montage_mode, time_range):
    if not rel_edf_path:
        return go.Figure(), 60

    abs_edf = os.path.join(CORPUS_ROOT, rel_edf_path)
    rec_path = os.path.splitext(abs_edf)[0] + '.rec'

    # Fast cached reads
    signals, fs, ch_names = cached_read_edf(abs_edf)
    events = cached_parse_rec(rec_path)

    if montage_mode == 'tcp':
        display_signals, display_names = build_tcp_montage(signals, ch_names)
    else:
        display_signals, display_names = signals, ch_names

    n_channels, total_samples = display_signals.shape
    total_sec = total_samples / fs

    start_sec, stop_sec = time_range
    start_sample = int(start_sec * fs)
    stop_sample = int(stop_sec * fs)
    stop_sample = min(stop_sample, total_samples)

    t = np.linspace(start_sec, stop_sample / fs, max(1, stop_sample - start_sample))

    fig = go.Figure()
    offset_spacing = 150.0  # uV offset per channel

    # Plot EEG Traces
    for i in range(n_channels):
        sig = display_signals[i, start_sample:stop_sample]
        # Fast scaling
        sig_scaled = (sig - np.mean(sig)) / (np.std(sig) + 1e-6) * 30.0 + (n_channels - 1 - i) * offset_spacing

        fig.add_trace(go.Scatter(
            x=t,
            y=sig_scaled,
            mode='lines',
            name=display_names[i],
            line=dict(width=1.2, color='#0284c7'),
            hoverinfo='x+name'
        ))

    # Add Prominent Event Overlay Shading & Badges
    events_in_range = []
    for ev in events:
        ev_start = ev['start_sec']
        ev_stop = ev['stop_sec']
        if ev_stop >= start_sec and ev_start <= stop_sec:
            events_in_range.append(ev)
            cls_str = ev['label_str']
            color = CLASS_COLORS.get(cls_str, 'rgba(148, 163, 184, 0.25)')

            # Highlight specific channel or full timeline
            ch_idx = ev['channel_idx']
            badge_text = f"<b>{cls_str.upper()}</b> ({ev_start:.1f}s - {ev_stop:.1f}s)"

            fig.add_vrect(
                x0=max(start_sec, ev_start),
                x1=min(stop_sec, ev_stop),
                fillcolor=color,
                opacity=0.45,
                layer="below",
                line_width=1,
                line_color=CLASS_HEX.get(cls_str, '#0284c7'),
                annotation_text=badge_text,
                annotation_position="top left",
                annotation_font=dict(color='#0f172a', size=11, family='sans-serif')
            )

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=140, r=40, t=50, b=50),
        title=dict(
            text=f"EEG Traces & Event Annotations ({len(events_in_range)} Events Visible in Range [{start_sec}s - {stop_sec}s])",
            font=dict(size=14, color='#0369a1')
        ),
        xaxis=dict(
            title='Time (Seconds)',
            showgrid=True,
            gridcolor='#e2e8f0',
            title_font=dict(size=13, color='#1e293b', family='sans-serif'),
            tickfont=dict(size=11, color='#1e293b')
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[(n_channels - 1 - i) * offset_spacing for i in range(n_channels)],
            ticktext=display_names,
            showgrid=True,
            gridcolor='#e2e8f0',
            tickfont=dict(size=12, color='#0f172a', family='sans-serif')
        ),
        showlegend=False
    )

    return fig, int(total_sec)


@app.callback(
    Output('htk-heatmap-graph', 'figure'),
    Output('htk-line-graph', 'figure'),
    Input('htk-ch-select', 'value')
)
def update_htk_plots(htk_path):
    if not htk_path or not os.path.exists(htk_path):
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_white', paper_bgcolor='#ffffff', plot_bgcolor='#ffffff')
        return empty_fig, empty_fig

    features, sample_period, parm_kind = parse_htk_file(htk_path)
    n_samples, n_feats = features.shape
    time_sec = np.arange(n_samples) * (sample_period * 1e-7)

    # Heatmap Figure
    fig_heat = go.Figure(data=go.Heatmap(
        z=features.T,
        x=time_sec,
        y=[f"Feat {i}" for i in range(n_feats)],
        colorscale='Blues'
    ))
    fig_heat.update_layout(
        title=f"HTK Differential Energy Feature Matrix ({os.path.basename(htk_path)})",
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=90, r=30, t=50, b=50),
        xaxis_title="Time (Seconds)",
        yaxis_title="Feature Dimension",
        font=dict(color='#0f172a')
    )

    # Line Plot Figure for First 3 Features
    fig_line = go.Figure()
    colors = ['#0284c7', '#10b981', '#f59e0b']
    for f_idx in range(min(3, n_feats)):
        fig_line.add_trace(go.Scatter(
            x=time_sec, y=features[:, f_idx], mode='lines', name=f"Feature {f_idx}",
            line=dict(width=1.5, color=colors[f_idx % len(colors)])
        ))
    fig_line.update_layout(
        title="Individual HTK Feature Trajectories",
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=70, r=30, t=50, b=50),
        xaxis_title="Time (Seconds)",
        yaxis_title="Feature Value",
        font=dict(color='#0f172a')
    )

    return fig_heat, fig_line


@app.callback(
    Output('rec-table', 'data'),
    Output('lab-table', 'data'),
    Input('edf-file-select', 'value')
)
def update_annotation_tables(rel_edf_path):
    if not rel_edf_path:
        return [], []

    abs_edf = os.path.join(CORPUS_ROOT, rel_edf_path)
    rec_path = os.path.splitext(abs_edf)[0] + '.rec'
    base_dir = os.path.dirname(abs_edf)
    base_name = os.path.splitext(os.path.basename(abs_edf))[0]
    lab_files = glob.glob(os.path.join(base_dir, f"{base_name}_ch*.lab"))

    rec_events = cached_parse_rec(rec_path)
    lab_events = []
    for lab_p in lab_files:
        lab_events.extend(parse_lab_file(lab_p))

    return rec_events, lab_events


@app.callback(
    Output('tensor-shape-x', 'children'),
    Output('tensor-shape-y', 'children'),
    Output('windows-count', 'children'),
    Output('pytorch-batch-graph', 'figure'),
    Input('edf-file-select', 'value'),
    Input('win-size-slider', 'value'),
    Input('stride-slider', 'value'),
    Input('target-fs-select', 'value'),
    Input('batch-size-select', 'value')
)
def update_pytorch_simulation(rel_edf_path, win_sec, stride_sec, target_fs, batch_size):
    if not rel_edf_path:
        return "N/A", "N/A", "0", go.Figure()

    abs_edf = os.path.join(CORPUS_ROOT, rel_edf_path)
    rec_path = os.path.splitext(abs_edf)[0] + '.rec'
    events = cached_parse_rec(rec_path)
    signals, fs, ch_names = cached_read_edf(abs_edf)
    montage_signals, _ = build_tcp_montage(signals, ch_names)

    window_samples = int(win_sec * fs)
    stride_samples = int(stride_sec * fs)
    n_samples = montage_signals.shape[1]

    n_windows = max(0, (n_samples - window_samples) // stride_samples + 1)
    target_samples = int(win_sec * target_fs)

    shape_x_str = f"({batch_size}, 22, {target_samples})"
    shape_y_str = f"({batch_size},)"

    fig = go.Figure()
    if n_windows > 0:
        sample_win = montage_signals[:, :window_samples]
        t = np.linspace(0, win_sec, target_samples)

        for ch in range(min(5, sample_win.shape[0])):
            fig.add_trace(go.Scatter(
                x=t, y=sample_win[ch, :target_samples], mode='lines',
                name=TCP_MONTAGE_DEFINITIONS[ch][1],
                line=dict(width=1.5)
            ))

    fig.update_layout(
        title=f"Sample PyTorch Window Tensor Iteration (Batch Size: {batch_size}, Window: {win_sec}s, Target FS: {target_fs}Hz)",
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=80, r=30, t=50, b=50),
        xaxis_title="Window Time (Seconds)",
        yaxis_title="Amplitude (uV)",
        font=dict(color='#0f172a')
    )

    return shape_x_str, shape_y_str, str(n_windows), fig


if __name__ == '__main__':
    print("=" * 70)
    print("Launching TUH EEG Event Corpus Plotly Dash Interactive Dashboard...")
    print("Access locally at: http://127.0.0.1:8050/")
    print("=" * 70)
    app.run(debug=False, port=8050)
