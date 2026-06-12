import os
import json
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
INPUT_CSV = "Dwarf_15Mpc.csv" #Update according to your csv file
OUTPUT_HTML = "index.html"
N_MC = 200

# Math Constants
D_AND = 0.783
L_AND, B_AND = 121.2, -21.6
VH_AND = -300.0
V_LSR = np.array([11.1, 12.2, 7.2])
FIDUCIAL_F_M = 0.67
FIDUCIAL_V0 = 255.2

REFERENCE_STRUCTURES = [
    {"name": "Milky Way", "l": 0.0, "b": 0.0, "D": 1e-5, "Vlos": 0.0, "color": "cyan", "size": 15},
    {"name": "M31 (Andromeda)", "l": L_AND, "b": B_AND, "D": D_AND, "Vlos": VH_AND, "color": "cyan", "size": 15},
    {"name": "M81", "l": 142.09, "b": 40.90, "D": 3.6, "Vlos": -34.0, "color": "orange", "size": 12},
    {"name": "IC342", "l": 138.17, "b": 10.58, "D": 3.3, "Vlos": 31.0, "color": "magenta", "size": 12},
    {"name": "Centaurus A", "l": 309.52, "b": 19.42, "D": 3.8, "Vlos": 547.0, "color": "yellow", "size": 12},
    {"name": "Virgo Cluster", "l": 284.0, "b": 74.0, "D": 16.5, "Vlos": 1000.0, "color": "red", "size": 25}
]

# ==========================================
# MATHEMATICAL KINEMATICS
# ==========================================
def galactic_unit_vector(l_deg: float, b_deg: float) -> np.ndarray:
    """Converts galactic coordinates to a 3D Cartesian unit vector."""
    l_rad, b_rad = np.radians(l_deg), np.radians(b_deg)
    return np.array([
        np.cos(b_rad) * np.cos(l_rad),
        np.cos(b_rad) * np.sin(l_rad),
        np.sin(b_rad)
    ])

def compute_solar_apex(f_m: float, V0: float) -> np.ndarray:
    """Computes the solar velocity vector with respect to the LG barycenter."""
    rA_hat = galactic_unit_vector(L_AND, B_AND)
    V0_vec = np.array([0, V0, 0])
    vA = VH_AND + np.dot(V0_vec + V_LSR, rA_hat)
    return V0_vec + V_LSR - (vA / (1 + f_m)) * rA_hat

def distance_modulus_to_mpc(mu: np.ndarray) -> np.ndarray:
    """Converts distance modulus to Megaparsecs."""
    return (10.0 ** ((mu + 5.0) / 5.0)) / 1e6

def transform_coordinates_single(l, b, D, Vh, f_m, V0):
    """Calculates both Heliocentric and LG Centric Cartesian coordinates."""
    l_rad, b_rad = np.radians(l), np.radians(b)
    
    # 1. Heliocentric Cartesian
    x_helio = D * np.cos(b_rad) * np.cos(l_rad)
    y_helio = D * np.cos(b_rad) * np.sin(l_rad)
    z_helio = D * np.sin(b_rad)
    gal_pos = np.vstack([x_helio, y_helio, z_helio])
    
    # 2. Local Group Barycenter Correction
    rA = galactic_unit_vector(L_AND, B_AND) * D_AND
    r_CM = (1.0 / (1 + f_m)) * rA
    
    r_LG_vec = (gal_pos - r_CM[:, None]) if gal_pos.ndim > 1 else (gal_pos - r_CM)
    x_lg, y_lg, z_lg = r_LG_vec[0], r_LG_vec[1], r_LG_vec[2]
    r_LG = np.linalg.norm(r_LG_vec, axis=0)
    
    # 3. Velocity Correction
    v_sun = compute_solar_apex(f_m, V0)
    gal_unit = np.vstack([np.cos(b_rad)*np.cos(l_rad), np.cos(b_rad)*np.sin(l_rad), np.sin(b_rad)])
    
    delta_v = np.dot(v_sun, gal_unit) if gal_unit.ndim > 1 else np.dot(v_sun, gal_unit.flatten())
    v_LG = Vh + delta_v
    
    return x_helio, y_helio, z_helio, x_lg, y_lg, z_lg, r_LG, v_LG

# ==========================================
# DATA PROCESSING
# ==========================================
def process_data(filepath: str) -> tuple:
    """Loads CSV, applies Monte Carlo sampling, and transforms coordinates."""
    print(f"Running Monte Carlo sampling (N={N_MC}) on {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found!")
        exit(1)
        
    num_objects = len(df)
    mc_arrays = {k: np.zeros((num_objects, N_MC)) for k in 
                 ['X_h', 'Y_h', 'Z_h', 'X_lg', 'Y_lg', 'Z_lg', 'r_lg', 'v_lg', 'D_h']}
    
    for i in range(N_MC):
        d_sampled = distance_modulus_to_mpc(np.random.normal(df['D_mod'], df['D_mod_err']))
        v_sampled = np.random.normal(df['Vlos'], df['Vlos_err'])
        
        xh, yh, zh, xlg, ylg, zlg, rlg, vlg = transform_coordinates_single(
            df['l'].values, df['b'].values, d_sampled, v_sampled, FIDUCIAL_F_M, FIDUCIAL_V0
        )
        
        mc_arrays['X_h'][:, i], mc_arrays['Y_h'][:, i], mc_arrays['Z_h'][:, i] = xh, yh, zh
        mc_arrays['X_lg'][:, i], mc_arrays['Y_lg'][:, i], mc_arrays['Z_lg'][:, i] = xlg, ylg, zlg
        mc_arrays['r_lg'][:, i], mc_arrays['v_lg'][:, i] = rlg, vlg
        mc_arrays['D_h'][:, i] = d_sampled

    # Assign means and standard deviations
    for key in ['X_h', 'Y_h', 'Z_h', 'X_lg', 'Y_lg', 'Z_lg']:
        df[key] = np.mean(mc_arrays[key], axis=1)
        
    df['r_LG'], df['r_LG_err'] = np.mean(mc_arrays['r_lg'], axis=1), np.std(mc_arrays['r_lg'], axis=1)
    df['v_LG'], df['v_LG_err'] = np.mean(mc_arrays['v_lg'], axis=1), np.std(mc_arrays['v_lg'], axis=1)
    df['D'], df['D_err'] = np.mean(mc_arrays['D_h'], axis=1), np.std(mc_arrays['D_h'], axis=1)
    df['Vlos_mean'] = df['Vlos'] 
    
    # Process References
    refs_df = pd.DataFrame(REFERENCE_STRUCTURES)
    rx_h, ry_h, rz_h, rx_lg, ry_lg, rz_lg, rr_lg, rv_lg = transform_coordinates_single(
        refs_df['l'].values, refs_df['b'].values, refs_df['D'].values, refs_df['Vlos'].values, FIDUCIAL_F_M, FIDUCIAL_V0
    )
    refs_df['X_h'], refs_df['Y_h'], refs_df['Z_h'] = rx_h, ry_h, rz_h
    refs_df['X_lg'], refs_df['Y_lg'], refs_df['Z_lg'] = rx_lg, ry_lg, rz_lg
    refs_df['r_LG'], refs_df['v_LG'] = rr_lg, rv_lg
    
    return df, refs_df

# ==========================================
# VISUALIZATION GENERATION
# ==========================================
def build_3d_map(df: pd.DataFrame, refs_df: pd.DataFrame) -> go.Figure:
    """Builds the interactive 3D Plotly Figure."""
    print("Building interactive 3D UI...")
    fig = go.Figure()

    def add_frame_traces(frame_suffix, is_visible):
        vel_col = 'v_LG' if frame_suffix == 'lg' else 'Vlos_mean'
        
        # 1. Dwarf Galaxies
        fig.add_trace(go.Scatter3d(
            x=df[f'Z_{frame_suffix}'], y=df[f'Y_{frame_suffix}'], z=df[f'X_{frame_suffix}'], 
            mode='markers', name='Dwarf Galaxies', visible=is_visible,
            marker=dict(
                size=4, color=df[vel_col], colorscale='RdBu_r', 
                colorbar=dict(title=f"Vel ({frame_suffix}) km/s", x=0.01, y=0.45, thickness=12, len=0.7), 
                opacity=0.8
            ),
            text=df['name'],
            customdata=np.stack((
                df['D' if frame_suffix=="h" else 'r_LG'], df['D_err' if frame_suffix=="h" else 'r_LG_err'],
                df[vel_col], df['Vlos_err' if frame_suffix=="h" else 'v_LG_err'],
                df['l'], df['b']
            ), axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br><br>" +
                f"Distance ({'Helio' if frame_suffix=='h' else 'LG'}): %{{customdata[0]:.2f}} ± %{{customdata[1]:.2f}} Mpc<br>" +
                f"Velocity ({'Helio' if frame_suffix=='h' else 'LG'}): %{{customdata[2]:.2f}} ± %{{customdata[3]:.2f}} km/s<br>" +
                "Coords (l, b): %{customdata[4]:.2f}°, %{customdata[5]:.2f}°<br><extra></extra>"
            )
        ))
        
        # 2. Reference Structures
        for _, row in refs_df.iterrows():
            vel_col_ref = 'v_LG' if frame_suffix == 'lg' else 'Vlos'
            fig.add_trace(go.Scatter3d(
                x=[row[f'Z_{frame_suffix}']], y=[row[f'Y_{frame_suffix}']], z=[row[f'X_{frame_suffix}']],
                mode='markers+text', name=row["name"], text=[row["name"]], textposition="top center", visible=is_visible,
                marker=dict(size=row["size"], color=row["color"], line=dict(color='white', width=1)),
                hovertemplate=f"<b>{row['name']}</b><br><br>Dist: {row['D' if frame_suffix=='h' else 'r_LG']:.2f} Mpc<br>Vel: {row[vel_col_ref]:.2f} km/s<br><extra></extra>"
            ))

        # 3. M31 Leapfrog Infall Track
        mw_row = refs_df[refs_df['name'] == 'Milky Way'].iloc[0]
        m31_row = refs_df[refs_df['name'] == 'M31 (Andromeda)'].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=[mw_row[f'Z_{frame_suffix}'], m31_row[f'Z_{frame_suffix}']],
            y=[mw_row[f'Y_{frame_suffix}'], m31_row[f'Y_{frame_suffix}']],
            z=[mw_row[f'X_{frame_suffix}'], m31_row[f'X_{frame_suffix}']],
            mode='lines', name='M31 Radial Track', visible=is_visible,
            line=dict(color='#00ffff', width=3, dash='dot'), hoverinfo='skip'
        ))

    add_frame_traces('h', True)   
    add_frame_traces('lg', False) 

    # UI Elements Setup
    num_traces = 2 + len(REFERENCE_STRUCTURES) # Dwarfs + M31 Track + References
    
    # Slider
    steps = [dict(
        method="relayout", label=f"{d} Mpc",
        args=[{"scene.xaxis.range": [-d, d], "scene.yaxis.range": [-d, d], "scene.zaxis.range": [-d, d]}]
    ) for d in [3, 5, 10, 15, 20]]

    sliders = [dict(
        active=3, currentvalue={"prefix": "Radius: ", "font": {"color": "white"}},
        pad={"t": 0, "b": 10}, x=0.99, y=0.05, xanchor="right", yanchor="bottom", len=0.175,
        font=dict(color="white"), bgcolor="rgba(255, 255, 255, 0.1)", 
        activebgcolor="#00ffff", bordercolor="rgba(0, 0, 0, 0)", steps=steps
    )]

    # Dynamic Buttons
    h_on, h_off = "<b><span style='color: #00ff00;'> Heliocentric </span></b>", " Heliocentric "
    lg_on, lg_off = "<b><span style='color: #00ff00;'> Local Group Centric </span></b>", " Local Group Centric "

    fig.update_layout(
        uirevision='constant',
        sliders=sliders,
        legend=dict(yanchor="top", y=0.85, xanchor="right", x=0.99, bgcolor="rgba(20, 20, 20, 0.7)", bordercolor="#444", borderwidth=1, font=dict(size=12)),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, y=1.0, xanchor="left", yanchor="top",
            showactive=False, bgcolor="#222222", bordercolor="#444", borderwidth=1, font=dict(color="white", size=12),
            buttons=list([
                dict(
                    label=h_on, method="update", 
                    args=[{"visible": [True]*num_traces + [False]*num_traces}, 
                          {"title": dict(text="Near-Field Cosmology (Heliocentric)", x=0.5, y=0.95, xanchor="center", yanchor="top"),
                           "updatemenus[0].buttons[0].label": h_on, "updatemenus[0].buttons[1].label": lg_off}]
                ),
                dict(
                    label=lg_off, method="update", 
                    args=[{"visible": [False]*num_traces + [True]*num_traces}, 
                          {"title": dict(text="Near-Field Cosmology (LG Barycenter)", x=0.5, y=0.95, xanchor="center", yanchor="top"),
                           "updatemenus[0].buttons[0].label": h_off, "updatemenus[0].buttons[1].label": lg_on}]
                ),
            ]),
        )],
        title=dict(text="Near-Field Cosmology (Heliocentric)", x=0.5, y=0.95, xanchor="center", yanchor="top"),
        paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white'),
        scene=dict(
            dragmode='turntable',
            xaxis_title='Z (Mpc)', yaxis_title='Y (Mpc)', zaxis_title='X (Mpc)',
            xaxis=dict(range=[-15, 15], gridcolor='#333', backgroundcolor='black'),
            yaxis=dict(range=[-15, 15], gridcolor='#333', backgroundcolor='black'),
            zaxis=dict(range=[-15, 15], gridcolor='#333', backgroundcolor='black'),
            aspectmode='data'
            
        ),
        margin=dict(l=0, r=0, b=0, t=0) 
    )
    return fig

# ==========================================
# MAIN EXECUTION WITH UI WRAPPER
# ==========================================
if __name__ == "__main__":
    # 1. Process the datasets
    df_main, refs_main = process_data(INPUT_CSV)
    
    # 2. Build the visual model
    figure = build_3d_map(df_main, refs_main)

    # 3. Create a JSON Dictionary of Galaxy Coordinates for the Search function
    search_dict = {}
    for idx, row in df_main.iterrows():
        # Plotly mapping logic used in the script maps: xaxis=Z, yaxis=Y, zaxis=X
        search_dict[row['name']] = {
            'h': {'x': row['Z_h'], 'y': row['Y_h'], 'z': row['X_h']},
            'lg': {'x': row['Z_lg'], 'y': row['Y_lg'], 'z': row['X_lg']}
        }
    for idx, row in refs_main.iterrows():
        search_dict[row['name']] = {
            'h': {'x': row['Z_h'], 'y': row['Y_h'], 'z': row['X_h']},
            'lg': {'x': row['Z_lg'], 'y': row['Y_lg'], 'z': row['X_lg']}
        }
    search_json = json.dumps(search_dict)

    # 4. Extract the Plotly HTML snippet
    plotly_html = figure.to_html(full_html=False, include_plotlyjs='cdn', div_id='plotly-graph')

    # 5. Build Custom HTML Wrapper with Search UI overlay (Using safe placeholders)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Near-Field Cosmology</title>
        <style>
            body { margin: 0; padding: 0; background-color: black; color: white; font-family: Arial, sans-serif; overflow: hidden; width: 100vw; height: 100vh; }
            
            /* FORCE PLOTLY FULL SCREEN */
            #plotly-graph { width: 100vw !important; height: 100vh !important; }
            
            #search-container {
                position: absolute;
                top: 50px;  /* Drops it right below the Plotly buttons */
                left: 10px; /* Aligns it to the left edge */
                z-index: 1000;
                background: rgba(30, 30, 30, 0.85);
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #444;
                box-shadow: 0 4px 6px rgba(0,0,0,0.5);
                display: flex; 
                align-items: center;
            }
            #searchBox { padding: 6px; width: 200px; background: #222; color: white; border: 1px solid #777; border-radius: 4px; }
            #viewBtn, #resetBtn { padding: 6px 12px; border: none; cursor: pointer; font-weight: bold; border-radius: 4px; margin-left: 5px; transition: 0.2s;}
            #viewBtn { background: #00ffff; color: black; }
            #viewBtn:hover { background: #00cccc; }
            #resetBtn { background: #444; color: white; }
            #resetBtn:hover { background: #666; }
        </style>
    </head>
    <body>
        <div id="search-container">
            <input type="text" id="searchBox" list="galaxy-list" placeholder="Search for a galaxy...">
            <datalist id="galaxy-list"></datalist>
            <button id="viewBtn">View</button>
            <button id="resetBtn">Reset View</button>
        </div>
        
        @@PLOTLY_GRAPH_HERE@@

        <script>
            const searchData = @@SEARCH_DATA_HERE@@;
            
            const datalist = document.getElementById('galaxy-list');
            for (let name in searchData) {
                let option = document.createElement('option');
                option.value = name;
                datalist.appendChild(option);
            }

            let highlightTraceIndex = -1;

            document.getElementById('viewBtn').addEventListener('click', function() {
                const name = document.getElementById('searchBox').value;
                if (!searchData[name]) {
                    alert("Galaxy not found! Please check the spelling or select from the dropdown.");
                    return;
                }

                const graph = document.getElementById('plotly-graph');
                const isLG = graph.layout.title.text.includes('LG Barycenter');
                const frame = isLG ? 'lg' : 'h';
                const coords = searchData[name][frame];

                // 1. Calculate the normalization factor for the camera
                const x_range = (graph.layout.scene && graph.layout.scene.xaxis && graph.layout.scene.xaxis.range) ? graph.layout.scene.xaxis.range : [-15, 15];
                const r_max = (x_range[1] - x_range[0]) / 2.0;
                
                const cx = coords.x / r_max;
                const cy = coords.y / r_max;
                const cz = coords.z / r_max;

                // 2. Point camera center AT the galaxy, and put the 'eye' close to it
                const zoomOffset = 0.08; 
                const update = {
                    'scene.camera.center': { x: cx, y: cy, z: cz },
                    'scene.camera.eye': { x: cx + zoomOffset, y: cy + zoomOffset, z: cz + zoomOffset }
                };
                Plotly.relayout('plotly-graph', update);

                // 3. Highlight Target
                const trace = {
                    x: [coords.x], y: [coords.y], z: [coords.z],
                    mode: 'markers+text',
                    text: ['🎯 ' + name],
                    textposition: 'top center',
                    textfont: {color: '#00ffff', size: 15, weight: 'bold'},
                    marker: {size: 15, color: 'rgba(0,0,0,0)', line: {color: '#00ffff', width: 3}},
                    name: 'Highlight',
                    showlegend: false,
                    hoverinfo: 'skip'
                };

                if (highlightTraceIndex !== -1) {
                    Plotly.deleteTraces('plotly-graph', highlightTraceIndex).then(() => {
                        Plotly.addTraces('plotly-graph', trace).then(() => {
                            highlightTraceIndex = graph.data.length - 1;
                        });
                    });
                } else {
                    Plotly.addTraces('plotly-graph', trace).then(() => {
                        highlightTraceIndex = graph.data.length - 1;
                    });
                }
            });

            document.getElementById('resetBtn').addEventListener('click', function() {
                document.getElementById('searchBox').value = '';
                if (highlightTraceIndex !== -1) {
                    Plotly.deleteTraces('plotly-graph', highlightTraceIndex);
                    highlightTraceIndex = -1;
                }
                
                // Reset the camera back to the default origin view
                Plotly.relayout('plotly-graph', {
                    'scene.camera.center': {x: 0, y: 0, z: 0},
                    'scene.camera.eye': {x: 1.25, y: 1.25, z: 1.25}
                });
            });
        </script>
    </body>
    </html>
    """

    # 6. Stream output directly to file to prevent MemoryError
    print(f"Exporting application to {OUTPUT_HTML}...")
    
    # Split using our new safe tokens
    part1, rest = html_template.split('@@PLOTLY_GRAPH_HERE@@')
    part2, part3 = rest.split('@@SEARCH_DATA_HERE@@')

    # Write sequentially to the hard drive
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(part1)
        f.write(plotly_html)
        f.write(part2)
        f.write(search_json)
        f.write(part3)
    
    # Launch automatically in default browser
    filepath = os.path.realpath(OUTPUT_HTML)
    webbrowser.open(f'file://{filepath}')
    print("Success! Process complete.")
