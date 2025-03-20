import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib
import mplcursors

# Use the TkAgg backend for GUI compatibility.
matplotlib.use("TkAgg")

# Conversion factor and threshold
MS_TO_KNOTS = 1.94384
# Only process rows with Device1_TrueWindSpeed (m/s) > MIN_TWS_MS
MIN_TWS_MS = 2.05778

# Global variables
data_folder = None
processed_data = None  # Combined and processed DataFrame from CSV files
# SOG pivot table (uses TWA (deg), rounded to nearest multiple of 5)
sog_table = None
vmg_table = None       # Extended VMG table (using TWA_raw (deg) for angles)
# Polar pivot table for plotting (uses TWA (deg), rounded to nearest multiple of 5)
polar_table = None
all_data = None        # Backup copy of the processed data
tws_vars = {}          # Dictionary to store TWS checkbox variables

# Declare GUI button variables globally.
export_sog_button = None
export_vmg_button = None
show_polar_button = None
save_polar_button = None
checkbox_frame = None

# -------------------- DATA PROCESSING FUNCTIONS -------------------- #


def process_csv_files(folder):
    """
    Process all CSV files in the given folder:
      - Reads only the required columns.
      - Filters out rows where Device1_TrueWindSpeed (m/s) is less than MIN_TWS_MS.
      - Converts SpeedOverGround from m/s to knots (rounded to 2 decimals).
      - Creates two wind angle columns:
            * 'TWA_raw (deg)' = Device1_TrueWindAngle (deg) rounded to the nearest integer 
              (used for the VMG table).
            * 'TWA (deg)' = Device1_TrueWindAngle (deg) rounded to the nearest multiple of 5 
              (used for the SOG table and now for the polar diagram).
      - Converts Device1_TrueWindSpeed (m/s) to TWS (knots) (rounded to nearest integer).
      - Converts Device1_VelocityMadeGood (m/s) to VMG (knots) (rounded to 2 decimals).

    Returns a DataFrame with:
      'SOG (knots)', 'TWA_raw (deg)', 'TWA (deg)', 'TWS (knots)', 'VMG (knots)'
    """
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    dataframes = []
    for file in csv_files:
        filepath = os.path.join(folder, file)
        try:
            df = pd.read_csv(filepath, usecols=[
                'SpeedOverGround (m/s)',
                'Device1_TrueWindAngle (deg)',
                'Device1_TrueWindSpeed (m/s)',
                'Device1_VelocityMadeGood (m/s)'
            ])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        # Filter out rows where true wind speed is less than MIN_TWS_MS.
        df = df[df['Device1_TrueWindSpeed (m/s)'] > MIN_TWS_MS]
        dataframes.append(df)
    if not dataframes:
        print("No CSV files found or no data passed the TWS filter.")
        return pd.DataFrame()
    combined = pd.concat(dataframes, ignore_index=True)
    combined['SOG (knots)'] = (
        combined['SpeedOverGround (m/s)'] * MS_TO_KNOTS).round(2)
    # For VMG table: use integer-rounded TWA.
    combined['TWA_raw (deg)'] = combined['Device1_TrueWindAngle (deg)'].round(
        0).astype(int)
    # For SOG table and polar diagram: use TWA rounded to the nearest multiple of 5.
    combined['TWA (deg)'] = (
        (combined['Device1_TrueWindAngle (deg)'] / 5).round() * 5).astype(int)
    combined['TWS (knots)'] = (
        combined['Device1_TrueWindSpeed (m/s)'] * MS_TO_KNOTS).round().astype(int)
    combined['VMG (knots)'] = (
        combined['Device1_VelocityMadeGood (m/s)'] * MS_TO_KNOTS).round(2)
    processed = combined[[
        'SOG (knots)', 'TWA_raw (deg)', 'TWA (deg)', 'TWS (knots)', 'VMG (knots)']]
    return processed

# -------------------- HELPER FUNCTION FOR ANGLE FILTERING -------------------- #


def apply_angle_filtering(df, angle_col):
    """
    Applies filtering to the DataFrame based on disallowed wind angle values.
    For the given angle_col, rows with angles below 30°, above 330°, or between 160° and 200°
    will have all SOG values (in columns other than the angle column) set to NaN.

    Returns the modified DataFrame.
    """
    mask = (df[angle_col] < 30) | (df[angle_col] > 330) | (
        (df[angle_col] >= 160) & (df[angle_col] <= 200))
    for col in df.columns[1:]:
        df.loc[mask, col] = np.nan
    return df

# -------------------- TABLE GENERATION FUNCTIONS -------------------- #


def generate_sog_table(df):
    """
    Generate a SOG pivot table from the processed DataFrame:
      - Rows: 'TWA (deg)' (rounded to the nearest multiple of 5).
      - Columns: TWS (knots) (only values from 4 knots upward).
      - Values: 95th percentile of SOG (knots) for each combination (rounded to 2 decimals).
      - Filters out rows with TWA values below 30°, above 330°, or strictly between 160° and 200°.

    Returns the pivot table DataFrame.
    """
    df_filtered = df[df['TWS (knots)'] >= 4]
    grouped = df_filtered.groupby(['TWA (deg)', 'TWS (knots)'])['SOG (knots)'].agg(
        lambda x: np.percentile(x, 95)).reset_index()
    grouped['SOG (knots)'] = grouped['SOG (knots)'].round(2)
    grouped = grouped.replace(0, np.nan)
    allowed_twa = sorted({t for t in df_filtered['TWA (deg)'].unique(
    ) if t >= 30 and t <= 330 and not (t > 160 and t < 200)})
    tws_values = sorted(df_filtered['TWS (knots)'].unique())
    full_grid = pd.DataFrame(list(itertools.product(allowed_twa, tws_values)),
                             columns=['TWA (deg)', 'TWS (knots)'])
    merged = pd.merge(full_grid, grouped, on=[
                      'TWA (deg)', 'TWS (knots)'], how='left')
    merged = merged.replace(0, np.nan)
    pivot = merged.pivot(
        index='TWA (deg)', columns='TWS (knots)', values='SOG (knots)')
    pivot.reset_index(inplace=True)
    return pivot


def generate_polar_table(df):
    """
    Generate a pivot table for the polar diagram from the processed DataFrame:
      - Rows: 'TWA (deg)' (rounded to the nearest multiple of 5).
      - Columns: TWS (knots) (only values from 4 knots upward).
      - Values: 95th percentile of SOG (knots) for each combination (rounded to 2 decimals).

    Returns the pivot table DataFrame.
    """
    df_filtered = df[df['TWS (knots)'] >= 4]
    grouped = df_filtered.groupby(['TWA (deg)', 'TWS (knots)'])['SOG (knots)'].agg(
        lambda x: np.percentile(x, 95)).reset_index()
    grouped['SOG (knots)'] = grouped['SOG (knots)'].round(2)
    grouped = grouped.replace(0, np.nan)
    angle_values = sorted(df_filtered['TWA (deg)'].unique())
    tws_values = sorted(df_filtered['TWS (knots)'].unique())
    full_grid = pd.DataFrame(list(itertools.product(angle_values, tws_values)),
                             columns=['TWA (deg)', 'TWS (knots)'])
    merged = pd.merge(full_grid, grouped, on=[
                      'TWA (deg)', 'TWS (knots)'], how='left')
    merged = merged.replace(0, np.nan)
    pivot = merged.pivot(
        index='TWA (deg)', columns='TWS (knots)', values='SOG (knots)')
    pivot.reset_index(inplace=True)
    return pivot


def get_best_vmg(sub_df, lower_bound, upper_bound, lower_inclusive=True, upper_inclusive=True, absolute=False):
    """
    From a subset of data for a given TWS, filter rows where TWA (deg) is between lower_bound and upper_bound.
    Returns a tuple (best_vmg, angle) for the row with the highest absolute VMG (if absolute=True) 
    or the maximum VMG otherwise. If no rows match, returns (np.nan, np.nan).
    Note: The angle returned is taken from the 'TWA_raw (deg)' column.
    """
    cond_lower = sub_df['TWA (deg)'] >= lower_bound if lower_inclusive else sub_df['TWA (deg)'] > lower_bound
    cond_upper = sub_df['TWA (deg)'] <= upper_bound if upper_inclusive else sub_df['TWA (deg)'] < upper_bound
    filtered = sub_df[cond_lower & cond_upper]
    if filtered.empty:
        return np.nan, np.nan
    idx = filtered['VMG (knots)'].abs().idxmax(
    ) if absolute else filtered['VMG (knots)'].idxmax()
    return filtered.loc[idx, 'VMG (knots)'], filtered.loc[idx, 'TWA_raw (deg)']


def generate_vmg_table(df):
    """
    Generate an extended VMG table from the processed DataFrame.
    For each unique TWS (>= 4 knots), compute four segments:
      - Starboard Beat: TWA from 0° to 90°.
      - Starboard Run: TWA from just above 90° to 180°.
      - Port Beat: TWA from 270° to 360°.
      - Port Run: TWA from 180° up to (but not including) 270°.
    For each segment, select the row with the highest absolute VMG and return its VMG (as a positive value)
    and the corresponding TWA_raw (deg).

    Returns a DataFrame with columns:
      'TWS (knots)',
      'Beat Angle Starboard (deg)', 'Beat VMG Starboard (knots)',
      'Run Angle Starboard (deg)', 'Run VMG Starboard (knots)',
      'Beat Angle Port (deg)', 'Beat VMG Port (knots)',
      'Run Angle Port (deg)', 'Run VMG Port (knots)'
    """
    vmg_rows = []
    unique_tws = sorted(df['TWS (knots)'].unique())
    for tws in unique_tws:
        if tws < 4:
            continue
        sub = df[df['TWS (knots)'] == tws]
        # Starboard side:
        beat_vmg_starboard, beat_angle_starboard = get_best_vmg(
            sub, 0, 90, True, True, absolute=False)
        run_vmg_starboard, run_angle_starboard = get_best_vmg(
            sub, 90, 180, False, True, absolute=True)
        run_vmg_starboard = abs(run_vmg_starboard)
        # Port side:
        beat_vmg_port, beat_angle_port = get_best_vmg(
            sub, 270, 360, True, True, absolute=False)
        run_vmg_port, run_angle_port = get_best_vmg(
            sub, 180, 270, True, False, absolute=True)
        run_vmg_port = abs(run_vmg_port)
        vmg_rows.append({
            'TWS (knots)': int(tws),
            'Beat Angle Starboard (deg)': int(beat_angle_starboard) if not pd.isna(beat_angle_starboard) else np.nan,
            'Beat VMG Starboard (knots)': round(beat_vmg_starboard, 2) if not pd.isna(beat_vmg_starboard) else np.nan,
            'Run Angle Starboard (deg)': int(run_angle_starboard) if not pd.isna(run_angle_starboard) else np.nan,
            'Run VMG Starboard (knots)': round(run_vmg_starboard, 2) if not pd.isna(run_vmg_starboard) else np.nan,
            'Beat Angle Port (deg)': int(beat_angle_port) if not pd.isna(beat_angle_port) else np.nan,
            'Beat VMG Port (knots)': round(beat_vmg_port, 2) if not pd.isna(beat_vmg_port) else np.nan,
            'Run Angle Port (deg)': int(run_angle_port) if not pd.isna(run_angle_port) else np.nan,
            'Run VMG Port (knots)': round(run_vmg_port, 2) if not pd.isna(run_vmg_port) else np.nan
        })
    vmg_df = pd.DataFrame(vmg_rows)
    # Convert the angle columns to strings to avoid the ".0" issue in Excel.
    angle_cols = ['Beat Angle Starboard (deg)', 'Run Angle Starboard (deg)',
                  'Beat Angle Port (deg)', 'Run Angle Port (deg)']
    for col in angle_cols:
        vmg_df[col] = vmg_df[col].apply(
            lambda x: '' if pd.isna(x) else str(int(x)))
    return vmg_df

# -------------------- SMOOTHING FUNCTION -------------------- #


def smooth_polar_data(pivot_df):
    """
    Aplica un ajuste parabólico (ajuste de segundo grado) a cada columna de SOG en la tabla polar,
    de forma segmentada sobre tramos contiguos de datos válidos.
    Se preservan los valores NaN (por ángulos desautorizados) para que no se modifiquen.

    Recibe un DataFrame cuyo primer columna es el ángulo (TWA (deg)) y las restantes son los SOG (knots)
    para cada TWS.
    """
    smoothed = pivot_df.copy()
    angle_col = smoothed.columns[0]
    x = smoothed[angle_col].values
    for col in smoothed.columns[1:]:
        y = smoothed[col].values.copy()
        is_valid = ~np.isnan(y)
        if not np.any(is_valid):
            continue
        # Identificar segmentos contiguos de índices válidos
        segments = []
        current_segment = []
        for i, valid in enumerate(is_valid):
            if valid:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        if current_segment:
            segments.append(current_segment)
        # Para cada segmento con al menos 3 puntos, ajustar un polinomio de segundo grado.
        for seg in segments:
            if len(seg) >= 3:
                seg_x = x[seg]
                seg_y = y[seg]
                coeffs = np.polyfit(seg_x, seg_y, 2)
                poly = np.poly1d(coeffs)
                y_fit = poly(seg_x)
                # Actualiza solo los índices de este segmento
                y[seg] = np.round(y_fit, 2)
        smoothed[col] = y
    return smoothed

# -------------------- CHECKBOX FILTERING FUNCTIONS -------------------- #


def update_tws_checkboxes(pivot_df):
    """
    Create checkboxes for each unique TWS value found in the pivot table.
    Note: Changing a checkbox does not automatically update the plot.
    """
    for widget in checkbox_frame.winfo_children():
        widget.destroy()
    tws_vars.clear()
    all_tws = [int(col) for col in pivot_df.columns if isinstance(
        col, (int, float)) or (str(col).isdigit())]
    for tws in sorted(all_tws):
        var = tk.IntVar()
        cb = tk.Checkbutton(checkbox_frame, text=str(tws), variable=var)
        cb.pack(side=tk.LEFT, padx=2)
        tws_vars[str(tws)] = var


def get_selected_tws():
    """
    Return a sorted list of selected TWS values from the checkboxes.
    If no checkbox is selected, return all available TWS values.
    """
    selected = [int(tws) for tws, var in tws_vars.items() if var.get() == 1]
    if not selected:
        return sorted([int(tws) for tws in sog_table.columns if tws != sog_table.columns[0]])
    return sorted(selected)


def filter_table_by_tws(df):
    """
    Filter the given DataFrame based on the selected TWS values.
    For pivot tables, the first column is assumed to be the angle column.
    If no checkbox is selected, the full DataFrame is returned.
    """
    selected = get_selected_tws()
    if 'TWA (deg)' in df.columns or 'TWA_raw (deg)' in df.columns:
        angle_col = 'TWA (deg)' if 'TWA (deg)' in df.columns else 'TWA_raw (deg)'
        cols = [angle_col] + [col for col in df.columns if col !=
                              angle_col and int(col) in selected]
        return df[cols]
    else:
        return df[df['TWS (knots)'].isin(selected)]

# -------------------- HELPER FUNCTION FOR POLAR PLOTTING -------------------- #


def setup_polar_axes():
    """
    Create and return a polar axis with common settings.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True)
    ax.set_thetagrids(range(0, 360, 10))
    return ax

# -------------------- POLAR DIAGRAM FUNCTIONS -------------------- #


def generate_polar_plot(pivot_df):
    """
    Generate a polar diagram using the polar pivot table.
    - Uses 'TWA (deg)' values (rounded to the nearest multiple of 5) for plotting.
    - Sets SOG values to NaN for disallowed angles:
         below 30°, above 330°, and between 160° and 200°.
    - Only plots lines for the TWS values selected via the checkboxes.
    - Aplica un ajuste parabólico para suavizar la representación.
    - Adds interactive hover annotations that show the valores de la tabla suavizada.
    """
    filtered = filter_table_by_tws(pivot_df)
    filtered = filtered.dropna(how="all", subset=filtered.columns[1:])
    angle_col = filtered.columns[0]  # Ahora es 'TWA (deg)'
    filtered = apply_angle_filtering(filtered, angle_col)
    # Aplicar suavizado parabólico en cada columna (respetando los NaN)
    filtered = smooth_polar_data(filtered)
    theta = np.deg2rad(filtered[angle_col].astype(float))
    tws_columns = [col for col in filtered.columns if col != angle_col]
    ax = setup_polar_axes()
    lines = []
    scatter_artists = []
    for tws in tws_columns:
        values = pd.to_numeric(filtered[tws], errors='coerce')
        # Plot continuous line for visual context.
        line, = ax.plot(theta, values, label=str(
            tws), linestyle='-', linewidth=0.8, alpha=0.7)
        lines.append(line)
        # Add invisible scatter markers at the actual pivot table points.
        sc = ax.scatter(theta, values, s=50, alpha=0)
        scatter_artists.append(sc)
    # Attach hover cursor only to the scatter markers.
    cursor = mplcursors.cursor(scatter_artists, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        twa_deg = np.rad2deg(x) % 360
        sel.annotation.set_text(f"TWA: {twa_deg:.0f}°\nSOG: {y:.2f} knots")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.set_title("Polar Diagram of Max SOG (Suavizado)")
    plt.tight_layout()
    plt.show()


def save_polar_image(pivot_df):
    """
    Save the polar diagram as a PNG image using the polar pivot table.
    Applies the same angle filtering and ajuste parabólico as in generate_polar_plot.
    """
    filtered = filter_table_by_tws(pivot_df)
    filtered = filtered.dropna(how="all", subset=filtered.columns[1:])
    angle_col = filtered.columns[0]
    filtered = apply_angle_filtering(filtered, angle_col)
    filtered = smooth_polar_data(filtered)
    theta = np.deg2rad(filtered[angle_col].astype(float))
    tws_columns = [col for col in filtered.columns if col != angle_col]
    ax = setup_polar_axes()
    for tws in tws_columns:
        values = pd.to_numeric(filtered[tws], errors='coerce')
        ax.plot(theta, values, label=str(tws),
                linestyle='-', linewidth=0.8, alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.set_title("Polar Diagram of Max SOG (Suavizado)")
    plt.tight_layout()
    file_path = filedialog.asksaveasfilename(title="Save Polar Diagram", defaultextension=".png",
                                             filetypes=[("PNG Files", "*.png")])
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Polar diagram saved to: {file_path}")
    plt.close()

# -------------------- EXPORT FUNCTIONS -------------------- #


def export_dataframe(df, title):
    """
    Export the given DataFrame to a CSV file using a file dialog.
    Applies filtering based on the selected TWS values.
    """
    filtered = filter_table_by_tws(df)
    file_path = filedialog.asksaveasfilename(title=title, defaultextension=".csv",
                                             filetypes=[("CSV Files", "*.csv")])
    if file_path:
        filtered.to_csv(file_path, index=False)
        print(f"Data exported to: {file_path}")

# -------------------- RESET FUNCTIONALITY -------------------- #


def reset_data():
    """
    Reset all global data and clear the checkboxes.
    Also disables the export and plot buttons until new data is processed.
    """
    global data_folder, processed_data, sog_table, vmg_table, polar_table, all_data, tws_vars
    global export_sog_button, export_vmg_button, show_polar_button, save_polar_button, checkbox_frame
    data_folder = None
    processed_data = None
    sog_table = None
    vmg_table = None
    polar_table = None
    all_data = None
    tws_vars.clear()
    # Clear the checkbox frame.
    for widget in checkbox_frame.winfo_children():
        widget.destroy()
    # Disable the buttons.
    export_sog_button.config(state='disabled')
    export_vmg_button.config(state='disabled')
    show_polar_button.config(state='disabled')
    save_polar_button.config(state='disabled')
    print("Data reset.")

# -------------------- MAIN GUI FUNCTIONALITY -------------------- #


def process_data_and_generate_tables():
    """
    Process CSV files from the selected folder and generate:
      - The processed data (in memory)
      - The SOG pivot table (using 'TWA (deg)', rounded to the nearest multiple of 5, with disallowed angles filtered out)
      - The extended VMG table (using 'TWA_raw (deg)' for angles)
      - The polar pivot table for plotting (using 'TWA (deg)', rounded to the nearest multiple of 5)
    Also updates the TWS checkboxes for filtering.
    """
    global processed_data, sog_table, vmg_table, all_data, polar_table
    processed_data = process_csv_files(data_folder)
    if processed_data.empty:
        print("No processed data available.")
        return
    all_data = processed_data.copy()
    sog_table = generate_sog_table(processed_data)
    vmg_table = generate_vmg_table(processed_data)
    polar_table = generate_polar_table(processed_data)
    update_tws_checkboxes(sog_table)
    print("SOG, VMG, and polar tables generated.")


def main():
    global export_sog_button, export_vmg_button, show_polar_button, save_polar_button, checkbox_frame

    root = tk.Tk()
    root.title("Sailing Data Processor")

    def select_folder_and_process():
        folder = filedialog.askdirectory(
            title="Select Folder Containing CSV Files")
        if folder:
            global data_folder
            data_folder = folder
            process_data_and_generate_tables()
            # Enable the buttons after processing data.
            export_sog_button.config(state='normal')
            export_vmg_button.config(state='normal')
            show_polar_button.config(state='normal')
            save_polar_button.config(state='normal')

    select_button = tk.Button(
        root, text="Select Folder and Process Data", command=select_folder_and_process)
    reset_button = tk.Button(root, text="Reset", command=reset_data)
    export_sog_button = tk.Button(root, text="Export SOG Table",
                                  command=lambda: export_dataframe(
                                      sog_table, "Save SOG Table"),
                                  state='disabled')
    export_vmg_button = tk.Button(root, text="Export VMG Table",
                                  command=lambda: export_dataframe(
                                      vmg_table, "Save VMG Table"),
                                  state='disabled')
    show_polar_button = tk.Button(root, text="Show Polar Diagram",
                                  command=lambda: generate_polar_plot(
                                      polar_table),
                                  state='disabled')
    save_polar_button = tk.Button(root, text="Save Polar Diagram",
                                  command=lambda: save_polar_image(
                                      polar_table),
                                  state='disabled')

    checkbox_frame = tk.LabelFrame(root, text="Filter by TWS (knots)")
    checkbox_frame.pack(pady=10)

    select_button.pack(pady=5)
    reset_button.pack(pady=5)
    export_sog_button.pack(pady=5)
    export_vmg_button.pack(pady=5)
    show_polar_button.pack(pady=5)
    save_polar_button.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
