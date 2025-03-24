import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib
import mplcursors

# Utilitza el backend TkAgg per a la compatibilitat amb la GUI.
matplotlib.use("TkAgg")

# Factor de conversió de m/s a nusos
MS_TO_KNOTS = 1.94384
# Processa només les files on Device1_TrueWindSpeed (m/s) és superior a MIN_TWS_MS (4 nusos)
MIN_TWS_MS = 2.05778

# Variables globals
data_folder = None
processed_data = None  # DataFrame combinat i processat dels fitxers CSV
# Taula dinàmica de SOG
sog_table = None
# Taula VMG
vmg_table = None
# Taula dinàmica per a la representació polar
polar_table = None
all_data = None        # Còpia de seguretat de les dades processades
tws_vars = {}          # Diccionari per emmagatzemar les variables de les caselles de TWS

# Variables dels botons de la GUI a nivell global
export_sog_button = None
export_vmg_button = None
show_polar_button = None
save_polar_button = None
checkbox_frame = None

# -------------------- FUNCIONS DE PROCESSAMENT DE DADES -------------------- #


def process_csv_files(folder):
    """
    Processa tots els fitxers CSV de la carpeta seleccionada:
      - Llegeix només les columnes necessàries.
      - Filtra les files on Device1_TrueWindSpeed (m/s) és inferior a MIN_TWS_MS.
      - Converteix SpeedOverGround de m/s a nusos (amb 2 decimals).
      - Crea dues columnes d'angle del vent:
            * 'TWA_raw (deg)' = Device1_TrueWindAngle (deg) arrodonit a l'unitat 
              (utilitzat per a la taula VMG).
            * 'TWA (deg)' = Device1_TrueWindAngle (deg) arrodonit al múltiple de 5 més proper 
              (utilitzat per a la taula SOG i per al diagrama polar).
      - Converteix Device1_TrueWindSpeed (m/s) a TWS (nusos) (arrodonit a l'unitat).
      - Converteix Device1_VelocityMadeGood (m/s) a VMG (nusos) amb 2 decimals.

    Retorna un DataFrame amb:
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
            print(f"Error al llegir {filepath}: {e}")
            continue
        # Filtra les files on la velocitat real del vent és inferior a 4 nusos.
        df = df[df['Device1_TrueWindSpeed (m/s)'] > MIN_TWS_MS]
        dataframes.append(df)
    if not dataframes:
        print("No s'han trobat fitxers CSV o cap dada ha passat el filtre de TWS.")
        return pd.DataFrame()
    combined = pd.concat(dataframes, ignore_index=True)
    combined['SOG (knots)'] = (
        combined['SpeedOverGround (m/s)'] * MS_TO_KNOTS).round(2)
    # Per a la taula VMG: utilitza el TWA arrodonit a l'unitat.
    combined['TWA_raw (deg)'] = combined['Device1_TrueWindAngle (deg)'].round(
        0).astype(int)
    # Per a la taula SOG i el diagrama polar: utilitza el TWA arrodonit al múltiple de 5.
    combined['TWA (deg)'] = (
        (combined['Device1_TrueWindAngle (deg)'] / 5).round() * 5).astype(int)
    combined['TWS (knots)'] = (
        combined['Device1_TrueWindSpeed (m/s)'] * MS_TO_KNOTS).round().astype(int)
    combined['VMG (knots)'] = (
        combined['Device1_VelocityMadeGood (m/s)'] * MS_TO_KNOTS).round(2)
    processed = combined[[
        'SOG (knots)', 'TWA_raw (deg)', 'TWA (deg)', 'TWS (knots)', 'VMG (knots)']]
    return processed

# -------------------- FUNCIÓ AUXILIAR PER AL FILTRAT D'ANGLES -------------------- #


def apply_angle_filtering(df, angle_col):
    """
    Aplica un filtre al DataFrame basat en valors d'angle de vent no permesos.
    Les files amb angles per sota de 30°, per sobre de 330° o entre 160° i 200°
    tindran tots els valors de SOG establerts a NaN.

    Retorna el DataFrame modificat.
    """
    mask = (df[angle_col] < 30) | (df[angle_col] > 330) | (
        (df[angle_col] >= 160) & (df[angle_col] <= 200))
    for col in df.columns[1:]:
        df.loc[mask, col] = np.nan
    return df

# -------------------- FUNCIONS DE GENERACIÓ DE TAULES -------------------- #


def generate_sog_table(df):
    """
    Genera una taula dinàmica SOG a partir del DataFrame processat:
      - Files: 'TWA (deg)' (múltiple de 5º)
      - Columnes: TWS (knots) (només valors a partir de 4 nusos).
      - Valors: Percentil 95 de SOG (knots) per a cada combinació (arrodonit a 2 decimals).
      - Filtra les files amb TWA no permesos (per sota de 30°, per sobre de 330° o entre 160° i 200°).

    Retorna el DataFrame de la taula dinàmica.
    """
    df_filtered = df[df['TWS (knots)'] >= 4]
    grouped = df_filtered.groupby(['TWA (deg)', 'TWS (knots)'])['SOG (knots)'].agg(
        lambda x: np.percentile(x, 95)).reset_index()
    grouped['SOG (knots)'] = grouped['SOG (knots)'].round(2)
    grouped = grouped.replace(0, np.nan)
    allowed_twa = sorted({t for t in df_filtered['TWA (deg)'].unique()
                          if t >= 30 and t <= 330 and not (t > 160 and t < 200)})
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
    Genera una taula dinàmica per al diagrama polar a partir del DataFrame processat:
      - Files: 'TWA (deg)' (arrodonit al múltiple de 5 més proper).
      - Columnes: TWS (knots) (només valors a partir de 4 nusos).
      - Valors: Percentil 95 de SOG (knots) per a cada combinació (arrodonit a 2 decimals).

    Retorna el DataFrame de la taula dinàmica.
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
    D'un subconjunt de dades per a un TWS donat, filtra les files on TWA (deg) està entre lower_bound i upper_bound.
    Retorna una dupla (best_vmg, angle) per a la fila amb el VMG absolut més alt. Si no hi ha cap fila que coincideixi, retorna (np.nan, np.nan).
    Nota: L'angle retornat es pren de la columna 'TWA_raw (deg)'.
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
    Genera una taula VMG a partir del DataFrame processat.
    Per a cada TWS únic (>= 4 nusos), calcula quatre segments:
      - Beat d'estribor: TWA de 0° a 90°.
      - Run d'estribor: TWA de 90° a 180°.
      - Beat de babor: TWA de 270° a 360°.
      - Run de babor: TWA de 180° a 270°.
    Per a cada segment, selecciona la fila amb el VMG absolut més alt i retorna el seu VMG (com a valor positiu)
    i el TWA_raw (deg) corresponent.

    Retorna un DataFrame amb les columnes:
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
        # Amura d'estribor:
        beat_vmg_starboard, beat_angle_starboard = get_best_vmg(
            sub, 0, 90, True, True, absolute=False)
        run_vmg_starboard, run_angle_starboard = get_best_vmg(
            sub, 90, 180, False, True, absolute=True)
        run_vmg_starboard = abs(run_vmg_starboard)
        # Amura de babor:
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
    # Converteix les columnes d'angle a strings per evitar el problema del ".0" en Excel.
    angle_cols = ['Beat Angle Starboard (deg)', 'Run Angle Starboard (deg)',
                  'Beat Angle Port (deg)', 'Run Angle Port (deg)']
    for col in angle_cols:
        vmg_df[col] = vmg_df[col].apply(
            lambda x: '' if pd.isna(x) else str(int(x)))
    return vmg_df

# -------------------- FUNCIONS DE SUAVITZAT (AJUST PARABÒLIC) -------------------- #


def smooth_polar_data(pivot_df):
    """
    Aplica un ajust parabòlic a cada columna de SOG de la taula polar,
    de forma segmentada sobre trams contigus de dades vàlides.
    Es mantenen els valors NaN (per als angles no permesos) sense modificar.

    Crea un DataFrame on la primera columna és l'angle (TWA (deg)) i les altres són els SOG (knots)
    per a cada TWS.
    """
    smoothed = pivot_df.copy()
    angle_col = smoothed.columns[0]
    x = smoothed[angle_col].values
    for col in smoothed.columns[1:]:
        y = smoothed[col].values.copy()
        is_valid = ~np.isnan(y)
        if not np.any(is_valid):
            continue
        # Identifica trams contigus d'índexs vàlids
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
        # Per a cada tram amb almenys 3 punts, s'ajusta un polinomi de segon grau.
        for seg in segments:
            if len(seg) >= 3:
                seg_x = x[seg]
                seg_y = y[seg]
                coeffs = np.polyfit(seg_x, seg_y, 2)
                poly = np.poly1d(coeffs)
                y_fit = poly(seg_x)
                # Actualitza només els índexs d'aquest tram
                y[seg] = np.round(y_fit, 2)
        smoothed[col] = y
    return smoothed

# -------------------- FUNCIONS DE FILTRAT AMB CASELLES DE VERIFICACIÓ -------------------- #


def update_tws_checkboxes(pivot_df):
    """
    Crea caselles per a cada valor de TWS trobat a la taula dinàmica.
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
    Retorna una llista ordenada dels valors TWS seleccionats a les caselles.
    Si no es selecciona cap, retorna tots els TWS disponibles.
    """
    selected = [int(tws) for tws, var in tws_vars.items() if var.get() == 1]
    if not selected:
        return sorted([int(tws) for tws in sog_table.columns if tws != sog_table.columns[0]])
    return sorted(selected)


def filter_table_by_tws(df):
    """
    Filtra el DataFrame donat en funció dels valors TWS seleccionats.
    Si no es selecciona cap casella, es retorna el DataFrame complet.
    """
    selected = get_selected_tws()
    if 'TWA (deg)' in df.columns or 'TWA_raw (deg)' in df.columns:
        angle_col = 'TWA (deg)' if 'TWA (deg)' in df.columns else 'TWA_raw (deg)'
        cols = [angle_col] + [col for col in df.columns if col !=
                              angle_col and int(col) in selected]
        return df[cols]
    else:
        return df[df['TWS (knots)'].isin(selected)]

# -------------------- FUNCIONS AUXILIARS PER A LA REPRESENTACIÓ POLAR -------------------- #


def setup_polar_axes():
    """
    Crea i retorna un eix polar amb les configuracions comunes.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True)
    ax.set_thetagrids(range(0, 360, 10))
    return ax

# -------------------- FUNCIONS DEL DIAGRAMA POLAR -------------------- #


def generate_polar_plot(pivot_df):
    """
    Genera un diagrama polar utilitzant la taula dinàmica polar.
    - Utilitza els valors de 'TWA (deg)' (arrodonits al múltiple de 5) per a la representació.
    - Estableix els valors de SOG a NaN per als angles no permesos (per sota de 30°, per sobre de 330° o entre 160° i 200°).
    - Només representa línies per als TWS seleccionats amb les caselles.
    - Aplica un ajust parabòlic per suavitzar la representació.
    - Afegeix anotacions interactives al passar el cursor que mostren els valors de la taula suavitzada.
    """
    filtered = filter_table_by_tws(pivot_df)
    filtered = filtered.dropna(how="all", subset=filtered.columns[1:])
    angle_col = filtered.columns[0]
    filtered = apply_angle_filtering(filtered, angle_col)
    # Aplica el suavitzat parabòlic a cada columna (respectant els NaN)
    filtered = smooth_polar_data(filtered)
    theta = np.deg2rad(filtered[angle_col].astype(float))
    tws_columns = [col for col in filtered.columns if col != angle_col]
    ax = setup_polar_axes()
    lines = []
    scatter_artists = []
    for tws in tws_columns:
        values = pd.to_numeric(filtered[tws], errors='coerce')
        # Representa una línia contínua per al context visual.
        line, = ax.plot(theta, values, label=str(
            tws), linestyle='-', linewidth=0.8, alpha=0.7)
        lines.append(line)
        # Afegeix marcadors dispersos invisibles als punts de la taula suavitzada.
        sc = ax.scatter(theta, values, s=50, alpha=0)
        scatter_artists.append(sc)
    cursor = mplcursors.cursor(scatter_artists, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        twa_deg = np.rad2deg(x) % 360
        sel.annotation.set_text(f"TWA: {twa_deg:.0f}°\nSOG: {y:.2f} knots")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.set_title("Diagrama Polar")
    plt.tight_layout()
    plt.show()


def save_polar_image(pivot_df):
    """
    Desa el diagrama polar com a imatge PNG utilitzant la taula dinàmica polar.
    Aplica el mateix filtrat d'angles i ajust parabòlic que en generate_polar_plot.
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
    ax.set_title("Diagrama Polar")
    plt.tight_layout()
    file_path = filedialog.asksaveasfilename(title="Desa Diagrama Polar", defaultextension=".png",
                                             filetypes=[("Fitxers PNG", "*.png")])
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Diagrama polar desat a: {file_path}")
    plt.close()

# -------------------- FUNCIONS D'EXPORTACIÓ -------------------- #


def export_dataframe(df, title):
    """
    Exporta el DataFrame a un fitxer CSV utilitzant un diàleg de fitxers.
    Aplica el filtrat basat en els TWS seleccionats.
    """
    filtered = filter_table_by_tws(df)
    file_path = filedialog.asksaveasfilename(title=title, defaultextension=".csv",
                                             filetypes=[("Fitxers CSV", "*.csv")])
    if file_path:
        filtered.to_csv(file_path, index=False)
        print(f"Dades exportades a: {file_path}")

# -------------------- FUNCIONALITAT DE REINICI -------------------- #


def reset_data():
    """
    Reinicia totes les dades globals i esborra les caselles de TWS.
    També desactiva els botons d'exportació i de gràfica fins que es processin noves dades.
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
    for widget in checkbox_frame.winfo_children():
        widget.destroy()
    export_sog_button.config(state='disabled')
    export_vmg_button.config(state='disabled')
    show_polar_button.config(state='disabled')
    save_polar_button.config(state='disabled')
    print("Dades reiniciades.")

# -------------------- FUNCIONALITAT PRINCIPAL DE LA GUI -------------------- #


def process_data_and_generate_tables():
    """
    Processa els fitxers CSV de la carpeta seleccionada i genera:
      - Les dades processades (en memòria)
      - La taula dinàmica SOG (utilitzant 'TWA (deg)' amb els angles no permesos filtrats)
      - La taula VMG (utilitzant 'TWA_raw (deg)' per als angles)
      - La taula dinàmica polar per a la representació (utilitzant 'TWA (deg)')
    També actualitza les caselles de TWS per al filtrat.
    """
    global processed_data, sog_table, vmg_table, all_data, polar_table
    processed_data = process_csv_files(data_folder)
    if processed_data.empty:
        print("No hi ha dades processades disponibles.")
        return
    all_data = processed_data.copy()
    sog_table = generate_sog_table(processed_data)
    vmg_table = generate_vmg_table(processed_data)
    polar_table = generate_polar_table(processed_data)
    update_tws_checkboxes(sog_table)
    print("S'han generat les taules SOG, VMG i el diagrama polar.")


def main():
    global export_sog_button, export_vmg_button, show_polar_button, save_polar_button, checkbox_frame

    root = tk.Tk()
    root.title("Generador de Polars")

    def select_folder_and_process():
        folder = filedialog.askdirectory(
            title="Selecciona la carpeta amb fitxers .csv")
        if folder:
            global data_folder
            data_folder = folder
            process_data_and_generate_tables()
            export_sog_button.config(state='normal')
            export_vmg_button.config(state='normal')
            show_polar_button.config(state='normal')
            save_polar_button.config(state='normal')

    select_button = tk.Button(
        root, text="Selecciona la carpeta", command=select_folder_and_process)
    reset_button = tk.Button(root, text="Reinicia", command=reset_data)
    export_sog_button = tk.Button(root, text="Exporta Taula SOG",
                                  command=lambda: export_dataframe(
                                      sog_table, "Desa Taula SOG"),
                                  state='disabled')
    export_vmg_button = tk.Button(root, text="Exporta Taula VMG",
                                  command=lambda: export_dataframe(
                                      vmg_table, "Desa Taula VMG"),
                                  state='disabled')
    show_polar_button = tk.Button(root, text="Diagrama Polar",
                                  command=lambda: generate_polar_plot(
                                      polar_table),
                                  state='disabled')
    save_polar_button = tk.Button(root, text="Desa Diagrama Polar",
                                  command=lambda: save_polar_image(
                                      polar_table),
                                  state='disabled')

    checkbox_frame = tk.LabelFrame(root, text="Filtra per TWS (kt)")
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
