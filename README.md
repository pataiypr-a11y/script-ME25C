# script-ME25C
sample1
!pip install python-docx

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from docx import Document
from google.colab import drive

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/modelsofstability/solution.xlsx'
sheet_name = '25C'

# Define the variations (The Grid)
x_values = [0.25, 0.20, 0.15, 0.10]
y_values = [0.80, 0.85, 0.90, 0.95]

try:
    # Reading Rows 3-14 (MA+ for 25C)
    # skiprows=2 skips rows 1 & 2. nrows=12 reads rows 3 through 14.
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="B:AG", skiprows=2, nrows=12, header=None)
    print(f"✅ Data Loaded from Rows 3-14 (Sheet 25C). Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ==========================================
# 2. MODEL DEFINITION (ME)
# ==========================================

FIXED_A = 100.0

def me_model(t, c1, c2):
    """
    Multi-Exponential Model: y = 100 * exp(c1*t + c2*t^2)
    """
    return FIXED_A * np.exp(c1 * t + c2 * t**2)

def residuals(params, t_data, y_data):
    c1, c2 = params
    y_model = me_model(t_data, c1, c2)
    return y_model - y_data

# ==========================================
# 3. LOOPING & PLOTTING
# ==========================================

fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.suptitle(r'MA$^{+}$ fitted for 25$^{\circ}$C', fontsize=22)

col_idx = 0
results_summary = []

for i, x_val in enumerate(x_values):
    for j, y_val in enumerate(y_values):
        ax = axes[i, j]

        try:
            # 1. Extract & Clean Data
            t_raw = df.iloc[:, col_idx]
            y_raw = df.iloc[:, col_idx+1]
            pair_df = pd.DataFrame({'t': t_raw, 'y': y_raw}).dropna()

            t_data = pair_df['t'].values.astype(float)
            y_data = pair_df['y'].values.astype(float)

            if len(t_data) < 3:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
                col_idx += 2
                continue

            # 2. Optimization
            # Initial guess (slower decay for 25C)
            initial_guess = [-0.05, 0.0]
            res = least_squares(residuals, initial_guess, args=(t_data, y_data), loss='soft_l1')
            c1, c2 = res.x

            # 3. Calculate Metrics
            y_model_at_data = me_model(t_data, c1, c2)
            sse = np.sum((y_data - y_model_at_data) ** 2)
            sst = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (sse / sst) if sst > 0 else 0

            results_summary.append({
                'x': x_val, 'y': y_val,
                'c1': c1, 'c2': c2,
                'A': FIXED_A,
                'R2': r_squared, 'SSE': sse
            })

            # 4. Generate Smooth Fit
            t_smooth = np.linspace(t_data[0], t_data[-1], 50)
            y_smooth = me_model(t_smooth, c1, c2)

            # 5. Plotting
            ax.scatter(t_data, y_data, color='red', s=40, label='Exp')
            ax.plot(t_smooth, y_smooth, color='green', label='ME')

            ax.set_title(f'x={x_val}, y={y_val}', fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)

            # Text Box
            text_str = (f'$c_1$: {c1:.3f}\n'
                        f'$c_2$: {c2:.3f}\n'
                        f'$R^2$: {r_squared:.4f}\n'
                        f'SSE: {sse:.4f}')

            ax.text(0.96, 0.72, text_str, transform=ax.transAxes, fontsize=9,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round'))

            if i == 3: ax.set_xlabel('t (year)')
            if j == 0: ax.set_ylabel(r'MA$^{+}$ (%)')

            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, "Error", ha='center')
            print(f"Error at pair {col_idx}: {e}")

        col_idx += 2

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("grid_4x4_ME_fit_MA_25C.png")
plt.show()

# ==========================================
# 4. EXPORT TO WORD DOCUMENT (MA+ 25°C)
# ==========================================

# Define the folder and file path
folder_path = '/content/drive/MyDrive/modelsofstability/thesistables'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

doc_path = os.path.join(folder_path, 'kinetics_parameters_MA_25C.docx')

# Load existing document or create a new one
if os.path.exists(doc_path):
    doc = Document(doc_path)
else:
    doc = Document()

doc.add_heading('PARAMETER & METRICS SUMMARY TABLE (MA+ 25°C)', level=2)

if results_summary:
    cols = ['x', 'y', 'c1', 'c2', 'A', 'R2', 'SSE']
    df_results = pd.DataFrame(results_summary)[cols]

    # Create the table in Word
    table = doc.add_table(rows=df_results.shape[0] + 1, cols=df_results.shape[1])
    table.style = 'Table Grid'

    # Add column headers
    for j, col_name in enumerate(df_results.columns):
        table.cell(0, j).text = col_name

    # Add data rows
    for i in range(df_results.shape[0]):
        for j in range(df_results.shape[1]):
            val = df_results.iat[i, j]
            # Format floating point numbers to 4 decimal places for neatness
            if isinstance(val, float):
                table.cell(i + 1, j).text = str(round(val, 4))
            else:
                table.cell(i + 1, j).text = str(val)

    # Save the document
    doc.save(doc_path)
    print(f"✅ MA+ Table successfully saved to: {doc_path}")
else:
    print("❌ No data available to save.")
!pip install python-docx

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from docx import Document
from google.colab import drive

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/modelsofstability/solution.xlsx'
sheet_name = '25C'

# Define the variations (The Grid)
x_values = [0.25, 0.20, 0.15, 0.10]
y_values = [0.80, 0.85, 0.90, 0.95]

try:
    # UPDATED: Reading Rows 18-29 (FA+ for 25C)
    # skiprows=17 skips rows 1-17. nrows=12 reads rows 18 through 29.
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="B:AG", skiprows=17, nrows=12, header=None)
    print(f"✅ Data Loaded from Rows 18-29 (Sheet 25C). Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ==========================================
# 2. MODEL DEFINITION (ME)
# ==========================================

FIXED_A = 100.0

def me_model(t, c1, c2):
    """
    Multi-Exponential Model: y = 100 * exp(c1*t + c2*t^2)
    """
    return FIXED_A * np.exp(c1 * t + c2 * t**2)

def residuals(params, t_data, y_data):
    c1, c2 = params
    y_model = me_model(t_data, c1, c2)
    return y_model - y_data

# ==========================================
# 3. LOOPING & PLOTTING
# ==========================================

fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.suptitle(r'FA$^{+}$ fitted for 25$^{\circ}$C', fontsize=22)

col_idx = 0
results_summary = []

for i, x_val in enumerate(x_values):
    for j, y_val in enumerate(y_values):
        ax = axes[i, j]

        try:
            # 1. Extract & Clean Data
            t_raw = df.iloc[:, col_idx]
            y_raw = df.iloc[:, col_idx+1]
            pair_df = pd.DataFrame({'t': t_raw, 'y': y_raw}).dropna()

            t_data = pair_df['t'].values.astype(float)
            y_data = pair_df['y'].values.astype(float)

            if len(t_data) < 3:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
                col_idx += 2
                continue

            # 2. Optimization
            # Initial guess (slower decay for 25C)
            initial_guess = [-0.05, 0.0]
            res = least_squares(residuals, initial_guess, args=(t_data, y_data), loss='soft_l1')
            c1, c2 = res.x

            # 3. Calculate Metrics
            y_model_at_data = me_model(t_data, c1, c2)
            sse = np.sum((y_data - y_model_at_data) ** 2)
            sst = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (sse / sst) if sst > 0 else 0

            results_summary.append({
                'x': x_val, 'y': y_val,
                'c1': c1, 'c2': c2,
                'A': FIXED_A,
                'R2': r_squared, 'SSE': sse
            })

            # 4. Generate Smooth Fit
            t_smooth = np.linspace(t_data[0], t_data[-1], 50)
            y_smooth = me_model(t_smooth, c1, c2)

            # 5. Plotting
            ax.scatter(t_data, y_data, color='red', s=40, label='Exp')
            ax.plot(t_smooth, y_smooth, color='green', label='ME')

            ax.set_title(f'x={x_val}, y={y_val}', fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)

            # Text Box
            text_str = (f'$c_1$: {c1:.3f}\n'
                        f'$c_2$: {c2:.3f}\n'
                        f'$R^2$: {r_squared:.4f}\n'
                        f'SSE: {sse:.4f}')

            ax.text(0.96, 0.72, text_str, transform=ax.transAxes, fontsize=9,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round'))

            if i == 3: ax.set_xlabel('t (year)')

            # --- UPDATED Y-AXIS LABEL ---
            if j == 0: ax.set_ylabel(r'FA$^{+}$ (%)')
            # ----------------------------

            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, "Error", ha='center')
            print(f"Error at pair {col_idx}: {e}")

        col_idx += 2

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("grid_4x4_ME_fit_FA_25C.png")
plt.show()

# ==========================================
# 4. EXPORT TO WORD DOCUMENT (FA+ 25°C)
# ==========================================

# Define the folder and file path
folder_path = '/content/drive/MyDrive/modelsofstability/thesistables'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

doc_path = os.path.join(folder_path, 'kinetics_parameters_FA_25C.docx')

# Load existing document or create a new one
if os.path.exists(doc_path):
    doc = Document(doc_path)
else:
    doc = Document()

doc.add_heading('PARAMETER & METRICS SUMMARY TABLE (FA+ 25°C)', level=2)

if results_summary:
    cols = ['x', 'y', 'c1', 'c2', 'A', 'R2', 'SSE']
    df_results = pd.DataFrame(results_summary)[cols]

    # Create the table in Word
    table = doc.add_table(rows=df_results.shape[0] + 1, cols=df_results.shape[1])
    table.style = 'Table Grid'

    # Add column headers
    for j, col_name in enumerate(df_results.columns):
        table.cell(0, j).text = col_name

    # Add data rows
    for i in range(df_results.shape[0]):
        for j in range(df_results.shape[1]):
            val = df_results.iat[i, j]
            # Format floating point numbers to 4 decimal places for neatness
            if isinstance(val, float):
                table.cell(i + 1, j).text = str(round(val, 4))
            else:
                table.cell(i + 1, j).text = str(val)

    # Save the document
    doc.save(doc_path)
    print(f"✅ FA+ Table successfully saved to: {doc_path}")
else:
    print("❌ No data available to save.")

!pip install python-docx

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from docx import Document
from google.colab import drive

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/modelsofstability/byproduct.xlsx'
sheet_name = '25C'

x_values = [0.25, 0.20, 0.15, 0.10]
y_values = [0.80, 0.85, 0.90, 0.95]

try:
    # Reading Rows 3-14 (MFA+ replaces MA+ location for 25C)
    # skiprows=2 skips rows 1 & 2. nrows=12 reads rows 3-14.
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="B:AG", skiprows=2, nrows=12, header=None)
    print(f"✅ Data Loaded from Rows 3-14 (Sheet 25C - MFA+). Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ==========================================
# 2. MODEL DEFINITION
# ==========================================

def byproduct_model(t, c1, c2, y_max):
    """
    Model: y = y_max * (1 - exp(c1*t + c2*t^2))
    """
    exponent = c1 * t + c2 * t**2
    exponent = np.clip(exponent, -700, 700)
    return y_max * (1 - np.exp(exponent))

def residuals(params, t_data, y_data, y_max):
    c1, c2 = params
    y_model = byproduct_model(t_data, c1, c2, y_max)
    return y_model - y_data

# ==========================================
# 3. LOOPING & PLOTTING
# ==========================================

fig, axes = plt.subplots(4, 4, figsize=(24, 20))
# Title: Font size 22
fig.suptitle(r'MFA$^{+}$ fitted for 25$^{\circ}$C', fontsize=22)

col_idx = 0
results_summary = []

for i, x_val in enumerate(x_values):
    for j, y_val in enumerate(y_values):

        ax = axes[i, j]

        try:
            # 1. Extract & Clean Data
            t_raw = df.iloc[:, col_idx]
            y_raw = df.iloc[:, col_idx+1]
            pair_df = pd.DataFrame({'t': t_raw, 'y': y_raw}).dropna()

            t_data = pair_df['t'].values.astype(float)
            y_data = pair_df['y'].values.astype(float)

            if len(t_data) < 3:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
                col_idx += 2
                continue

            # --- DEFINE Y_MAX ---
            local_y_max = np.max(y_data)

            # 2. Optimization
            # Initial guess: slower kinetics at 25C
            initial_guess = [-0.01, 0.0]

            res = least_squares(residuals, initial_guess, args=(t_data, y_data, local_y_max), loss='soft_l1')
            c1, c2 = res.x

            # 3. Calculate Metrics
            y_model_at_data = byproduct_model(t_data, c1, c2, local_y_max)
            sse = np.sum((y_data - y_model_at_data) ** 2)
            sst = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (sse / sst) if sst > 0 else 0

            results_summary.append({
                'x': x_val, 'y': y_val,
                'c1': c1, 'c2': c2,
                'y_max': local_y_max,
                'R2': r_squared, 'SSE': sse
            })

            # 4. Generate Smooth Fit
            t_smooth = np.linspace(0, t_data.max() * 1.1, 50)
            y_smooth = byproduct_model(t_smooth, c1, c2, local_y_max)

            # 5. Plotting
            ax.scatter(t_data, y_data, color='purple', s=40, label='MFA Exp')
            ax.plot(t_smooth, y_smooth, color='orange', label='MFA Fit')

            ax.set_title(f'x={x_val}, y={y_val}', fontsize=10, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)

            # Text Box - Shifted Up (y=0.15)
            text_str = (f'$c_1$: {c1:.3f}\n'
                        f'$c_2$: {c2:.3f}\n'
                        f'$y_{{max}}$: {local_y_max:.2f}\n'
                        f'$R^2$: {r_squared:.4f}\n'
                        f'SSE: {sse:.4f}')

            ax.text(0.96, 0.15, text_str, transform=ax.transAxes, fontsize=9,
                    ha='right', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round'))

            if i == 3: ax.set_xlabel('t (year)')
            if j == 0: ax.set_ylabel(r'MFA$^{+}$ (%)')

            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, "Error", ha='center')
            print(f"Error at pair {col_idx}: {e}")

        col_idx += 2

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("MFA_Fit_25C.png")
plt.show()

# ==========================================
# 4. EXPORT TO WORD DOCUMENT (MFA+ 25°C)
# ==========================================

# Define the folder and file path
folder_path = '/content/drive/MyDrive/modelsofstability/thesistables'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

doc_path = os.path.join(folder_path, 'kinetics_parameters_MFA_25C.docx')

# Load existing document or create a new one
if os.path.exists(doc_path):
    doc = Document(doc_path)
else:
    doc = Document()

doc.add_heading('PARAMETER & METRICS SUMMARY TABLE (MFA+ 25°C)', level=2)

if results_summary:
    cols = ['x', 'y', 'c1', 'c2', 'y_max', 'R2', 'SSE']
    df_results = pd.DataFrame(results_summary)[cols]

    # Create the table in Word
    table = doc.add_table(rows=df_results.shape[0] + 1, cols=df_results.shape[1])
    table.style = 'Table Grid'

    # Add column headers
    for j, col_name in enumerate(df_results.columns):
        table.cell(0, j).text = col_name

    # Add data rows
    for i in range(df_results.shape[0]):
        for j in range(df_results.shape[1]):
            val = df_results.iat[i, j]
            # Format floating point numbers to 4 decimal places for neatness
            if isinstance(val, float):
                table.cell(i + 1, j).text = str(round(val, 4))
            else:
                table.cell(i + 1, j).text = str(val)

    # Save the document
    doc.save(doc_path)
    print(f"✅ MFA+ Table successfully saved to: {doc_path}")
else:
    print("❌ No data available to save.")

!pip install python-docx

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from docx import Document
from google.colab import drive

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/modelsofstability/byproduct.xlsx'
sheet_name = '25C'

x_values = [0.25, 0.20, 0.15, 0.10]
y_values = [0.80, 0.85, 0.90, 0.95]

try:
    # Reading Rows 18-29 (DMFA+ replaces FA+ location for 25C)
    # skiprows=17 skips rows 1-17. nrows=12 reads rows 18-29.
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="B:AG", skiprows=17, nrows=12, header=None)
    print(f"✅ Data Loaded from Rows 18-29 (Sheet 25C - DMFA+). Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ==========================================
# 2. MODEL DEFINITION
# ==========================================

def byproduct_model(t, c1, c2, y_max):
    """
    Model: y = y_max * (1 - exp(c1*t + c2*t^2))
    """
    exponent = c1 * t + c2 * t**2
    exponent = np.clip(exponent, -700, 700)
    return y_max * (1 - np.exp(exponent))

def residuals(params, t_data, y_data, y_max):
    c1, c2 = params
    y_model = byproduct_model(t_data, c1, c2, y_max)
    return y_model - y_data

# ==========================================
# 3. LOOPING & PLOTTING
# ==========================================

fig, axes = plt.subplots(4, 4, figsize=(24, 20))
# Title: Font size 22
fig.suptitle(r'DMFA$^{+}$ fitted for 25$^{\circ}$C', fontsize=22)

col_idx = 0
results_summary = []

for i, x_val in enumerate(x_values):
    for j, y_val in enumerate(y_values):

        ax = axes[i, j]

        try:
            # 1. Extract & Clean Data
            t_raw = df.iloc[:, col_idx]
            y_raw = df.iloc[:, col_idx+1]
            pair_df = pd.DataFrame({'t': t_raw, 'y': y_raw}).dropna()

            t_data = pair_df['t'].values.astype(float)
            y_data = pair_df['y'].values.astype(float)

            if len(t_data) < 3:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
                col_idx += 2
                continue

            # --- DEFINE Y_MAX ---
            local_y_max = np.max(y_data)

            # 2. Optimization
            # Initial guess: slower kinetics at 25C
            initial_guess = [-0.01, 0.0]

            res = least_squares(residuals, initial_guess, args=(t_data, y_data, local_y_max), loss='soft_l1')
            c1, c2 = res.x

            # 3. Calculate Metrics
            y_model_at_data = byproduct_model(t_data, c1, c2, local_y_max)
            sse = np.sum((y_data - y_model_at_data) ** 2)
            sst = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (sse / sst) if sst > 0 else 0

            results_summary.append({
                'x': x_val, 'y': y_val,
                'c1': c1, 'c2': c2,
                'y_max': local_y_max,
                'R2': r_squared, 'SSE': sse
            })

            # 4. Generate Smooth Fit
            t_smooth = np.linspace(0, t_data.max() * 1.1, 50)
            y_smooth = byproduct_model(t_smooth, c1, c2, local_y_max)

            # 5. Plotting
            ax.scatter(t_data, y_data, color='purple', s=40, label='DMFA Exp')
            ax.plot(t_smooth, y_smooth, color='orange', label='DMFA Fit')

            ax.set_title(f'x={x_val}, y={y_val}', fontsize=10, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)

            # Text Box - Shifted Up (y=0.15)
            text_str = (f'$c_1$: {c1:.3f}\n'
                        f'$c_2$: {c2:.3f}\n'
                        f'$y_{{max}}$: {local_y_max:.2f}\n'
                        f'$R^2$: {r_squared:.4f}\n'
                        f'SSE: {sse:.4f}')

            ax.text(0.96, 0.15, text_str, transform=ax.transAxes, fontsize=9,
                    ha='right', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round'))

            if i == 3: ax.set_xlabel('t (year)')
            if j == 0: ax.set_ylabel(r'DMFA$^{+}$ (%)')

            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, "Error", ha='center')
            print(f"Error at pair {col_idx}: {e}")

        col_idx += 2

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("DMFA_Fit_25C.png")
plt.show()

# ==========================================
# 4. EXPORT TO WORD DOCUMENT (DMFA+ 25°C)
# ==========================================

# Define the folder and file path
folder_path = '/content/drive/MyDrive/modelsofstability/thesistables'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

doc_path = os.path.join(folder_path, 'kinetics_parameters_DMFA_25C.docx')

# Load existing document or create a new one
if os.path.exists(doc_path):
    doc = Document(doc_path)
else:
    doc = Document()

doc.add_heading('PARAMETER & METRICS SUMMARY TABLE (DMFA+ 25°C)', level=2)

if results_summary:
    cols = ['x', 'y', 'c1', 'c2', 'y_max', 'R2', 'SSE']
    df_results = pd.DataFrame(results_summary)[cols]

    # Create the table in Word
    table = doc.add_table(rows=df_results.shape[0] + 1, cols=df_results.shape[1])
    table.style = 'Table Grid'

    # Add column headers
    for j, col_name in enumerate(df_results.columns):
        table.cell(0, j).text = col_name

    # Add data rows
    for i in range(df_results.shape[0]):
        for j in range(df_results.shape[1]):
            val = df_results.iat[i, j]
            # Format floating point numbers to 4 decimal places for neatness
            if isinstance(val, float):
                table.cell(i + 1, j).text = str(round(val, 4))
            else:
                table.cell(i + 1, j).text = str(val)

    # Save the document
    doc.save(doc_path)
    print(f"✅ DMFA+ Table successfully saved to: {doc_path}")
else:
    print("❌ No data available to save.")

