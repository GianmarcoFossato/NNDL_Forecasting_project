import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import glob
    import os
    from pathlib import Path
    import re
    import subprocess
    import zipfile
    import pandas as pd
    import torch
    import marimo as mo

    return Path, glob, mo, os, pd, re, subprocess, torch, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # HyPT: Improving TimesNet for Long-Term Electricity Forecasting
    This notebook clones the "NNDL_Forecasting_project" repository, downloads the Electricity (ECL) dataset, trains models and fine-tunes HyPT, and allows you to display and download results.

    ---
    ### Marimo Execution & Sharing Guidelines
    Marimo automatically re-evaluates dependent cells when UI state changes. Use the **Runtime reactivity** panel below to enable your desired behaviour.
    _Recommended setting: Deactivate "On startup", activate "On cell change", and set "On module change" to lazy._
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPU
    Configure the GPU in the **Configure Compute** panel above.
    """)
    return


@app.cell
def _(torch):
    # Sanity check for GPU
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Running on CPU.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clone Repository
    """)
    return


@app.cell
def _(os, subprocess):
    NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(NOTEBOOK_DIR)

    # Repository of the project
    REPO_NAME = "NNDL_Forecasting_project"
    REPO_OWNER = "GianmarcoFossato"
    REPO_BRANCH = "model3-fix-due-to-ablation"

    # Prevent nested cloning if the cell is re-run: 
    if os.path.basename(os.getcwd()) == REPO_NAME:
        os.chdir("..")

    # Public URL
    repo_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

    if os.path.exists(REPO_NAME):
        print(f"Repository '{REPO_NAME}' already exists. Resetting local changes and pulling...")
        subprocess.run(["git", "-C", REPO_NAME, "fetch", "origin"])
        subprocess.run(["git", "-C", REPO_NAME, "reset", "--hard", f"origin/{REPO_BRANCH}"])
    else:
        print(f"Cloning repository: {REPO_NAME}...")
        subprocess.run(["git", "clone", "-b", REPO_BRANCH, repo_url])
        print("Repository cloned successfully.")

    os.chdir(REPO_NAME)
    return (REPO_NAME,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install required packages
    """)
    return


@app.cell
def _(subprocess):
    print("Installing requirements...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fetch Benchmark Data
    Downloads the official Electricity (ECL) compressed dataset records into the expected local path framework.
    """)
    return


@app.cell
def _(os, subprocess):
    os.makedirs("dataset/electricity", exist_ok=True)

    print("Downloading electricity.zip from Google Drive...")
    subprocess.run(
        [
            "gdown",
            "1FHH0S3d6IK_UOpg6taBRavx4MragRLo1",
            "-O",
            "dataset/electricity/electricity.zip",
        ]
    )

    print("Unzipping dataset...")
    subprocess.run(
        [
            "unzip",
            "-q",
            "dataset/electricity/electricity.zip",
            "-d",
            "dataset/",
        ]
    )

    if os.path.exists("dataset/electricity/electricity.zip"):
        os.remove("dataset/electricity/electricity.zip")
    if os.path.exists("dataset/electricity/.DS_Store"):
        os.remove("dataset/electricity/.DS_Store")

    print("Verifying directory structure:")
    subprocess.run(["ls", "-lh", "dataset/electricity/"])
    return


@app.cell(hide_code=True)
def _(mo):
    execution_mode = mo.ui.radio(
        options=["Hyperparameter Tuning", "Model Evaluation", "Ablation", "Idle"],
        value="Idle",
        label="**Select Execution Workflow:**"
    )

    model_selector = mo.ui.multiselect(
        options=["DLinear", "TimesNet", "TimeXer", "TimeMixer", "iTransformer", "HyPT"],
        value=["HyPT"],
        label="**Select Models to Run (in order):**"
    )
    return execution_mode, model_selector


@app.cell(hide_code=True)
def _(execution_mode, mo, model_selector):
    control_panel = mo.vstack([
        mo.md("## Workflow & Model Execution Control Panel"),
        execution_mode,
        mo.md("---"),
        model_selector if execution_mode.value == "Model Evaluation" else mo.md("_Switch to **Model Evaluation** mode to select active individual models._")
    ])

    control_panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fine Tuning a Model
    The following cell blocks allow fine-tuning of a specific model.
    """)
    return


@app.cell
def _(REPO_NAME, execution_mode, mo, subprocess):
    import optuna
    import optuna.visualization as vis

    mo.stop(
        execution_mode.value != "Hyperparameter Tuning",
        mo.md("*Hyperparameter Tuning is currently disabled by the Control Panel selection.*")
    )

    print("Starting hyperparameter tuning...")
    subprocess.run(["chmod", "+x", REPO_NAME + "/scripts/long_term_forecast/Tuning/HyPT_hp.sh"])
    subprocess.run(["bash", REPO_NAME + "/scripts/long_term_forecast/Tuning/HyPT_hp.sh"])
    print("Tuning completed!")
    return optuna, vis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Execute Training Experiments
    Executes scripts assigned to run evaluation processes for implemented models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DLinear
    """)
    return


@app.cell
def _(execution_mode, mo, model_selector, subprocess):
    mo.stop(
        execution_mode.value != "Model Evaluation" or "DLinear" not in model_selector.value,
        mo.md("*DLinear execution skipped.*")
    )

    subprocess.run(["chmod", "+x", "scripts/long_term_forecast/ECL_script/DLinear.sh"])
    subprocess.run(["bash", "scripts/long_term_forecast/ECL_script/DLinear.sh"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TimesNet
    """)
    return


@app.cell
def _(execution_mode, mo, model_selector, subprocess):
    mo.stop(
        execution_mode.value != "Model Evaluation" or "TimesNet" not in model_selector.value,
        mo.md("*TimesNet execution skipped.*")
    )

    subprocess.run(["chmod", "+x", "scripts/long_term_forecast/ECL_script/TimesNet.sh"])
    subprocess.run(["bash", "scripts/long_term_forecast/ECL_script/TimesNet.sh"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TimeXer
    """)
    return


@app.cell
def _(execution_mode, mo, model_selector, subprocess):
    mo.stop(
        execution_mode.value != "Model Evaluation" or "TimeXer" not in model_selector.value,
        mo.md("*TimeXer execution skipped.*")
    )

    subprocess.run(["chmod", "+x", "scripts/long_term_forecast/ECL_script/TimeXer.sh"])
    subprocess.run(["bash", "scripts/long_term_forecast/ECL_script/TimeXer.sh"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TimeMixer
    """)
    return


@app.cell
def _(execution_mode, mo, model_selector, subprocess):
    mo.stop(
        execution_mode.value != "Model Evaluation" or "TimeMixer" not in model_selector.value,
        mo.md("*TimeMixer execution skipped.*")
    )

    subprocess.run(["chmod", "+x", "scripts/long_term_forecast/ECL_script/TimeMixer.sh"])
    subprocess.run(["bash", "scripts/long_term_forecast/ECL_script/TimeMixer.sh"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### iTransformer
    """)
    return


@app.cell
def _(execution_mode, mo, model_selector, subprocess):
    mo.stop(
        execution_mode.value != "Model Evaluation" or "iTransformer" not in model_selector.value,
        mo.md("*iTransformer execution skipped.*")
    )

    subprocess.run(["chmod", "+x", "scripts/long_term_forecast/ECL_script/iTransformer.sh"])
    subprocess.run(["bash", "scripts/long_term_forecast/ECL_script/iTransformer.sh"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### HyPT
    """)
    return


@app.cell
def _(execution_mode, mo, model_selector, subprocess):
    mo.stop(
        execution_mode.value != "Model Evaluation" or "HyPT" not in model_selector.value,
        mo.md("*HyPT execution skipped.*")
    )

    subprocess.run(["chmod", "+x", "scripts/long_term_forecast/ECL_script/HyPT.sh"])
    subprocess.run(["bash", "scripts/long_term_forecast/ECL_script/HyPT.sh"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run HyPT Ablation Study

    This pipeline runs the ablation scripts, comparing individual and combined architectural components of HyPT across seeds 2021, 2022, and 2023.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seed 2021
    """)
    return


@app.cell
def _(execution_mode, mo, subprocess):
    mo.stop(
        execution_mode.value != "Ablation",
        mo.md("*Ablation execution skipped.*")
    )

    script_path_one = "scripts/long_term_forecast/Ablation/HyPT_2021.sh"

    print(f"Starting HyPT Ablation Study using '{script_path_one}'...")
    subprocess.run(["chmod", "+x", script_path_one])
    subprocess.run(["bash", script_path_one])
    print("Ablation study completed!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seed 2022
    """)
    return


@app.cell
def _(execution_mode, mo, subprocess):
    mo.stop(
        execution_mode.value != "Ablation",
        mo.md("*Ablation execution skipped.*")
    )

    script_path_two = "scripts/long_term_forecast/Ablation/HyPT_2022.sh"

    print(f"Starting HyPT Ablation Study using '{script_path_two}'...")
    subprocess.run(["chmod", "+x", script_path_two])
    subprocess.run(["bash", script_path_two])
    print("Ablation study completed!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seed 2023
    """)
    return


@app.cell
def _(execution_mode, mo, subprocess):
    mo.stop(
        execution_mode.value != "Ablation",
        mo.md("*Ablation execution skipped.*")
    )

    script_path_three = "scripts/long_term_forecast/Ablation/HyPT_2023.sh"

    print(f"Starting HyPT Ablation Study using '{script_path_three}'...")
    subprocess.run(["chmod", "+x", script_path_three])
    subprocess.run(["bash", script_path_three])
    print("Ablation study completed!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    results_source = mo.ui.radio(
        options={
            "Paper Results": "paper_test_results",
            "Local Test Results": "test_results"
        },
        value="Paper Results",
        label="**Select Data Source to Display:**"
    )

    results_source
    return (results_source,)


@app.cell(hide_code=True)
def _(Path, execution_mode, mo, pd, re, results_source):
    mo.stop(
        execution_mode.value == "Idle",
        mo.md("*Table display skipped while in Idle mode.*")
    )

    mode_subfolder_map = {
        "Model Evaluation": "evaluation",
        "Ablation": "ablation",
        "Hyperparameter Tuning": "tuning",
    }
    subfolder = mode_subfolder_map.get(execution_mode.value, "")
    base_dir = Path(results_source.value) / subfolder

    # Shared CSS for standard Marimo table look + red/blue highlights + rotated Electricity label
    COMMON_STYLE = """
    <style>
        .marimo-style-table {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            border-collapse: collapse;
            width: auto;
            margin: 16px 0;
            font-size: 14px;
            color: inherit;
        }
        .marimo-style-table th, .marimo-style-table td {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 8px 12px;
            text-align: center;
        }
        .marimo-style-table th {
            background-color: rgba(128, 128, 128, 0.08);
            font-weight: 600;
        }
        .marimo-style-table .seq-col {
            font-weight: 600;
            background-color: rgba(128, 128, 128, 0.03);
        }
        .marimo-style-table td.dataset-label {
            position: relative;
            font-weight: 600;
            font-size: 13px;
            width: 28px;
            padding: 0;
            background-color: rgba(128, 128, 128, 0.05);
        }
        .marimo-style-table td.dataset-label .label-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) rotate(180deg);
            writing-mode: vertical-lr;
            white-space: nowrap;
        }
        .marimo-style-table .avg-row td {
            border-top: 2px solid rgba(128, 128, 128, 0.4);
            font-weight: bold;
            background-color: rgba(128, 128, 128, 0.05);
        }
        .marimo-style-table .grand-avg-row td {
            border-top: 2px solid rgba(128, 128, 128, 0.6);
            border-bottom: 2px solid rgba(128, 128, 128, 0.6);
            font-weight: bold;
            background-color: rgba(128, 128, 128, 0.08);
        }
        .best-val {
            color: #dc2626;
            font-weight: bold;
        }
        .second-val {
            color: #2563eb;
            text-decoration: underline;
        }
    </style>
    """

    # -------------------------------------------------------------
    # 1. EVALUATION TABLE FORMATTER
    # -------------------------------------------------------------
    def format_evaluation_results(b_dir: Path) -> str:
        log_files = list(b_dir.glob("**/output.log"))
        if not log_files:
            return f"<p>No evaluation log files found in <code>{b_dir}</code>.</p>"

        folder_pattern = re.compile(r"long_term_forecast_ECL_96_(\d+)_([A-Za-z0-9]+)_")
        metric_pattern = re.compile(r"mse:\s*([-\d.eE+]+),\s*mae:\s*([-\d.eE+]+)")

        records = []
        for log_path in sorted(log_files):
            folder_name = log_path.parent.name
            match = folder_pattern.search(folder_name)
            if not match:
                continue

            seq_len = int(match.group(1))
            model = match.group(2)
            model_display = "HyPT\n(Mine)" if model == "HyPT" else model

            try:
                content = log_path.read_text(encoding="utf-8")
                metrics = metric_pattern.findall(content)
                if metrics:
                    mse_val, mae_val = metrics[-1]
                    records.append({
                        "Seq": seq_len,
                        "Model": model_display,
                        "MSE": float(mse_val),
                        "MAE": float(mae_val)
                    })
            except Exception:
                continue

        if not records:
            return f"<p>No valid evaluation metrics found in <code>{b_dir}</code> logs.</p>"

        df = pd.DataFrame(records)
        df = df.groupby(["Seq", "Model"], as_index=False).mean()

        pivoted = df.pivot(
            index="Seq", columns="Model", values=["MSE", "MAE"]
        ).swaplevel(axis=1)

        all_models = list(df["Model"].unique())
        if "HyPT\n(Mine)" in all_models:
            all_models.remove("HyPT\n(Mine)")
            all_models = ["HyPT\n(Mine)"] + sorted(all_models)

        col_order = pd.MultiIndex.from_product(
            [all_models, ["MSE", "MAE"]], names=["Model", "Metric"]
        )
        pivoted = pivoted.reindex(columns=col_order)

        avg_series = pivoted.mean(axis=0)
        pivoted.loc["Avg"] = avg_series

        styled_pivoted = pivoted.copy()
        for idx, row in pivoted.iterrows():
            for metric in ["MSE", "MAE"]:
                metric_vals = [row[(m, metric)] for m in all_models if (m, metric) in row and not pd.isna(row[(m, metric)])]
                if not metric_vals:
                    continue
                sorted_vals = sorted(metric_vals)
                best_val = sorted_vals[0]
                second_val = sorted_vals[1] if len(sorted_vals) > 1 else best_val

                for m in all_models:
                    val = row.get((m, metric))
                    if val is None or pd.isna(val):
                        styled = "-"
                    else:
                        formatted_val = f"{val:.3f}"
                        if abs(val - best_val) < 1e-6:
                            styled = f'<span class="best-val">{formatted_val}</span>'
                        elif abs(val - second_val) < 1e-6:
                            styled = f'<span class="second-val">{formatted_val}</span>'
                        else:
                            styled = formatted_val
                    styled_pivoted.loc[idx, (m, f"{metric}_styled")] = styled

        html = COMMON_STYLE + """
        <table class="marimo-style-table">
            <thead>
                <tr>
                    <th colspan="2">Models</th>
        """

        for idx, m in enumerate(all_models):
            m_label = m.replace("\n", " ")
            html += f'<th colspan="2">{m_label}</th>'
        html += "</tr><tr>"

        html += '<th colspan="2">Metric</th>'
        for idx, m in enumerate(all_models):
            html += "<th>MSE</th><th>MAE</th>"
        html += "</tr></thead><tbody>"

        seqs = [s for s in [96, 192, 336, 720] if s in pivoted.index] + ["Avg"]
        rowspan_cnt = len([s for s in [96, 192, 336, 720] if s in pivoted.index])

        for idx, seq in enumerate(seqs):
            if seq == "Avg":
                html += '<tr class="avg-row"><td></td>'
            else:
                html += "<tr>"
                if idx == 0 and rowspan_cnt > 0:
                    html += f'<td rowspan="{rowspan_cnt}" class="dataset-label"><span class="label-text">Electricity</span></td>'

            html += f'<td class="seq-col">{seq}</td>'

            for m_idx, m in enumerate(all_models):
                mse_s = styled_pivoted.loc[seq, (m, "MSE_styled")] if (m, "MSE_styled") in styled_pivoted.columns else "-"
                mae_s = styled_pivoted.loc[seq, (m, "MAE_styled")] if (m, "MAE_styled") in styled_pivoted.columns else "-"
                html += f"<td>{mse_s}</td><td>{mae_s}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        return html

    # -------------------------------------------------------------
    # 2. ABLATION TABLE FORMATTER (Per Seed + Seed Average)
    # -------------------------------------------------------------
    def format_ablation_results(b_dir: Path) -> str:
        log_files = list(b_dir.glob("**/output.log"))
        if not log_files:
            return f"<p>No ablation log files found in <code>{b_dir}</code>.</p>"

        folder_pattern = re.compile(r"long_term_forecast_ECL_96_(\d+)_([a-zA-Z0-9_]+)_s(\d+)_")
        metric_pattern = re.compile(r"mse:\s*([-\d.eE+]+),\s*mae:\s*([-\d.eE+]+)")

        records = []
        for log_path in sorted(log_files):
            folder_name = log_path.parent.name
            match = folder_pattern.search(folder_name)
            if not match:
                continue
            pred_len = int(match.group(1))
            variant = match.group(2)
            seed = int(match.group(3))

            try:
                content = log_path.read_text(encoding="utf-8")
                m_match = metric_pattern.findall(content)
                if m_match:
                    records.append({
                        "pred_len": pred_len,
                        "seed": seed,
                        "variant": variant,
                        "mse": float(m_match[-1][0]),
                        "mae": float(m_match[-1][1])
                    })
            except Exception:
                continue

        if not records:
            return f"<p>No valid ablation metrics found in <code>{b_dir}</code> logs.</p>"

        df_rec = pd.DataFrame(records)

        variant_order = ["none", "branch_a", "branch_b", "both", "both_nodrop"]
        found_variants = [v for v in variant_order if v in df_rec["variant"].unique()]
        for v in df_rec["variant"].unique():
            if v not in found_variants:
                found_variants.append(v)

        variant_names = {
            "none": "None",
            "branch_a": "Branch A",
            "branch_b": "Branch B",
            "both": "Both",
            "both_nodrop": "Both (no drop)"
        }

        horizons = sorted(df_rec["pred_len"].unique())
        seeds = sorted(df_rec["seed"].unique())

        data = {}
        for _, row in df_rec.iterrows():
            data[(row["pred_len"], f"Seed {row['seed']}", row["variant"])] = (row["mse"], row["mae"])

        horizon_avgs = {}
        for h in horizons:
            for v in found_variants:
                mses = [data[(h, f"Seed {s}", v)][0] for s in seeds if (h, f"Seed {s}", v) in data]
                maes = [data[(h, f"Seed {s}", v)][1] for s in seeds if (h, f"Seed {s}", v) in data]
                if mses and maes:
                    horizon_avgs[(h, "Avg", v)] = (sum(mses) / len(mses), sum(maes) / len(maes))

        grand_avg = {}
        for v in found_variants:
            mses = [horizon_avgs[(h, "Avg", v)][0] for h in horizons if (h, "Avg", v) in horizon_avgs]
            maes = [horizon_avgs[(h, "Avg", v)][1] for h in horizons if (h, "Avg", v) in horizon_avgs]
            if mses and maes:
                grand_avg[v] = (sum(mses) / len(mses), sum(maes) / len(maes))

        def format_row_cells(row_vals):
            res = {}
            for metric_idx in [0, 1]:
                valid_vals = {
                    v: row_vals[v][metric_idx]
                    for v in found_variants
                    if v in row_vals and row_vals[v] is not None
                }
                if not valid_vals:
                    for v in found_variants:
                        if v not in res:
                            res[v] = ["-", "-"]
                        else:
                            res[v].append("-")
                    continue

                sorted_v = sorted(valid_vals.values())
                best = sorted_v[0]
                second = sorted_v[1] if len(sorted_v) > 1 else best

                for v in found_variants:
                    val = valid_vals.get(v)
                    if val is None:
                        formatted = "-"
                    else:
                        f_val = f"{val:.3f}"
                        if abs(val - best) < 1e-6:
                            formatted = f'<span class="best-val">{f_val}</span>'
                        elif abs(val - second) < 1e-6:
                            formatted = f'<span class="second-val">{f_val}</span>'
                        else:
                            formatted = f_val

                    if v not in res:
                        res[v] = [formatted]
                    else:
                        res[v].append(formatted)
            return res

        html = COMMON_STYLE + """
        <table class="marimo-style-table">
            <thead>
                <tr>
                    <th colspan="2">Ablation Variant</th>
        """

        for idx, v in enumerate(found_variants):
            v_label = variant_names.get(v, v)
            html += f'<th colspan="2">{v_label}</th>'
        html += "</tr><tr>"

        html += '<th class="seq-col">Horizon</th><th class="seq-col">Seed</th>'
        for idx, v in enumerate(found_variants):
            html += "<th>MSE</th><th>MAE</th>"
        html += "</tr></thead><tbody>"

        for h in horizons:
            num_seed_rows = len(seeds) + 1
            for s_idx, s in enumerate(seeds):
                html += "<tr>"
                if s_idx == 0:
                    html += f'<td rowspan="{num_seed_rows}" class="seq-col" style="vertical-align: middle;">{h}</td>'
                html += f'<td class="seq-col">Seed {s}</td>'

                row_vals = {v: data.get((h, f"Seed {s}", v)) for v in found_variants}
                styled_cells = format_row_cells(row_vals)

                for v_idx, v in enumerate(found_variants):
                    mse_str, mae_str = styled_cells[v]
                    html += f"<td>{mse_str}</td><td>{mae_str}</td>"
                html += "</tr>"

            html += '<tr class="avg-row"><td class="seq-col">Avg</td>'
            avg_vals = {v: horizon_avgs.get((h, "Avg", v)) for v in found_variants}
            styled_avg_cells = format_row_cells(avg_vals)
            for v_idx, v in enumerate(found_variants):
                mse_str, mae_str = styled_avg_cells[v]
                html += f"<td>{mse_str}</td><td>{mae_str}</td>"
            html += "</tr>"

        html += '<tr class="grand-avg-row"><td colspan="2" class="seq-col">Grand Avg</td>'
        grand_vals = {v: grand_avg.get(v) for v in found_variants}
        styled_grand_cells = format_row_cells(grand_vals)
        for v_idx, v in enumerate(found_variants):
            mse_str, mae_str = styled_grand_cells[v]
            html += f"<td>{mse_str}</td><td>{mae_str}</td>"
        html += "</tr>"

        html += "</tbody></table>"
        return html

    # -------------------------------------------------------------
    # 3. HYPERPARAMETER TUNING FORMATTER
    # -------------------------------------------------------------
    def format_tuning_results(b_dir: Path) -> str:
        log_files = list(b_dir.glob("**/output.log")) + list(b_dir.glob("output.log"))
        if not log_files and not (b_dir.exists() and list(b_dir.iterdir())):
            return f"<p>No tuning directory found at <code>{b_dir}</code>.</p>"

        combined_log_text = ""
        for lp in set(log_files):
            try:
                combined_log_text += "\n" + lp.read_text(encoding="utf-8")
            except Exception:
                continue

        metric_pattern = re.compile(r"mse:\s*([-\d.eE+]+),\s*mae:\s*([-\d.eE+]+)")
        horizons = [96, 192, 336, 720]
        results_by_horizon = {}

        subdirs = [d for d in b_dir.iterdir() if d.is_dir()] if b_dir.exists() else []

        for h in horizons:
            matching_dirs = [
                d for d in subdirs
                if f"_pl{h}_" in d.name or f"_{h}_" in d.name or d.name.endswith(f"_{h}")
            ]

            config_str = "Optimal Config"
            mse_val = "-"
            mae_val = "-"

            if matching_dirs:
                folder = matching_dirs[0]
                dm = re.search(r"_dm(\d+)_", folder.name)
                nh = re.search(r"_nh(\d+)_", folder.name)
                el = re.search(r"_el(\d+)_", folder.name)
                df = re.search(r"_df(\d+)_", folder.name)

                config_parts = []
                if dm: config_parts.append(f"d_model={dm.group(1)}")
                if nh: config_parts.append(f"n_heads={nh.group(1)}")
                if el: config_parts.append(f"e_layers={el.group(1)}")
                if df: config_parts.append(f"d_ff={df.group(1)}")
                if config_parts:
                    config_str = ", ".join(config_parts)

                folder_log = folder / "output.log"
                if folder_log.exists():
                    try:
                        f_content = folder_log.read_text(encoding="utf-8")
                        m = metric_pattern.findall(f_content)
                        if m:
                            mse_val = f"{float(m[-1][0]):.3f}"
                            mae_val = f"{float(m[-1][1]):.3f}"
                    except Exception:
                        pass

            if mse_val == "-" and combined_log_text:
                h_matches = re.findall(
                    rf"(?:pl{h}_|ECL_96_{h}_).*?mse:\s*([-\d.eE+]+),\s*mae:\s*([-\d.eE+]+)",
                    combined_log_text,
                    flags=re.DOTALL
                )
                if h_matches:
                    mse_val = f"{float(h_matches[-1][0]):.3f}"
                    mae_val = f"{float(h_matches[-1][1]):.3f}"

            results_by_horizon[h] = {
                "config": config_str,
                "mse": mse_val,
                "mae": mae_val
            }

        html = COMMON_STYLE + """
        <h3>Optimal HyPT Model Results</h3>
        <table class="marimo-style-table">
            <thead>
                <tr>
                    <th>Horizon</th>
                    <th>Optimal Config</th>
                    <th>MSE</th>
                    <th>MAE</th>
                </tr>
            </thead>
            <tbody>
        """

        for h in horizons:
            info = results_by_horizon[h]
            html += f"""
                <tr>
                    <td class="seq-col">{h}</td>
                    <td><code>{info['config']}</code></td>
                    <td>{info['mse']}</td>
                    <td>{info['mae']}</td>
                </tr>
            """

        html += "</tbody></table>"
        return html

    # Marimo Execution Dispatch
    if not base_dir.exists():
        table_html = f"<p>Directory <code>{base_dir}</code> does not exist.</p>"
    elif execution_mode.value == "Ablation":
        table_html = format_ablation_results(base_dir)
    elif execution_mode.value == "Hyperparameter Tuning":
        table_html = format_tuning_results(base_dir)
    else:
        table_html = format_evaluation_results(base_dir)

    mo.Html(table_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optuna Dashboard & Hyperparameter Visualization
    """)
    return


@app.cell
def _(REPO_NAME, execution_mode, glob, mo, optuna):
    mo.stop(
        execution_mode.value != "Hyperparameter Tuning",
        mo.md("*Optuna dashboard is disabled outside Hyperparameter Tuning mode.*")
    )

    def scan_optuna_studies(db_paths):
        names = []
        s_map = {}
        for db_path in db_paths:
            target_url = f"sqlite:///{db_path}"
            try:
                summaries = optuna.get_all_study_summaries(storage=target_url)
                for summary in summaries:
                    names.append(summary.study_name)
                    s_map[summary.study_name] = target_url
            except Exception:
                continue
        return names, s_map

    # Primary search
    primary_paths = sorted(list(set(
        glob.glob("test_results/tuning/hp_search_*/optuna_study.db") +
        glob.glob("test_results/tuning/**/optuna_study.db", recursive=True)
    )))
    study_names, storage_map = scan_optuna_studies(primary_paths)

    # Fallback search
    if not study_names:
        fallback_patterns = [
            "paper_test_results/tuning/hp_search_*/optuna_study.db",
            "paper_test_results/tuning/**/optuna_study.db",
            f"{REPO_NAME}/paper_test_results/tuning/hp_search_*/optuna_study.db",
            f"{REPO_NAME}/paper_test_results/tuning/**/optuna_study.db",
        ]
        fallback_paths = []
        for pattern in fallback_patterns:
            fallback_paths.extend(glob.glob(pattern, recursive=True))

        fallback_paths = sorted(list(set(fallback_paths)))
        study_names, storage_map = scan_optuna_studies(fallback_paths)

    if study_names:
        study_dropdown = mo.ui.dropdown(options=study_names, value=study_names[0], label="Select Study:")
        selector_ui = mo.vstack([mo.md("### Optuna Study Inspector"), study_dropdown])
    else:
        study_dropdown = None
        selector_ui = mo.md("No Optuna studies found yet. Waiting for tuning process to generate database...")

    selector_ui
    return storage_map, study_dropdown


@app.cell
def _(execution_mode, mo, optuna, storage_map, study_dropdown, vis):
    mo.stop(execution_mode.value != "Hyperparameter Tuning")

    if study_dropdown is not None and study_dropdown.value:
        try:
            active_storage = storage_map[study_dropdown.value]
            study = optuna.load_study(study_name=study_dropdown.value, storage=active_storage)

            fig_history = vis.plot_optimization_history(study)
            fig_importances = vis.plot_param_importances(study)
            fig_parallel = vis.plot_parallel_coordinate(study)

            dashboard = mo.vstack([
                mo.md(f"**Best Value (MSE):** `{study.best_value}`"),
                mo.md(f"**Best Params:** `{study.best_params}`"),
                mo.hstack([fig_history, fig_importances]),
                fig_parallel
            ])
        except Exception as e:
            dashboard = mo.md(f"Waiting for trials to complete... (Error: {e})")
    else:
        dashboard = mo.md("")

    dashboard
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save Results
    Creates a downloadable `.zip` archive of the `test_results` folder without heavy model weights and array files.
    """)
    return


@app.cell
def _(mo, os, zipfile):
    _source_dir = "test_results"
    _output_zip = "test_results_archive.zip"

    if not os.path.exists(_source_dir):
        download_button = mo.md(f"**{_source_dir}** folder not found. Please run the training script first.")
    else:
        with zipfile.ZipFile(_output_zip, 'w', zipfile.ZIP_DEFLATED) as _zipf:
            for _root, _dirs, _files in os.walk(_source_dir):
                if "results" in _dirs:
                    _dirs.remove("results")

                for _file in _files:
                    if _file.endswith('.npy') or _file.endswith('.pth'):
                        continue

                    _full_path = os.path.join(_root, _file)
                    _arc_name = os.path.relpath(_full_path, _source_dir)
                    _zipf.write(_full_path, _arc_name)

        with open(_output_zip, "rb") as _f:
            _zip_bytes = _f.read()

        download_button = mo.download(
            data=_zip_bytes, 
            filename="test_results.zip", 
            label="Download Test Results"
        )

    download_button
    return


if __name__ == "__main__":
    app.run()
