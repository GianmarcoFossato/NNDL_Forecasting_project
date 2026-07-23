import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import re
    import pandas as pd
    import torch
    import os
    import subprocess
    from dotenv import load_dotenv
    import glob
    import zipfile
    from pathlib import Path

    return Path, glob, mo, os, pd, re, subprocess, torch, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # HyPT: Improving TimesNet for Long-Term Electricity Forecasting
    This notebook clones the "NNDL_Forecasting_project" repository, downloads the Electricity (ECL) dataset, trains some models and fine tuning HyPT, and allows you to download some of the results.

    ---
    ### Marimo Execution & Sharing Guidelines
    Marimo automatically re-evaluates dependent cells when UI state changes. Use the **Runtime reactivity** panel below to enable your desired behaviour.
    _I recommend deactivating "On startup", activating "On cell change" and set "On module change" to lazy._
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

    # Check if the repository directory already exists in the current working directory
    if os.path.exists(REPO_NAME):
        print(f"Repository '{REPO_NAME}' already exists. Resetting local changes and pulling...")
        # Fetch the latest changes from remote without merging yet
        subprocess.run(["git", "-C", REPO_NAME, "fetch", "origin"])
        # Force reset the local branch to match the remote branch perfectly
        subprocess.run(["git", "-C", REPO_NAME, "reset", "--hard", f"origin/{REPO_BRANCH}"])
    else:
        print(
            f"Cloning repository: {REPO_NAME}..."
        )
        subprocess.run(["git", "clone","-b", REPO_BRANCH, repo_url])
        print("Repository cloned succesfully.")

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
    subprocess.run(["pip", "install", "-r","requirements.txt"])
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
        mo.md("##Workflow & Model Execution Control Panel"),
        execution_mode,
        mo.md("---"),
        model_selector if execution_mode.value == "Model Evaluation" else mo.md("_Switch to **Model Evaluation** mode to select active individual models._")
    ])

    control_panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Fine Tuning a model

    The following cell blocks allow to fine tune a specific model.
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

    # Grant permissions and run the tuning script
    print("Starting hyperparameter tuning...")
    subprocess.run(["chmod", "+x", REPO_NAME+"/scripts/long_term_forecast/Tuning/HyPT_hp.sh"])
    subprocess.run(["bash", REPO_NAME+"/scripts/long_term_forecast/Tuning/HyPT_hp.sh"])
    print("Tuning completed!")
    return optuna, vis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the SQLite database generated by the script.
    """)
    return


@app.cell
def _(REPO_NAME, execution_mode, glob, mo, optuna):
    mo.stop(
        execution_mode.value != "Hyperparameter Tuning",
        mo.md("*Optuna dashboard is disabled in Model Evaluation mode.*")
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
                # Shield notebook execution flow from half-written or locked DB instances
                continue
        return names, s_map

    # Primary Search: test_results directory
    primary_paths = sorted(list(set(
        glob.glob("test_results/hp_search_*/optuna_study.db") +
        glob.glob("test_results/**/optuna_study.db", recursive=True)
    )))
    study_names, storage_map = scan_optuna_studies(primary_paths)

    # Fallback Search: paper_test_results/Tuning directory
    if not study_names:
        fallback_patterns = [
            "paper_test_results/Tuning/hp_search_*/optuna_study.db",
            "paper_test_results/Tuning/**/optuna_study.db",
            f"{REPO_NAME}/paper_test_results/Tuning/hp_search_*/optuna_study.db",
            f"{REPO_NAME}/paper_test_results/Tuning/**/optuna_study.db",
        ]
        fallback_paths = []
        for pattern in fallback_patterns:
            fallback_paths.extend(glob.glob(pattern, recursive=True))

        fallback_paths = sorted(list(set(fallback_paths)))
        study_names, storage_map = scan_optuna_studies(fallback_paths)

    # Render UI selector based on findings
    if study_names:
        study_dropdown = mo.ui.dropdown(options=study_names, value=study_names[0], label="Select Study:")
        selector_ui = mo.vstack([mo.md("### Optuna Dashboard"), study_dropdown])
    else:
        study_dropdown = None
        selector_ui = mo.md("No Optuna studies found yet. Wait for the tuning script to start generating trials.")

    selector_ui

    '''# Dynamically scan the restructured directory tree to support arbitrary model IDs
    db_paths = glob.glob("test_results/hp_search_*/optuna_study.db")

    study_names = []
    storage_map = {}

    # Extract study instances across discoverable targets to populate the workspace registry
    for db_path in db_paths:
        target_url = f"sqlite:///{db_path}"
        try:
            summaries = optuna.get_all_study_summaries(storage=target_url)
            for summary in summaries:
                study_names.append(summary.study_name)
                storage_map[summary.study_name] = target_url
        except Exception:
            # Shield notebook execution flow from half-written or locked DB instances
            continue

    if study_names:
        study_dropdown = mo.ui.dropdown(options=study_names, value=study_names[0], label="Select Study:")
        selector_ui = mo.vstack([mo.md("### Optuna Dashboard"), study_dropdown])
    else:
        study_dropdown = None
        selector_ui = mo.md("No Optuna studies found yet. Wait for the tuning script to start generating trials.")

    selector_ui'''
    return storage_map, study_dropdown


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Render Optuna's visualization module
    """)
    return


@app.cell
def _(execution_mode, mo, optuna, storage_map, study_dropdown, vis):
    mo.stop(execution_mode.value != "Hyperparameter Tuning")

    if study_dropdown is not None and study_dropdown.value:
        try:
            # Resolve the accurate contextual storage path based on selection state mapping
            active_storage = storage_map[study_dropdown.value]
            study = optuna.load_study(study_name=study_dropdown.value, storage=active_storage)

            # Generate Plotly figures natively
            fig_history = vis.plot_optimization_history(study)
            fig_importances = vis.plot_param_importances(study)
            fig_parallel = vis.plot_parallel_coordinate(study)

            # Layout the dashboard
            dashboard = mo.vstack([
                mo.md(f"**Best Value (MSE):** `{study.best_value}`"),
                mo.md(f"**Best Params:** `{study.best_params}`"),
                mo.hstack([fig_history, fig_importances]),
                fig_parallel
            ])
        except Exception as e:
            dashboard = mo.md(f"Waiting for more trials to complete to generate charts... (Error: {e})")
    else:
        dashboard = mo.md("")

    dashboard
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Execute Training Experiments
    Executes execution scripts assigned to run evaluation processes for some implemented models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###DLinear
    """)
    return


@app.cell
def _(execution_mode, mo, model_selector, subprocess):
    mo.stop(
        execution_mode.value != "Model Evaluation" or "DLinear" not in model_selector.value,
        mo.md("*DLinear execution skipped.*")
    )

    # Grant execution permissions explicitly to bash wrapper scripts
    subprocess.run(
        ["chmod", "+x", "scripts/long_term_forecast/ECL_script/DLinear.sh"]
    )

    # Run training pipelines via script paths
    subprocess.run(["bash", "scripts/long_term_forecast/ECL_script/DLinear.sh"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###TimesNet
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
    ###TimeMixer
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
    ###iTransformer
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
    ###HyPT
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

    This pipeline runs the full ablation suite comparing the individual and combined architectural components of HyPT:
    * **Full Hybrid Model (`both`)**: Periodicity (Branch A) + Cross-Variate (Branch B)
    * **Periodicity Branch Only (`branch_a`)**: Evaluates time-patch periodicity modeling only
    * **Cross-Variate Branch Only (`branch_b`)**: Evaluates cross-channel attention modeling only
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
    ###Seed 2022
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
    ###Seed 2023
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
    # Radio buttons to choose between pre-computed paper results or local run results
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


@app.cell
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

    # Base folder path based on chosen source and active workflow
    base_dir = Path(results_source.value) / subfolder
    output_file = base_dir / "result_long_term_forecast.txt"

    # Regex to match the required metric line structure
    metrics_pattern = re.compile(
        r"^mse:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?, "
        r"mae:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?, "
        r"dtw:(?:Not calculated|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
    )

    results = []

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory '{base_dir}' does not exist.")

    # Sort subfolders to keep output clean and predictable
    subfolders = sorted([f for f in base_dir.iterdir() if f.is_dir()])

    for folder in subfolders:
        log_file = folder / "output.log"

        if not log_file.exists():
            raise FileNotFoundError(f"Missing 'output.log' in {folder.name}")

        # Read lines and strip trailing whitespace/newlines
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"The log file in '{folder.name}' is empty.")

        last_line = lines[-1]

        # Validate that the last line matches the expected pattern
        if not metrics_pattern.match(last_line):
            raise ValueError(
                f"Validation error in folder '{folder.name}':\n"
                f"Expected metrics line, but got:\n'{last_line}'"
            )

        results.append(f"{folder.name}\n{last_line}\n")

    # Write all gathered entries into the output file
    output_content = "\n".join(results)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    def parse_and_format_results(file_path: Path) -> str:
        # Read and parse log file data
        text = file_path.read_text()
        pattern = re.compile(
            r"long_term_forecast_ECL_96_(\d+)_([A-Za-z0-9]+)_.*?\nmse:([\d.]+),\s*mae:([\d.]+)"
        )

        records = []
        for seq_len, model, mse, mae in pattern.findall(text):
            records.append(
                {
                    "Seq": int(seq_len),
                    "Model": "HyPT\n(Mine)" if model == "HyPT" else model,
                    "MSE": float(mse),
                    "MAE": float(mae),
                }
            )

        df = pd.DataFrame(records)

        # Pivot table
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

        # Calculate Average row
        avg_series = pivoted.mean(axis=0)
        pivoted.loc["Avg"] = avg_series

        # Highlight best (Red Bold) and second best (Blue Underline) per row
        def style_cell(val, best, second):
            formatted_val = f"{val:.3f}"
            if abs(val - best) < 1e-6:
                return f'<span style="color: red; font-weight: bold;">{formatted_val}</span>'
            elif abs(val - second) < 1e-6:
                return f'<span style="color: blue; text-decoration: underline;">{formatted_val}</span>'
            return formatted_val

        for idx, row in pivoted.iterrows():
            for metric in ["MSE", "MAE"]:
                metric_vals = [row[(m, metric)] for m in all_models]
                sorted_vals = sorted(metric_vals)
                best_val = sorted_vals[0]
                second_val = sorted_vals[1] if len(sorted_vals) > 1 else best_val

                for m in all_models:
                    val = row[(m, metric)]
                    styled = style_cell(val, best_val, second_val)
                    pivoted.loc[idx, (m, f"{metric}_styled")] = styled

        # Build HTML Table
        html = """
        <style>
            .paper-table {
                font-family: 'Times New Roman', Times, serif;
                border-collapse: collapse;
                text-align: center;
                font-size: 14px;
                margin: 20px 0;
            }
            .paper-table th, .paper-table td {
                padding: 4px 8px;
            }

            /* Top & Bottom thick horizontal borders */
            .paper-table thead tr:first-child th {
                border-top: 2px solid black;
            }
            .paper-table tbody tr:last-child td {
                border-bottom: 2px solid black;
            }

            /* Model header bottom line */
            .paper-table .model-header {
                border-bottom: 1px solid black;
            }

            /* Metric header row lines */
            .paper-table .metric-header-row th {
                border-bottom: 2px solid black;
            }
            .paper-table .models-title {
                border-bottom: 1px solid black;
                font-weight: bold;
            }

            /* Vertical model separators */
            .paper-table .model-end {
                border-right: 1px solid black;
            }

            /* Column dividing sequence lengths from metrics */
            .paper-table .seq-col {
                border-right: 1px solid black;
                font-weight: bold;
            }

            /* Rotated "Electricity" label pinned to the right side of its cell */
            .paper-table td.dataset-label {
                position: relative;
                font-weight: bold;
                font-size: 13px;
                width: 1px !important;
                padding: 0 !important;
                white-space: nowrap;
                border-left: hidden;
            }
            .paper-table td.dataset-label .label-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) rotate(180deg);
                writing-mode: vertical-lr;
            }

            /* Average row top divider line (excludes the empty far-left cell) */
            .paper-table .avg-row td:not(.no-border) {
                border-top: 1px solid black;
            }
        </style>
        <table class="paper-table">
            <thead>
                <tr>
                    <th colspan="2" class="models-title" style="border-right: 1px solid black; border-bottom: hidden" >Models</th>
        """

        for idx, m in enumerate(all_models):
            is_last = " model-end" if idx < len(all_models) - 1 else ""
            html += f'<th colspan="2" class="model-header{is_last}">{m}</th>'
        html += "</tr><tr class='metric-header-row'>"

        html += '<th colspan="2" class="seq-col">Metric</th>'
        for idx, m in enumerate(all_models):
            is_last = " model-end" if idx < len(all_models) - 1 else ""
            html += f"<th>MSE</th><th class='{is_last}'>MAE</th>"
        html += "</tr></thead><tbody>"

        # Data Rows
        seqs = [96, 192, 336, 720, "Avg"]
        for idx, seq in enumerate(seqs):
            if seq == "Avg":
                html += '<tr class="avg-row">'
                # Blank far-left cell under Electricity (no left border, no top border)
                html += '<td class="no-border"></td>'
            else:
                html += "<tr>"
                # Centered rotated Electricity label spanning rows 96-720
                if idx == 0:
                    html += '<td rowspan="4" class="dataset-label"><span class="label-text">Electricity</span></td>'

            html += f'<td class="seq-col">{seq}</td>'

            for m_idx, m in enumerate(all_models):
                is_last = " model-end" if m_idx < len(all_models) - 1 else ""
                mse_s = pivoted.loc[seq, (m, "MSE_styled")]
                mae_s = pivoted.loc[seq, (m, "MAE_styled")]
                html += f"<td>{mse_s}</td><td class='{is_last}'>{mae_s}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        return html

    # Marimo Execution
    table_html = parse_and_format_results(output_file)
    mo.Html(table_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save results
    Creates a downloadable `.zip` of the `test_results` folder, without model weights and array files.
    """)
    return


@app.cell(hide_code=True)
def _(REPO_NAME, mo, os, zipfile):
    _source_dir = REPO_NAME+"/test_results"
    _output_zip = "test_results_archive.zip"

    if not os.path.exists(_source_dir):
        # Graceful handling if the training script hasn't generated the folder yet
        download_button = mo.md(f"**{_source_dir}** folder not found. Please run the training script first.")
    else:
        # Walk through and compress only the intended files
        with zipfile.ZipFile(_output_zip, 'w', zipfile.ZIP_DEFLATED) as _zipf:
            for _root, _dirs, _files in os.walk(_source_dir):
                # Explicitly exclude the 'results' directory if it exists as a subfolder
                if "results" in _dirs:
                    _dirs.remove("results")

                for _file in _files:
                    # Skip the model files and arrays
                    if _file.endswith('.npy') or _file.endswith('.pth'):
                        continue

                    _full_path = os.path.join(_root, _file)
                    # Create a clean relative path inside the zip file
                    _arc_name = os.path.relpath(_full_path, _source_dir)
                    _zipf.write(_full_path, _arc_name)
                # Add result_long_term_forecast.txt from the repo root if it exists
                if os.path.exists(_root_txt_file):
                    _zipf.write(_root_txt_file, arcname="result_long_term_forecast.txt")

        # Read the streamlined zip into memory
        with open(_output_zip, "rb") as _f:
            _zip_bytes = _f.read()

        # Provide the clean download button
        download_button = mo.download(
            data=_zip_bytes,
            filename="test_results.zip",
            label="Download Test Results"
        )

    # Render the button or the warning message in the Marimo UI
    download_button
    return


if __name__ == "__main__":
    app.run()