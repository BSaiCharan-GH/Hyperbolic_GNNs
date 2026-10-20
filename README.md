# Hyperbolic Graph Neural Networks for Forensic Analytics

A research prototype for transaction-network analysis using **the original Lorentz (hyperboloid) model** with graph-attention message passing, applied to the Elliptic Bitcoin transaction dataset. The compact learned representation is mapped to two dimensions for Poincaré-disk visualization.

## What was changed for deployment

The original ZIP was kept as the architectural source of truth: `HyperbolicOps`, `HypGAT`, `HypLinear`, and `ForensicDetector` still use the Lorentz path and its 2-D Poincaré projection. The former `train.py` only built a synthetic balanced tree with random labels, so it has been replaced with a real Elliptic graph training/evaluation script. A robust CSV loader, a real model-output exporter, a Streamlit interface, dependency files, and a Colab workflow have been added.

**Checkpoint compatibility matters:** the checkpoint produced during the earlier deployment work belongs to a different Poincaré-ball model and is not compatible with this original Lorentz detector. This package intentionally does not reuse it or present its metrics as results from the Lorentz model. Run the supplied Colab notebook once to train this original architecture on the real dataset and export its genuine predictions.

## Project files

- `src/models/detector.py` — original Lorentz hyperbolic detector, ending in a 2-D projection and binary classifier.
- `src/layers/hyp_gat.py` — graph-attention updates in tangent space, mapped back to the Lorentz manifold.
- `src/layers/hyp_linear.py` and `src/utils/hyperbolic_ops.py` — original hyperbolic linear layer and Lorentz manifold operations.
- `src/utils/real_data_loader.py` — Elliptic graph loading, train-period feature scaling, 165 model features, labels, timesteps, and edges.
- `train.py` — real graph training, chronological label masks, validation threshold selection, checkpoint, and evaluation metrics.
- `export_predictions.py` — uses a **Lorentz checkpoint from this version** to export transaction scores, embeddings, degree, and evaluation metrics without retraining.
- `app.py` — Streamlit interface for exploring real exported transaction scores, threshold-based triage, held-out evaluation, and the 2-D embedding. It never generates placeholder transactions.
- `notebooks/Train_and_Export_Lorentz_GNN.ipynb` — Colab workflow to train the original model on the dataset in Google Drive and produce the outputs needed by the app.
- `publish_to_github.ps1` — syncs these files to the existing GitHub repository using your local Git authentication.

## Data

The raw Elliptic files should remain in Drive/local storage and **must not be committed** to GitHub. Expected files:

- `elliptic_txs_features.csv` (no header: transaction ID, timestep, 165 features)
- `elliptic_txs_classes.csv` (`txId`, `class`)
- `elliptic_txs_edgelist.csv` (`txId1`, `txId2`)

The loader excludes timestep from model features and fits `StandardScaler` using labelled-period input features from timesteps 1–34 only. It clips/scales the standardized inputs to limit numerical instability in the Lorentz exponential map.

## Run locally

For the web interface:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To train on a local copy of the dataset:

```bash
pip install -r requirements-training.txt
python train.py --data-dir "/path/to/elliptic_bitcoin_dataset" --epochs 30 --patience 6 --output-dir artifacts
python export_predictions.py --data-dir "/path/to/elliptic_bitcoin_dataset" --checkpoint artifacts/forensic_detector.pt --output-dir artifacts
```

The exporter creates `artifacts/predictions.csv` and `artifacts/deployment_metrics.json`. The app can also load a real `predictions.csv` through the sidebar uploader, but bundling the export in `artifacts/` makes it available to every visitor.

## Recommended workflow with Google Drive / Colab

1. **Sync the deployment source to GitHub.** Extract this ZIP on Windows, ensure Git for Windows is installed and authenticated to GitHub, then open PowerShell in the extracted folder and run:

   ```powershell
   .\publish_to_github.ps1
   ```

   It clones your existing public repository, copies the final source files, removes the old random-tree demo assets, commits, and pushes to `main`. It does not request a token or password in the script; Git Credential Manager may open its standard sign-in flow.

2. **Train the original Lorentz model in Colab.** Open `notebooks/Train_and_Export_Lorentz_GNN.ipynb` from this repository in Colab and run all cells. It mounts Drive, discovers the three real CSV files, trains the Lorentz model, exports predictions/metrics, and downloads a ZIP of the output artifacts. This is a new training run because the old saved Poincaré checkpoint is a different architecture; do not substitute it.

3. **Add the real outputs to the project.** Extract the downloaded `lorentz_deployment_artifacts.zip` into this folder's `artifacts/` directory, preserving filenames (`predictions.csv`, `deployment_metrics.json`, and `training_history.json`). Then run the PowerShell sync script again from the project folder to commit the real outputs.

4. **Deploy the app.** In Streamlit Community Cloud, create an app for repository `BSaiCharan-GH/Hyperbolic_GNNs`, branch `main`, main file `app.py`.

The app can start before the prediction export is added, but it will explicitly show that results are pending rather than inventing outputs. For a portfolio-quality demo, complete the Colab export before publishing the final app URL.

## Evaluation protocol and limitations

Training uses labelled transactions from timesteps 1–34, validation/threshold selection uses 35–41, and the held-out labelled test window is 42–49. All graph edges are available for message passing, so this is a **transductive chronological-label evaluation**, not strict future-isolated generalization. The app calls scores model ranking signals, not calibrated probabilities. A high score is not evidence of criminal activity. Compare against sensible baselines and run stricter temporal/graph-isolated evaluation before making strong scientific claims.
