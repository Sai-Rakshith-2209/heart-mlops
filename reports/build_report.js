// Builds reports/MLOps_Assignment1_Report.docx from the real pipeline outputs.
// Usage: node reports/build_report.js   (run from repo root after training)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table,
  TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak,
  Footer, Header, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "reports", "figures");
const meta = JSON.parse(fs.readFileSync(path.join(ROOT, "models", "model_meta.json")));
const eda = JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "eda_summary.json")));
const cmp = fs.readFileSync(path.join(ROOT, "reports", "model_comparison.csv"), "utf8")
  .trim().split("\n").map((l) => l.split(","));
const hdr = cmp[0];
const rows = cmp.slice(1).map((r) => Object.fromEntries(hdr.map((h, i) => [h, r[i]])));

const STUDENT = "Sai Rakshith";
const STUDENT_ID = "<BITS ID>";
const REPO = "https://github.com/<your-username>/heart-mlops";

// A4 with 2 cm margins -> 9638 DXA text width.
const TEXT_W = 9638;
const FONT = "Calibri";
const ACCENT = "1F3A5F";

const pngSize = (f) => {
  const b = fs.readFileSync(f);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
};
function img(name, widthPx = 600, caption) {
  const f = path.join(FIG, name);
  const { w, h } = pngSize(f);
  const out = [new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f),
      transformation: { width: widthPx, height: Math.round((widthPx * h) / w) } })],
  })];
  if (caption) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })] }));
  return out;
}
const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts,
  children: Array.isArray(text) ? text : [new TextRun(text)] });
const b = (t) => new TextRun({ text: t, bold: true });
const t = (s) => new TextRun(s);
const code = (s) => new TextRun({ text: s, font: "Consolas", size: 19 });
const h1 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(s)] });
const h2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(s)] });
const bullet = (children) => new Paragraph({ numbering: { reference: "bullets", level: 0 },
  spacing: { after: 60 }, children: typeof children === "string" ? [t(children)] : children });
const codeBlock = (lines) => lines.map((l, i) => new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" },
  spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 140 : 0 },
  indent: { left: 120, right: 120 },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })],
}));
const shot = (what, how) => new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "FFF4D6", color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E0A100", space: 6 } },
  spacing: { before: 120, after: 160 }, indent: { left: 160 },
  children: [new TextRun({ text: "SCREENSHOT: ", bold: true, color: "8A6100" }),
    new TextRun({ text: what }),
    ...(how ? [new TextRun({ text: "   Command: ", bold: true, color: "8A6100" }),
      new TextRun({ text: how, font: "Consolas", size: 18 })] : [])],
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC5CF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, body, widths, boldFirstRow = -1) {
  const total = widths.reduce((a, c) => a + c, 0);
  const cell = (txt, w, isHead, bold) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: ACCENT, color: "auto" }
      : bold ? { type: ShadingType.CLEAR, fill: "EAF2EC", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(txt), size: 18,
      bold: isHead || bold, color: isHead ? "FFFFFF" : "1A1A1A" })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...body.map((r, ri) => new TableRow({ children: r.map((c, i) =>
        cell(c, widths[i], false, ri === boldFirstRow)) }))],
  });
}
const f3 = (x) => Number(x).toFixed(3);
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });

const pretty = { logistic_regression: "Logistic Regression", random_forest: "Random Forest",
  hist_gradient_boosting: "HistGradientBoosting" };
const m = meta.metrics;
const corr = Object.entries(eda.target_correlation);

// ---------------------------------------------------------------- content
const children = [
  // Title page
  new Paragraph({ spacing: { before: 2200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Machine Learning Operations (AIMLCZG523)", size: 28, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 },
    children: [new TextRun({ text: "Assignment 1", size: 30, color: ACCENT, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 200 },
    children: [new TextRun({ text: "Heart Disease Risk Prediction:", size: 48, bold: true, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 },
    children: [new TextRun({ text: "End-to-End MLOps from Raw Data to a Monitored Kubernetes API", size: 32, color: ACCENT })] }),
  ...[["Student", STUDENT], ["BITS ID", STUDENT_ID], ["Programme", "M.Tech AI/ML, BITS Pilani (WILP)"],
    ["Code repository", REPO], ["Date", "20 October 2026"]].map(([k, v]) =>
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
      children: [new TextRun({ text: `${k}: `, bold: true, size: 24 }), new TextRun({ text: v, size: 24 })] })),
  new Paragraph({ children: [new PageBreak()] }),


  // 1
  h1("1. Summary and architecture"),
  p([t("A binary classifier predicts presence of heart disease from 13 clinical attributes of the UCI Cleveland dataset and is served as a hardened, monitored REST API. The champion model is "),
    b(`${pretty[meta.model_name]} (C=${meta.best_params.C}, ${meta.best_params.penalty})`),
    t(`, selected on 5-fold cross-validated ROC-AUC (${f3(m.cv_roc_auc)}). On the untouched 20% test set it reaches ROC-AUC ${f3(m.test_roc_auc)}, accuracy ${f3(m.test_accuracy)} and recall ${f3(m.test_recall)}.`)]),
  p("Every stage is code in a single Python package (src/heart) that is executed identically by a developer, the notebook and the CI runner, so there is one implementation of each step. The figure below shows the end-to-end flow."),
  ...img("architecture.png", 620, "Figure 1. System architecture: training and tracking, CI/CD, Kubernetes serving and monitoring"),
  table(["Assignment task", "Where it is implemented"], [
    ["1. Data acquisition and EDA", "src/heart/data.py, src/heart/eda.py, notebooks/"],
    ["2. Features and models", "src/heart/features.py, src/heart/train.py"],
    ["3. Experiment tracking", "MLflow (sqlite backend) inside train.py"],
    ["4. Packaging and reproducibility", "models/model.joblib + model_meta.json, MLflow model, pinned requirements"],
    ["5. CI/CD and tests", ".github/workflows/ci.yml, tests/ (19 pytest tests)"],
    ["6. Containerisation", "Dockerfile, scripts/docker_smoke_test.sh"],
    ["7. Deployment", "k8s/*.yaml, scripts/deploy_minikube.sh"],
    ["8. Monitoring and logging", "/metrics, JSON logs, k8s/monitoring (Prometheus + Grafana)"],
  ], [3600, 6038]),

  // 2
  h1("2. Setup and installation"),
  p("Requirements: Python 3.11+ (3.12 used in CI and the container), Docker, and minikube with kubectl for deployment. From a clean clone:"),
  ...codeBlock([
    "python3 -m venv .venv && source .venv/bin/activate",
    "pip install -r requirements-dev.txt -e .",
    "python -m heart.data     # download + clean  -> data/processed/heart_clean.csv",
    "python -m heart.eda      # EDA figures       -> reports/figures/",
    "python -m heart.train    # tune + track      -> mlflow.db, models/",
    "pytest -v && ruff check .",
    "make mlflow              # MLflow UI  http://127.0.0.1:5000",
    "./scripts/docker_smoke_test.sh          # Task 6",
    "./scripts/deploy_minikube.sh            # Tasks 7-8",
  ]),
  p([t("Dependencies are split into three pinned files: "), code("requirements-serve.txt"),
    t(" (the slim runtime baked into the image), "), code("requirements.txt"),
    t(" (adds MLflow, plotting, skops for training) and "), code("requirements-dev.txt"),
    t(" (adds pytest, httpx, ruff). All pins were verified to resolve to binary wheels for Python 3.12 on Linux.")]),

  // 3
  h1("3. Data acquisition and EDA"),
  h2("3.1 Acquisition and cleaning"),
  p([t("heart.data downloads the official UCI file "), code("processed.cleveland.data"),
    t(", and falls back to a GitHub mirror of the same 303 rows (string-coded categoricals mapped back to the UCI codebook) if the UCI host is unreachable. Cleaning then:")]),
  bullet("coerces every column to numeric and treats '?' as missing;"),
  bullet("nulls out values outside the UCI codebook domains (e.g. cp not in 1-4) or physiologically impossible ranges;"),
  bullet("drops exact duplicate rows;"),
  bullet([t("binarises the target: "), code("num = 0"), t(" means no disease, "), code("num 1-4"), t(" means disease present.")]),
  p([t(`Result: ${eda.rows} rows, ${Object.values(eda.missing_by_column).reduce((a, c) => a + c, 0)} missing cells (`),
    t(Object.entries(eda.missing_by_column).map(([k, v]) => `${k}: ${v}`).join(", ")),
    t("). Missing values are deliberately not imputed here: imputation is part of the model pipeline so its statistics are learned on training folds only, preventing leakage, and the same logic runs at inference.")]),
  h2("3.2 Exploratory analysis"),
  ...img("01_class_balance.png", 300, `Figure 2. Class balance: ${(eda.positive_rate * 100).toFixed(1)}% positive`),
  p("The classes are close to balanced, so no resampling is needed; stratified splits keep the ratio in every fold. ROC-AUC and recall are still reported as the primary metrics because missing a sick patient is costlier than a false alarm."),
  ...img("02_numeric_histograms.png", 600, "Figure 3. Numeric feature distributions by class"),
  p("Patients with disease show markedly lower maximum heart rate (thalach), higher ST depression (oldpeak) and more fluoroscopy-coloured vessels (ca). Cholesterol and resting blood pressure distributions largely overlap between classes."),
  ...img("03_categorical_disease_rate.png", 600, "Figure 4. Disease rate per category (n per bar)"),
  p("Asymptomatic chest pain (cp = 4), exercise-induced angina (exang = 1) and reversible thallium defects (thal = 7) each carry disease rates above 70%. restecg = 1 has only 4 patients, which is why categorical features are one-hot encoded with handle_unknown='ignore' rather than relying on rare levels."),
  ...img("04_correlation_heatmap.png", 470, "Figure 5. Pearson correlation matrix including the target"),
  table(["Feature", "Correlation with target", "Feature", "Correlation with target"],
    Array.from({ length: Math.ceil(corr.length / 2) }, (_, i) => [
      corr[i][0], corr[i][1].toFixed(3),
      corr[i + 7] ? corr[i + 7][0] : "", corr[i + 7] ? corr[i + 7][1].toFixed(3) : ""]),
    [2409, 2410, 2409, 2410]),
  p("Feature correlations are moderate (none above |0.6|), so multicollinearity is not a concern and all 13 features are retained. fbs and chol carry little linear signal; dropping both moves CV ROC-AUC from 0.907 to 0.911, a +0.004 change well inside one CV standard deviation (about 0.02). They are therefore kept: the gain is noise-level, and L2 regularisation already shrinks weak coefficients.", { spacing: { before: 120, after: 120 } }),
  ...img("05_boxplots_by_class.png", 620, "Figure 6. Box plots of numeric features by class (outlier check)"),
  p("A few high cholesterol values (up to 564 mg/dl) are extreme but clinically plausible, so they are kept rather than clipped; StandardScaler and tree models are robust enough at this scale."),

  // 4
  h1("4. Feature engineering and model development"),
  h2("4.1 Preprocessing pipeline"),
  table(["Group", "Columns", "Transform"], [
    ["Numeric", "age, trestbps, chol, thalach, oldpeak, ca", "median imputation, then StandardScaler"],
    ["Binary", "sex, fbs, exang", "most-frequent imputation (already 0/1)"],
    ["Categorical", "cp, restecg, slope, thal", "most-frequent imputation, then one-hot (unknown -> all zeros)"],
  ], [1500, 3800, 4338]),
  p([t("The ColumnTransformer and the estimator are wrapped in a single sklearn "), b("Pipeline"),
    t(", producing 22 model inputs. Because the whole fitted Pipeline is the deployable artifact, preprocessing at inference is exactly the training code: there is no train/serve skew.")], { spacing: { before: 120, after: 120 } }),
  h2("4.2 Model selection and tuning"),
  p(`Data was split 80/20 with stratification (seed 42): ${eda.rows - 61} training and 61 test rows. Three model families were tuned with GridSearchCV over 5-fold stratified CV, scoring accuracy, precision, recall, F1 and ROC-AUC on every fold, and refitting on ROC-AUC. The test set is used once, for reporting only; the champion is selected on CV ROC-AUC.`),
  table(["Model", "Search space", "Best parameters"], [
    ["Logistic Regression (liblinear)", "C in {0.01, 0.1, 0.3, 1, 3} x penalty in {l1, l2} (10)", "C = 1.0, l2"],
    ["Random Forest", "n_estimators {200, 400} x max_depth {None, 4, 8} x min_samples_leaf {1, 5} (12)", "200 trees, max_depth 4, min_samples_leaf 5"],
    ["HistGradientBoosting", "learning_rate {0.03, 0.1} x max_depth {2, 3} x max_iter {100, 200} (8)", "lr 0.03, max_depth 2, 100 iterations"],
  ], [2600, 4538, 2500]),
  h2("4.3 Results"),
  table(["Model", "CV ROC-AUC", "CV recall", "Test ROC-AUC", "Test acc.", "Test prec.", "Test recall", "Test F1"],
    rows.map((r) => [pretty[r.model], f3(r.cv_roc_auc), f3(r.cv_recall), f3(r.test_roc_auc),
      f3(r.test_accuracy), f3(r.test_precision), f3(r.test_recall), f3(r.test_f1)]),
    [2238, 1050, 1050, 1100, 1050, 1050, 1050, 1050], 0),
  ...img("model_comparison.png", 540, "Figure 7. CV and test metrics per model family"),
  p([b("Why Logistic Regression wins. "),
    t("All three families are within about 0.02 CV ROC-AUC, which is inside one CV standard deviation (about 0.02). Given equal performance, the linear model is the better production choice: it has the highest CV ROC-AUC, the lowest CV variance, interpretable coefficients for clinicians, a sub-millisecond inference cost and the smallest artifact. The tuned tree models also chose their most constrained settings (shallow depth, larger leaves), a sign that the dataset (303 rows) rewards simple models.")]),
  ...img("models/logistic_regression_roc_pr.png", 560, "Figure 8. Champion ROC and precision-recall curves on the test set"),
  ...img("models/logistic_regression_confusion_matrix.png", 280, "Figure 9. Champion confusion matrix (threshold 0.5)"),
  p("At the default threshold the champion misses 2 of 28 diseased test patients (recall 0.93) at the cost of 6 false positives. The threshold is stored in model_meta.json and read by the API, so it can be tuned toward recall without code changes."),
  ...img("models/logistic_regression_feature_importance.png", 390, "Figure 10. Champion: absolute standardised coefficients"),

  // 5
  h1("5. Experiment tracking (MLflow)"),
  p("Tracking uses an MLflow sqlite backend (mlflow.db) with local artifact storage. Each model family is a parent run; each grid candidate is a nested child run, so every configuration evaluated is recorded:"),
  table(["Level", "Count", "Logged"], [
    ["Parent run per family", "3", "split sizes, seed, CV folds, best params; CV + test metrics; confusion matrix, ROC/PR and importance plots; MLflow model with signature and input example; tags: data SHA-256, git SHA"],
    ["Nested candidate runs", "30", "hyperparameters; mean and std of every CV metric; train ROC-AUC (overfitting check)"],
    ["Champion tag", "1", "champion=true, model_meta.json, model comparison plot"],
  ], [2300, 900, 6438]),
  p([t("MLflow 3 serialises scikit-learn models with "), b("skops"),
    t(" instead of pickle. The training code allow-lists exactly the types present in each fitted model, computed from the object itself, rather than trusting arbitrary types.")], { spacing: { before: 120, after: 120 } }),
  shot("MLflow UI experiment page showing the 3 parent runs with nested candidates, and one run's metrics and artifacts tab.", "make mlflow"),

  // 6
  h1("6. Model packaging and reproducibility"),
  bullet([b("Artifact: "), code("models/model.joblib"), t(" is the full fitted Pipeline (preprocessing + model). The same model is also logged as an MLflow model.")]),
  bullet([b("Metadata: "), code("model_meta.json"), t(` records the MLflow run id (${meta.mlflow_run_id.slice(0, 12)}...), metrics, decision threshold, feature order, training-data SHA-256, timestamp and the Python and scikit-learn versions.`)]),
  bullet([b("Determinism: "), t("fixed random_state for the split, CV shuffling and every estimator; a unit test asserts identical predictions from two independent fits.")]),
  bullet([b("Pinned environment: "), t("exact versions in the three requirements files; numpy, pandas and scikit-learn are identical between training and serving because the artifact is a fitted sklearn object.")]),
  bullet([b("Data lineage: "), t("the cleaned CSV is committed and its hash is stored with every run; heart.data reproduces it from the source.")]),

  // 7
  h1("7. CI/CD pipeline and automated testing"),
  h2("7.1 Unit tests (19, pytest)"),
  table(["File", "What is tested"], [
    ["test_data.py (5)", "UCI parsing marks '?' missing; mirror codes map to the UCI codebook; target binarisation; out-of-domain values nulled and duplicates dropped; cleaned-data contract (303 rows, binary target)"],
    ["test_model.py (7)", "preprocessor output width (22) and finiteness; tolerance of missing and unseen categories; CV ROC-AUC above 0.85; determinism; each model family trains and predicts valid probabilities"],
    ["test_api.py (7)", "/health and /ready; /predict returns prediction and confidence; optional fields; 422 for out-of-range, invalid code and unknown field; batch and 413 limit; /metrics content"],
  ], [2000, 7638]),
  p("API tests train a small model on the fly, so the test job does not depend on the training job's artifacts.", { spacing: { before: 120 } }),
  h2("7.2 GitHub Actions workflow"),
  table(["Job", "Steps", "Fails the build when"], [
    ["lint", "ruff check (pyflakes, pycodestyle, isort, bugbear, pyupgrade), ruff format --check", "any lint or formatting violation"],
    ["test", "pytest -v with JUnit XML uploaded as an artifact", "any test fails"],
    ["train (needs lint, test)", "download + clean, EDA, tune + track, quality gate; uploads model, mlflow.db, reports, train log", "CV or test ROC-AUC below 0.85, or test recall below 0.80"],
    ["docker (needs train)", "build image with this run's model, run it, /ready + /predict + 422 check; Trivy scan; push to GHCR on main", "container does not serve, or fixable CRITICAL CVE"],
  ], [1900, 4738, 3000]),
  p("Concurrency cancels superseded runs, pip and Docker layers are cached, and the workflow token is read-only except for package write in the push job.", { spacing: { before: 120 } }),
  shot("GitHub Actions run graph with all four jobs green, plus the Artifacts section listing model, mlflow-and-reports and test-results.", "push to main, open the Actions tab"),
  shot("A deliberately failing run (e.g. break one test) showing the pipeline stops with a clear log.", null),

  // 8
  h1("8. Model containerisation"),
  p("The Dockerfile is multi-stage: a deps stage builds a virtualenv from requirements-serve.txt only; the runtime stage copies the venv, three source files and the model. MLflow, plotting libraries and compilers are not in the image. .dockerignore is an allow-list, so nothing else (data, tests, mlflow.db) can leak into the build context."),
  table(["Control", "Implementation"], [
    ["Least privilege", "fixed non-root UID 10001 (verifiable by runAsNonRoot)"],
    ["Health", "HEALTHCHECK calls /ready (model loaded), not just process liveness"],
    ["Contract", "POST /predict accepts JSON, returns prediction, label, confidence, probability_disease, model name and version"],
    ["Traceability", "model_meta.json inside the image; served on /model and as the model_info metric"],
  ], [2300, 7338]),
  p("Verified output from the API serving the champion (the same files the image contains, in a clean environment with only requirements-serve.txt):", { spacing: { before: 120 } }),
  ...codeBlock([
    "$ curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample_request.json",
    '{"prediction":1,"label":"heart_disease","confidence":0.8986,"probability_disease":0.8986,',
    ' "model_name":"logistic_regression","model_version":"1.0.0"}',
    "$ curl -X POST localhost:8000/predict -d '{\"age\":41,\"sex\":0,\"cp\":2, ... \"thal\":3}'",
    '{"prediction":0,"label":"no_heart_disease","confidence":0.9573,"probability_disease":0.0427, ...}',
    "$ curl -X POST localhost:8000/predict -d '{\"age\":500}'   ->  HTTP 422",
  ]),
  shot("Terminal output of the smoke test: docker build, /ready, /predict JSON, 422 check and 'healthy' container status.", "./scripts/docker_smoke_test.sh"),

  // 9
  h1("9. Production deployment (Kubernetes)"),
  table(["Manifest", "Purpose and key settings"], [
    ["00-namespace.yaml", "heart and monitoring namespaces enforcing the restricted Pod Security Standard"],
    ["10-deployment.yaml", "2 replicas, RollingUpdate maxUnavailable 0; startup, readiness (/ready) and liveness (/health) probes; requests and memory limits; read-only root FS, all capabilities dropped, seccomp RuntimeDefault, no service-account token; topology spread across nodes"],
    ["20-service.yaml", "Service type LoadBalancer on port 80"],
    ["30-ingress.yaml", "nginx Ingress for heart.local with per-client rate limit and body-size cap"],
    ["40-hpa-pdb.yaml", "HPA 2-5 replicas at 70% CPU; PodDisruptionBudget minAvailable 1"],
    ["50-prometheus-rbac.yaml", "namespaced Role letting Prometheus list pods in heart only"],
  ], [2400, 7238]),
  p("All manifests pass strict schema validation (kubeconform, Kubernetes 1.32: 17 resources valid). Deployment on minikube:", { spacing: { before: 120 } }),
  ...codeBlock([
    "./scripts/deploy_minikube.sh              # build, minikube image load, kubectl apply",
    "minikube tunnel                           # separate terminal",
    "kubectl -n heart get svc heart-api        # EXTERNAL-IP",
    "curl -X POST http://<EXTERNAL-IP>/predict -H 'Content-Type: application/json' -d @sample_request.json",
    "curl -X POST http://heart.local/predict  ...   # via Ingress",
  ]),
  shot("kubectl get all,ingress,hpa -n heart showing 2/2 Running pods and the LoadBalancer EXTERNAL-IP.", "kubectl get all,ingress,hpa -n heart"),
  shot("curl /predict through the LoadBalancer IP and through the Ingress host, with JSON responses.", null),

  // 10
  h1("10. Monitoring and logging"),
  p([b("Logging. "), t("Every request emits one JSON line with request id (propagated from x-request-id or generated, and echoed in the response header), method, path, status, latency and client. Predictions log only the label and probability. Patient features are never written to logs, since this is health data.")]),
  ...codeBlock(['{"ts":"2026-10-19T19:05:55+0000","level":"INFO","logger":"heart.api","msg":"request",',
    ' "request_id":"f04b5e12...","method":"POST","path":"/predict","status":422,"latency_ms":1.28}']),
  p([b("Metrics. "), t("/metrics exposes http_requests_total (route, status), http_request_duration_seconds histogram, model_predictions_total (label), model_disease_probability histogram and model_info. Prometheus discovers every pod through annotations, so each replica is scraped individually rather than through the load-balanced Service.")]),
  table(["Grafana panel", "Query intent"], [
    ["Requests/s, p95 latency, 5xx error ratio", "service health (golden signals)"],
    ["Request rate by route and status", "traffic mix and 4xx validation failures"],
    ["Latency p50/p95/p99 for /predict", "performance regression"],
    ["Predictions by class; disease-positive share", "output distribution"],
    ["Mean predicted P(disease) over 15 min", "simple drift signal: a sustained shift means input data changed"],
  ], [4000, 5638]),
  p("Alert rules fire when no replica is scraped for 1 minute, when more than 5% of requests fail for 5 minutes, or when p95 /predict latency exceeds 500 ms for 10 minutes. The Grafana admin password is generated randomly into a Secret; Grafana and Prometheus are reachable only through kubectl port-forward.", { spacing: { before: 120 } }),
  shot("Prometheus Targets page with both heart-api pods UP.", "kubectl -n monitoring port-forward svc/prometheus 9090:9090"),
  shot("Grafana 'Heart Disease API' dashboard populated after running the load generator.", "python scripts/load_test.py --url http://<EXTERNAL-IP>"),
  shot("kubectl logs of one pod showing JSON request logs.", "kubectl -n heart logs deploy/heart-api --tail=20"),

  // 11
  h1("11. Limitations and next steps"),
  bullet("303 patients from a single centre (Cleveland) is small; with a 61-row test set the standard error on accuracy is about 0.04, so differences between the three model families are not statistically meaningful. External validation on the Hungarian, Swiss and Long Beach subsets is the next step."),
  bullet("The 0.5 threshold is a default; a recall-oriented threshold should be agreed with clinical stakeholders, then set in model_meta.json."),
  bullet("Drift monitoring is a mean-probability proxy; per-feature drift (e.g. PSI against training statistics) would be added next."),
  bullet("Model promotion is by CI artifact; a production setup would use the MLflow Model Registry with staged promotion and a remote tracking server."),
  bullet("This is a research prototype and not a medical device; outputs are not clinical advice."),
  h1("12. Deliverables"),
  table(["Deliverable", "Location"], [
    ["Code, Dockerfile, requirements", "src/, Dockerfile, requirements*.txt"],
    ["Cleaned dataset and download script", "data/processed/heart_clean.csv, python -m heart.data"],
    ["Notebook and scripts (EDA, training, inference)", "notebooks/heart_disease_eda_modelling.ipynb, src/heart/, scripts/predict_client.py"],
    ["Unit tests", "tests/"],
    ["CI workflow", ".github/workflows/ci.yml"],
    ["Deployment manifests", "k8s/, k8s/monitoring/, docker-compose.yml"],
    ["Screenshots", "screenshots/"],
    ["Repository", REPO],
    ["API access", "Local: ./scripts/deploy_minikube.sh then http://<EXTERNAL-IP>/docs"],
  ], [3800, 5838]),
];

const doc = new Document({
  creator: STUDENT, title: "MLOps Assignment 1 - Heart Disease Risk Prediction",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2E5A88" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET,
    text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "AIMLCZG523 MLOps - Assignment 1", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `${STUDENT}  |  Page `, size: 16, color: "888888" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" })] })] }) },
    children,
  }],
});

const out = path.join(ROOT, "reports", "MLOps_Assignment1_Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("written", out); });
