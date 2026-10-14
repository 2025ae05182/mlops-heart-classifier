# mlops-heart-classifier

mlops-heart-classifier/
│
├── data/                     # Raw & cleaned datasets
│   ├── raw/                  # Original UCI dataset
│   └── processed/            # Cleaned/preprocessed data
│
├── notebooks/                # Jupyter notebooks for EDA & prototyping
│   ├── 01_eda.ipynb
│   └── 02_model_dev.ipynb
│
├── src/                      # Python scripts (production-ready code)
│   ├── data_loader.py        # Data acquisition & preprocessing
│   ├── preprocess.py         # Feature engineering pipeline
│   ├── train.py              # Model training & MLflow logging
│   ├── inference.py          # Model inference logic
│   └── utils.py              # Helper functions
│
├── api/                      # Model serving API
│   ├── app.py                # FastAPI/Flask app with /predict endpoint
│   └── Dockerfile            # Containerization setup
│
├── tests/                    # Unit tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── ci-cd/                    # GitHub Actions / Jenkins pipelines
│   └── workflow.yml
│
├── k8s/                      # Kubernetes manifests / Helm charts
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
│
├── monitoring/               # Prometheus & Grafana configs
│   └── prometheus.yml
│
├── requirements.txt          # Python dependencies
├── README.md                 # Setup instructions & project overview
├── report.pdf                # Final 10-page assignment report
└── screenshots/              # Deployment & workflow screenshots

