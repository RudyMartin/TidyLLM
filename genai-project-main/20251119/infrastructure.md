⏺ Done. tidyllm-infrastructure is now the main entry point.

  New Architecture

  pip install tidyllm-infrastructure

  This single install gets you everything:

  | What You Get   | Description                           |
  |----------------|---------------------------------------|
  | services       | RAG, flows, DSPy (from tidyllm)       |
  | tlm            | Pure Python math operations           |
  | embeddings     | TF-IDF, LSA, transformers             |
  | Infrastructure | AWS, Bedrock, PostgreSQL, credentials |

  Usage

  from tidyllm_infrastructure import services, tlm, embeddings, yaml_loader

  # Embeddings
  vectors = embeddings.tfidf_fit_transform(documents)

  # Infrastructure  
  config = yaml_loader.load_settings("settings.yaml")

  Note: The infrastructure modules (boto3, polars, etc.) bring in scipy/numpy as their dependencies. The pure
  packages (tlm, tidyllm-embeddings) remain numpy-free when imported directly for lightweight use cases.
