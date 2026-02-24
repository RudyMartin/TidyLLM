

# 🔌 Streamlit Portal for Team Apps

This repository provides a **central portal** that lets teams publish and discover their own **Streamlit micro-apps**.

The portal:

* Reads a **registry of app manifests** (`registry.yaml`).
* Displays each app as a **tile** with icon, title, and description.
* Enforces **roles/permissions** from manifests.
* Provides a **consistent entry point** for all apps.

Each **team app**:

* Lives in its own repo (isolation of code & dependencies).
* Publishes an `app_manifest.yaml`.
* Is registered in the portal by adding its manifest URL to `registry.yaml`.

---

## 📂 Repository Structure

```
portal/
├── app.py                  # Main Streamlit portal
├── registry.yaml           # Central registry of app manifests
├── requirements.txt        # Python dependencies
├── Dockerfile              # Build the portal container
├── docker-compose.yml      # Dev setup with Traefik router
├── manifest_schema.json    # Schema to validate app manifests
└── README.md               # Developer guide
```

---

## 1. Main Portal App (`app.py`)

```python
import streamlit as st, yaml, requests

st.set_page_config(page_title="Team Apps Portal", layout="wide")
st.title("🔌 Team Apps Portal")

REGISTRY_URL = "https://raw.githubusercontent.com/your-org/portal/main/registry.yaml"

@st.cache_data(ttl=60)
def load_yaml(url):
    return yaml.safe_load(requests.get(url, timeout=5).text)

# Load registry of app manifests
reg = load_yaml(REGISTRY_URL)
user_roles = {"analyst", "viewer"}  # replace with roles from OIDC

cols = st.columns(3)
i = 0
for item in reg["apps"]:
    try:
        m = load_yaml(item["manifest_url"])
    except Exception as e:
        st.error(f"Failed to load manifest: {item} ({e})")
        continue

    allowed = set(m.get("permissions", {}).get("roles_allowed", []))
    if allowed and not (user_roles & allowed):
        continue

    with cols[i % 3]:
        st.markdown(f"### {m.get('icon','🧩')} {m['title']}")
        st.caption(m.get("description",""))
        st.link_button("Open", m["route"])
    i += 1
```

---

## 2. Registry of Apps (`registry.yaml`)

```yaml
apps:
  - manifest_url: "https://raw.githubusercontent.com/org/qa-workbench/main/app_manifest.yaml"
  - manifest_url: "https://raw.githubusercontent.com/org/risk-dashboard/main/app_manifest.yaml"
```

---

## 3. Requirements (`requirements.txt`)

```
streamlit==1.37.0
pyyaml
requests
jsonschema
```

---

## 4. Dockerfile (`Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501", "--server.enableCORS=false"]
```

---

## 5. Local Dev with Docker Compose (`docker-compose.yml`)

```yaml
version: "3.8"
services:
  portal:
    build: .
    container_name: portal
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.portal.rule=PathPrefix(`/`)"
      - "traefik.http.services.portal.loadbalancer.server.port=8501"
    depends_on: [traefik]

  traefik:
    image: traefik:v3.0
    command:
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
    ports: ["80:80"]
    volumes: ["/var/run/docker.sock:/var/run/docker.sock:ro"]
```

This runs:

* **Portal** at `/`
* Additional apps (with labels) under `/apps/<id>`

---

## 6. Manifest Schema (`manifest_schema.json`)

```json
{
  "type": "object",
  "required": ["app_id", "title", "route", "permissions"],
  "properties": {
    "app_id": {"type": "string"},
    "title": {"type": "string"},
    "description": {"type": "string"},
    "route": {"type": "string"},
    "icon": {"type": "string"},
    "permissions": {
      "type": "object",
      "properties": {
        "roles_allowed": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

---

## 7. Example Team App Manifest

```yaml
app_id: qa-workbench
title: "QA Workbench"
owner_team: "Validation"
category: "Analytics"
description: "Custom workflows for QA scoring and reporting."
icon: "🛠️"
route: "/apps/qa-workbench"
permissions:
  roles_allowed: ["analyst","manager"]
healthcheck: "/healthz"
version: "0.1.0"
```

---

## 8. Example Team App (`app.py`)

```python
import streamlit as st
st.set_page_config(page_title="QA Workbench", layout="wide")

st.title("🛠️ QA Workbench")
st.write("This is your custom app. Add your workflows here.")

tab1, tab2 = st.tabs(["Score Docs", "Generate Report"])
with tab1:
    st.write("Upload docs → run checks → view findings.")
with tab2:
    st.write("Assemble compliance-ready reports.")

# Healthcheck
if st.query_params.get("healthz") == ["1"]:
    st.write("OK")
```

---

## 9. Adding a New App to the Portal

1. Create a repo for your app.
2. Add:

   * `app.py`
   * `requirements.txt`
   * `Dockerfile`
   * `app_manifest.yaml`
3. Deploy your app container.
4. Add your `manifest_url` to `registry.yaml` in the **portal** repo.
5. Commit + push → portal will show your app automatically.

---

## 10. Auth & Roles

* Use **OIDC (Cognito/Auth0/AD)** at the proxy or ALB level.
* Pass user claims to apps via headers (e.g., `X-User`, `X-Roles`).
* Apps filter content using `roles_allowed` in their manifests.

---

## 11. Deployment (AWS Example)

* **ECS Fargate**: run each app as its own service.
* **ALB**:

  * `/` → Portal service
  * `/apps/<id>` → Target group for each app
* **Cognito or AD**: OIDC auth on ALB.
* **Secrets**: use AWS Secrets Manager.

---

## 12. Governance Rails

* Template repo for new apps.
* Manifest validation with JSON schema.
* Healthchecks required for every app.
* CI/CD (GitHub Actions → Docker build → push ECR → deploy ECS).

---

# ✅ Next Steps for Teammates

1. **Clone this repo** and run locally:

   ```bash
   git clone https://github.com/your-org/portal.git
   cd portal
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **Create your own app repo**:

   * Copy the example `app.py`, `Dockerfile`, `requirements.txt`, `app_manifest.yaml`.

3. **Add your app’s manifest URL** to the `registry.yaml` in this repo.

4. **Test locally with Docker Compose**.

5. **Deploy to AWS/ECS (or your platform)**.

