# TidyLLM Demos

A consolidated collection of demos showcasing the TidyLLM ecosystem capabilities.

## 🏗️ Structure

```
tidyllm-demos/
├── README.md                 # This file
├── requirements.txt          # Common dependencies
├── settings.yaml            # Shared configuration
├── launcher.py              # Main demo launcher
├── demos/
│   ├── live-ticker/         # Live AI Question Ticker
│   ├── gateway-control/     # MLflow Gateway Control Dashboard
│   ├── settings-config/     # Settings Configuration Interface
│   └── mvr-demo/           # Model Validation Report Demo
├── shared/
│   ├── utils.py            # Shared utilities
│   ├── database.py         # Database connection utilities
│   └── gateway.py          # Gateway connection utilities
└── docs/
    ├── installation.md     # Installation guide
    └── architecture.md     # System architecture
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e ../tidyllm-gateway
   ```

2. **Configure settings:**
   ```bash
   cp settings.yaml.example settings.yaml
   # Edit settings.yaml with your configuration
   ```

3. **Launch demos:**
   ```bash
   python launcher.py
   ```

## 📊 Available Demos

### 1. Live AI Ticker
**Location:** `demos/live-ticker/`
**Description:** Real-time AI question processing with cost tracking and budget controls
**Features:**
- Real LLM requests through TidyLLM Gateway
- Comprehensive cost tracking and budget management
- Time period analysis and model breakdown
- Professional expander interface with prompt transparency

### 2. Gateway Control Dashboard
**Location:** `demos/gateway-control/`
**Description:** MLflow Gateway management and monitoring interface
**Features:**
- Connection status monitoring
- Route management
- Request tracking
- Performance metrics

### 3. Settings Configuration
**Location:** `demos/settings-config/`
**Description:** Interactive settings management interface
**Features:**
- YAML configuration editor
- Environment variable management
- Database connection testing
- Credential management

### 4. MVR Demo
**Location:** `demos/mvr-demo/`
**Description:** Model Validation Report processing and analysis
**Features:**
- Document processing and analysis
- SPARSE code execution
- Compliance tracking
- Report generation

## 🔧 Configuration

### Settings File (`settings.yaml`)
```yaml
# Database settings
database:
  host: localhost
  port: 5432
  name: your_database_name
  user: your_username
  password: your_password

# MLflow settings
mlflow:
  tracking_uri: http://localhost:5000
  gateway_uri: http://localhost:5000

# TidyLLM Gateway settings
tidyllm_gateway:
  mlflow_gateway_uri: http://localhost:5000
  default_provider: claude
  default_model: claude-3-5-sonnet
```

### Environment Variables
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=your_database_name
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## 🛠️ Development

### Adding a New Demo
1. Create a new directory in `demos/`
2. Add your demo files
3. Update `launcher.py` to include your demo
4. Add documentation in `docs/`

### Shared Utilities
Use the utilities in `shared/` for common functionality:
- Database connections
- Gateway setup
- Configuration management
- Error handling

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your demo or improvements
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated:** August 30, 2024
**Version:** 1.0.0


