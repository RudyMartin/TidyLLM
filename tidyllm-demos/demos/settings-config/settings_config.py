#!/usr/bin/env python3
"""
Settings Configurator - Streamlit Interface
Simple interface for adjusting settings.yaml configuration
"""
import streamlit as st
import yaml
import os
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Settings Configurator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_settings() -> Dict[str, Any]:
    """Load settings from demo app's settings.yaml"""
    try:
        settings_paths = [
            "../settings.yaml",  # Demo app's settings
            "settings.yaml",     # Fallback to local if needed
        ]
        for path in settings_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        return create_default_settings()
    except Exception as e:
        st.error(f"Error loading settings: {e}")
        return create_default_settings()

def create_default_settings() -> Dict[str, Any]:
    """Create default settings structure"""
    return {
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "database": "demo_db",
            "username": "demo_user",
            "password": "demo_pass",
            "ssl_mode": "prefer",
            "connection_pool_size": 10,
            "timeout": 30
        },
        "aws": {
            "region": "us-east-1",
            "profile": "default",
            "access_key_id": "",
            "secret_access_key": "",
            "session_token": "",
            "kms_key_id": ""
        },
        "integrations": {
            "mlflow": {
                "enabled": True,
                "tracking_uri": "file://./mlruns",
                "experiment_name": "demo-experiment"
            }
        }
    }

def save_settings(settings: Dict[str, Any]) -> bool:
    """Save settings to demo app's settings.yaml"""
    try:
        with open("../settings.yaml", 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving settings: {e}")
        return False

def render_postgres_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render PostgreSQL configuration section"""
    st.subheader("🗄️ PostgreSQL Configuration")
    
    postgres = settings.get("postgres", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        postgres["host"] = st.text_input("Host", postgres.get("host", "localhost"))
        postgres["port"] = st.number_input("Port", value=postgres.get("port", 5432), min_value=1, max_value=65535)
        postgres["database"] = st.text_input("Database", postgres.get("database", "demo_db"))
        postgres["username"] = st.text_input("Username", postgres.get("username", "demo_user"))
    
    with col2:
        postgres["password"] = st.text_input("Password", postgres.get("password", "demo_pass"), type="password")
        postgres["ssl_mode"] = st.selectbox("SSL Mode", ["prefer", "require", "verify-ca", "verify-full"],
                                            index=["prefer", "require", "verify-ca", "verify-full"].index(postgres.get("ssl_mode", "prefer")))
        postgres["connection_pool_size"] = st.number_input("Connection Pool Size", value=postgres.get("connection_pool_size", 10), min_value=1, max_value=100)
        postgres["timeout"] = st.number_input("Timeout (seconds)", value=postgres.get("timeout", 30), min_value=1, max_value=300)
    
    settings["postgres"] = postgres
    return settings

def render_aws_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render AWS configuration section"""
    st.subheader("☁️ AWS Configuration")
    
    aws = settings.get("aws", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        aws["region"] = st.text_input("AWS Region", aws.get("region", "us-east-1"))
        aws["profile"] = st.text_input("AWS Profile", aws.get("profile", "default"))
        aws["access_key_id"] = st.text_input("Access Key ID", aws.get("access_key_id", ""), type="password")
    
    with col2:
        aws["secret_access_key"] = st.text_input("Secret Access Key", aws.get("secret_access_key", ""), type="password")
        aws["session_token"] = st.text_input("Session Token", aws.get("session_token", ""), type="password")
        aws["kms_key_id"] = st.text_input("KMS Key ID", aws.get("kms_key_id", ""))
    
    settings["aws"] = aws
    return settings

def render_mlflow_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render MLflow configuration section"""
    st.subheader("📊 MLflow Configuration")
    
    integrations = settings.get("integrations", {})
    mlflow = integrations.get("mlflow", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        mlflow["enabled"] = st.checkbox("Enable MLflow", mlflow.get("enabled", True))
        mlflow["tracking_uri"] = st.text_input("Tracking URI", mlflow.get("tracking_uri", "file://./mlruns"))
        mlflow["experiment_name"] = st.text_input("Experiment Name", mlflow.get("experiment_name", "demo-experiment"))
    
    with col2:
        st.write("**Environment Settings**")
        st.info("Configure MLflow for different environments")
    
    integrations["mlflow"] = mlflow
    settings["integrations"] = integrations
    return settings

def handle_file_upload():
    """Handle file uploads to tmp_input folder"""
    st.subheader("📁 File Upload")
    
    uploaded_files = st.file_uploader(
        "Upload files to tmp_input folder", 
        accept_multiple_files=True,
        help="Upload any files - they will be saved to the tmp_input directory"
    )
    
    if uploaded_files:
        # Ensure tmp_input directory exists
        tmp_input_path = Path("../../tmp_input")
        tmp_input_path.mkdir(exist_ok=True)
        
        for uploaded_file in uploaded_files:
            try:
                # Save the file
                file_path = tmp_input_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ Saved: {uploaded_file.name} ({len(uploaded_file.getbuffer())} bytes)")
                st.info(f"📍 Location: {file_path.absolute()}")
                
            except Exception as e:
                st.error(f"❌ Error saving {uploaded_file.name}: {e}")
    
    # Show existing files in tmp_input
    tmp_input_path = Path("../../tmp_input")
    if tmp_input_path.exists():
        files = list(tmp_input_path.glob("*"))
        if files:
            st.subheader("📂 Files in tmp_input")
            for file_path in files:
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    st.text(f"📄 {file_path.name} ({file_size} bytes)")

def main():
    """Main application"""
    st.title("⚙️ Settings Configurator")
    st.markdown("Configure your demo app settings easily with this interface.")
    
    # Load current settings
    settings = load_settings()
    
    # Sidebar for navigation
    st.sidebar.title("Configuration Sections")
    
    sections = {
        "PostgreSQL": render_postgres_config,
        "AWS": render_aws_config,
        "MLflow": render_mlflow_config,
        "File Upload": lambda settings: handle_file_upload() or settings
    }
    
    selected_section = st.sidebar.selectbox("Select Section", list(sections.keys()))
    
    # Render selected section
    settings = sections[selected_section](settings)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            if save_settings(settings):
                st.success("Settings saved successfully!")
            else:
                st.error("Failed to save settings")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            settings = create_default_settings()
            st.success("Settings reset to defaults!")
            st.rerun()
    
    with col3:
        if st.button("📋 Show JSON"):
            st.json(settings)
    
    # Show current settings file location
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Settings File:**")
    settings_file = "../settings.yaml"
    if os.path.exists(settings_file):
        st.sidebar.success(f"✅ {settings_file}")
    else:
        st.sidebar.warning(f"⚠️ {settings_file} (will be created)")
    
    # Show file size if exists
    if os.path.exists(settings_file):
        size = os.path.getsize(settings_file)
        st.sidebar.markdown(f"**File Size:** {size} bytes")
    

if __name__ == "__main__":
    main()
