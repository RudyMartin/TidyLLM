"""
Dashboard Helper Utilities
==========================

Utility functions and helpers for the AI Dropzone Manager dashboard.
Includes data formatting, file operations, and common UI components.
"""

import streamlit as st
import polars as pl
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import base64

class DashboardUtils:
    """Utility functions for dashboard operations."""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def get_status_color(status: str) -> str:
        """Get color code for status indicators."""
        colors = {
            "active": "#22c55e",
            "processing": "#f59e0b", 
            "completed": "#3b82f6",
            "failed": "#ef4444",
            "idle": "#6b7280",
            "queued": "#8b5cf6"
        }
        return colors.get(status.lower(), "#6b7280")
    
    @staticmethod
    def get_status_icon(status: str) -> str:
        """Get emoji icon for status."""
        icons = {
            "active": "🟢",
            "processing": "🟡",
            "completed": "✅",
            "failed": "❌", 
            "idle": "⚪",
            "queued": "🔵"
        }
        return icons.get(status.lower(), "⚪")
    
    @staticmethod
    def create_download_link(data: Union[str, bytes, pl.DataFrame], 
                           filename: str, 
                           text: str = "Download") -> str:
        """Create a download link for data."""
        if isinstance(data, pl.DataFrame):
            data = data.to_csv()
            
        if isinstance(data, str):
            data = data.encode()
            
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
        return href

class ConfigManager:
    """Configuration management for dashboard."""
    
    def __init__(self, config_path: str = "tidyllm/web/config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Failed to load config: {e}")
                
        return self.get_default_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save config: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "dashboard": {
                "refresh_interval": 30,
                "auto_refresh": True,
                "theme": "light",
                "items_per_page": 10
            },
            "monitoring": {
                "enable_alerts": True,
                "alert_thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "queue_length": 20,
                    "error_rate": 10
                }
            },
            "drop_zones": {
                "base_path": "tidyllm/drop_zones",
                "supported_formats": [".pdf", ".txt", ".docx"],
                "max_file_size": 100  # MB
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        self.save_config()

class AlertManager:
    """Alert and notification management."""
    
    def __init__(self):
        self.alerts = []
        
    def add_alert(self, 
                  level: str, 
                  title: str, 
                  message: str, 
                  component: str = "System"):
        """Add an alert to the queue."""
        alert = {
            "id": len(self.alerts) + 1,
            "level": level,
            "title": title, 
            "message": message,
            "component": component,
            "timestamp": datetime.now(),
            "acknowledged": False
        }
        self.alerts.append(alert)
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active (unacknowledged) alerts."""
        return [alert for alert in self.alerts if not alert["acknowledged"]]
    
    def acknowledge_alert(self, alert_id: int):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                break
    
    def render_alerts(self):
        """Render alerts in the UI."""
        active_alerts = self.get_active_alerts()
        
        if active_alerts:
            st.sidebar.markdown("### 🚨 Active Alerts")
            
            for alert in active_alerts[-3:]:  # Show last 3 alerts
                level_color = {
                    "error": "#ef4444",
                    "warning": "#f59e0b", 
                    "info": "#3b82f6"
                }.get(alert["level"], "#6b7280")
                
                with st.sidebar.container():
                    st.markdown(f"""
                    <div style="border-left: 4px solid {level_color}; padding: 8px; margin: 4px 0;">
                        <strong>{alert["title"]}</strong><br>
                        <small>{alert["message"]}</small><br>
                        <small style="color: #6b7280;">{alert["timestamp"].strftime('%H:%M')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Acknowledge", key=f"ack_{alert['id']}"):
                        self.acknowledge_alert(alert["id"])
                        st.experimental_rerun()

class DataExporter:
    """Data export utilities for dashboard."""
    
    @staticmethod
    def export_processing_history(history_df: pl.DataFrame) -> str:
        """Export processing history as CSV."""
        return history_df.to_csv()
    
    @staticmethod  
    def export_system_metrics(metrics: Dict[str, Any]) -> str:
        """Export system metrics as JSON."""
        return json.dumps(metrics, indent=2, default=str)
    
    @staticmethod
    def export_worker_status(workers: List[Dict[str, Any]]) -> str:
        """Export worker status as JSON."""
        return json.dumps(workers, indent=2, default=str)
    
    @staticmethod
    def create_system_report(
        metrics: Dict[str, Any],
        workers: List[Dict[str, Any]], 
        processing_history: pl.DataFrame
    ) -> str:
        """Create comprehensive system report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_metrics": metrics,
            "worker_status": workers,
            "processing_summary": {
                "total_processed": processing_history.height,
                "success_rate": (processing_history.filter(pl.col('status') == 'completed').height / processing_history.height) * 100,
                "avg_processing_time": processing_history.select(pl.col('processing_time').str.replace('s', '').cast(pl.Float64)).mean().item()
            }
        }
        
        return json.dumps(report, indent=2, default=str)

class FileOperations:
    """File operations for drop zone management."""
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get detailed file information."""
        if not file_path.exists():
            return None
            
        stats = file_path.stat()
        
        return {
            "name": file_path.name,
            "size": stats.st_size,
            "size_formatted": DashboardUtils.format_file_size(stats.st_size),
            "modified": datetime.fromtimestamp(stats.st_mtime),
            "extension": file_path.suffix.lower(),
            "is_supported": file_path.suffix.lower() in ['.pdf', '.txt', '.docx']
        }
    
    @staticmethod
    def move_file(source: Path, destination: Path) -> bool:
        """Move file from source to destination."""
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            return True
        except Exception as e:
            st.error(f"Failed to move file: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path: Path) -> bool:
        """Delete a file."""
        try:
            file_path.unlink()
            return True
        except Exception as e:
            st.error(f"Failed to delete file: {e}")
            return False

# Global instances for easy access
config_manager = ConfigManager()
alert_manager = AlertManager()