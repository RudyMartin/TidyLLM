"""
Drop Zone Monitor Component
==========================

Reusable Streamlit component for monitoring drop zone status and activity.
Provides real-time file detection, processing status, and interactive controls.
"""

import streamlit as st
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class DropzoneMonitor:
    """Component for monitoring individual drop zones."""
    
    def __init__(self, zone_path: str, display_name: str, bracket_command: str):
        self.zone_path = Path(zone_path)
        self.display_name = display_name
        self.bracket_command = bracket_command
        
    def render(self) -> Dict[str, Any]:
        """Render the drop zone monitor component."""
        
        # Get current status
        files = self._get_files()
        file_count = len(files)
        
        # Status indicator
        if file_count > 0:
            status = "🟢 Active"
            status_color = "#22c55e"
        else:
            status = "⚪ Empty"
            status_color = "#6b7280"
            
        # Header with status
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### {self.display_name}")
            
        with col2:
            st.markdown(f"""
            <div style="text-align: center;">
                <span style="color: {status_color}; font-weight: bold;">{status}</span><br>
                <small>{file_count} files</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div style="text-align: center;">
                <code>{self.bracket_command}</code>
            </div>
            """, unsafe_allow_html=True)
        
        # File listing
        if files:
            st.markdown("**Files in drop zone:**")
            
            for file_path in files:
                with st.container():
                    col_file, col_size, col_actions = st.columns([3, 1, 2])
                    
                    with col_file:
                        st.write(f"📄 {file_path.name}")
                        
                    with col_size:
                        size_kb = file_path.stat().st_size / 1024
                        st.write(f"{size_kb:.1f} KB")
                        
                    with col_actions:
                        if st.button(f"🚀", key=f"process_{file_path.name}", help="Process Now"):
                            return {
                                "action": "process",
                                "file": file_path.name,
                                "command": self.bracket_command
                            }
        else:
            st.info(f"Drop files here to trigger {self.bracket_command}")
            
        # Directory info
        with st.expander("ℹ️ Directory Info"):
            st.code(f"Path: {self.zone_path}")
            st.write(f"Monitoring: {self.bracket_command}")
            
            if self.zone_path.exists():
                stats = self.zone_path.stat()
                st.write(f"Last modified: {datetime.fromtimestamp(stats.st_mtime)}")
            
        return {"action": None}
        
    def _get_files(self) -> List[Path]:
        """Get list of files in the drop zone."""
        if not self.zone_path.exists():
            return []
            
        files = []
        for pattern in ["*.pdf", "*.txt", "*.docx"]:
            files.extend(self.zone_path.glob(pattern))
            
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)


class ProcessingStatusMonitor:
    """Component for monitoring processing status across all zones."""
    
    def __init__(self, base_path: str = "tidyllm/drop_zones"):
        self.base_path = Path(base_path)
        
    def render(self):
        """Render processing status overview."""
        
        st.markdown("### 📊 Processing Overview")
        
        # Get status counts
        status = self._get_processing_status()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📥 Queued", status["queued"])
            
        with col2:
            st.metric("⚙️ Processing", status["processing"]) 
            
        with col3:
            st.metric("✅ Completed", status["completed"])
            
        with col4:
            st.metric("❌ Failed", status["failed"])
            
        # Processing queue
        if status["processing"] > 0 or status["queued"] > 0:
            st.markdown("**Current Processing Queue:**")
            
            # Simulate processing queue (replace with real data)
            queue_data = [
                {"file": "mvr_report.pdf", "command": "[Process MVR]", "status": "processing", "progress": 75},
                {"file": "contract_v2.pdf", "command": "[Contract Review]", "status": "queued", "progress": 0},
                {"file": "financial_q3.pdf", "command": "[Financial Analysis]", "status": "queued", "progress": 0}
            ]
            
            for item in queue_data:
                col_file, col_cmd, col_progress = st.columns([2, 2, 1])
                
                with col_file:
                    st.write(f"📄 {item['file']}")
                    
                with col_cmd:
                    st.write(f"`{item['command']}`")
                    
                with col_progress:
                    if item['status'] == 'processing':
                        st.progress(item['progress'] / 100)
                    else:
                        st.write("⏳ Queued")
        
        # Recent completions
        recent_completions = self._get_recent_completions()
        if recent_completions:
            st.markdown("**Recently Completed:**")
            
            for completion in recent_completions[:5]:
                col_file, col_time, col_result = st.columns([2, 1, 1])
                
                with col_file:
                    st.write(f"📄 {completion['file']}")
                    
                with col_time:
                    st.write(completion['completed_at'])
                    
                with col_result:
                    if completion['success']:
                        st.write("✅ Success")
                    else:
                        st.write("❌ Failed")
    
    def _get_processing_status(self) -> Dict[str, int]:
        """Get current processing status counts."""
        # In real implementation, this would query the AI Dropzone Manager
        return {
            "queued": len(list((self.base_path / "mvr_analysis").glob("*"))) if (self.base_path / "mvr_analysis").exists() else 0,
            "processing": len(list((self.base_path / "processing").glob("*"))) if (self.base_path / "processing").exists() else 1,
            "completed": len(list((self.base_path / "completed").glob("*"))) if (self.base_path / "completed").exists() else 15,
            "failed": len(list((self.base_path / "failed").glob("*"))) if (self.base_path / "failed").exists() else 2
        }
    
    def _get_recent_completions(self) -> List[Dict[str, Any]]:
        """Get recent completion history."""
        # Simulate recent completions (replace with real data from processing logs)
        return [
            {"file": "compliance_doc.pdf", "completed_at": "10:45", "success": True},
            {"file": "audit_report.pdf", "completed_at": "10:32", "success": True},
            {"file": "risk_assessment.pdf", "completed_at": "10:18", "success": False},
            {"file": "policy_update.pdf", "completed_at": "09:55", "success": True}
        ]


class BracketCommandPanel:
    """Component for executing bracket commands manually."""
    
    def __init__(self):
        self.commands = self._get_available_commands()
        
    def render(self) -> Optional[Dict[str, Any]]:
        """Render bracket command execution panel."""
        
        st.markdown("### 🎯 Execute Bracket Command")
        
        # Command selection
        selected_command = st.selectbox(
            "Select Command",
            options=[cmd["command"] for cmd in self.commands],
            format_func=lambda x: f"{x} ({self._get_command_priority(x)})"
        )
        
        # File upload or selection
        upload_tab, select_tab = st.tabs(["📁 Upload File", "🗂️ Select from Drop Zone"])
        
        with upload_tab:
            uploaded_file = st.file_uploader(
                "Choose file to process",
                type=['pdf', 'txt', 'docx'],
                help="Upload a document to process with the selected bracket command"
            )
            
            if uploaded_file and selected_command:
                if st.button("🚀 Process Uploaded File", type="primary"):
                    return {
                        "action": "process_upload",
                        "file": uploaded_file.name,
                        "command": selected_command,
                        "file_data": uploaded_file
                    }
        
        with select_tab:
            # List files from drop zones
            zone_files = self._get_zone_files()
            
            if zone_files:
                selected_file = st.selectbox(
                    "Select file from drop zones",
                    options=[f"{zone}: {file}" for zone, files in zone_files.items() for file in files]
                )
                
                if selected_file and st.button("🚀 Process Selected File", type="primary"):
                    zone, file = selected_file.split(": ", 1)
                    return {
                        "action": "process_existing",
                        "file": file,
                        "zone": zone,
                        "command": selected_command
                    }
            else:
                st.info("No files found in drop zones")
        
        return None
    
    def _get_available_commands(self) -> List[Dict[str, str]]:
        """Get list of available bracket commands."""
        return [
            {"command": "[Process MVR]", "priority": "High", "category": "Compliance"},
            {"command": "[Financial Analysis]", "priority": "Normal", "category": "Analysis"},
            {"command": "[Contract Review]", "priority": "High", "category": "Legal"},
            {"command": "[Quality Check]", "priority": "Normal", "category": "QA"},
            {"command": "[Peer Review]", "priority": "Critical", "category": "Review"},
            {"command": "[Hybrid Analysis]", "priority": "High", "category": "Advanced"}
        ]
    
    def _get_command_priority(self, command: str) -> str:
        """Get priority level for a command."""
        for cmd in self.commands:
            if cmd["command"] == command:
                return cmd["priority"]
        return "Normal"
    
    def _get_zone_files(self) -> Dict[str, List[str]]:
        """Get files available in drop zones."""
        zones = {
            "MVR Analysis": "tidyllm/drop_zones/mvr_analysis",
            "Financial Analysis": "tidyllm/drop_zones/financial_analysis",
            "Contract Review": "tidyllm/drop_zones/contract_review"
        }
        
        zone_files = {}
        for zone_name, zone_path in zones.items():
            path = Path(zone_path)
            if path.exists():
                files = []
                for pattern in ["*.pdf", "*.txt", "*.docx"]:
                    files.extend([f.name for f in path.glob(pattern)])
                if files:
                    zone_files[zone_name] = files
                    
        return zone_files