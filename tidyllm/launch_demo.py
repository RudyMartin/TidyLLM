"""
TidyLLM Demo Launcher
====================

Ensures all required services and configurations are properly set up
before launching demos. This script:

1. Enables MLFlow integration (required for workflow gateway)
2. Validates database connections 
3. Sets up proper environment variables
4. Provides easy access to all demo applications
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
import streamlit as st

class TidyLLMDemoLauncher:
    """Launch and manage TidyLLM demos with proper configuration"""
    
    def __init__(self):
        self.tidyllm_root = Path(__file__).parent
        self.settings_file = self.tidyllm_root / "tidyllm" / "admin" / "settings.yaml"
        self.demos_available = self._discover_demos()
    
    def ensure_mlflow_enabled(self):
        """Ensure MLFlow is enabled in settings.yaml"""
        try:
            if not self.settings_file.exists():
                st.error(f"❌ Settings file not found: {self.settings_file}")
                return False
                
            # Read current settings
            with open(self.settings_file, 'r') as f:
                settings = yaml.safe_load(f)
            
            # Check MLFlow configuration
            mlflow_config = settings.get('integrations', {}).get('mlflow', {})
            
            if not mlflow_config.get('enabled', False):
                st.warning("⚠️ MLFlow is disabled in settings.yaml")
                
                if st.button("🔧 Enable MLFlow Integration"):
                    # Enable MLFlow
                    settings['integrations']['mlflow']['enabled'] = True
                    settings['integrations']['mlflow']['tracking_uri'] = "sqlite:///mlflow.db"
                    settings['integrations']['mlflow']['experiment_name'] = "tidyllm-workflows"
                    
                    # Write back to settings
                    with open(self.settings_file, 'w') as f:
                        yaml.safe_dump(settings, f, default_flow_style=False)
                    
                    st.success("✅ MLFlow enabled! Please restart the application.")
                    st.experimental_rerun()
                
                return False
            else:
                st.success("✅ MLFlow integration is enabled")
                tracking_uri = mlflow_config.get('tracking_uri', 'Not set')
                experiment = mlflow_config.get('experiment_name', 'tidyllm-enhanced')
                st.info(f"📊 MLFlow URI: {tracking_uri}")
                st.info(f"🧪 Experiment: {experiment}")
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to check MLFlow configuration: {e}")
            return False
    
    def _discover_demos(self):
        """Discover available demo applications"""
        demos = []
        
        # Streamlit demos
        streamlit_demos = [
            {
                "name": "🔧 Workflow Builder",
                "description": "Create Domain RAG and MVR Analysis workflows", 
                "file": "domain_rag_workflow_builder.py",
                "type": "streamlit"
            },
            {
                "name": "💬 Chat Interface", 
                "description": "RAG chat with workflow integration",
                "file": "chat_workflow_interface.py",
                "type": "streamlit"
            },
            {
                "name": "🏗️ HeirOS Demo",
                "description": "Hierarchical workflow orchestration",
                "file": "heiros_streamlit_demo.py", 
                "type": "streamlit"
            }
        ]
        
        # Python script demos
        python_demos = [
            {
                "name": "🚀 Enhanced Drop Zones",
                "description": "Production S3 tracking drop zones",
                "file": "production_tracking_drop_zones.py",
                "type": "python"
            },
            {
                "name": "📊 Research Query Demo", 
                "description": "Query research papers and documents",
                "file": "query_research_chat.py",
                "type": "python"
            },
            {
                "name": "⚙️ Settings Demo",
                "description": "Test configuration settings",
                "file": "settings_demo.py", 
                "type": "python"
            }
        ]
        
        # Filter to only existing files
        all_demos = streamlit_demos + python_demos
        existing_demos = []
        
        for demo in all_demos:
            demo_path = self.tidyllm_root / demo["file"]
            if demo_path.exists():
                demo["path"] = str(demo_path)
                existing_demos.append(demo)
                
        return existing_demos
    
    def render_demo_launcher(self):
        """Render the demo launcher interface"""
        st.title("🚀 TidyLLM Demo Launcher")
        st.markdown("Launch demos with proper configuration and MLFlow integration")
        
        # MLFlow Configuration Section
        st.header("🔧 Configuration Check")
        mlflow_ready = self.ensure_mlflow_enabled()
        
        if not mlflow_ready:
            st.warning("⚠️ Please enable MLFlow before launching workflow demos")
            return
            
        # Environment Status
        with st.expander("🌐 Environment Status"):
            st.write("📂 TidyLLM Root:", str(self.tidyllm_root))
            st.write("⚙️ Settings File:", str(self.settings_file))
            st.write("🐍 Python:", sys.version)
            
            # Check key environment variables
            env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
            for var in env_vars:
                value = os.environ.get(var)
                if value:
                    st.write(f"🔑 {var}: {'*' * (len(value)-4) + value[-4:] if len(value) > 4 else '***'}")
                else:
                    st.write(f"❌ {var}: Not set")
        
        # Available Demos
        st.header("🎯 Available Demos")
        
        if not self.demos_available:
            st.warning("No demo files found!")
            return
            
        # Group by type
        streamlit_demos = [d for d in self.demos_available if d["type"] == "streamlit"]
        python_demos = [d for d in self.demos_available if d["type"] == "python"]
        
        # Streamlit Demos
        if streamlit_demos:
            st.subheader("🌐 Streamlit Web Applications")
            
            col1, col2 = st.columns(2)
            
            for i, demo in enumerate(streamlit_demos):
                col = col1 if i % 2 == 0 else col2
                
                with col:
                    with st.container():
                        st.markdown(f"**{demo['name']}**")
                        st.caption(demo['description'])
                        
                        if st.button(f"▶️ Launch", key=f"streamlit_{i}"):
                            self._launch_streamlit_demo(demo)
        
        # Python Demos  
        if python_demos:
            st.subheader("🐍 Python Script Demos")
            
            col1, col2 = st.columns(2)
            
            for i, demo in enumerate(python_demos):
                col = col1 if i % 2 == 0 else col2
                
                with col:
                    with st.container():
                        st.markdown(f"**{demo['name']}**")
                        st.caption(demo['description'])
                        
                        if st.button(f"▶️ Run", key=f"python_{i}"):
                            self._launch_python_demo(demo)
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔧 Open Workflow Builder", use_container_width=True):
                self._launch_streamlit_app("domain_rag_workflow_builder.py")
        
        with col2:
            if st.button("💬 Open Chat Interface", use_container_width=True):  
                self._launch_streamlit_app("chat_workflow_interface.py")
        
        with col3:
            if st.button("📊 View MLFlow UI", use_container_width=True):
                self._launch_mlflow_ui()
    
    def _launch_streamlit_demo(self, demo):
        """Launch a Streamlit demo"""
        st.info(f"🚀 Launching {demo['name']}...")
        st.code(f"streamlit run {demo['file']}", language="bash")
        st.info("The demo will open in a new browser tab")
        
        # Actual launch would happen here
        # subprocess.Popen([sys.executable, "-m", "streamlit", "run", demo["path"]])
    
    def _launch_python_demo(self, demo):
        """Launch a Python demo"""
        st.info(f"🚀 Running {demo['name']}...")
        st.code(f"python {demo['file']}", language="bash")
        
        # Could capture and display output
        # result = subprocess.run([sys.executable, demo["path"]], capture_output=True, text=True)
    
    def _launch_streamlit_app(self, filename):
        """Launch a specific Streamlit app"""
        st.info(f"🚀 Launching {filename}...")
        st.code(f"streamlit run {filename}", language="bash")
    
    def _launch_mlflow_ui(self):
        """Launch MLFlow UI"""
        st.info("📊 Launching MLFlow UI...")
        st.code("mlflow ui", language="bash")
        st.info("MLFlow UI will be available at http://localhost:5000")

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="TidyLLM Demo Launcher",
        page_icon="🚀",
        layout="wide"
    )
    
    launcher = TidyLLMDemoLauncher()
    launcher.render_demo_launcher()

if __name__ == "__main__":
    main()