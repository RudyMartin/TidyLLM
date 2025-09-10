"""
TidyLLM Onboarding Sidebar Component
====================================

Navigation sidebar for the onboarding system.
"""

import streamlit as st

def render_sidebar():
    """Render the navigation sidebar."""
    
    st.sidebar.title("🏢 TidyLLM Onboarding")
    st.sidebar.markdown("---")
    
    # Navigation menu
    pages = [
        "Connection Config",
        "System Setup",
        "Chat Test", 
        "Knowledge Management",
        "Workflows",
        "Test Workflow",
        "Dashboard"
    ]
    
    selected_page = st.sidebar.selectbox(
        "Select Section",
        pages,
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("System Status")
    
    # Check if session manager is available
    if hasattr(st.session_state, 'session_manager') and st.session_state.session_manager:
        st.sidebar.success("✅ Session Manager Active")
    else:
        st.sidebar.error("❌ Session Manager Not Available")
    
    # Check if gateways are available - CRITICAL DEPENDENCY
    if hasattr(st.session_state, 'gateways') and st.session_state.gateways:
        # Count successful gateways
        successful_gateways = sum(1 for gw in st.session_state.gateways.values() if gw is not None)
        total_gateways = len(st.session_state.gateways)
        
        if successful_gateways == total_gateways:
            st.sidebar.success(f"✅ All {total_gateways} Core Gateways Ready")
        elif successful_gateways > 0:
            st.sidebar.warning(f"⚠️ {successful_gateways}/{total_gateways} Core Gateways Ready")
            
            # Show which gateways are failing
            st.sidebar.markdown("**Core Gateways:**")
            for name, gateway in st.session_state.gateways.items():
                if gateway is None:
                    st.sidebar.error(f"❌ {name}")
                else:
                    st.sidebar.success(f"✅ {name}")
        else:
            st.sidebar.error("❌ No Core Gateways Ready")
    else:
        st.sidebar.error("🚨 CRITICAL: Core Gateways Not Initialized")
        st.sidebar.markdown("**AWS connection required!**")
        st.sidebar.markdown("Configure connections first →")
    
    # Check services separately
    if hasattr(st.session_state, 'services') and st.session_state.services:
        st.sidebar.markdown("**Utility Services:**")
        for name, service in st.session_state.services.items():
            if service is None:
                st.sidebar.error(f"❌ {name}")
            else:
                st.sidebar.success(f"✅ {name}")
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.subheader("Quick Actions")
    
    if st.sidebar.button("🔄 Refresh System"):
        # Force refresh gateways and services
        from onboarding.core.session_manager import SessionManager
        st.session_state.gateways = SessionManager.get_gateways()
        st.session_state.services = SessionManager.get_services()
        st.rerun()
    
    if st.sidebar.button("🧪 Run Tests"):
        st.session_state.run_tests = True
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Help section
    st.sidebar.subheader("Help")
    st.sidebar.markdown("""
    **Getting Started:**
    1. Configure connections in Connection Config
    2. Test AI models in Chat Test
    3. Set up knowledge domains
    4. Configure workflows
    5. Test end-to-end processing
    6. Monitor system in Dashboard
    """)
    
    return selected_page
