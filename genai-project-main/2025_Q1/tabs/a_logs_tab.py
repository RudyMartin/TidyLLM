import streamlit as st
import json
import logging
from collections import Counter

from core.s3_utils import list_s3_files_with_extension, cleanup_log_files
from core.logging_helper import list_log_files, read_log_file, cleanup_logs, get_logger

# --- Logging Setup ---
logger = get_logger("logs_tab")
logger.setLevel(logging.INFO)

# --- Main Logs Tab Logic ---
def render():
    st.header("🧾 Log Viewer")

    # Lazy import of get_tab_context to avoid circular imports
    from core.init_context import get_tab_context

    # Unified context
    tab_context = get_tab_context()
    clients = tab_context.clients
    config = tab_context.config

    # Use dictionary access for S3 client
    s3_client = clients.get("s3")
    if s3_client is None:
        st.error("S3 client not found.")
        return

    # Retrieve configuration values (ensure consistent key usage)
    log_bucket = config.get("bucket_name", "")
    log_prefix = config.get("log_directory", config.get("log_folder", "dev/logs"))

    # --- Log Cleanup UI ---
    st.subheader("Log Cleanup")
    dry_run = st.checkbox("Perform Dry Run", value=False)
    confirm = st.checkbox("Delete log files permanently", value=False)
    
    if confirm or dry_run:
        cleanup_log_files(s3_client, config, dry_run=dry_run)
        st.success("Cleanup initiated.")

    # --- List log files from S3 ---
    log_files = list_s3_files_with_extension(log_bucket, config.get("log_folder", "dev/logs"), "log")
    
    if not log_files:
        st.warning("⚠️ No log files found.")
    else:
        st.write(f"✅ Found {len(log_files)} log files.")
        st.write(log_files[:10])  # Limit display to first 10 results

    # --- Log File Selection ---
    selected_log = st.text_input("Enter the log file name to view (or leave blank to skip):")
    
    if selected_log:
        try:
            raw = read_log_file(selected_log)
            log_content = raw if isinstance(raw, str) else str(raw)

            # Count log entries for INFO, WARNING, ERROR levels
            entry_counts = Counter()
            for line in log_content.splitlines():
                if "[INFO]" in line:
                    entry_counts["INFO"] += 1
                elif "[WARNING]" in line:
                    entry_counts["WARNING"] += 1
                elif "[ERROR]" in line:
                    entry_counts["ERROR"] += 1

            st.subheader("🔢 Entry Counts")
            st.write(f"INFO: {entry_counts.get('INFO', 0)}")
            st.write(f"WARNING: {entry_counts.get('WARNING', 0)}")
            st.write(f"ERROR: {entry_counts.get('ERROR', 0)}")

            st.subheader("📜 Log Content")
            st.text_area("Log File Content", log_content, height=300)

        except Exception as e:
            st.error(f"❌ Error loading log file: {e}")
            logger.error(f"Error reading log file {selected_log}: {e}")

# Main function call
if __name__ == "__main__":
    render()
