"""
Model Discovery Scheduler
=========================

Automatically runs model discovery on a schedule to keep configurations
up-to-date with AWS Bedrock model changes.

Features:
- Scheduled discovery (daily/weekly/custom)
- Notification system for changes
- Safe updates with rollback capability
- Integration with existing systems
"""

import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional
import json
import os
from pathlib import Path

from .dynamic_model_discovery import get_model_discovery, auto_update_embeddings_config

logger = logging.getLogger(__name__)

class ModelDiscoveryScheduler:
    """Handles scheduled model discovery and updates"""
    
    def __init__(self, 
                 config_path: str = None,
                 discovery_interval: str = "daily",
                 notification_callback: Callable = None,
                 enable_auto_updates: bool = True):
        """
        Initialize model discovery scheduler
        
        Args:
            config_path: Path to embeddings config file
            discovery_interval: "daily", "weekly", or cron-like schedule
            notification_callback: Function to call when models change
            enable_auto_updates: Whether to automatically update config files
        """
        self.config_path = config_path
        self.discovery_interval = discovery_interval
        self.notification_callback = notification_callback
        self.enable_auto_updates = enable_auto_updates
        
        self.scheduler_thread = None
        self.running = False
        self.last_run = None
        self.last_results = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for scheduler"""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "model_discovery.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def start_scheduler(self):
        """Start the background scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        logger.info(f"Starting model discovery scheduler with {self.discovery_interval} interval")
        
        # Schedule discovery
        if self.discovery_interval == "daily":
            schedule.every().day.at("02:00").do(self._run_discovery)
        elif self.discovery_interval == "weekly":
            schedule.every().sunday.at("02:00").do(self._run_discovery)
        elif self.discovery_interval == "hourly":  # For testing
            schedule.every().hour.do(self._run_discovery)
        else:
            # Default to daily
            schedule.every().day.at("02:00").do(self._run_discovery)
        
        # Start scheduler thread
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # Run initial discovery
        self._run_discovery()
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        logger.info("Stopping model discovery scheduler")
        self.running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
    
    def _run_scheduler(self):
        """Background scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _run_discovery(self):
        """Run model discovery and update configuration"""
        try:
            logger.info("Running scheduled model discovery...")
            
            # Get model discovery instance
            discovery = get_model_discovery()
            
            # Discover models
            models = discovery.discover_models(force_refresh=True)
            
            # Generate compatibility report
            report = discovery.get_model_compatibility_report()
            
            # Update configuration if auto-updates enabled
            updates = None
            if self.enable_auto_updates:
                updates = discovery.update_embeddings_config(self.config_path)
            
            # Store results
            self.last_run = datetime.now()
            self.last_results = {
                "models_discovered": len(models),
                "compatibility_report": report,
                "updates": updates,
                "timestamp": self.last_run.isoformat()
            }
            
            # Log results
            self._log_discovery_results()
            
            # Send notifications if changes detected
            if updates and (updates.get("new_models") or updates.get("deprecated_models")):
                self._send_notifications(updates)
            
            logger.info("Scheduled model discovery completed successfully")
            
        except Exception as e:
            logger.error(f"Scheduled model discovery failed: {e}")
            
            # Store error result
            self.last_results = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _log_discovery_results(self):
        """Log discovery results"""
        if not self.last_results:
            return
        
        results = self.last_results
        
        logger.info(f"Discovery Results:")
        logger.info(f"  Models discovered: {results.get('models_discovered', 0)}")
        
        if 'compatibility_report' in results:
            report = results['compatibility_report']
            logger.info(f"  Available models: {report.get('available_models', 0)}")
            logger.info(f"  Deprecated models: {report.get('deprecated_models', 0)}")
            logger.info(f"  New models: {report.get('new_models', 0)}")
        
        if 'updates' in results and results['updates']:
            updates = results['updates']
            if updates.get('new_models'):
                logger.info(f"  New models added: {len(updates['new_models'])}")
                for new_model in updates['new_models']:
                    logger.info(f"    - {new_model['model']['model_id']}")
            
            if updates.get('deprecated_models'):
                logger.info(f"  Models deprecated: {len(updates['deprecated_models'])}")
                for deprecated_id in updates['deprecated_models']:
                    logger.warning(f"    - {deprecated_id} is deprecated")
    
    def _send_notifications(self, updates: Dict[str, Any]):
        """Send notifications about model changes"""
        try:
            if self.notification_callback:
                self.notification_callback(updates)
            else:
                # Default notification - log important changes
                if updates.get("new_models"):
                    logger.info(f"🆕 NEW MODELS AVAILABLE:")
                    for new_model in updates["new_models"]:
                        model = new_model["model"]
                        logger.info(f"   • {model.model_id} ({model.native_dimension} dims)")
                
                if updates.get("deprecated_models"):
                    logger.warning(f"⚠️ MODELS DEPRECATED:")
                    for model_id in updates["deprecated_models"]:
                        logger.warning(f"   • {model_id}")
                        
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            "running": self.running,
            "discovery_interval": self.discovery_interval,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_results": self.last_results,
            "next_scheduled": self._get_next_scheduled_time(),
            "auto_updates_enabled": self.enable_auto_updates
        }
    
    def _get_next_scheduled_time(self) -> Optional[str]:
        """Get next scheduled run time"""
        try:
            jobs = schedule.get_jobs()
            if jobs:
                next_job = min(jobs, key=lambda job: job.next_run)
                return next_job.next_run.isoformat()
        except Exception:
            pass
        return None
    
    def force_discovery(self) -> Dict[str, Any]:
        """Force an immediate model discovery run"""
        logger.info("Forcing immediate model discovery...")
        self._run_discovery()
        return self.last_results or {}

# Global scheduler instance
_scheduler_instance = None

def get_model_scheduler(config_path: str = None, 
                       discovery_interval: str = "daily",
                       notification_callback: Callable = None) -> ModelDiscoveryScheduler:
    """Get global model discovery scheduler"""
    global _scheduler_instance
    
    if _scheduler_instance is None:
        _scheduler_instance = ModelDiscoveryScheduler(
            config_path=config_path,
            discovery_interval=discovery_interval,
            notification_callback=notification_callback
        )
    
    return _scheduler_instance

def start_auto_discovery(config_path: str = None, 
                        discovery_interval: str = "daily") -> ModelDiscoveryScheduler:
    """Start automatic model discovery with default settings"""
    scheduler = get_model_scheduler(config_path, discovery_interval)
    scheduler.start_scheduler()
    return scheduler

# Integration with existing notification systems
def slack_notification(updates: Dict[str, Any]):
    """Example: Send Slack notifications for model changes"""
    try:
        # This would integrate with your Slack webhook
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not webhook_url:
            return
        
        message = "🤖 *AWS Bedrock Model Updates*\n"
        
        if updates.get("new_models"):
            message += f"\n🆕 *New Models ({len(updates['new_models'])}):*\n"
            for new_model in updates["new_models"]:
                model = new_model["model"]
                message += f"• `{model.model_id}` - {model.native_dimension} dimensions\n"
        
        if updates.get("deprecated_models"):
            message += f"\n⚠️ *Deprecated Models ({len(updates['deprecated_models'])}):*\n"
            for model_id in updates["deprecated_models"]:
                message += f"• `{model_id}`\n"
        
        # Send to Slack (implement actual webhook call)
        logger.info(f"Would send to Slack: {message}")
        
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")

def email_notification(updates: Dict[str, Any]):
    """Example: Send email notifications for model changes"""
    try:
        # This would integrate with your email system
        email_recipients = os.getenv('MODEL_UPDATE_EMAIL_RECIPIENTS', '').split(',')
        if not email_recipients or not email_recipients[0]:
            return
        
        subject = f"AWS Bedrock Model Updates - {datetime.now().strftime('%Y-%m-%d')}"
        body = "AWS Bedrock has updated their available models:\n\n"
        
        if updates.get("new_models"):
            body += f"New Models Available ({len(updates['new_models'])}):\n"
            for new_model in updates["new_models"]:
                model = new_model["model"]
                body += f"- {model.model_id} ({model.native_dimension} dimensions)\n"
            body += "\n"
        
        if updates.get("deprecated_models"):
            body += f"Deprecated Models ({len(updates['deprecated_models'])}):\n"
            for model_id in updates["deprecated_models"]:
                body += f"- {model_id}\n"
        
        # Send email (implement actual email sending)
        logger.info(f"Would send email to {email_recipients}: {subject}")
        
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")