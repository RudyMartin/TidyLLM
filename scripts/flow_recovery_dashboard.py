#!/usr/bin/env python3
"""
Flow Recovery Dashboard - File Purgatory Management Interface
============================================================

Interactive dashboard for monitoring and recovering stuck files in drop zone flows.
Provides real-time monitoring, manual recovery tools, and system health insights.

Features:
- Real-time stalled file detection
- Manual recovery actions  
- Flow health monitoring
- Recovery statistics and reporting
- Emergency intervention tools

Usage:
    python scripts/flow_recovery_dashboard.py
    python scripts/flow_recovery_dashboard.py --monitor-only
    python scripts/flow_recovery_dashboard.py --recover-file <submission_id>
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tidyllm.infrastructure.workers.flow_recovery_worker import (
        FlowRecoveryWorker,
        RecoveryAction,
        RecoveryRequest,
        StalledFile,
        FileStatus
    )
    RECOVERY_WORKER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Recovery worker not available: {e}")
    RECOVERY_WORKER_AVAILABLE = False


class FlowRecoveryDashboard:
    """Interactive dashboard for flow recovery management."""
    
    def __init__(self, drop_zones_path: str = None):
        """Initialize dashboard."""
        self.drop_zones_path = drop_zones_path or str(project_root / "drop_zones")
        self.recovery_worker = None
        self.monitoring_active = False
        
        print("🩺 Flow Recovery Dashboard")
        print("=" * 50)
    
    async def start_recovery_worker(self) -> bool:
        """Start the recovery worker."""
        if not RECOVERY_WORKER_AVAILABLE:
            print("❌ Recovery worker not available")
            return False
        
        try:
            self.recovery_worker = FlowRecoveryWorker(
                drop_zones_path=self.drop_zones_path,
                stale_threshold_minutes=30,
                max_auto_retries=3,
                monitoring_interval=60.0
            )
            
            await self.recovery_worker.initialize()
            await self.recovery_worker.start()
            
            print("✅ Recovery worker started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recovery worker: {e}")
            return False
    
    async def stop_recovery_worker(self) -> None:
        """Stop the recovery worker."""
        if self.recovery_worker:
            await self.recovery_worker.stop()
            print("✅ Recovery worker stopped")
    
    async def show_dashboard(self) -> None:
        """Show interactive dashboard."""
        while True:
            await self._clear_screen()
            await self._show_header()
            await self._show_stalled_files()
            await self._show_recovery_stats()
            await self._show_menu()
            
            choice = input("\nEnter your choice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == '1':
                await self._monitor_flows()
            elif choice == '2':
                await self._manual_recovery_menu()
            elif choice == '3':
                await self._show_detailed_report()
            elif choice == '4':
                await self._emergency_tools_menu()
            elif choice == '5':
                await self._export_reports()
            elif choice == 'r':
                continue  # Refresh
            else:
                print("Invalid choice. Press Enter to continue...")
                input()
    
    async def _clear_screen(self) -> None:
        """Clear terminal screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    async def _show_header(self) -> None:
        """Show dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("🩺 TIDYLLM FLOW RECOVERY DASHBOARD")
        print("=" * 60)
        print(f"Drop Zones Path: {self.drop_zones_path}")
        print(f"Current Time: {now}")
        
        if self.recovery_worker:
            stats = self.recovery_worker.get_recovery_statistics()
            monitoring_status = "🟢 ACTIVE" if stats["monitoring_active"] else "🔴 INACTIVE"
            print(f"Recovery Worker: ✅ RUNNING | Monitoring: {monitoring_status}")
        else:
            print("Recovery Worker: ❌ NOT STARTED")
        
        print("=" * 60)
    
    async def _show_stalled_files(self) -> None:
        """Show current stalled files."""
        print("\n📋 STALLED FILES SUMMARY")
        print("-" * 40)
        
        if not self.recovery_worker:
            print("❌ Recovery worker not available")
            return
        
        stalled_files = self.recovery_worker.get_stalled_files_report()
        
        if not stalled_files:
            print("✅ No stalled files detected")
            return
        
        print(f"🚨 {len(stalled_files)} stalled files detected:")
        print()
        
        for i, file_info in enumerate(stalled_files[:10], 1):  # Show first 10
            submission_id = file_info["submission_id"][:20] + "..."
            status = file_info["status"]
            stage = file_info["current_stage"]
            stall_time = file_info["stall_duration_minutes"]
            retry_count = file_info["retry_count"]
            
            print(f"{i:2d}. {submission_id} | {status:12} | {stage:10} | {stall_time:6.1f}min | {retry_count} retries")
        
        if len(stalled_files) > 10:
            print(f"    ... and {len(stalled_files) - 10} more files")
    
    async def _show_recovery_stats(self) -> None:
        """Show recovery statistics."""
        print("\n📊 RECOVERY STATISTICS")
        print("-" * 40)
        
        if not self.recovery_worker:
            print("❌ Recovery worker not available")
            return
        
        stats = self.recovery_worker.get_recovery_statistics()
        
        print(f"Files Monitored: {stats['recovery_stats']['files_monitored']}")
        print(f"Stalled Files: {stats['recovery_stats']['stalled_files_detected']}")
        print(f"Successful Recoveries: {stats['recovery_stats']['successful_recoveries']}")
        print(f"Failed Recoveries: {stats['recovery_stats']['failed_recoveries']}")
        print(f"Manual Interventions: {stats['recovery_stats']['manual_interventions']}")
        
        # Calculate success rate
        total_recoveries = stats['recovery_stats']['successful_recoveries'] + stats['recovery_stats']['failed_recoveries']
        if total_recoveries > 0:
            success_rate = (stats['recovery_stats']['successful_recoveries'] / total_recoveries) * 100
            print(f"Success Rate: {success_rate:.1f}%")
    
    async def _show_menu(self) -> None:
        """Show main menu options."""
        print("\n🎛️  DASHBOARD MENU")
        print("-" * 40)
        print("1. Monitor Flows (Real-time)")
        print("2. Manual Recovery")
        print("3. Detailed Report")
        print("4. Emergency Tools")
        print("5. Export Reports")
        print("R. Refresh")
        print("Q. Quit")
    
    async def _monitor_flows(self) -> None:
        """Real-time flow monitoring."""
        print("\n🔍 REAL-TIME FLOW MONITORING")
        print("Press Ctrl+C to stop monitoring")
        print("-" * 40)
        
        if not self.recovery_worker:
            print("❌ Recovery worker not available")
            input("Press Enter to continue...")
            return
        
        self.monitoring_active = True
        
        try:
            while self.monitoring_active:
                # Show current time
                now = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{now}] Monitoring flows...")
                
                # Get latest stalled files
                stalled_files = self.recovery_worker.get_stalled_files_report()
                
                if stalled_files:
                    print(f"🚨 {len(stalled_files)} stalled files detected:")
                    for file_info in stalled_files[:5]:  # Show first 5
                        submission_id = file_info["submission_id"][:15]
                        stage = file_info["current_stage"]
                        stall_time = file_info["stall_duration_minutes"]
                        print(f"  • {submission_id} in {stage} for {stall_time:.1f} minutes")
                else:
                    print("✅ No stalled files")
                
                # Show stats
                stats = self.recovery_worker.get_recovery_statistics()
                print(f"📊 Monitored: {stats['recovery_stats']['files_monitored']} | "
                      f"Recovered: {stats['recovery_stats']['successful_recoveries']} | "
                      f"Failed: {stats['recovery_stats']['failed_recoveries']}")
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            self.monitoring_active = False
            print("\n✅ Monitoring stopped")
        
        input("Press Enter to continue...")
    
    async def _manual_recovery_menu(self) -> None:
        """Manual recovery options menu."""
        while True:
            await self._clear_screen()
            print("🔧 MANUAL RECOVERY MENU")
            print("=" * 40)
            
            if not self.recovery_worker:
                print("❌ Recovery worker not available")
                input("Press Enter to continue...")
                return
            
            # Show stalled files
            stalled_files = self.recovery_worker.get_stalled_files_report()
            
            if not stalled_files:
                print("✅ No stalled files to recover")
                input("Press Enter to continue...")
                return
            
            print("Select a file to recover:")
            for i, file_info in enumerate(stalled_files, 1):
                submission_id = file_info["submission_id"][:30]
                status = file_info["status"]
                stage = file_info["current_stage"]
                print(f"{i:2d}. {submission_id} | {status} | {stage}")
            
            print(f"{len(stalled_files) + 1:2d}. Back to main menu")
            
            try:
                choice = input("\nEnter file number: ").strip()
                if choice == str(len(stalled_files) + 1):
                    break
                
                file_index = int(choice) - 1
                if 0 <= file_index < len(stalled_files):
                    file_info = stalled_files[file_index]
                    await self._recover_specific_file(file_info)
                else:
                    print("Invalid selection")
                    input("Press Enter to continue...")
                    
            except ValueError:
                print("Invalid input")
                input("Press Enter to continue...")
    
    async def _recover_specific_file(self, file_info: Dict[str, Any]) -> None:
        """Recover a specific file."""
        submission_id = file_info["submission_id"]
        
        print(f"\n🔧 RECOVERY OPTIONS FOR: {submission_id[:30]}")
        print("-" * 50)
        print(f"Current Status: {file_info['status']}")
        print(f"Current Stage: {file_info['current_stage']}")
        print(f"Stall Duration: {file_info['stall_duration_minutes']:.1f} minutes")
        print(f"Retry Count: {file_info['retry_count']}")
        
        if file_info['error_messages']:
            print(f"Errors: {', '.join(file_info['error_messages'][:2])}")
        
        print("\nRecovery Actions:")
        print("1. Retry Current Stage")
        print("2. Restart from Beginning")
        print("3. Skip to Next Stage")
        print("4. Manual Intervention")
        print("5. Quarantine File")
        print("6. Cancel")
        
        choice = input("\nSelect recovery action: ").strip()
        
        action_map = {
            "1": RecoveryAction.RETRY_CURRENT_STAGE,
            "2": RecoveryAction.RESTART_FROM_BEGINNING,
            "3": RecoveryAction.SKIP_TO_NEXT_STAGE,
            "4": RecoveryAction.MANUAL_INTERVENTION,
            "5": RecoveryAction.QUARANTINE,
        }
        
        if choice == "6":
            return
        
        action = action_map.get(choice)
        if not action:
            print("Invalid choice")
            input("Press Enter to continue...")
            return
        
        print(f"\n🚀 Executing recovery action: {action.value}")
        
        try:
            # Submit recovery task
            task_id = await self.recovery_worker.recover_file(
                submission_id=submission_id,
                action=action,
                force_recovery=True
            )
            
            print(f"✅ Recovery task submitted: {task_id}")
            print("Monitoring task progress...")
            
            # Wait for task completion
            await self._wait_for_recovery_completion(task_id)
            
        except Exception as e:
            print(f"❌ Recovery failed: {e}")
        
        input("Press Enter to continue...")
    
    async def _wait_for_recovery_completion(self, task_id: str) -> None:
        """Wait for recovery task to complete."""
        max_wait = 60  # 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            task_status = self.recovery_worker.get_task_status(task_id)
            
            if task_status:
                if task_status.get("completed_at"):
                    # Find the completed task
                    for completed_task in self.recovery_worker.completed_tasks:
                        if completed_task.task_id == task_id:
                            result = completed_task.result
                            if completed_task.error_message:
                                print(f"❌ Task failed: {completed_task.error_message}")
                            elif result:
                                if result.success:
                                    print(f"✅ Recovery successful: {result.new_status.value}")
                                    if result.messages:
                                        for msg in result.messages:
                                            print(f"   • {msg}")
                                else:
                                    print(f"❌ Recovery failed: {result.new_status.value}")
                                    if result.errors:
                                        for error in result.errors:
                                            print(f"   • {error}")
                            return
                    
                    print("❌ Task completed but result not found")
                    return
            
            print(".", end="", flush=True)
            await asyncio.sleep(2)
        
        print(f"\n⏰ Recovery task timed out after {max_wait} seconds")
    
    async def _show_detailed_report(self) -> None:
        """Show detailed recovery report."""
        print("\n📄 DETAILED RECOVERY REPORT")
        print("=" * 60)
        
        if not self.recovery_worker:
            print("❌ Recovery worker not available")
            input("Press Enter to continue...")
            return
        
        stalled_files = self.recovery_worker.get_stalled_files_report()
        stats = self.recovery_worker.get_recovery_statistics()
        
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Drop Zones Path: {self.drop_zones_path}")
        print()
        
        # Summary statistics
        print("SUMMARY STATISTICS:")
        print("-" * 20)
        for key, value in stats['recovery_stats'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print()
        
        # Detailed file information
        if stalled_files:
            print("STALLED FILES DETAILS:")
            print("-" * 22)
            
            for file_info in stalled_files:
                print(f"Submission ID: {file_info['submission_id']}")
                print(f"File Path: {file_info['file_path']}")
                print(f"Status: {file_info['status']}")
                print(f"Current Stage: {file_info['current_stage']}")
                print(f"Submitted: {file_info['submitted_at']}")
                print(f"Last Activity: {file_info['last_activity']}")
                print(f"Stall Duration: {file_info['stall_duration_minutes']:.1f} minutes")
                print(f"Retry Count: {file_info['retry_count']}")
                
                if file_info['error_messages']:
                    print(f"Errors: {', '.join(file_info['error_messages'])}")
                
                print("-" * 40)
        
        input("Press Enter to continue...")
    
    async def _emergency_tools_menu(self) -> None:
        """Emergency recovery tools."""
        print("\n🚨 EMERGENCY RECOVERY TOOLS")
        print("⚠️  Use with caution!")
        print("-" * 40)
        print("1. Force Restart All Stalled Files")
        print("2. Quarantine All Files > 24 Hours")
        print("3. Clear Processing Folders")
        print("4. Reset All Audit Logs")
        print("5. Back to Main Menu")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "5":
            return
        
        confirm = input("⚠️  This is an emergency action. Type 'CONFIRM' to proceed: ").strip()
        
        if confirm != "CONFIRM":
            print("❌ Emergency action cancelled")
            input("Press Enter to continue...")
            return
        
        if choice == "1":
            await self._emergency_restart_all()
        elif choice == "2":
            await self._emergency_quarantine_old_files()
        elif choice == "3":
            await self._emergency_clear_processing()
        elif choice == "4":
            await self._emergency_reset_audit_logs()
        else:
            print("Invalid choice")
        
        input("Press Enter to continue...")
    
    async def _emergency_restart_all(self) -> None:
        """Emergency restart all stalled files."""
        print("\n🚨 EMERGENCY: Restarting all stalled files...")
        
        if not self.recovery_worker:
            print("❌ Recovery worker not available")
            return
        
        stalled_files = self.recovery_worker.get_stalled_files_report()
        
        if not stalled_files:
            print("✅ No stalled files to restart")
            return
        
        success_count = 0
        
        for file_info in stalled_files:
            try:
                task_id = await self.recovery_worker.recover_file(
                    submission_id=file_info["submission_id"],
                    action=RecoveryAction.RESTART_FROM_BEGINNING,
                    force_recovery=True
                )
                success_count += 1
                print(f"✅ Restarted: {file_info['submission_id'][:30]}")
                
            except Exception as e:
                print(f"❌ Failed to restart {file_info['submission_id'][:30]}: {e}")
        
        print(f"\n📊 Emergency restart complete: {success_count}/{len(stalled_files)} files restarted")
    
    async def _emergency_quarantine_old_files(self) -> None:
        """Emergency quarantine files older than 24 hours."""
        print("\n🚨 EMERGENCY: Quarantining files older than 24 hours...")
        
        if not self.recovery_worker:
            print("❌ Recovery worker not available")
            return
        
        stalled_files = self.recovery_worker.get_stalled_files_report()
        old_files = [f for f in stalled_files if f["stall_duration_minutes"] > 24 * 60]
        
        if not old_files:
            print("✅ No old files to quarantine")
            return
        
        success_count = 0
        
        for file_info in old_files:
            try:
                task_id = await self.recovery_worker.recover_file(
                    submission_id=file_info["submission_id"],
                    action=RecoveryAction.QUARANTINE,
                    force_recovery=True,
                    recovery_options={"reason": "Emergency quarantine - file older than 24 hours"}
                )
                success_count += 1
                print(f"✅ Quarantined: {file_info['submission_id'][:30]}")
                
            except Exception as e:
                print(f"❌ Failed to quarantine {file_info['submission_id'][:30]}: {e}")
        
        print(f"\n📊 Emergency quarantine complete: {success_count}/{len(old_files)} files quarantined")
    
    async def _emergency_clear_processing(self) -> None:
        """Emergency clear processing folders."""
        print("\n🚨 EMERGENCY: Clearing processing folders...")
        print("⚠️  This action is not yet implemented for safety")
        print("Manual clearing required:")
        print(f"  Processing folders in: {self.drop_zones_path}")
    
    async def _emergency_reset_audit_logs(self) -> None:
        """Emergency reset audit logs."""
        print("\n🚨 EMERGENCY: Resetting audit logs...")
        print("⚠️  This action is not yet implemented for safety")
        print("Manual reset required:")
        print(f"  Audit folders in: {self.drop_zones_path}")
    
    async def _export_reports(self) -> None:
        """Export recovery reports."""
        print("\n📤 EXPORT REPORTS")
        print("-" * 20)
        
        if not self.recovery_worker:
            print("❌ Recovery worker not available")
            input("Press Enter to continue...")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export stalled files report
        stalled_files = self.recovery_worker.get_stalled_files_report()
        stats = self.recovery_worker.get_recovery_statistics()
        
        report = {
            "export_timestamp": datetime.now().isoformat(),
            "drop_zones_path": self.drop_zones_path,
            "recovery_statistics": stats,
            "stalled_files": stalled_files
        }
        
        report_file = project_root / f"flow_recovery_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Report exported to: {report_file}")
            print(f"📊 Exported {len(stalled_files)} stalled files and statistics")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
        
        input("Press Enter to continue...")


async def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description="Flow Recovery Dashboard")
    parser.add_argument("--drop-zones-path", help="Path to drop zones directory")
    parser.add_argument("--monitor-only", action="store_true", help="Monitor only mode")
    parser.add_argument("--recover-file", help="Recover specific file by submission ID")
    args = parser.parse_args()
    
    dashboard = FlowRecoveryDashboard(args.drop_zones_path)
    
    # Start recovery worker
    if not await dashboard.start_recovery_worker():
        print("❌ Cannot start dashboard without recovery worker")
        return 1
    
    try:
        if args.monitor_only:
            await dashboard._monitor_flows()
        elif args.recover_file:
            # Direct recovery mode
            print(f"🔧 Recovering file: {args.recover_file}")
            # Implementation for direct recovery
        else:
            # Interactive dashboard
            await dashboard.show_dashboard()
    
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return 1
    
    finally:
        await dashboard.stop_recovery_worker()
    
    print("\n✅ Dashboard shutdown complete")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))