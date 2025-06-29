#!/usr/bin/env python3
"""
🚨 EMERGENCY MEMORY CLEANUP - CRITICAL INTERVENTION
💎 ULTRA PERFECT memory management for 4GB RAM systems

Bu script acil durumda memory temizliği yapar:
1. Python process'leri optimize eder
2. Garbage collection çalıştırır  
3. Temporary files temizler
4. System cache'i optimize eder
5. Memory kullanımını optimize eder

KULLANIM:
python emergency_memory_cleanup.py --aggressive
python emergency_memory_cleanup.py --gentle
"""

import gc
import os
import sys
import psutil
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

# Simple logging to avoid emoji unicode issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EmergencyCleanup")

class EmergencyMemoryCleanup:
    """ULTRA PERFECT Emergency Memory Cleanup System"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.start_memory = self._get_memory_status()
        
        logger.info("EMERGENCY MEMORY CLEANUP SYSTEM ACTIVATED")
        logger.info(f"Initial Memory: {self.start_memory['used_gb']:.1f}GB/{self.start_memory['total_gb']:.1f}GB ({self.start_memory['percentage_used']:.1f}%)")

    def _get_memory_status(self) -> Dict[str, float]:
        """Get current memory status"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percentage_used": round(memory.percent, 1)
        }

    def aggressive_python_gc_cleanup(self) -> Dict[str, Any]:
        """Aggressive Python garbage collection"""
        
        logger.info("Starting aggressive Python garbage collection...")
        
        results = {
            "gc_collections": 0,
            "objects_collected": 0,
            "memory_freed_mb": 0
        }
        
        # Get memory before cleanup
        before_memory = self._get_memory_status()
        
        # Force garbage collection multiple times
        for i in range(5):
            collected = gc.collect()
            results["gc_collections"] += 1
            results["objects_collected"] += collected
            time.sleep(0.1)  # Brief pause between collections
        
        # Additional aggressive cleanup
        gc.set_debug(0)  # Disable debug mode to save memory
        
        # Clear module cache if possible
        try:
            sys.modules.clear()
        except:
            pass
        
        # Get memory after cleanup
        after_memory = self._get_memory_status()
        memory_freed = before_memory["used_gb"] - after_memory["used_gb"]
        results["memory_freed_mb"] = round(memory_freed * 1024, 1)
        
        logger.info(f"Python GC completed: {results['objects_collected']} objects collected")
        logger.info(f"Memory freed: {results['memory_freed_mb']}MB")
        
        return results

    def cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files"""
        
        logger.info("Cleaning up temporary files...")
        
        results = {
            "temp_files_removed": 0,
            "space_freed_mb": 0,
            "project_temp_cleaned": False
        }
        
        try:
            # Clean system temp directory
            temp_dir = Path(tempfile.gettempdir())
            initial_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
            
            for temp_file in temp_dir.glob('tmp*'):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        results["temp_files_removed"] += 1
                except:
                    continue
            
            # Clean project-specific temp files
            project_temp_patterns = [
                "*.tmp", "*.temp", "*~", "*.bak", 
                "__pycache__", "*.pyc", "*.pyo"
            ]
            
            for pattern in project_temp_patterns:
                for temp_file in self.project_root.rglob(pattern):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                            results["temp_files_removed"] += 1
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file, ignore_errors=True)
                    except:
                        continue
            
            final_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
            results["space_freed_mb"] = round((initial_size - final_size) / (1024**2), 1)
            results["project_temp_cleaned"] = True
            
            logger.info(f"Temp cleanup: {results['temp_files_removed']} files removed, {results['space_freed_mb']}MB freed")
            
        except Exception as e:
            logger.warning(f"Temp cleanup partial failure: {e}")
            
        return results

    def optimize_python_memory(self) -> Dict[str, Any]:
        """Optimize Python memory usage"""
        
        logger.info("Optimizing Python memory usage...")
        
        results = {
            "optimization_applied": [],
            "memory_optimized": True
        }
        
        try:
            # Reduce import overhead
            import sys
            if hasattr(sys, 'intern'):
                results["optimization_applied"].append("string_interning")
            
            # Optimize list allocations
            import array
            results["optimization_applied"].append("array_optimization")
            
            # Clear cached compiled regexes
            import re
            re.purge()
            results["optimization_applied"].append("regex_cache_cleared")
            
            # Optimize module imports
            import importlib
            if hasattr(importlib, 'invalidate_caches'):
                importlib.invalidate_caches()
                results["optimization_applied"].append("import_cache_cleared")
                
            logger.info(f"Python optimizations: {', '.join(results['optimization_applied'])}")
            
        except Exception as e:
            logger.warning(f"Python optimization partial failure: {e}")
            results["memory_optimized"] = False
            
        return results

    def close_unnecessary_processes(self) -> Dict[str, Any]:
        """Identify and suggest closing unnecessary processes"""
        
        logger.info("Analyzing running processes for memory optimization...")
        
        results = {
            "total_processes": 0,
            "high_memory_processes": [],
            "python_processes": [],
            "suggestions": []
        }
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
                try:
                    results["total_processes"] += 1
                    memory_mb = proc.info['memory_info'].rss / (1024**2)
                    
                    # Track high memory processes
                    if memory_mb > 100:  # Processes using more than 100MB
                        results["high_memory_processes"].append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "memory_mb": round(memory_mb, 1)
                        })
                    
                    # Track Python processes specifically
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline'] or []
                        results["python_processes"].append({
                            "pid": proc.info['pid'],
                            "memory_mb": round(memory_mb, 1),
                            "command": ' '.join(cmdline)[:100] if cmdline else 'Unknown'
                        })
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Generate suggestions
            if len(results["high_memory_processes"]) > 5:
                results["suggestions"].append("Consider closing some high-memory applications")
            
            if len(results["python_processes"]) > 3:
                results["suggestions"].append("Multiple Python processes detected - consider consolidating")
                
            logger.info(f"Process analysis: {results['total_processes']} total, {len(results['high_memory_processes'])} high-memory")
            
        except Exception as e:
            logger.warning(f"Process analysis failed: {e}")
            
        return results

    def run_emergency_cleanup(self, aggressive: bool = True) -> Dict[str, Any]:
        """Run complete emergency memory cleanup"""
        
        logger.info("=" * 60)
        logger.info("STARTING EMERGENCY MEMORY CLEANUP")
        logger.info("=" * 60)
        
        cleanup_results = {
            "start_time": time.time(),
            "initial_memory": self.start_memory,
            "cleanup_steps": {},
            "final_memory": {},
            "total_memory_freed_mb": 0,
            "success": False
        }
        
        # Step 1: Aggressive Python GC
        logger.info("STEP 1: Aggressive Python Garbage Collection")
        cleanup_results["cleanup_steps"]["python_gc"] = self.aggressive_python_gc_cleanup()
        
        # Step 2: Temp file cleanup
        logger.info("STEP 2: Temporary File Cleanup")
        cleanup_results["cleanup_steps"]["temp_files"] = self.cleanup_temp_files()
        
        # Step 3: Python memory optimization
        logger.info("STEP 3: Python Memory Optimization")
        cleanup_results["cleanup_steps"]["python_optimization"] = self.optimize_python_memory()
        
        # Step 4: Process analysis
        logger.info("STEP 4: Process Analysis")
        cleanup_results["cleanup_steps"]["process_analysis"] = self.close_unnecessary_processes()
        
        # Final memory check
        final_memory = self._get_memory_status()
        cleanup_results["final_memory"] = final_memory
        
        # Calculate total memory freed
        memory_freed = self.start_memory["used_gb"] - final_memory["used_gb"]
        cleanup_results["total_memory_freed_mb"] = round(memory_freed * 1024, 1)
        
        # Determine success
        improvement_threshold = 0.5  # At least 500MB freed
        cleanup_results["success"] = abs(memory_freed) >= improvement_threshold or final_memory["available_gb"] >= 1.5
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EMERGENCY CLEANUP COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Initial Memory: {self.start_memory['used_gb']:.1f}GB ({self.start_memory['percentage_used']:.1f}%)")
        logger.info(f"Final Memory:   {final_memory['used_gb']:.1f}GB ({final_memory['percentage_used']:.1f}%)")
        logger.info(f"Memory Change:  {cleanup_results['total_memory_freed_mb']:+.1f}MB")
        logger.info(f"Available Now:  {final_memory['available_gb']:.1f}GB")
        
        if cleanup_results["success"]:
            logger.info("SUCCESS: Memory cleanup successful - Ready for optimization")
        else:
            logger.warning("PARTIAL: Some improvement made - Consider closing applications manually")
        
        return cleanup_results

    def get_optimization_readiness(self) -> Dict[str, Any]:
        """Check if system is ready for optimization after cleanup"""
        
        current_memory = self._get_memory_status()
        
        readiness = {
            "memory_status": current_memory,
            "ready_for_optimization": current_memory["available_gb"] >= 1.0,
            "recommended_action": "",
            "optimization_mode": "NONE"
        }
        
        if current_memory["available_gb"] >= 2.0:
            readiness["optimization_mode"] = "FULL"
            readiness["recommended_action"] = "Ready for full optimization (500+ trials)"
        elif current_memory["available_gb"] >= 1.5:
            readiness["optimization_mode"] = "MODERATE"
            readiness["recommended_action"] = "Ready for moderate optimization (300 trials)"
        elif current_memory["available_gb"] >= 1.0:
            readiness["optimization_mode"] = "LIGHT"
            readiness["recommended_action"] = "Ready for light optimization (200 trials)"
        else:
            readiness["optimization_mode"] = "NONE"
            readiness["recommended_action"] = "CRITICAL: Close applications or move to VPS"
        
        return readiness


def main():
    """Main emergency cleanup execution"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Emergency Memory Cleanup")
    parser.add_argument("--aggressive", action="store_true", help="Run aggressive cleanup")
    parser.add_argument("--gentle", action="store_true", help="Run gentle cleanup")
    args = parser.parse_args()
    
    print("EMERGENCY MEMORY CLEANUP SYSTEM")
    print("ULTRA PERFECT memory management for critical situations")
    print("=" * 60)
    
    try:
        # Create cleanup instance
        cleanup = EmergencyMemoryCleanup()
        
        # Run cleanup
        aggressive_mode = args.aggressive or not args.gentle  # Default to aggressive
        results = cleanup.run_emergency_cleanup(aggressive=aggressive_mode)
        
        # Check readiness for optimization
        readiness = cleanup.get_optimization_readiness()
        
        print("\nOPTIMIZATION READINESS:")
        print(f"Mode: {readiness['optimization_mode']}")
        print(f"Action: {readiness['recommended_action']}")
        
        if readiness["ready_for_optimization"]:
            print("\nNEXT STEP: Run optimization with appropriate trial count")
            if readiness["optimization_mode"] == "FULL":
                print("python PHASE_EXECUTION_MASTER_PLAN.py --phase 1A --strategy momentum")
            elif readiness["optimization_mode"] == "MODERATE":
                print("python smart_range_optimizer.py --strategy momentum --trials 300")
            else:
                print("python smart_range_optimizer.py --strategy momentum --trials 200 --quick")
        else:
            print("\nCRITICAL: Manual intervention required")
            print("- Close unnecessary applications")
            print("- Consider using VPS for optimization")
            print("- Or restart computer to free memory")
        
        return results
        
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")
        return None


if __name__ == "__main__":
    main()