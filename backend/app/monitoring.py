import psutil
import logging
import time
from functools import wraps

class SimpleMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        uptime = time.time() - self.start_time
        
        logging.info(f"MONITOR {context}: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%, Uptime={uptime:.0f}s")
        
        # Simple warning for high memory usage
        if memory_mb > 1000:  # 1GB threshold
            logging.warning(f"High memory usage detected: {memory_mb:.1f}MB")
    
    def memory_check(self, func):
        """Decorator to monitor memory usage of functions"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            mem_before = self.process.memory_info().rss / 1024 / 1024
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                mem_after = self.process.memory_info().rss / 1024 / 1024
                mem_diff = mem_after - mem_before
                
                if mem_diff > 50:  # Log if memory increased by more than 50MB
                    logging.info(f"MEMORY: {func.__name__} used {mem_diff:.1f}MB")
        
        return wrapper

# Global monitor instance
monitor = SimpleMonitor()