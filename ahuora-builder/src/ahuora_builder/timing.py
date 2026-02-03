from contextvars import ContextVar
import time


class TimingDebugHandler:
    def __init__(self, key: str = "root") -> None:
        self.timing: dict[str, dict] = {}
        self.pointer_stack = [self.timing]
        self.step_into(key)
    
    def current_item(self):
        item = self.current_pointer()
        if len(item) == 0:
            return None
        return item[list(item.keys())[-1]]
    
    def current_pointer(self):
        return self.pointer_stack[-1]
    
    def update_last_timing(self):
        if self.current_item():
            # update the duration of the previous key
            self.current_item()["duration"] = time.time() - self.current_item()["start_time"]

    def add_timing(self, key: str):
        self.update_last_timing()
        self.current_pointer()[key] = {
            "start_time": time.time(),
            "duration": None,  # calculated at the next timing
            "children": {}
        }
    
    def step_into(self, key=None):
        self.add_timing(key)
        self.pointer_stack.append(self.current_item()["children"])
    
    def step_out(self):
        self.update_last_timing()
        self.pointer_stack.pop()

    def close(self):
        # update root duration
        self.step_out()
        self.update_last_timing()
        return self.timing


def start_timing(key: str = "root"):
    """
    start a new timing handler
    """
    handler = TimingDebugHandler(key)
    timing.set(handler)
    return handler


def get_timer():
    """
    get the current timing handler
    """
    return timing.get()


# Create a context variable to store the timing information
timing: ContextVar[TimingDebugHandler | None] = ContextVar("timing", default=None)
