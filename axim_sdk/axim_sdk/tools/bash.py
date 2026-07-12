from typing import List, Dict, Any, Optional
import os
import shlex
from ..shell import Shell

class BashTool:
    """
    Executes a bash command in a stateful shell.
    This version includes basic security and result formatting.
    """
    
    def __init__(self, shell: Optional[Shell] = None):
        self.shell = shell or Shell()
        self.cwd = os.getcwd()
        self.history: List[str] = []

    def execute(self, command: str) -> Dict[str, Any]:
        """
        Executes a bash command and returns a result dictionary.
        """
        # Security check: identify potentially dangerous commands
        is_safe, warning = self._check_security(command)
        if not is_safe:
            return {
                "status": "denied",
                "message": f"Command denied: {warning}",
                "command": command
            }

        self.history.append(command)
        
        # Execute command in the shell
        # In a real tool, we would handle stateful CWD changes here
        # For now, we prepend 'cd {cwd} && ' to each command
        wrapped_command = f"cd {shlex.quote(self.cwd)} && {command} && pwd"
        output = self.shell.run(wrapped_command)
        
        # Parse output to find the new CWD
        lines = output.strip().split("\n")
        if lines:
            new_cwd = lines[-1]
            if os.path.isdir(new_cwd):
                self.cwd = new_cwd
                output = "\n".join(lines[:-1])
        
        return {
            "status": "success",
            "output": output,
            "cwd": self.cwd,
            "command": command
        }

    def _check_security(self, command: str) -> (bool, Optional[str]):
        """
        Perform basic security checks.
        """
        # Simplified version: block direct rm -rf / or similar
        forbidden = ["rm -rf /", "mkfs", "dd if=/dev/zero", ":(){ :|:& };:"]
        for f in forbidden:
            if f in command:
                return False, f"Potentially destructive command detected: {f}"
        return True, None

    def __str__(self):
        return f"BashTool(cwd={self.cwd})"
