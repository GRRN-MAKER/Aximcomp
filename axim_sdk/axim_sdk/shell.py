import os
import subprocess
from typing import Optional, Protocol, List

class ShellProvider(Protocol):
    def execute(self, command: str) -> str:
        ...

class BashShellProvider:
    def __init__(self, shell: str = "/bin/bash"):
        # This will eventually use the Rust core once built
        try:
            from axim_core import ShellExecutor  # provided by the Rust crate
            self.executor = ShellExecutor(shell)
            self.use_rust = True
        except ImportError:
            self.shell = shell
            self.use_rust = False

    def execute(self, command: str) -> str:
        if self.use_rust:
            return self.executor.execute(command)
        
        # Fallback to standard subprocess if Rust core is not built
        result = subprocess.run(
            [self.shell, "-c", command],
            capture_output=True,
            text=True
        )
        output = result.stdout
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}"
        return output

class Shell:
    def __init__(self, provider: Optional[ShellProvider] = None):
        self.provider = provider or BashShellProvider()

    def run(self, command: str) -> str:
        return self.provider.execute(command)
