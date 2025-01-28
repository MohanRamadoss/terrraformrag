import os
import json
from pathlib import Path

class ModuleHandler:
    def __init__(self, modules_dir="custom_modules"):
        self.modules_dir = Path(modules_dir)
        self.modules_dir.mkdir(exist_ok=True)
        
    def save_module(self, name, code, description, version="1.0.0"):
        """Save a custom module with versioning."""
        module_dir = self.modules_dir / name / version
        module_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main module code
        with open(module_dir / "main.tf", "w") as f:
            f.write(code)
            
        # Save module metadata
        metadata = {
            "name": name,
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        with open(module_dir / "module.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def get_module(self, name, version="latest"):
        """Retrieve a module by name and version."""
        module_dir = self.modules_dir / name
        if not module_dir.exists():
            return None
            
        if version == "latest":
            # Get latest version
            versions = [v.name for v in module_dir.iterdir() if v.is_dir()]
            if not versions:
                return None
            version = sorted(versions)[-1]
            
        module_path = module_dir / version
        if not module_path.exists():
            return None
            
        # Read module code and metadata
        with open(module_path / "main.tf") as f:
            code = f.read()
        with open(module_path / "module.json") as f:
            metadata = json.load(f)
            
        return {
            "code": code,
            "metadata": metadata
        }
