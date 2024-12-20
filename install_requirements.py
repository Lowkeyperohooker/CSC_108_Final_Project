import json
import subprocess

# Load the requirements from the JSON file
with open('requirements.json', 'r') as file:
    requirements = json.load(file)

# Install each package using pip
for package in requirements["packages"]:
    name = package["name"]
    version = package.get("version", "latest")
    if version == "latest":
        subprocess.check_call(["pip", "install", name])
    else:
        subprocess.check_call(["pip", "install", f"{name}=={version}"])

print("All packages have been installed successfully!")
