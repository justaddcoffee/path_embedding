#!/usr/bin/env python3
"""Retry the Clofibrate evaluation that timed out."""
import subprocess
import os

# The pathway that timed out
query = """Here is a pathway that describes how a drug might affect a disease:

**Pathway:** SmallMolecule: Clofibrate | affects | Gene: ALB | affects | SmallMolecule: Myricetin | contributes_to | Disease: metabolic dysfunction-associated steatotic liver disease

**Drug:** Clofibrate
**Disease:** metabolic dysfunction-associated steatotic liver disease

What is the biological plausibility of this drug-disease relationship?

Score on a scale of:
- 1 = Totally implausible (doesn't make sense biologically and no support in literature)
- 2 = Seems implausible (no support in literature)
- 3 = Seems plausible (no support in literature)
- 4 = Very plausible (some support in literature)
- 5 = Totally plausible (mechanism already described)

Consider:
- Has this exact mechanism been described in the literature?
- Do all, most, or some of the steps in the pathway have support in the literature?
- Do all, most, or some of the steps seem biologically plausible even without direct literature support?
"""

output_file = "results/deep_research/bottom_04_rank9952_Clofibrate_metabolic_dysfunction-associated_steatotic_liver_disease.md"
api_key_file = '/Users/jtr4v/openai.key.another'

# Read API key
with open(api_key_file) as f:
    api_key = f.read().strip()

# Run deep-research-client via uvx with 20 minute timeout
cmd = [
    'uvx', 'deep-research-client', 'research',
    query,
    '--output', output_file
]

env = os.environ.copy()
env['OPENAI_API_KEY'] = api_key

print(f"Starting deep research for: Clofibrate → metabolic dysfunction-associated steatotic liver disease")
print(f"Timeout: 20 minutes")
print(f"Output: {output_file}")
print()

try:
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200  # 20 minute timeout (increased from 10)
    )

    if result.returncode == 0:
        print(f"✓ Research completed successfully!")
        print(f"✓ Output saved to: {output_file}")
    else:
        print(f"✗ Research failed")
        print(f"  Error: {result.stderr}")

except subprocess.TimeoutExpired:
    print(f"✗ Research timed out after 20 minutes")
except Exception as e:
    print(f"✗ Error running research: {e}")
