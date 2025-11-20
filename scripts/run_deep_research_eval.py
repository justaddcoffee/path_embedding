#!/usr/bin/env python3
"""Run deep research evaluation on top and bottom ranked Pathfinder paths."""
import pandas as pd
import subprocess
import os
from pathlib import Path

def run_deep_research(query: str, output_file: str, api_key_file: str) -> bool:
    """Run deep-research-client via uvx.

    Args:
        query: Research question
        output_file: Path to save markdown output
        api_key_file: Path to OpenAI API key file

    Returns:
        True if successful, False otherwise
    """
    # Read API key
    with open(api_key_file) as f:
        api_key = f.read().strip()

    # Run deep-research-client via uvx
    cmd = [
        'uvx', 'deep-research-client', 'research',
        query,
        '--output', output_file
    ]

    env = os.environ.copy()
    env['OPENAI_API_KEY'] = api_key

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"✓ Research completed: {output_file}")
            return True
        else:
            print(f"✗ Research failed for {output_file}")
            print(f"  Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Research timed out for {output_file}")
        return False
    except Exception as e:
        print(f"✗ Error running research: {e}")
        return False


def main():
    """Run deep research on top 5 and bottom 5 unique Pathfinder paths."""
    # Load scored paths
    df = pd.read_csv('results/scored_paths_10k.csv')
    print(f"Loaded {len(df):,} scored paths\n")

    # Select top 5 and bottom 5 with unique drug-disease pairs
    n_eval = 5

    # Get unique top paths
    top_paths = []
    seen_pairs = set()
    for _, row in df.sort_values('rf_score', ascending=False).iterrows():
        pair = (row['drug_name'], row['disease_name'])
        if pair not in seen_pairs:
            top_paths.append(row)
            seen_pairs.add(pair)
            if len(top_paths) == n_eval:
                break

    # Get unique bottom paths (excluding pairs already in top paths)
    bottom_paths = []
    # Note: seen_pairs already contains top path pairs, so we continue using it
    for _, row in df.sort_values('rf_score', ascending=True).iterrows():
        pair = (row['drug_name'], row['disease_name'])
        if pair not in seen_pairs:
            bottom_paths.append(row)
            seen_pairs.add(pair)
            if len(bottom_paths) == n_eval:
                break

    # Create output directory
    output_dir = Path('results/deep_research')
    output_dir.mkdir(exist_ok=True, parents=True)

    api_key_file = '/Users/jtr4v/openai.key.another'

    print("=" * 80)
    print(f"EVALUATING TOP {n_eval} HIGHEST-SCORED PATHS (UNIQUE DRUG-DISEASE PAIRS)")
    print("=" * 80)

    for i, row in enumerate(top_paths, 1):
        drug = row['drug_name']
        disease = row['disease_name']
        score = row['rf_score']
        rank = row['rf_rank']

        print(f"\n[{i}/{n_eval}] Rank {rank} | Score: {score:.3f}")
        print(f"    {drug} → {disease}")

        # Construct research query
        query = f"""Here is a pathway that describes how a drug might affect a disease:

**Pathway:** {row['path_text']}

**Drug:** {drug}
**Disease:** {disease}

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

        # Output file
        filename = f"top_{i:02d}_rank{rank:04d}_{drug}_{disease}.md"
        filename = filename.replace(' ', '_').replace('/', '_')
        output_file = str(output_dir / filename)

        # Run research
        run_deep_research(query, output_file, api_key_file)

    print("\n" + "=" * 80)
    print(f"EVALUATING BOTTOM {n_eval} LOWEST-SCORED PATHS (UNIQUE DRUG-DISEASE PAIRS)")
    print("=" * 80)

    for i, row in enumerate(bottom_paths, 1):
        drug = row['drug_name']
        disease = row['disease_name']
        score = row['rf_score']
        rank = row['rf_rank']

        print(f"\n[{i}/{n_eval}] Rank {rank} | Score: {score:.3f}")
        print(f"    {drug} → {disease}")

        # Construct research query
        query = f"""Here is a pathway that describes how a drug might affect a disease:

**Pathway:** {row['path_text']}

**Drug:** {drug}
**Disease:** {disease}

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

        # Output file
        filename = f"bottom_{i:02d}_rank{rank:04d}_{drug}_{disease}.md"
        filename = filename.replace(' ', '_').replace('/', '_')
        output_file = str(output_dir / filename)

        # Run research
        run_deep_research(query, output_file, api_key_file)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
