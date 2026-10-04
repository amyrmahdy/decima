"""Write a plain copy of agentbench-gold.toml (fake credentials decoded) for the jabr/classifier-benchmark harness.

    python release/agent/decode_gold.py > /path/to/classifier-benchmark/sources/samples/agentbench.toml
"""
import re
import sys
from pathlib import Path

s = (Path(__file__).parent / "agentbench-gold.toml").read_text()
sys.stdout.write(re.sub(r"⟦rev:([^⟧]+)⟧", lambda m: m.group(1)[::-1], s))
