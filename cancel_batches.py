#!/usr/bin/env python3
# cancel_batches.py
from openai import OpenAI
import sys

client = OpenAI()

if len(sys.argv) == 1:
    # list all recent jobs so you can see IDs
    jobs = client.batches.list(limit=10)
    for j in jobs.data:
        print(f"ID={j.id}  status={j.status}")
    print("\nUsage: python cancel_batches.py <batch_id> [<batch_id> ...]")
    sys.exit(0)

for bid in sys.argv[1:]:
    job = client.batches.cancel(bid)
    print(f"Cancelled {bid}: status={job.status}")
