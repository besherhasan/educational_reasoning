#!/usr/bin/env python3
# check_batches.py
from openai import OpenAI
import sys, json

client = OpenAI()

# list the most recent 5 jobs
print("=== Recent Batch Jobs ===")
jobs = client.batches.list(limit=5)
for job in jobs.data:
    print(f"ID={job.id}  status={job.status}  created={job.created_at}")

print("\nUsage:")
print("  python check_batches.py <batch_id>   # to see details and download results\n")

if len(sys.argv) > 1:
    batch_id = sys.argv[1]
    job = client.batches.retrieve(batch_id)
    print("\n=== Job Details ===")
    print(json.dumps(job.to_dict(), indent=2, ensure_ascii=False))

    if job.status == "completed" and job.output_file_id:
        print(f"\nDownloading results file {job.output_file_id} ...")
        content = client.files.content(job.output_file_id).content
        outpath = f"{batch_id}_results.jsonl"
        with open(outpath, "wb") as f:
            f.write(content)
        print(f"Saved to {outpath}")
    else:
        print("\nThis job has not finished yet or has no output file.")
