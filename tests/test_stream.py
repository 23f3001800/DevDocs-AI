"""
Manual streaming test — NOT a pytest unit test.

Usage:
    python -m tests.test_stream

This calls the live Anthropic API and requires ANTHROPIC_API_KEY.
Renamed main() to avoid pytest auto-collection.
"""

import asyncio

from app.chain import ask_stream


async def run_stream_test():
    print("Starting stream...\n")

    async for chunk in ask_stream("how to design streaming function"):
        print(chunk, end="", flush=True)

    print("\n\nDone.")


if __name__ == "__main__":
    asyncio.run(run_stream_test())
