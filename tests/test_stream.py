import asyncio

from app.chain import ask_stream


async def main():
    print("Starting stream...\n")

    async for chunk in ask_stream("how to design streaming function"):
        print(chunk, end="", flush=True)

    print("\n\nDone.")


asyncio.run(main())
