import asyncio
from satoriengine.veda.engine import Engine

async def main():
    await Engine.create()
    await asyncio.Event().wait()
    
asyncio.run(main())