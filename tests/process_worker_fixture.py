"""Importable spawn target for tests; no cameras, CUDA, network or database."""

import asyncio


def fake_worker(role, config, channels):
    """Use real IPC with deterministic slow, failed and healthy worker modes."""
    asyncio.run(run(role, config, channels))


async def run(role, config, channels):
    """Mirror the production queue direction and completion messages."""
    pulse = asyncio.create_task(channels.pulse())
    try:
        if config.frame_ingest.folder_path == "crash" and role == "vision":
            channels.failed.set()
            return
        await channels.report("ready")
        if role == "vision":
            while True:
                received, job = await channels.receive(channels.triggers)
                if not received:
                    continue
                if job is None:
                    await channels.send(channels.audit, None)
                    break
                await channels.report("started", sequence=job["sequence"])
                if config.frame_ingest.folder_path == "hang":
                    await asyncio.sleep(60)
                await channels.send(channels.audit, job)
                await channels.report("completed", sequence=job["sequence"])
        else:
            while True:
                received, command = await channels.receive(channels.commands)
                if received:
                    await channels.report(
                        "roll_ready", request_id=command["request_id"], roll_id="test-roll",
                    )
                    break
                if channels.abort.is_set():
                    return
                # No-roll shutdown must also be able to consume a drain marker.
                got_job, job = await channels.receive(channels.audit)
                if got_job and job is None:
                    await channels.report("stopped")
                    return
            while True:
                received, job = await channels.receive(channels.audit)
                if not received:
                    continue
                if job is None:
                    break
                await channels.report("started", sequence=job["sequence"])
                await asyncio.sleep(0.15)
                await channels.report("dashboard", event={"sequence": job["sequence"]})
                await channels.report("completed", sequence=job["sequence"])
        await channels.report("stopped")
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)
