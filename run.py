from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.utils.profiler import get_profiler, profiler_section
    from vipe.streams.base import StreamList

    profiler_cfg = getattr(args, "profiler", None)
    profiler = get_profiler()
    profiler.reset()

    profiling_enabled = bool(getattr(profiler_cfg, "enabled", False)) if profiler_cfg is not None else False
    if profiling_enabled:
        profiler.enable()
    else:
        profiler.disable()

    # Gather all video streams
    stream_list = StreamList.make(args.streams)
    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Process each video stream
    logger = configure_logging()
    with profiler_section("Vipe"):
        for stream_idx in range(len(stream_list)):
            video_stream = stream_list[stream_idx]
            logger.info(
                f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
            )
            pipeline = make_pipeline(args.pipeline)
            with profiler_section(f"pipeline.run[{video_stream.name()}]"):
                pipeline.run(video_stream)
            logger.info(f"Finished processing {video_stream.name()}")

    if profiling_enabled:
        min_percentage = float(getattr(profiler_cfg, "min_percentage", 0.0)) if profiler_cfg is not None else 0.0
        max_depth_cfg = getattr(profiler_cfg, "max_depth", None) if profiler_cfg is not None else None
        max_depth = int(max_depth_cfg) if max_depth_cfg not in (None, "null") else None
        report = profiler.report(min_percentage=min_percentage, max_depth=max_depth)
        output_path = getattr(profiler_cfg, "output", None) if profiler_cfg is not None else None
        output_path += f"{stream_list[0].name()}.txt"
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report)
            logger.info("Profiler report written to %s", path.resolve())
        logger.info("Profiling summary:\n%s", report)


if __name__ == "__main__":
    run()
