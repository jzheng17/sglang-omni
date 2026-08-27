"""Launch a PD or colocated thinker for the comparison matrix."""
import argparse

from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.serve.launcher import launch_server

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", choices=["pd", "colo"], required=True)
    ap.add_argument("--prefill-gpu", type=int, default=0)
    ap.add_argument("--decode-gpu", type=int, default=1)
    ap.add_argument("--prefill-fraction", type=float, default=None)
    ap.add_argument("--decode-fraction", type=float, default=None)
    ap.add_argument("--thinker-fraction", type=float, default=0.68)
    ap.add_argument("--encoder-fraction", type=float, default=0.025)
    ap.add_argument("--max-running-requests", type=int, default=4)
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--no-share-weights", action="store_true")
    ap.add_argument("--max-inflight-handoffs", type=int, default=0)
    args = ap.parse_args()

    cfg = Qwen3OmniPipelineConfig(model_path=MODEL)
    stages = {s.name: s for s in cfg.stages}

    engine = {"max_running_requests": args.max_running_requests}
    if not args.cuda_graph:
        engine["disable_cuda_graph"] = True

    th = stages["thinker"]
    th.gpu_memory_fraction = args.thinker_fraction
    th.engine = type(th.engine or object)() if False else th.engine
    from sglang_omni.config.schema import EngineArgs, PDConfig, PDStagePlacement

    th.engine = (th.engine or EngineArgs()).model_copy(update=engine)

    for name in ("image_encoder", "audio_encoder"):
        if name in stages:
            stages[name].gpu_memory_fraction = args.encoder_fraction

    if args.mode == "pd":
        pd = PDConfig(
            prefill=PDStagePlacement(
                gpu=args.prefill_gpu, memory_fraction=args.prefill_fraction
            ),
            decode=PDStagePlacement(
                gpu=args.decode_gpu, memory_fraction=args.decode_fraction
            ),
            share_weights=not args.no_share_weights,
        )
        if args.max_inflight_handoffs > 0:
            pd = pd.model_copy(
                update={"max_inflight_handoffs": args.max_inflight_handoffs}
            )
        th.pd_disaggregation = pd
        print("PD:", th.pd_disaggregation, flush=True)
    else:
        print("colocated, thinker gpu:", th.gpu, flush=True)

    print("engine:", th.engine.overrides(), flush=True)
    launch_server(cfg, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
