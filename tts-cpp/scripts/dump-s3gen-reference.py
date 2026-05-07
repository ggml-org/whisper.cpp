#!/usr/bin/env python3
"""
Dump every intermediate tensor from the Python S3Gen inference path.

These tensors are used to verify the C++/ggml implementation stage by stage.

Output layout (directory passed via --out):
    text_tokens.npy           int32  (N_text,)
    speech_tokens.npy         int32  (N_speech,)
    speech_tokens_padded.npy  int32  (N_speech_with_silence,)
    flow_input_tokens.npy     int32  (N_prompt+N_speech+sil,)     concat of prompt_token + trimmed speech_tokens
    embedding.npy             float32 (192,)
    speaker_emb_normalized.npy float32 (192,)
    speaker_emb_affine.npy    float32 (80,)
    input_embedded.npy        float32 (N_tokens, 512)
    encoder_output.npy        float32 (2*N_tokens, 512)
    encoder_proj.npy          float32 (2*N_tokens, 80)             mu
    conds_filled.npy          float32 (80, 2*N_tokens)             prompt_feat + zeros
    cfm_z0.npy                float32 (80, 2*N_tokens)             initial noise (fixed seed)
    cfm_step0_dxdt.npy        float32 (80, 2*N_tokens)
    cfm_step1_dxdt.npy        float32 (80, 2*N_tokens)
    mel_output.npy            float32 (80, 2*N_speech)
    waveform.npy              float32 (N_samples,)
    prompt_token.npy          int32  (250,)
    prompt_feat.npy           float32 (500, 80)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from chatterbox.models.s3gen.const import S3GEN_SIL


def make_hook(storage: dict, name: str, multi_call: bool = False):
    """If multi_call, each call saves with a sequence suffix _call0, _call1, ..."""
    counter = {"n": 0}
    def hook(_module, _inputs, output):
        key = f"{name}_call{counter['n']}" if multi_call else name
        if isinstance(output, torch.Tensor):
            storage[key] = output.detach().clone().cpu()
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    storage[f"{key}_tup{i}"] = o.detach().clone().cpu()
        counter["n"] += 1
    return hook


def make_first_call_hook(storage: dict, name: str, transform=None):
    """Capture only the FIRST forward call's output (with optional transform).

    Used for stage_G2 intermediates (cfm_h_conv, cfm_h_ln) which the C++
    test harness expects with no _callN suffix and only needs from CFM step 0.
    """
    seen = {"n": 0}
    def hook(_module, _inputs, output):
        if seen["n"] > 0:
            return
        if isinstance(output, torch.Tensor):
            t = output if transform is None else transform(output)
            storage[name] = t.detach().clone().cpu()
        seen["n"] += 1
    return hook


def save(t, path: Path):
    if torch.is_tensor(t):
        arr = t.detach().cpu().contiguous().numpy()
    else:
        arr = t
    arr = np.ascontiguousarray(arr)  # force C-order for predictable byte layout
    np.save(path, arr)
    print(f"  {path.name:38s} shape={tuple(arr.shape)} dtype={arr.dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello from ggml.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    print(f"Loading model on {args.device}...")
    tts = ChatterboxTurboTTS.from_pretrained(device=args.device)
    assert tts.conds is not None

    # ------------------------------------------------------------------
    # 1. Text tokens
    # ------------------------------------------------------------------
    print("\n--- Text tokenization ---")
    normalized = punc_norm(args.text)
    text_tokens = tts.tokenizer(normalized, return_tensors="pt").input_ids.to(tts.device)
    save(text_tokens.squeeze(0).int(), args.out / "text_tokens.npy")

    # ------------------------------------------------------------------
    # 2. Speech tokens from T3 (deterministic, greedy)
    # ------------------------------------------------------------------
    print("\n--- T3 speech token generation ---")
    torch.manual_seed(args.seed)  # reset before T3 forward
    speech_tokens = tts.t3.inference_turbo(
        t3_cond=tts.conds.t3, text_tokens=text_tokens,
        temperature=1.0, top_k=1, top_p=1.0, repetition_penalty=1.0,
        max_gen_len=args.n_predict,
    )
    save(speech_tokens.squeeze(0).int(), args.out / "speech_tokens.npy")

    # Remove OOV tokens and add silence (matches tts_turbo.py generate())
    speech_tokens_trim = speech_tokens[speech_tokens < 6561]
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(tts.device)
    speech_tokens_padded = torch.cat([speech_tokens_trim, silence])
    save(speech_tokens_padded.int(), args.out / "speech_tokens_padded.npy")
    print(f"Trimmed+silence: {speech_tokens_padded.shape}")

    # ------------------------------------------------------------------
    # 3. S3Gen flow inference, with hooks to capture intermediates
    # ------------------------------------------------------------------
    print("\n--- S3Gen flow inference (with hooks) ---")
    flow = tts.s3gen.flow

    storage = {}
    hooks = [
        flow.input_embedding.register_forward_hook(make_hook(storage, "input_embedded")),
        flow.spk_embed_affine_layer.register_forward_hook(make_hook(storage, "speaker_emb_affine")),
        flow.encoder.register_forward_hook(make_hook(storage, "encoder_out")),
        flow.encoder_proj.register_forward_hook(make_hook(storage, "encoder_proj")),
        # Intermediate encoder stages
        flow.encoder.embed.register_forward_hook(make_hook(storage, "encoder_embed")),
        flow.encoder.pre_lookahead_layer.register_forward_hook(make_hook(storage, "pre_lookahead")),
        flow.encoder.up_layer.register_forward_hook(make_hook(storage, "up_layer")),
        flow.encoder.up_embed.register_forward_hook(make_hook(storage, "up_embed")),
        flow.encoder.after_norm.register_forward_hook(make_hook(storage, "after_norm")),
    ]
    for i in range(6):
        hooks.append(flow.encoder.encoders[i].register_forward_hook(make_hook(storage, f"enc_block{i}")))
    for i in range(4):
        hooks.append(flow.encoder.up_encoders[i].register_forward_hook(make_hook(storage, f"up_block{i}")))

    # Extra hooks inside block 0 to debug attention
    b0 = flow.encoder.encoders[0]
    hooks.append(b0.norm_mha.register_forward_hook(make_hook(storage, "b0_norm_mha")))
    hooks.append(b0.self_attn.linear_q.register_forward_hook(make_hook(storage, "b0_q")))
    hooks.append(b0.self_attn.linear_k.register_forward_hook(make_hook(storage, "b0_k")))
    hooks.append(b0.self_attn.linear_v.register_forward_hook(make_hook(storage, "b0_v")))
    hooks.append(b0.self_attn.linear_pos.register_forward_hook(make_hook(storage, "b0_p")))
    hooks.append(b0.self_attn.linear_out.register_forward_hook(make_hook(storage, "b0_attn_out")))
    hooks.append(b0.norm_ff.register_forward_hook(make_hook(storage, "b0_norm_ff")))
    hooks.append(b0.feed_forward.register_forward_hook(make_hook(storage, "b0_ff_out")))

    # Hooks inside CFM decoder for stage-by-stage validation
    est = flow.decoder.estimator
    # Time MLP path - multi_call=True since called per step (and per t/r)
    hooks.append(est.time_embeddings.register_forward_hook(make_hook(storage, "cfm_t_sinemb", multi_call=True)))
    hooks.append(est.time_mlp.register_forward_hook(make_hook(storage, "cfm_t_mlp", multi_call=True)))
    hooks.append(est.time_embed_mixer.register_forward_hook(make_hook(storage, "cfm_t_mix", multi_call=True)))
    # First down_block (multi_call to capture step 0 and step 1)
    d0_rn = est.down_blocks[0][0]                  # CausalResnetBlock1D
    hooks.append(d0_rn.block1.register_forward_hook(make_hook(storage, "cfm_d0_rn_b1", multi_call=True)))
    hooks.append(d0_rn.block2.register_forward_hook(make_hook(storage, "cfm_d0_rn_b2", multi_call=True)))
    hooks.append(d0_rn.mlp.register_forward_hook(make_hook(storage, "cfm_d0_rn_mlp", multi_call=True)))
    hooks.append(d0_rn.res_conv.register_forward_hook(make_hook(storage, "cfm_d0_rn_res", multi_call=True)))
    hooks.append(d0_rn.register_forward_hook(make_hook(storage, "cfm_d0_rn", multi_call=True)))
    # G2-gap fix: capture h_conv (after CausalConv1d, before LN) and h_ln
    # (after LayerNorm, after the second Transpose back to (B, C, T)).  Only
    # the first call (CFM step 0) is captured because stage_G2 in
    # test_s3gen.cpp loads cfm_step0_* inputs and expects matching G2
    # intermediates.  block1.block layout per CausalBlock1D:
    #   [0] CausalConv1d   -> (B, C, T)
    #   [1] Transpose(1,2) -> (B, T, C)
    #   [2] LayerNorm      -> (B, T, C)
    #   [3] Transpose(1,2) -> (B, C, T)   <- save here for h_ln in (C, T) layout
    #   [4] Mish           -> (B, C, T)   (already captured by block1 hook -> cfm_d0_rn_b1)
    d0_b1_seq = d0_rn.block1.block
    hooks.append(d0_b1_seq[0].register_forward_hook(
        make_first_call_hook(storage, "cfm_h_conv")))
    hooks.append(d0_b1_seq[3].register_forward_hook(
        make_first_call_hook(storage, "cfm_h_ln")))
    # First transformer block in down_block 0
    d0_t0 = est.down_blocks[0][1][0]               # BasicTransformerBlock
    hooks.append(d0_t0.norm1.register_forward_hook(make_hook(storage, "cfm_d0_t0_n1", multi_call=True)))
    hooks.append(d0_t0.attn1.register_forward_hook(make_hook(storage, "cfm_d0_t0_attn", multi_call=True)))
    hooks.append(d0_t0.norm3.register_forward_hook(make_hook(storage, "cfm_d0_t0_n3", multi_call=True)))
    hooks.append(d0_t0.ff.register_forward_hook(make_hook(storage, "cfm_d0_t0_ff", multi_call=True)))
    hooks.append(d0_t0.register_forward_hook(make_hook(storage, "cfm_d0_t0", multi_call=True)))
    # Whole down_block[0] output (after downsample)
    hooks.append(est.down_blocks[0][2].register_forward_hook(make_hook(storage, "cfm_d0_downsample", multi_call=True)))
    # Last mid_block output
    hooks.append(est.mid_blocks[-1].register_forward_hook(make_hook(storage, "cfm_mid_last", multi_call=True)))
    # Up_block
    hooks.append(est.up_blocks[0][0].register_forward_hook(make_hook(storage, "cfm_u0_rn", multi_call=True)))
    hooks.append(est.up_blocks[0][2].register_forward_hook(make_hook(storage, "cfm_u0_upsample", multi_call=True)))
    hooks.append(est.final_block.register_forward_hook(make_hook(storage, "cfm_final_block", multi_call=True)))
    hooks.append(est.final_proj.register_forward_hook(make_hook(storage, "cfm_final_proj", multi_call=True)))

    # Also save the raw embedding + normalized
    ref = tts.conds.gen
    embedding = ref["embedding"]
    prompt_token = ref["prompt_token"]
    prompt_feat = ref["prompt_feat"]

    save(embedding.squeeze(0).cpu(), args.out / "embedding.npy")
    save(torch.nn.functional.normalize(embedding, dim=1).squeeze(0).cpu(), args.out / "speaker_emb_normalized.npy")
    save(prompt_token.squeeze(0).int().cpu(), args.out / "prompt_token.npy")
    save(prompt_feat.squeeze(0).cpu(), args.out / "prompt_feat.npy")

    # Build the input flow sees: prompt_token + speech_tokens_padded
    flow_input_tokens = torch.cat([prompt_token.squeeze(0).cpu(),
                                   speech_tokens_padded.cpu()])
    save(flow_input_tokens.int(), args.out / "flow_input_tokens.npy")

    # Capture CFM internals: z (initial noise) via torch.randn_like monkeypatch,
    # and per-step dxdt via a hook on cfm.estimator.
    cfm = flow.decoder
    captured = {}

    # basic_euler calls self.estimator.forward directly -> hooks don't fire.
    # So we monkeypatch the forward method instead.
    step_idx = [0]
    estimator = cfm.estimator
    orig_est_forward = estimator.forward

    def estimator_forward_capture(x, mask=None, mu=None, t=None, spks=None, cond=None, r=None):
        captured[f"cfm_step{step_idx[0]}_x_in"] = x.detach().clone().cpu()
        captured[f"cfm_step{step_idx[0]}_mu"] = mu.detach().clone().cpu()
        captured[f"cfm_step{step_idx[0]}_t"] = t.detach().clone().cpu() if t is not None else None
        captured[f"cfm_step{step_idx[0]}_r"] = r.detach().clone().cpu() if r is not None else None
        captured[f"cfm_step{step_idx[0]}_spks"] = spks.detach().clone().cpu() if spks is not None else None
        captured[f"cfm_step{step_idx[0]}_cond"] = cond.detach().clone().cpu() if cond is not None else None
        captured[f"cfm_step{step_idx[0]}_mask"] = mask.detach().clone().cpu() if mask is not None else None
        # G2-gap fix: replicate the pack([x, mu, spks_bc, cond], dim=1)
        # done inside ConditionalDecoder.forward so stage_G2 has its
        # `cfm_concat.npy` reference.  Only capture from CFM step 0.
        if step_idx[0] == 0:
            try:
                from einops import pack as _pack, repeat as _repeat
                xc = _pack([x, mu], "b * t")[0]
                if spks is not None:
                    spks_bc = _repeat(spks, "b c -> b c t", t=x.shape[-1])
                    xc = _pack([xc, spks_bc], "b * t")[0]
                if cond is not None:
                    xc = _pack([xc, cond], "b * t")[0]
                captured["cfm_concat"] = xc.detach().clone().cpu()
            except Exception as e:
                print(f"  cfm_concat capture skipped: {e}")
        out = orig_est_forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
        captured[f"cfm_step{step_idx[0]}_dxdt"] = out.detach().clone().cpu()
        step_idx[0] += 1
        return out
    estimator.forward = estimator_forward_capture

    # Monkeypatch torch.randn_like only inside the CFM call to capture z
    import torch as _torch
    orig_randn_like = _torch.randn_like
    z_captured = {"count": 0}
    def randn_like_capture(x, *a, **kw):
        out = orig_randn_like(x, *a, **kw)
        if z_captured["count"] == 0:  # first call inside CFM is the z init
            captured["cfm_z0_raw"] = out.detach().clone().cpu()
            z_captured["count"] += 1
        return out
    _torch.randn_like = randn_like_capture

    # Also monkeypatch torch.randn to capture the flow_inference noise (meanflow noise buffer)
    orig_randn = _torch.randn
    noise_captured = {"count": 0}
    def randn_capture(*a, **kw):
        out = orig_randn(*a, **kw)
        if noise_captured["count"] == 0:
            captured["cfm_noised_mels"] = out.detach().clone().cpu()
            noise_captured["count"] += 1
        return out
    _torch.randn = randn_capture

    try:
        torch.manual_seed(args.seed)  # reset before CFM noise
        mel = tts.s3gen.flow_inference(
            speech_tokens=speech_tokens_padded.unsqueeze(0).to(tts.device),
            ref_dict=ref,
            n_cfm_timesteps=2,
            finalize=True,
        )
    finally:
        _torch.randn_like = orig_randn_like
        _torch.randn = orig_randn
        estimator.forward = orig_est_forward
        for h in hooks:
            h.remove()

    # Save hooked tensors
    for name, t in storage.items():
        if t is None: continue
        save(t.squeeze(0) if t.ndim > 1 else t, args.out / f"{name}.npy")
    for name, t in captured.items():
        if t is None: continue
        save(t.squeeze(0) if t.ndim > 1 else t, args.out / f"{name}.npy")

    save(mel.squeeze(0).cpu(), args.out / "mel_output.npy")

    # ------------------------------------------------------------------
    # 4. HiFTGenerator inference (re-register hooks since CFM ones were removed)
    # ------------------------------------------------------------------
    print("\n--- HiFT vocoder inference ---")
    hift = tts.s3gen.mel2wav
    hift_storage = {}
    hift_hooks = [
        hift.f0_predictor.register_forward_hook(make_hook(hift_storage, "hift_f0")),
        hift.f0_upsamp.register_forward_hook(make_hook(hift_storage, "hift_f0_upsamp")),
        hift.m_source.register_forward_hook(make_hook(hift_storage, "hift_msource")),
        hift.m_source.l_sin_gen.register_forward_hook(make_hook(hift_storage, "hift_sinegen")),
        hift.conv_pre.register_forward_hook(make_hook(hift_storage, "hift_conv_pre")),
        hift.conv_post.register_forward_hook(make_hook(hift_storage, "hift_conv_post")),
    ]
    for i in range(3):
        hift_hooks.append(hift.ups[i].register_forward_hook(make_hook(hift_storage, f"hift_ups{i}")))
        hift_hooks.append(hift.source_downs[i].register_forward_hook(make_hook(hift_storage, f"hift_sd{i}")))
        hift_hooks.append(hift.source_resblocks[i].register_forward_hook(make_hook(hift_storage, f"hift_sr{i}")))
    for i in range(9):
        hift_hooks.append(hift.resblocks[i].register_forward_hook(make_hook(hift_storage, f"hift_rb{i}")))

    # Monkeypatch SineGen's random bits so we can reproduce in C++
    sg = hift.m_source.l_sin_gen
    orig_sg_forward = sg.forward
    def sinegen_capture(f0_):
        # Save inputs
        hift_storage["hift_sinegen_input"] = f0_.detach().clone().cpu()
        # Also capture the uniform phase and gaussian noise before forward calls them
        out = orig_sg_forward(f0_)
        return out
    sg.forward = sinegen_capture

    # Also capture the Uniform sample used for phase and the randn used for noise in SineGen.
    from torch.distributions.uniform import Uniform as _Uniform
    orig_uniform_sample = _Uniform.sample
    uniform_seen = {"count": 0}
    def uniform_sample_capture(self, sample_shape=torch.Size()):
        s = orig_uniform_sample(self, sample_shape)
        if uniform_seen["count"] == 0:
            hift_storage["hift_sinegen_phase_vec"] = s.detach().clone().cpu()
            uniform_seen["count"] += 1
        return s
    _Uniform.sample = uniform_sample_capture

    orig_randn_like2 = _torch.randn_like
    rl_seen = {"count": 0}
    def randn_like_capture2(x, *a, **kw):
        s = orig_randn_like2(x, *a, **kw)
        if rl_seen["count"] == 0:
            hift_storage["hift_sinegen_noise"] = s.detach().clone().cpu()
            rl_seen["count"] += 1
        return s
    _torch.randn_like = randn_like_capture2

    # Also capture the randn used for the m_source noise branch
    orig_randn_like3 = _torch.randn_like
    # Note: m_source calls randn_like once more outside SineGen.
    # We use a counter to distinguish: first call is inside SineGen, second is the outer noise branch.

    # G2-gap fix: capture s_stft (the cat'd real+imag STFT of the source
    # signal).  HiFTGenerator.decode() does:
    #   real, imag = self._stft(s.squeeze(1))
    #   s_stft = torch.cat([real, imag], dim=1)
    # The C++ stage_H3 / stage_H4 harnesses load `hift_s_stft.npy`, so
    # capture it here by monkeypatching _stft.
    orig_hift_stft = hift._stft
    stft_seen = {"count": 0}
    def _stft_capture(x):
        real, imag = orig_hift_stft(x)
        if stft_seen["count"] == 0:
            s_stft = torch.cat([real, imag], dim=1)
            hift_storage["hift_s_stft"] = s_stft.detach().clone().cpu()
            stft_seen["count"] += 1
        return real, imag
    hift._stft = _stft_capture

    try:
        torch.manual_seed(args.seed + 1)  # Different seed so HiFT random is reproducible per run
        hift_cache = torch.zeros(1, 1, 0).to(tts.device)
        wavs, _ = tts.s3gen.mel2wav.inference(speech_feat=mel, cache_source=hift_cache)
    finally:
        sg.forward = orig_sg_forward
        _Uniform.sample = orig_uniform_sample
        _torch.randn_like = orig_randn_like2
        hift._stft = orig_hift_stft
        for h in hift_hooks:
            h.remove()

    for name, t in hift_storage.items():
        if t is None: continue
        save(t.squeeze(0) if t.ndim > 1 else t, args.out / f"{name}.npy")

    wav = wavs.squeeze(0).squeeze(0).cpu()
    save(wav, args.out / "waveform.npy")

    # ------------------------------------------------------------------
    # 5. Save also as WAV for auditory check
    # ------------------------------------------------------------------
    try:
        import torchaudio as ta
        ta.save(str(args.out / "reference.wav"), wav.unsqueeze(0), tts.sr)
        print(f"  reference.wav written ({tts.sr} Hz, {wav.numel()} samples)")
    except Exception as e:
        print(f"  WAV write failed: {e}")

    print(f"\nAll reference tensors dumped to {args.out}")


if __name__ == "__main__":
    main()
