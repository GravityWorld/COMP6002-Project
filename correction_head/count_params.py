"""Standalone parameter-count check for correction_head/model.py's
architecture, computed by plain arithmetic -- no torch import, so this
runs even in an environment without torch/opencood (it was written and
run in exactly that situation: neither is installed in the cloud sandbox
or the remote-device Linux VM this was drafted from).

This is NOT a substitute for instantiating the real nn.Module and calling
count_parameters() on it in the opencood conda env -- it's a cross-check
that the two numbers should agree, so a mismatch there means the model.py
source and this spec have drifted apart, not that either one is "right".

Run: python correction_head/count_params.py
"""

IN_CH = 384
HEAD_CH = 128
FUSE_HIDDEN = 128
EMB_DIM = 64
PAIR_HIDDEN = 128
GROUPS = 8
FRAG_AUX = 9  # Wed 7 Oct 2026 frag_extent_norm feature, dataset.py/model.py
PAIR_AUX = 2
GATE_HIDDEN = 16  # Tue 7 Oct 2026 learned distance gate, model.py PairwiseAffinityHead.gate


def conv2d_params(in_ch, out_ch, k):
    return in_ch * out_ch * k * k + out_ch  # weight + bias


def linear_params(in_f, out_f):
    return in_f * out_f + out_f


def groupnorm_params(channels):
    return 2 * channels  # affine gamma + beta


def main():
    encoder = 0
    encoder += conv2d_params(IN_CH, HEAD_CH, 1)          # reduce
    encoder += groupnorm_params(HEAD_CH)                  # gn1
    encoder += conv2d_params(HEAD_CH, HEAD_CH, 3)         # conv1
    encoder += groupnorm_params(HEAD_CH)                  # gn2
    encoder += conv2d_params(HEAD_CH, HEAD_CH, 3)         # conv2
    encoder += groupnorm_params(HEAD_CH)                  # gn3
    encoder += linear_params(HEAD_CH + FRAG_AUX, FUSE_HIDDEN)  # fuse[0]
    encoder += linear_params(FUSE_HIDDEN, EMB_DIM)              # fuse[2]

    pair_head = 0
    # 2*EMB_DIM, not EMB_DIM: forward() concatenates two EMB_DIM-wide
    # symmetric combinations (emb_i+emb_j and |emb_i-emb_j|), not one.
    # (This was originally written as EMB_DIM + PAIR_AUX here -- wrong,
    # and model.py's __init__ had the matching bug, so the two matched
    # each other but not the real forward pass. Caught by a real run of
    # verify_on_device.py; see model.py's PairwiseAffinityHead docstring.)
    pair_head += linear_params(2 * EMB_DIM + PAIR_AUX, PAIR_HIDDEN)  # mlp[0]
    pair_head += linear_params(PAIR_HIDDEN, 1)                        # mlp[2]
    # Learned distance/size-gap gate (Tue 7 Oct 2026, "option 2", decisions.md):
    # a small separate network on pair_aux alone, subtracted from mlp's logit.
    pair_head += linear_params(PAIR_AUX, GATE_HIDDEN)  # gate[0]
    pair_head += linear_params(GATE_HIDDEN, 1)         # gate[2]

    total = encoder + pair_head

    print(f"FragmentEncoder params: {encoder:,}")
    print(f"PairwiseAffinityHead params: {pair_head:,}")
    print(f"TOTAL: {total:,}")
    print()
    print("Target band (scope_and_timeline_oct19.md): hundreds of thousands")
    print("to low millions. Comms-budget estimate already on record")
    print("(task1_datasets_pipeline_dfl.md Sec 4) assumes ~300K-500K params.")
    in_band_ceiling = 100_000 <= total <= 2_000_000
    in_band_comms = 300_000 <= total <= 500_000
    print(f"Within hundreds-of-thousands-to-low-millions ceiling: {in_band_ceiling}")
    print(f"Within the ~300K-500K comms-estimate band: {in_band_comms}")

    fp32_mb = total * 4 / 1e6
    int8_kb = total * 1 / 1e3
    bit1_kb = total / 8 / 1e3
    print()
    print(f"Approx payload per round: fp32={fp32_mb:.2f}MB  int8={int8_kb:.0f}KB  1-bit={bit1_kb:.0f}KB")


if __name__ == "__main__":
    main()
