# Inkling-2-Mini — spécification d'architecture (phase recherche)

Sources d'autorité lues le 2026-11-24 :
- `config.json` de `thinkingmachines/Inkling-2-Mini` (HF)
- `src/models/inkling2.cpp` + `src/models/inkling2-loop.cpp` de `ggml-org/llama.cpp@master` (PR #32694)
- `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py` de `huggingface/transformers@main`
- en-tête GGUF parsé de `unsloth/Inkling-2-Mini-GGUF` `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`

## Hyperparamètres (text_config)

| Paramètre | Valeur |
|---|---|
| hidden_size H | 2304 |
| couches physiques | 24 (`block_count=24`) |
| boucle | `loop.start=6`, `loop.end=18`, `loop.count=3` → **48 visites logiques** (6 + 12×3 + 6) |
| vocab | 201024 padded / 200058 unpadded |
| têtes Q / KV | 24 / 8 (GQA 3:1), head_dim 128 |
| d_rel / rel_extent | 16 / 1024 (pas de RoPE, biais de position relative) |
| sliding window | 1024 sur 20 couches locales ; **globales = couches 5, 11, 17, 23** |
| sconv kernel | 4 (`shortconv.l_cache=4`) |
| attn_res block | 6 (`attn_res.block_size=6`) |
| MoE | 128 experts routés, top-8, 0 shared, intermediate 1152 |
| route_scale | 8.0 (`expert_weights_scale`) |
| SiTU | β=4.0 (gate), β₂=25.0 (up) |
| qk_l2_clip | 22.627416610717773 ; rel_l2_clip 8.0 |
| log scale | floor 8192, α=0.1 (couches globales uniquement) |
| final logit softcap | 30.0 |
| rms_norm_eps | 1e-6 |
| contexte | 1 048 576 |
| eos | 200006 |

## Tenseurs par couche physique `blk.N` (GGUF, dims au format ggml : [col, row, expert])

Communs : `attn_q [2304,3072]`, `attn_k [2304,1024]`, `attn_v [2304,1024]`,
`attn_output [3072,2304]`, `attn_gate [2304,3072]` (gate sigmoïde de sortie d'attention),
`attn_sink_proj [2304,24]`, `attn_rel [2304,384]` (24×16), `attn_rel_bias [16,1024]` (extent SWA=1024 ou 1024 global),
`attn_rel_clip [16,24]`, `attn_q_clip [128,24]`,
`attn_k_conv [4,1024]`, `attn_v_conv [4,1024]`, `attn_post_conv [4,2304]`, `ffn_post_conv [4,2304]`,
`ffn_norm [2304]`, `ffn_gate_inp [2304,128]` (router), `exp_probs_b.bias [128]`,
`ffn_gate_up_exps [2304,2304,128]` NVFP4 + `ffn_gate_up_exps.scale [128]` F32,
`ffn_down_exps [1152,2304,128]` NVFP4 + scale, `ffn_res_query [2304]` F32,
`attn_res_query [2304]` (SWA) ou `[2304,3]` (globales 5/11/17/23).

Spécifique couche **locale (SWA)** : `attn_norm [2304]`.
Spécifique couche **globale** : `attn_q_inp_norm`, `attn_k_inp_norm`, `attn_v_inp_norm [2304]`,
`attn_k_conv_proj [2304,32]`, `attn_v_conv_proj [2304,32]` (taps dynamiques du sconv).

Couches du core (6..17) : `loop_msg_query [2304,4]` + par ctrl ∈ {q,k,v,ffn} :
`loop_{c}_ctrl_down [4608,64]`, `loop_{c}_ctrl_gain [64,2304]`, `loop_{c}_ctrl_forget [64,2304]` (rank 64).

Globaux : `token_embd [2304,201024]` BF16, `token_embd_norm [2304]` F32,
`output [2304,201024]` BF16 (lm_head), `output_norm [2304]`, `output_res_query [2304]`.

## Forward d'une visite de couche (build_layer, llama.cpp)

1. **attn_res (attention sur la profondeur)** : si la couche a `attn_res_query`,
   `mixed = mix(blocks, cur, query)` — softmax(1/√H) entre `cur` rms-normé et chaque bloc
   `blocks` (entrées des couches où `ip % 6 == 0`), pondéré par la requête apprise.
   SWA : 1 requête ; global : 3 requêtes → entrées Q/K/V distinctes.
2. **contrôle de boucle** (passes 2/3 des couches 6..17) : `inputs[j] = control(inputs[j], msg[j])`
   avec `control(cur,msg) = cur - forget⊙cur + gain⊙msg`, `gain/forget = tanh(ctrl_gain/forget · (ctrl_down · concat(rms(cur),rms(msg))))`.
3. **norms** : SWA → `attn_norm` partagée Q/K/V ; global → `attn_{q,k,v}_inp_norm` séparées.
4. **projections** : Q = wq·in₀ ; K = sconv(wk·in₁, attn_k_conv [+ taps dynamiques conv_proj si global]) ;
   V = sconv(wv·in₂, attn_v_conv [+ taps]) ; R = attn_rel·in₀.
   `sconv` = conv1d causale kernel 4 + residual, sortie scalée par `1/√((dyn?2:1)·4+1)`, état récurrent de 3 pas.
5. **clips** : Q × attn_q_clip (par canal) puis L2-clip à qk_clip ; K L2-clip à qk_clip ;
   V → rms_norm sans gamma ; R × attn_rel_clip puis L2-clip à rel_clip.
   Global : Q,R × τ(pos), τ = 1 + α·log(max(1,(pos+1)/8192)).
6. **biais relatif + sinks** : `table = pad(attn_rel_bias,+1) @ R` → gather par distance relative
   (clampée à la fenêtre SWA ou à rel_extent) → `kq_b [head, kv, q]` ; on soustrait
   `sinks = attn_sink_proj·in₀` ; softmax avec un sink fictif de logit 0 (dénominateur +1).
7. **attention** : softmax(Q·K/√128 + kq_b, masque causal/SWA) · V.
8. **sortie attention** : `out = attn_out ⊙ sigmoid(attn_gate·in₀)`, puis `sconv(wo·out, attn_post_conv)`.
9. **residual par bloc de 6** : si `ip % 6 == 0`, `blocks.push(inp)` (nouveau stream) ; sinon `cur = inp + out`.
10. **FFN MoE** : `cur = mix(blocks, cur, ffn_res_query)` [+ control FFN si boucle] → `ffn_norm` →
    logits router = `ffn_gate_inp·x + exp_probs_b` (F32), top-8, sigmoïde, × route_scale 8.0 ;
    experts : `down( SiTU(gate_up(x)) )` avec **SiTU(g,u) = β·tanh(g/β)·sigmoid(g)·β₂·tanh(u/β₂)** ;
    puis `sconv(ffn_out, ffn_post_conv)` ; `cur = ffn_inp + out`.
11. **sortie modèle** : `mix(blocks, cur, output_res_query)` → `output_norm` → lm_head →
    softcap 30 (`tanh(x/30)·30`) → tronqué à 200058 tokens.

## Boucle (inkling2-loop.cpp)

- 24 blocs physiques ; les visites 24..47 réutilisent les poids des couches 6..17 (`layers[il] = layers[layer_phys(il)]`), chaque visite a **son propre KV cache et son propre état sconv**.
- Passe 0 : couches 0..23 séquentielles. Passes 1/2 : on repart de la sortie de la couche 5 (prelude),
  `bank` = blocs accumulés après le premier bloc du core + sortie de la passe précédente ;
  `messages = mix(bank, bank_out, loop_msg_query)` (4 messages Q/K/V/FFN) injectés via `control`.
- KV cache et état conv dimensionnés pour **48 visites**.

## État récurrent sconv par visite

canaux = 2·1024 (K,V) + 2·2304 (attn_post, ffn_post) = 6656 ; history = 3 pas ;
offsets : K@0, V@1024, attn_post@2048, ffn_post@4352.
