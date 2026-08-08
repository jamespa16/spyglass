// H1 probe: does Qwen3.6-27B's residual-stream channel banding ([0,1024) /
// [1024,4096) / [4096,5120)) correspond to image-token vs text-token usage?
//
// Runs a single image+text prompt through the real model via mtmd, and for
// every layer's residual-stream output ("l_out-N") plus the final
// post-output_norm hidden state ("result_norm"), accumulates per-channel
// activation statistics (mean, mean|.|, std) split by whether the token
// position came from the image chunk or a text chunk.
//
// Usage:
//   PROBE_OUT=out.json ./qwen_modality_probe -m <model.gguf> --mmproj <mmproj.gguf> \
//       --image <image.jpg> [-p "custom prompt containing <__media__>"] [-ngl 99]

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

enum probe_label { LABEL_TEXT = 0, LABEL_IMAGE = 1, LABEL_COUNT = 2 };

struct channel_stats {
    std::vector<double> sum;
    std::vector<double> sumsq;
    std::vector<double> sum_abs;
    int64_t n = 0;
};

struct tensor_accum {
    channel_stats stats[LABEL_COUNT];
};

struct probe_state {
    std::unordered_map<std::string, tensor_accum> tensors;
    probe_label current_label = LABEL_TEXT;
    std::vector<uint8_t> scratch;
};

static bool is_interesting_tensor(const std::string & name) {
    return name.rfind("l_out-", 0) == 0 || name == "result_norm" || name == "model.input_embed";
}

static float get_f32(const uint8_t * data, const size_t * nb, int64_t i0, int64_t i1) {
    return *(const float *) (data + i1 * nb[1] + i0 * nb[0]);
}

static bool probe_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (probe_state *) user_data;
    std::string name = t->name;

    if (!is_interesting_tensor(name)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "probe: skipping %s, unexpected type %s\n", name.c_str(), ggml_type_name(t->type));
        return true;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data;
    if (!is_host) {
        size_t n_bytes = ggml_nbytes(t);
        state->scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, state->scratch.data(), 0, n_bytes);
        data = state->scratch.data();
    } else {
        data = (const uint8_t *) t->data;
    }

    const int64_t n_embd = t->ne[0];
    const int64_t n_tok  = t->ne[1];

    auto & accum = state->tensors[name];
    auto & stats = accum.stats[state->current_label];
    if ((int64_t) stats.sum.size() != n_embd) {
        stats.sum.assign(n_embd, 0.0);
        stats.sumsq.assign(n_embd, 0.0);
        stats.sum_abs.assign(n_embd, 0.0);
    }

    for (int64_t tk = 0; tk < n_tok; tk++) {
        for (int64_t c = 0; c < n_embd; c++) {
            double v = get_f32(data, t->nb, c, tk);
            stats.sum[c]     += v;
            stats.sumsq[c]   += v * v;
            stats.sum_abs[c] += std::fabs(v);
        }
    }
    stats.n += n_tok;

    return true;
}

static void dump_json(const probe_state & state, int64_t n_embd, const std::string & out_path) {
    json root;
    root["n_embd"] = n_embd;
    root["labels"] = { "text", "image" };

    json tensors = json::object();
    for (const auto & [name, accum] : state.tensors) {
        json entry;
        for (int lbl = 0; lbl < LABEL_COUNT; lbl++) {
            const auto & s = accum.stats[lbl];
            json lbl_json;
            lbl_json["n"] = s.n;
            std::vector<double> mean(n_embd, 0.0), mean_abs(n_embd, 0.0), stdev(n_embd, 0.0);
            for (int64_t c = 0; c < n_embd && c < (int64_t) s.sum.size(); c++) {
                if (s.n > 0) {
                    mean[c]     = s.sum[c] / s.n;
                    mean_abs[c] = s.sum_abs[c] / s.n;
                    double var  = s.sumsq[c] / s.n - mean[c] * mean[c];
                    stdev[c]    = std::sqrt(std::max(0.0, var));
                }
            }
            lbl_json["mean"]     = mean;
            lbl_json["mean_abs"] = mean_abs;
            lbl_json["std"]      = stdev;
            entry[lbl == LABEL_TEXT ? "text" : "image"] = std::move(lbl_json);
        }
        tensors[name] = std::move(entry);
    }
    root["tensors"] = std::move(tensors);

    std::ofstream f(out_path);
    f << root.dump();
    fprintf(stderr, "probe: wrote %s (%zu tensors)\n", out_path.c_str(), state.tensors.size());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) {
        return 1;
    }
    if (params.mmproj.path.empty()) {
        fprintf(stderr, "probe: --mmproj is required\n");
        return 1;
    }
    if (params.image.empty()) {
        fprintf(stderr, "probe: --image is required\n");
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(params.numa);

    probe_state state;
    params.cb_eval = probe_cb_eval;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    if (!model || !lctx) {
        fprintf(stderr, "probe: failed to load model\n");
        return 1;
    }
    const int64_t n_embd = llama_model_n_embd(model);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = params.mmproj_use_gpu;
    mparams.n_threads       = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    mparams.warmup          = false;

    mtmd_context * ctx_vision = mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams);
    if (!ctx_vision) {
        fprintf(stderr, "probe: failed to load mmproj from %s\n", params.mmproj.path.c_str());
        return 1;
    }

    auto bm = mtmd_helper_bitmap_init_from_file(ctx_vision, params.image[0].c_str(), false);
    if (!bm.bitmap) {
        fprintf(stderr, "probe: failed to load image %s\n", params.image[0].c_str());
        return 1;
    }

    std::string prompt = params.prompt.empty()
        ? std::string("Describe this newspaper front page in detail, including the headline text.")
        : params.prompt;
    if (prompt.find(mtmd_default_marker()) == std::string::npos) {
        prompt = std::string(mtmd_default_marker()) + prompt;
    }

    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.text_len       = prompt.size();
    text.add_special    = true;
    text.parse_special  = true;

    const mtmd_bitmap * bitmaps_c[1] = { bm.bitmap };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int32_t res = mtmd_tokenize(ctx_vision, chunks, &text, bitmaps_c, 1);
    if (res != 0) {
        fprintf(stderr, "probe: mtmd_tokenize failed, res=%d\n", res);
        return 1;
    }

    llama_pos n_past = 0;
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    fprintf(stderr, "probe: prompt tokenized into %zu chunks\n", n_chunks);

    for (size_t i = 0; i < n_chunks; i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        auto ctype = mtmd_input_chunk_get_type(chunk);
        size_t n_tok_chunk = mtmd_input_chunk_get_n_tokens(chunk);

        state.current_label = (ctype == MTMD_INPUT_CHUNK_TYPE_IMAGE) ? LABEL_IMAGE : LABEL_TEXT;

        fprintf(stderr, "probe: chunk %zu/%zu type=%s n_tokens=%zu n_past=%d\n",
                i + 1, n_chunks, ctype == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "IMAGE" : "TEXT",
                n_tok_chunk, n_past);

        if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            llama_pos new_n_past = n_past;
            res = mtmd_helper_eval_chunk_single(ctx_vision, lctx, chunk, n_past, 0, params.n_batch, false, &new_n_past);
            if (res != 0) {
                fprintf(stderr, "probe: text chunk %zu decode failed, res=%d\n", i, res);
                return 1;
            }
            n_past = new_n_past;
        } else {
            res = mtmd_encode_chunk(ctx_vision, chunk);
            if (res != 0) {
                fprintf(stderr, "probe: image chunk %zu encode failed, res=%d\n", i, res);
                return 1;
            }
            float * embd = mtmd_get_output_embd(ctx_vision);
            llama_pos new_n_past = n_past;
            res = mtmd_helper_decode_image_chunk(ctx_vision, lctx, chunk, embd, n_past, 0, params.n_batch,
                                                  &new_n_past, nullptr, nullptr);
            if (res != 0) {
                fprintf(stderr, "probe: image chunk %zu decode failed, res=%d\n", i, res);
                return 1;
            }
            n_past = new_n_past;
        }
    }

    fprintf(stderr, "probe: done, n_past=%d, n_embd=%lld\n", n_past, (long long) n_embd);

    const char * out_env = std::getenv("PROBE_OUT");
    dump_json(state, n_embd, out_env ? out_env : "probe_output.json");

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bm.bitmap);
    mtmd_free(ctx_vision);
    llama_backend_free();

    return 0;
}
