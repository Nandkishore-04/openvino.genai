// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/text2speech_pipeline.h"

#define CHECK_STATUS(return_status)                                                           \
    if (return_status != 0) {                                                                 \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", (int)return_status, __LINE__); \
        goto err;                                                                             \
    }

// Minimal WAV header structure
// This removes struct padding to ensure the Wav Header matches the exact binary layout
// required by the WAV specification, ensuring the output file is valid.
#pragma pack(push, 1)
typedef struct {
    char chunk_id[4];
    uint32_t chunk_size;
    char format[4];
    char subchunk1_id[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char subchunk2_id[4];
    uint32_t subchunk2_size;
} WavHeader;
#pragma pack(pop)

void save_to_wav(const float* waveform_ptr, size_t waveform_size, const char* file_path) {
    uint32_t sample_rate = 16000;
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 32;  // IEEE Float

    WavHeader header;
    memcpy(header.chunk_id, "RIFF", 4);
    header.chunk_size = 36 + waveform_size * sizeof(float);
    memcpy(header.format, "WAVE", 4);
    memcpy(header.subchunk1_id, "fmt ", 4);
    header.subchunk1_size = 16;
    header.audio_format = 3;  // IEEE Float
    header.num_channels = num_channels;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * num_channels * sizeof(float);
    header.block_align = num_channels * sizeof(float);
    header.bits_per_sample = bits_per_sample;
    memcpy(header.subchunk2_id, "data", 4);
    header.subchunk2_size = waveform_size * sizeof(float);

    FILE* file = fopen(file_path, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", file_path);
        return;
    }
    fwrite(&header, sizeof(WavHeader), 1, file);
    fwrite(waveform_ptr, sizeof(float), waveform_size, file);
    fclose(file);
}

ov_tensor_t* read_speaker_embedding(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open speaker embedding file: %s\n", file_path);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size != 512 * sizeof(float)) {
        fprintf(stderr, "Speaker embedding file must be 512 floats (2048 bytes).\n");
        fclose(file);
        return NULL;
    }

    float* data = (float*)malloc(file_size);
    fread(data, 1, file_size, file);
    fclose(file);

    ov_tensor_t* tensor = NULL;
    ov_shape_t shape;
    shape.rank = 2;
    int64_t dims[] = {1, 512};
    shape.dims = dims;

    // Create tensor from host ptr
    ov_tensor_create(f32, shape, &tensor);
    void* tensor_data = NULL;
    ov_tensor_data(tensor, &tensor_data);
    memcpy(tensor_data, data, file_size);
    free(data);

    return tensor;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<PROMPT>\" [<SPEAKER_EMBEDDING_BIN_FILE>]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* model_dir = argv[1];
    const char* prompt = argv[2];
    const char* speaker_embedding_path = (argc == 4) ? argv[3] : NULL;
    const char* device = "CPU";

    ov_genai_text2speech_pipeline* pipeline = NULL;
    ov_genai_text2speech_decoded_results* results = NULL;
    ov_tensor_t* speaker_embedding = NULL;

    CHECK_STATUS(ov_genai_text2speech_pipeline_create(model_dir, device, 0, &pipeline));

    if (speaker_embedding_path) {
        speaker_embedding = read_speaker_embedding(speaker_embedding_path);
        if (!speaker_embedding)
            goto err;
    }

    const char* texts[] = {prompt};
    CHECK_STATUS(ov_genai_text2speech_pipeline_generate(pipeline, texts, 1, speaker_embedding, 0, &results));

    size_t count = 0;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speeches_count(results, &count));
    if (count != 1) {
        fprintf(stderr, "Expected exactly one decoded waveform\n");
        goto err;
    }

    ov_tensor_t* speech_tensor = NULL;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speech_at(results, 0, &speech_tensor));

    void* waveform_data = NULL;
    ov_tensor_data(speech_tensor, &waveform_data);

    ov_shape_t shape;
    ov_tensor_get_shape(speech_tensor, &shape);
    size_t waveform_size = 1;
    for (size_t i = 0; i < shape.rank; ++i)
        waveform_size *= shape.dims[i];
    ov_shape_free(&shape);

    const char* output_file = "output_audio.wav";
    save_to_wav((const float*)waveform_data, waveform_size, output_file);
    printf("[Info] Text successfully converted to audio file \"%s\".\n", output_file);

    ov_tensor_free(speech_tensor);

err:
    if (results)
        ov_genai_text2speech_decoded_results_free(results);
    if (pipeline)
        ov_genai_text2speech_pipeline_free(pipeline);
    if (speaker_embedding)
        ov_tensor_free(speaker_embedding);

    return EXIT_SUCCESS;
}
