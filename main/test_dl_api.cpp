#include "dl_model_base.hpp"
#include "dl_module_add.hpp"
#include "dl_module_creator.hpp"
#include "dl_module_relu.hpp"
#include "driver/sdmmc_host.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_jpeg_dec.h"
#include "esp_vfs_fat.h"
#include "hwcheck_input.h"
#include "sdmmc_cmd.h"
#include "unity.h"
#include <sys/stat.h>
#include <type_traits>
static const char *TAG = "TEST DL MODEL";

using namespace dl;

uint8_t key[16] = {0x8a, 0x7f, 0xc9, 0x61, 0xe4, 0xe6, 0xff, 0x0a, 0xd2, 0x64, 0x36, 0x95, 0x28, 0x75, 0xae, 0x4a};

typedef struct {
    const char *name;
    bool requested_param_copy;
    bool effective_param_copy;
    int total_ram_used;
    int internal_ram_used;
    int psram_used;
    int64_t avg_infer_us;
} model_storage_perf_t;

static model_storage_perf_t benchmark_model_storage(const char *name, bool param_copy)
{
    const int warmup_runs = 2;
    const int timed_runs = 10;
    const int total_ram_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    const int internal_ram_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    const int psram_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);

    Model *model = new Model("model",
                             0,
                             fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
                             0,
                             MEMORY_MANAGER_GREEDY,
                             nullptr,
                             param_copy);
    TEST_ASSERT_NOT_NULL(model);
    TEST_ASSERT_NOT_NULL(model->get_fbs_model());
    TEST_ASSERT_EQUAL(param_copy, model->get_fbs_model()->m_param_copy);

    const int total_ram_after_load = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    const int internal_ram_after_load = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    const int psram_after_load = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    const bool effective_param_copy = model->get_fbs_model()->m_param_copy;

    for (int i = 0; i < warmup_runs; i++) {
        model->run();
    }

    int64_t total_infer_us = 0;
    for (int i = 0; i < timed_runs; i++) {
        const int64_t start_us = esp_timer_get_time();
        model->run();
        total_infer_us += esp_timer_get_time() - start_us;
    }

    delete model;
    module::ModuleCreator::get_instance()->clear();

    const int total_ram_after_delete = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_after_delete + 1024 >= total_ram_before);

    model_storage_perf_t result;
    result.name = name;
    result.requested_param_copy = param_copy;
    result.effective_param_copy = effective_param_copy;
    result.total_ram_used = total_ram_before - total_ram_after_load;
    result.internal_ram_used = internal_ram_before - internal_ram_after_load;
    result.psram_used = psram_before - psram_after_load;
    result.avg_infer_us = total_infer_us / timed_runs;

    ESP_LOGI(TAG,
             "%s: requested_param_copy=%d, effective_param_copy=%d, avg_infer=%lld us, total_ram=%d B, internal_ram=%d B, psram=%d B",
             result.name,
             result.requested_param_copy,
             result.effective_param_copy,
             (long long)result.avg_infer_us,
             result.total_ram_used,
             result.internal_ram_used,
             result.psram_used);

    return result;
}

TEST_CASE("Test dl model API: load()", "[api]")
{
    ESP_LOGI(TAG, "Test dl model API: load()");
    int internal_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    Model *model = new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, key);
    delete model;
    module::ModuleCreator::get_instance()->clear();

    int internal_ram_size_second = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    fbs::FbsLoader *fbs_loader = new fbs::FbsLoader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    fbs::FbsModel *fbs_model = fbs_loader->load(0, key);
    Model *model2 = new Model(fbs_model);
    delete model2;
    delete fbs_loader;
    delete fbs_model;
    module::ModuleCreator::get_instance()->clear();

    int internal_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    int psram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);

    ESP_LOGI(TAG,
             "internal ram size before: %d, second: %d, end:%d",
             internal_ram_size_before,
             internal_ram_size_second,
             internal_ram_size_end);
    ESP_LOGI(TAG, "psram size before: %d, end:%d", psram_size_before, psram_size_end);
    TEST_ASSERT_EQUAL(true, internal_ram_size_before - internal_ram_size_second < 1000);
    TEST_ASSERT_EQUAL(true, internal_ram_size_second == internal_ram_size_end);
    TEST_ASSERT_EQUAL(true, psram_size_before == psram_size_end);
}

TEST_CASE("Test dl model API: profile()", "[api]")
{
    ESP_LOGI(TAG, "Test dl model API: run()");
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    delete model;

    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

    model->profile();
    model->minimize();
    model->profile_memory();
    delete model;

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before <= total_ram_size_end);
}

TEST_CASE("Test dl model API: run()", "[api]")
{
    ESP_LOGI(TAG, "Test dl model API: run()");
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    delete model;

    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    dl::tool::Latency latency;
    for (int i = 0; i < 15; i++) {
        model = new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, (i % 2) * 100000);

        latency.start();
        model->run();
        latency.end();
        printf("run:%ld ms\n", latency.get_period() / 1000);
        delete model;
    }

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}

TEST_CASE("Test dl model API: single-core vs multi-core runtime", "[coretest]")
{
    Model *model = new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, nullptr, true);
    TEST_ASSERT_NOT_NULL(model);

    const int warmup_runs = 1;
    const int timed_runs = 5;

    runtime_mode_t modes[] = {RUNTIME_MODE_SINGLE_CORE, RUNTIME_MODE_MULTI_CORE, RUNTIME_MODE_AUTO};
    const char *names[] = {"SINGLE_CORE", "MULTI_CORE", "AUTO"};

    for (int m = 0; m < 3; m++) {
        for (int i = 0; i < warmup_runs; i++) {
            model->run(modes[m]);
        }
        int64_t total_us = 0;
        for (int i = 0; i < timed_runs; i++) {
            int64_t t0 = esp_timer_get_time();
            model->run(modes[m]);
            int64_t dt = esp_timer_get_time() - t0;
            total_us += dt;
            printf("CORETEST[%s][%d] %lld us\n", names[m], i, (long long)dt);
        }
        printf("CORETEST_SUMMARY[%s] avg_us=%lld\n", names[m], (long long)(total_us / timed_runs));
        fflush(stdout);
    }

    delete model;
}

TEST_CASE("Test dl model API: inference with weights in RAM vs flash", "[api][perf]")
{
    ESP_LOGI(TAG, "Test dl model API: inference with weights in RAM vs flash");

    model_storage_perf_t weights_in_ram = benchmark_model_storage("weights_in_ram", true);
    model_storage_perf_t weights_in_flash = benchmark_model_storage("weights_in_flash", false);

    ESP_LOGI(TAG,
             "weights_in_ram avg=%lld us, weights_in_flash avg=%lld us, delta=%lld us",
             (long long)weights_in_ram.avg_infer_us,
             (long long)weights_in_flash.avg_infer_us,
             (long long)(weights_in_flash.avg_infer_us - weights_in_ram.avg_infer_us));
    ESP_LOGI(TAG,
             "weights_in_ram total_ram=%d B, weights_in_flash total_ram=%d B, saved=%d B",
             weights_in_ram.total_ram_used,
             weights_in_flash.total_ram_used,
             weights_in_ram.total_ram_used - weights_in_flash.total_ram_used);

    TEST_ASSERT_TRUE(weights_in_flash.total_ram_used <= weights_in_ram.total_ram_used + 4096);
}

TEST_CASE("HW vs Python: fixed input comparison", "[hwcheck]")
{
    // 10 real images (sense_224/images_x_y_224, first 10 deterministic eval
    // images), each run through PPQ's TorchExecutor int8 simulation for
    // modelo_01/eq on Colab (esp-ppq, equalization x4 + bias_correct).
    // Reference int8sim values are printed alongside each device result by
    // the accompanying comparison script; see espdl-hwcheck/hwcheck10.json.
    Model *model = new Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    TEST_ASSERT_NOT_NULL(model);

    for (int img = 0; img < hwcheck_num_images; img++) {
        TensorBase *input = new TensorBase({1, 224, 224, 3}, hwcheck_inputs[img], -7, DATA_TYPE_INT8);
        TEST_ASSERT_EQUAL(hwcheck_input_len, input->get_size());

        model->run(input);

        auto &outputs = model->get_outputs();
        TEST_ASSERT_TRUE(outputs.find("obstaculo") != outputs.end());
        TEST_ASSERT_TRUE(outputs.find("desvio") != outputs.end());

        TensorBase *obstaculo = outputs.at("obstaculo");
        TensorBase *desvio = outputs.at("desvio");
        float obstaculo_scale = exp2f((float)obstaculo->exponent);
        float desvio_scale = exp2f((float)desvio->exponent);

        printf("HWCHECK[%d] obstaculo_q: %d %d (exponent %d, scale %f)\n",
               img,
               obstaculo->get_element<int8_t>(0),
               obstaculo->get_element<int8_t>(1),
               obstaculo->exponent,
               obstaculo_scale);
        printf("HWCHECK[%d] obstaculo_dequant: %f %f\n",
               img,
               obstaculo->get_element<int8_t>(0) * obstaculo_scale,
               obstaculo->get_element<int8_t>(1) * obstaculo_scale);
        printf("HWCHECK[%d] desvio_q: %d %d %d (exponent %d, scale %f)\n",
               img,
               desvio->get_element<int8_t>(0),
               desvio->get_element<int8_t>(1),
               desvio->get_element<int8_t>(2),
               desvio->exponent,
               desvio_scale);
        printf("HWCHECK[%d] desvio_dequant: %f %f %f\n",
               img,
               desvio->get_element<int8_t>(0) * desvio_scale,
               desvio->get_element<int8_t>(1) * desvio_scale,
               desvio->get_element<int8_t>(2) * desvio_scale);

        delete input;
    }

    delete model;
}

#define STREAM_IMG_BYTES (224 * 224 * 3)
// 3072 divides STREAM_IMG_BYTES evenly (49 chunks/image) and divides evenly
// by 3, so base64 needs no padding: 3072 raw bytes -> exactly 4096 chars.
#define STREAM_CHUNK_RAW_BYTES 3072
#define STREAM_CHUNK_B64_CHARS 4096

// Chunks travel as base64 TEXT lines via fgets/printf, not raw bytes via
// fread(): the console VFS applies CR/LF line-ending translation to stdio
// reads (CONFIG_LIBC_STDIN_LINE_ENDING_CR), which silently mangles any 0x0D
// byte in a raw binary payload and desyncs the stream. Base64's alphabet
// never contains 0x0D/0x0A, so it rides through the already-working
// fgets/printf text path with no corruption risk.
static int b64_val(char c)
{
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

static size_t b64_decode(const char *in, uint8_t *out)
{
    // '=' padding matters here: unlike the original fixed 150528-byte buffers
    // (always an exact multiple of the 3072-byte chunk size, so base64 never
    // needed padding), arbitrary files (e.g. JPEGs) almost never end on a
    // multiple-of-3 boundary. Silently skipping '=' (treating it as just
    // another invalid char to ignore) permanently drops the last partial
    // group's byte(s), under-reporting the decoded length forever and
    // deadlocking the caller's "read until N bytes" loop.
    size_t out_len = 0;
    int vals[4];
    int n = 0;
    for (const char *p = in; *p && *p != '\n' && *p != '\r'; p++) {
        if (*p == '=') {
            break;
        }
        int v = b64_val(*p);
        if (v < 0) {
            continue;
        }
        vals[n++] = v;
        if (n == 4) {
            out[out_len++] = (uint8_t)((vals[0] << 2) | (vals[1] >> 4));
            out[out_len++] = (uint8_t)(((vals[1] & 0xF) << 4) | (vals[2] >> 2));
            out[out_len++] = (uint8_t)(((vals[2] & 0x3) << 6) | vals[3]);
            n = 0;
        }
    }
    if (n == 2) {
        out[out_len++] = (uint8_t)((vals[0] << 2) | (vals[1] >> 4));
    } else if (n == 3) {
        out[out_len++] = (uint8_t)((vals[0] << 2) | (vals[1] >> 4));
        out[out_len++] = (uint8_t)(((vals[1] & 0xF) << 4) | (vals[2] >> 2));
    }
    return out_len;
}

// Streaming test: the host sends one image at a time over the same serial
// link used for logging (COM12, raw pyserial, not idf.py monitor), instead
// of embedding the whole eval set as flash-resident C arrays. 100 real
// 224x224x3 int8 images would be ~14.7 MB of literal source data, which does
// not fit in the flash left free after the app + model partitions, so
// streaming is the only way to cover the entire test set on this board.
// Protocol per image: host writes "I\n", then the 150528-byte NHWC int8
// buffer in STREAM_CHUNK_BYTES chunks; device ACKs each chunk with a single
// 'A' byte so the host never outruns the UART RX ring buffer. Host sends
// "DONE\n" instead of "I\n" to end the run.
TEST_CASE("Stream test: full aval set over serial", "[stream]")
{
    // The console's stdin is non-blocking by default (fgets() returns EOF
    // instantly if nothing is queued yet), so it never actually waits for
    // the host's "I\n"/image data -- it just falls straight through. Only
    // switching stdio to the interrupt-driven UART driver makes reads block
    // until data arrives.
    if (!uart_is_driver_installed(UART_NUM_0)) {
        uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    }
    uart_vfs_dev_use_driver(UART_NUM_0);

    int psram_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    int internal_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

    // param_copy=true (explicit) copies model weights from flash into PSRAM
    // at load time -- also esp-dl's default, spelled out here since this run
    // is meant to characterize PSRAM-backed inference.
    Model *model =
        new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, nullptr, true);
    TEST_ASSERT_NOT_NULL(model);
    TEST_ASSERT_EQUAL(true, model->get_fbs_model()->m_param_copy);

    int psram_after = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    int internal_after = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    printf("STREAM_META param_copy=1 psram_used_by_model=%d internal_used_by_model=%d\n",
           psram_before - psram_after,
           internal_before - internal_after);

    uint8_t *img_buf = (uint8_t *)heap_caps_malloc(STREAM_IMG_BYTES, MALLOC_CAP_SPIRAM);
    TEST_ASSERT_NOT_NULL(img_buf);

    printf("STREAM_READY\n");
    fflush(stdout);

    int idx = 0;
    int64_t first_infer_us = -1;
    int64_t total_infer_us = 0;

    while (true) {
        char cmd[8] = {0};
        if (fgets(cmd, sizeof(cmd), stdin) == nullptr) {
            break;
        }
        if (cmd[0] == 'D') { // "DONE\n"
            break;
        }

        static char line[STREAM_CHUNK_B64_CHARS + 4];
        size_t received = 0;
        bool ok = true;
        while (received < STREAM_IMG_BYTES) {
            if (fgets(line, sizeof(line), stdin) == nullptr) {
                printf("STREAM_ERR short_read img=%d at %d\n", idx, (int)received);
                ok = false;
                break;
            }
            size_t got = b64_decode(line, img_buf + received);
            received += got;
            printf("A\n");
            fflush(stdout);
        }
        TEST_ASSERT_TRUE(ok);
        TEST_ASSERT_EQUAL(STREAM_IMG_BYTES, received);

        TensorBase *input = new TensorBase({1, 224, 224, 3}, (int8_t *)img_buf, -7, DATA_TYPE_INT8);

        int64_t t0 = esp_timer_get_time();
        model->run(input, RUNTIME_MODE_MULTI_CORE); // ~1.85x faster than SINGLE_CORE (measured); AUTO does not pick it
        int64_t dt = esp_timer_get_time() - t0;
        if (idx == 0) {
            first_infer_us = dt;
        }
        total_infer_us += dt;

        auto &outputs = model->get_outputs();
        TEST_ASSERT_TRUE(outputs.find("obstaculo") != outputs.end());
        TEST_ASSERT_TRUE(outputs.find("desvio") != outputs.end());
        TensorBase *obstaculo = outputs.at("obstaculo");
        TensorBase *desvio = outputs.at("desvio");
        float obstaculo_scale = exp2f((float)obstaculo->exponent);
        float desvio_scale = exp2f((float)desvio->exponent);

        printf("STREAM[%d] latency_us=%lld\n", idx, (long long)dt);
        printf("STREAM[%d] obstaculo_q: %d %d (exponent %d, scale %f)\n",
               idx,
               obstaculo->get_element<int8_t>(0),
               obstaculo->get_element<int8_t>(1),
               obstaculo->exponent,
               obstaculo_scale);
        printf("STREAM[%d] obstaculo_dequant: %f %f\n",
               idx,
               obstaculo->get_element<int8_t>(0) * obstaculo_scale,
               obstaculo->get_element<int8_t>(1) * obstaculo_scale);
        printf("STREAM[%d] desvio_q: %d %d %d (exponent %d, scale %f)\n",
               idx,
               desvio->get_element<int8_t>(0),
               desvio->get_element<int8_t>(1),
               desvio->get_element<int8_t>(2),
               desvio->exponent,
               desvio_scale);
        printf("STREAM[%d] desvio_dequant: %f %f %f\n",
               idx,
               desvio->get_element<int8_t>(0) * desvio_scale,
               desvio->get_element<int8_t>(1) * desvio_scale,
               desvio->get_element<int8_t>(2) * desvio_scale);
        fflush(stdout);

        delete input;
        idx++;
    }

    printf("STREAM_SUMMARY count=%d first_infer_us=%lld avg_infer_us=%lld\n",
           idx,
           (long long)first_infer_us,
           (long long)(idx ? total_infer_us / idx : 0));
    fflush(stdout);

    heap_caps_free(img_buf);
    delete model;
}

// SD card pilot: this board's microSD slot is wired for SDMMC 1-line mode
// (CLK=39, CMD=38, D0=40 -- the common ESP32-S3-CAM N16R8/OV5640 reference
// design). Measures whether pre-loading images onto the card (instead of
// streaming each one over serial) is fast enough to cover the full
// 3971-image test set: receives 5 images over serial (same base64 protocol
// as [stream]), times writing each to /sdcard, then times reading each back
// + running inference, separately from transfer time.
#define SD_MOUNT_POINT "/sdcard"
#define SD_TEST_IMAGES 5

TEST_CASE("SD card pilot: write+read+infer timing", "[sdtest]")
{
    if (!uart_is_driver_installed(UART_NUM_0)) {
        uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    }
    uart_vfs_dev_use_driver(UART_NUM_0);

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED; // "most velocity": 40 MHz, the max for 1-bit SD (non-MMC)

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1;
    slot_config.clk = GPIO_NUM_39;
    slot_config.cmd = GPIO_NUM_38;
    slot_config.d0 = GPIO_NUM_40;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = true, // format the card if it doesn't mount cleanly
        .max_files = 8,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_card_t *card = nullptr;
    esp_err_t mret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &card);
    TEST_ASSERT_EQUAL(ESP_OK, mret);
    printf("SDTEST_MOUNT ok freq_khz=%d card_type=%s sectors=%llu sector_size=%d\n",
           host.max_freq_khz,
           (card->is_mmc ? "MMC" : (card->is_sdio ? "SDIO" : "SD")),
           (unsigned long long)card->csd.capacity,
           card->csd.sector_size);
    fflush(stdout);

    Model *model =
        new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, nullptr, true);
    TEST_ASSERT_NOT_NULL(model);

    uint8_t *img_buf = (uint8_t *)heap_caps_malloc(STREAM_IMG_BYTES, MALLOC_CAP_SPIRAM);
    TEST_ASSERT_NOT_NULL(img_buf);

    printf("SDTEST_READY\n");
    fflush(stdout);

    static char line[STREAM_CHUNK_B64_CHARS + 4];
    char path[64];

    // Phase 1: receive each image over serial, time writing it to the SD card.
    for (int idx = 0; idx < SD_TEST_IMAGES; idx++) {
        char cmd[8] = {0};
        TEST_ASSERT_NOT_NULL(fgets(cmd, sizeof(cmd), stdin)); // "I\n"

        size_t received = 0;
        while (received < STREAM_IMG_BYTES) {
            TEST_ASSERT_NOT_NULL(fgets(line, sizeof(line), stdin));
            received += b64_decode(line, img_buf + received);
            printf("A\n");
            fflush(stdout);
        }
        TEST_ASSERT_EQUAL(STREAM_IMG_BYTES, received);

        snprintf(path, sizeof(path), SD_MOUNT_POINT "/img%d.bin", idx);
        int64_t t0 = esp_timer_get_time();
        FILE *f = fopen(path, "wb");
        TEST_ASSERT_NOT_NULL(f);
        size_t wrote = fwrite(img_buf, 1, STREAM_IMG_BYTES, f);
        fclose(f);
        int64_t write_us = esp_timer_get_time() - t0;
        TEST_ASSERT_EQUAL(STREAM_IMG_BYTES, wrote);

        printf("SDTEST_WRITE[%d] write_us=%lld\n", idx, (long long)write_us);
        fflush(stdout);
    }

    // Phase 2: read each image back from the SD card and run inference,
    // isolating card-read time from serial-transfer time.
    int64_t first_read_us = -1, first_infer_us = -1;
    for (int idx = 0; idx < SD_TEST_IMAGES; idx++) {
        snprintf(path, sizeof(path), SD_MOUNT_POINT "/img%d.bin", idx);

        int64_t t0 = esp_timer_get_time();
        FILE *f = fopen(path, "rb");
        TEST_ASSERT_NOT_NULL(f);
        size_t got = fread(img_buf, 1, STREAM_IMG_BYTES, f);
        fclose(f);
        int64_t read_us = esp_timer_get_time() - t0;
        TEST_ASSERT_EQUAL(STREAM_IMG_BYTES, got);
        if (idx == 0) {
            first_read_us = read_us;
        }

        TensorBase *input = new TensorBase({1, 224, 224, 3}, (int8_t *)img_buf, -7, DATA_TYPE_INT8);
        int64_t t1 = esp_timer_get_time();
        model->run(input, RUNTIME_MODE_MULTI_CORE); // ~1.85x faster than SINGLE_CORE (measured); AUTO does not pick it
        int64_t infer_us = esp_timer_get_time() - t1;
        if (idx == 0) {
            first_infer_us = infer_us;
        }

        auto &outputs = model->get_outputs();
        TensorBase *obstaculo = outputs.at("obstaculo");
        TensorBase *desvio = outputs.at("desvio");
        float obstaculo_scale = exp2f((float)obstaculo->exponent);
        float desvio_scale = exp2f((float)desvio->exponent);

        printf("SDTEST_READ[%d] read_us=%lld infer_us=%lld\n", idx, (long long)read_us, (long long)infer_us);
        printf("SDTEST_READ[%d] obstaculo_dequant: %f %f\n",
               idx,
               obstaculo->get_element<int8_t>(0) * obstaculo_scale,
               obstaculo->get_element<int8_t>(1) * obstaculo_scale);
        printf("SDTEST_READ[%d] desvio_dequant: %f %f %f\n",
               idx,
               desvio->get_element<int8_t>(0) * desvio_scale,
               desvio->get_element<int8_t>(1) * desvio_scale,
               desvio->get_element<int8_t>(2) * desvio_scale);
        fflush(stdout);

        delete input;
    }

    printf("SDTEST_SUMMARY count=%d first_read_us=%lld first_infer_us=%lld\n",
           SD_TEST_IMAGES,
           (long long)first_read_us,
           (long long)first_infer_us);
    fflush(stdout);

    heap_caps_free(img_buf);
    delete model;
    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
}

// Full test set on SD: separates the one-time image write from the many
// inference reads, both independently checkpointable/resumable from the host
// side. Protocol, one line per command:
//   "W <index>\n" then STREAM_IMG_BYTES as base64 lines -> write /sdcard/img<index>.bin
//   "R <index>\n"                                        -> read + infer that file
//   "DONE\n"                                              -> exit
TEST_CASE("SD full test set: indexed write/read/infer", "[sdfull]")
{
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1;
    slot_config.clk = GPIO_NUM_39;
    slot_config.cmd = GPIO_NUM_38;
    slot_config.d0 = GPIO_NUM_40;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = true,
        .max_files = 8,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_card_t *card = nullptr;
    esp_err_t mret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &card);
    TEST_ASSERT_EQUAL(ESP_OK, mret);
    printf("SDFULL_MOUNT ok freq_khz=%d\n", host.max_freq_khz);

    if (!uart_is_driver_installed(UART_NUM_0)) {
        uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    }
    uart_vfs_dev_use_driver(UART_NUM_0);

    Model *model =
        new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, nullptr, true);
    TEST_ASSERT_NOT_NULL(model);

    uint8_t *img_buf = (uint8_t *)heap_caps_malloc(STREAM_IMG_BYTES, MALLOC_CAP_SPIRAM);
    TEST_ASSERT_NOT_NULL(img_buf);

    printf("SDFULL_READY\n");
    fflush(stdout);

    static char line[STREAM_CHUNK_B64_CHARS + 4];
    char path[64];

    while (true) {
        char cmd[16] = {0};
        if (fgets(cmd, sizeof(cmd), stdin) == nullptr) {
            break;
        }
        if (cmd[0] == 'D') { // "DONE\n"
            break;
        }

        int index = atoi(cmd + 2); // "W 123\n" / "R 123\n"
        snprintf(path, sizeof(path), SD_MOUNT_POINT "/img%d.bin", index);

        if (cmd[0] == 'W') {
            size_t received = 0;
            while (received < STREAM_IMG_BYTES) {
                if (fgets(line, sizeof(line), stdin) == nullptr) {
                    printf("SDFULL_ERR short_read idx=%d at %d\n", index, (int)received);
                    break;
                }
                received += b64_decode(line, img_buf + received);
                printf("A\n");
                fflush(stdout);
            }
            TEST_ASSERT_EQUAL(STREAM_IMG_BYTES, received);

            FILE *f = fopen(path, "wb");
            TEST_ASSERT_NOT_NULL(f);
            size_t wrote = fwrite(img_buf, 1, STREAM_IMG_BYTES, f);
            fclose(f);
            TEST_ASSERT_EQUAL(STREAM_IMG_BYTES, wrote);
            printf("SDFULL_WRITTEN[%d]\n", index);
            fflush(stdout);
        } else if (cmd[0] == 'R') {
            FILE *f = fopen(path, "rb");
            TEST_ASSERT_NOT_NULL(f);
            size_t got = fread(img_buf, 1, STREAM_IMG_BYTES, f);
            fclose(f);
            TEST_ASSERT_EQUAL(STREAM_IMG_BYTES, got);

            TensorBase *input = new TensorBase({1, 224, 224, 3}, (int8_t *)img_buf, -7, DATA_TYPE_INT8);
            int64_t t0 = esp_timer_get_time();
            model->run(input, RUNTIME_MODE_MULTI_CORE);
            int64_t infer_us = esp_timer_get_time() - t0;

            auto &outputs = model->get_outputs();
            TensorBase *obstaculo = outputs.at("obstaculo");
            TensorBase *desvio = outputs.at("desvio");
            float obstaculo_scale = exp2f((float)obstaculo->exponent);
            float desvio_scale = exp2f((float)desvio->exponent);

            printf("SDFULL_RESULT[%d] latency_us=%lld\n", index, (long long)infer_us);
            printf("SDFULL_RESULT[%d] obstaculo_dequant: %f %f\n",
                   index,
                   obstaculo->get_element<int8_t>(0) * obstaculo_scale,
                   obstaculo->get_element<int8_t>(1) * obstaculo_scale);
            printf("SDFULL_RESULT[%d] desvio_dequant: %f %f %f\n",
                   index,
                   desvio->get_element<int8_t>(0) * desvio_scale,
                   desvio->get_element<int8_t>(1) * desvio_scale,
                   desvio->get_element<int8_t>(2) * desvio_scale);
            fflush(stdout);

            delete input;
        }
    }

    heap_caps_free(img_buf);
    delete model;
    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
}

// Reads real JPEGs already sitting on the SD card (sense_teste_224/images_x_y_224,
// copied there directly from the PC while the card was still in a reader) and
// decodes + quantizes + infers entirely on-device -- no image data crosses
// serial at all. Host only sends filenames; device reports decode time,
// quantize time, and inference time separately.
#define JPEG_IMG_DIR SD_MOUNT_POINT "/sense_teste_224/images_x_y_224"
#define SUBSET_DIR SD_MOUNT_POINT "/subset_microcontrolador"

TEST_CASE("SD JPEG pilot: decode+quantize+infer timing from real files", "[sdjpeg]")
{
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1;
    slot_config.clk = GPIO_NUM_39;
    slot_config.cmd = GPIO_NUM_38;
    slot_config.d0 = GPIO_NUM_40;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false, // real data already on the card -- never reformat here
        .max_files = 8,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_card_t *card = nullptr;
    esp_err_t mret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &card);
    TEST_ASSERT_EQUAL(ESP_OK, mret);
    printf("SDJPEG_MOUNT ok freq_khz=%d\n", host.max_freq_khz);

    mkdir(SUBSET_DIR, 0777); // ignore EEXIST -- fine if it already exists
    // per-resolution subfolders so images from different (resolution, alpha)
    // groups don't collide on filename (e.g. "990.jpg" exists in all of them)
    mkdir(SUBSET_DIR "/32", 0777);
    mkdir(SUBSET_DIR "/64", 0777);
    mkdir(SUBSET_DIR "/128", 0777);
    mkdir(SUBSET_DIR "/224", 0777);

    if (!uart_is_driver_installed(UART_NUM_0)) {
        uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    }
    uart_vfs_dev_use_driver(UART_NUM_0);

    Model *model =
        new Model("model", 0, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, MEMORY_MANAGER_GREEDY, nullptr, true);
    TEST_ASSERT_NOT_NULL(model);

    // 224x224x3 int8 NHWC input buffer, and a scratch buffer for the file
    // bytes + decoded RGB888 (also 224*224*3, decoder writes RGB888 directly
    // in NHWC-compatible row-major interleaved order).
    int8_t *quant_buf = (int8_t *)heap_caps_malloc(STREAM_IMG_BYTES, MALLOC_CAP_SPIRAM);
    uint8_t *rgb_buf = (uint8_t *)jpeg_calloc_align(STREAM_IMG_BYTES, 16);
    uint8_t *file_buf = (uint8_t *)heap_caps_malloc(64 * 1024, MALLOC_CAP_SPIRAM); // JPEGs here are a few KB-tens of KB
    TEST_ASSERT_NOT_NULL(quant_buf);
    TEST_ASSERT_NOT_NULL(rgb_buf);
    TEST_ASSERT_NOT_NULL(file_buf);

    const float in_scale = 0.0078125f; // model's input scale (exponent -7), from modelo_XX.json

    printf("SDJPEG_READY\n");
    fflush(stdout);

    static char line[256];
    while (true) {
        if (fgets(line, sizeof(line), stdin) == nullptr) {
            break;
        }
        if (line[0] == 'D') { // "DONE\n"
            break;
        }

        if (line[0] == 'J') { // "J <filename> <length>\n" -- write a JPEG file to SUBSET_DIR
            char name[192];
            long jlen = 0;
            sscanf(line + 2, "%191s %ld", name, &jlen);

            char jpath[320];
            snprintf(jpath, sizeof(jpath), "%s/%s", SUBSET_DIR, name);

            static char b64in[4100];
            size_t received = 0;
            FILE *jf = fopen(jpath, "wb");
            TEST_ASSERT_NOT_NULL(jf);
            while (received < (size_t)jlen) {
                if (fgets(b64in, sizeof(b64in), stdin) == nullptr) {
                    printf("SDJPEG_ERR short_write %s at %d\n", name, (int)received);
                    break;
                }
                size_t got = b64_decode(b64in, file_buf);
                if (received + got > (size_t)jlen) {
                    got = (size_t)jlen - received;
                }
                fwrite(file_buf, 1, got, jf);
                received += got;
                printf("A\n");
                fflush(stdout);
            }
            fclose(jf);
            printf("SDJPEG_WRITTEN[%s] bytes=%d\n", name, (int)received);
            fflush(stdout);
            continue;
        }

        // "F <filename>\n" (JPEG_IMG_DIR) or "S <filename>\n" (SUBSET_DIR) or "Q <filename>\n" (raw RGB dump, JPEG_IMG_DIR)
        bool from_subset = (line[0] == 'S');
        char *name = line + 2;
        char *nl = strchr(name, '\n');
        if (nl) {
            *nl = '\0';
        }

        char path[320];
        snprintf(path, sizeof(path), "%s/%s", from_subset ? SUBSET_DIR : JPEG_IMG_DIR, name);

        int64_t t0 = esp_timer_get_time();
        FILE *f = fopen(path, "rb");
        TEST_ASSERT_NOT_NULL(f);
        size_t file_len = fread(file_buf, 1, 64 * 1024, f);
        fclose(f);
        int64_t read_us = esp_timer_get_time() - t0;

        int64_t t1 = esp_timer_get_time();
        jpeg_dec_config_t jpeg_cfg = DEFAULT_JPEG_DEC_CONFIG();
        jpeg_dec_handle_t jpeg_dec = nullptr;
        TEST_ASSERT_EQUAL(JPEG_ERR_OK, jpeg_dec_open(&jpeg_cfg, &jpeg_dec));

        jpeg_dec_io_t io = {};
        io.inbuf = file_buf;
        io.inbuf_len = file_len;
        io.outbuf = rgb_buf;

        jpeg_dec_header_info_t header_info = {};
        TEST_ASSERT_EQUAL(JPEG_ERR_OK, jpeg_dec_parse_header(jpeg_dec, &io, &header_info));
        TEST_ASSERT_EQUAL(JPEG_ERR_OK, jpeg_dec_process(jpeg_dec, &io));
        jpeg_dec_close(jpeg_dec);
        int64_t decode_us = esp_timer_get_time() - t1;

        if (line[0] == 'Q') { // "Q <filename>\n" -- dump raw decoded RGB888, no quantize/infer
            printf("SDJPEG_RAWRGB_BEGIN[%s] w=%d h=%d bytes=%d\n", name, header_info.width, header_info.height,
                   STREAM_IMG_BYTES);
            static char b64line[4100];
            int b64_n = 0;
            static const char *b64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for (int off = 0; off < STREAM_IMG_BYTES; off += 3072) {
                b64_n = 0;
                for (int j = off; j < off + 3072; j += 3) {
                    uint8_t b0 = rgb_buf[j], b1 = rgb_buf[j + 1], b2 = rgb_buf[j + 2];
                    b64line[b64_n++] = b64_table[b0 >> 2];
                    b64line[b64_n++] = b64_table[((b0 & 0x3) << 4) | (b1 >> 4)];
                    b64line[b64_n++] = b64_table[((b1 & 0xF) << 2) | (b2 >> 6)];
                    b64line[b64_n++] = b64_table[b2 & 0x3F];
                }
                b64line[b64_n] = '\0';
                printf("%s\n", b64line);
            }
            printf("SDJPEG_RAWRGB_END[%s]\n", name);
            fflush(stdout);
            continue;
        }

        int img_bytes = (int)header_info.width * (int)header_info.height * 3;
        int64_t t2 = esp_timer_get_time();
        for (int i = 0; i < img_bytes; i++) {
            float normalized = ((float)rgb_buf[i] - 127.5f) / 127.5f;
            int q = (int)lroundf(normalized / in_scale);
            if (q > 127) {
                q = 127;
            } else if (q < -128) {
                q = -128;
            }
            quant_buf[i] = (int8_t)q;
        }
        int64_t quant_us = esp_timer_get_time() - t2;

        TensorBase *input =
            new TensorBase({1, (int)header_info.height, (int)header_info.width, 3}, quant_buf, -7, DATA_TYPE_INT8);
        int64_t t3 = esp_timer_get_time();
        model->run(input, RUNTIME_MODE_MULTI_CORE);
        int64_t infer_us = esp_timer_get_time() - t3;

        auto &outputs = model->get_outputs();
        TensorBase *obstaculo = outputs.at("obstaculo");
        TensorBase *desvio = outputs.at("desvio");
        float obstaculo_scale = exp2f((float)obstaculo->exponent);
        float desvio_scale = exp2f((float)desvio->exponent);

        printf("SDJPEG_RESULT[%s] w=%d h=%d read_us=%lld decode_us=%lld quant_us=%lld infer_us=%lld\n",
               name,
               header_info.width,
               header_info.height,
               (long long)read_us,
               (long long)decode_us,
               (long long)quant_us,
               (long long)infer_us);
        printf("SDJPEG_RESULT[%s] obstaculo_dequant: %f %f\n",
               name,
               obstaculo->get_element<int8_t>(0) * obstaculo_scale,
               obstaculo->get_element<int8_t>(1) * obstaculo_scale);
        printf("SDJPEG_RESULT[%s] desvio_dequant: %f %f %f\n",
               name,
               desvio->get_element<int8_t>(0) * desvio_scale,
               desvio->get_element<int8_t>(1) * desvio_scale,
               desvio->get_element<int8_t>(2) * desvio_scale);
        fflush(stdout);

        delete input;
    }

    heap_caps_free(quant_buf);
    jpeg_free_align(rgb_buf);
    heap_caps_free(file_buf);
    delete model;
    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
}

TEST_CASE("Test dl module API: run()", "[api]")
{
    ESP_LOGI(TAG, "Test dl module API: run()");
    int total_ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    TensorBase *input1 = new TensorBase({1, 3, 64, 64}, nullptr, 0, DATA_TYPE_INT8);
    TensorBase *input2 = new TensorBase({1, 1, 1, 64}, nullptr, 0, DATA_TYPE_INT8);
    TensorBase *output = new TensorBase({1, 3, 64, 64}, nullptr, 0, DATA_TYPE_INT8);

    // single input and single output
    module::Module *relu_op = new module::Relu("relu", MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
    relu_op->run(input1, output);
    for (int i = 0; i < output->get_size(); i++) {
        int8_t in = input1->get_element<int8_t>(i);
        int8_t out = output->get_element<int8_t>(i);
        if (in > 0) {
            TEST_ASSERT_EQUAL(true, in == out);
        } else {
            TEST_ASSERT_EQUAL(true, out == 0);
        }
    }

    // multiple inputs and multiple outputs
    module::Module *add_op = new module::Add("add", MODULE_NON_INPLACE, QUANT_TYPE_SYMM_8BIT);
    add_op->run({input1, input2}, {output});

    for (int i = 0; i < output->get_size(); i++) {
        int8_t in1 = input1->get_element<int8_t>(i);
        int8_t in2 = input1->get_element<int8_t>(i % 64);
        int8_t out = output->get_element<int8_t>(i);
        TEST_ASSERT_EQUAL(true, in1 + in2 == out);
    }

    delete input1;
    delete input2;
    delete output;
    delete relu_op;
    delete add_op;

    int total_ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, total_ram_size_before == total_ram_size_end);
}
