#include "dl_model_base.hpp"
#include "dl_module_creator.hpp"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "unity.h"

static const char *TAG = "TEST_DL_POWER";

using namespace dl;

// Seconds of idle (one core, other halted) emitted before inference starts so
// the analyzer captures a baseline. Trigger the probe on the current step from
// this baseline up to the inference plateau -- no GPIO marker is used.
#ifndef POWER_BASELINE_S
#define POWER_BASELINE_S 3
#endif

TEST_CASE("Probe current during inference", "[power]")
{
    // ---- Constraint 2: only one core active ---------------------------------
    // Halt the core we are NOT running on. With both watchdogs already disabled
    // (CONFIG_ESP_INT_WDT=n, CONFIG_ESP_TASK_WDT_EN=n) the stalled core's idle
    // task never needs servicing, so a gap-free inference loop is safe.
    const int run_core = esp_cpu_get_core_id();
    const int other_core = run_core == 0 ? 1 : 0;
    esp_cpu_stall(other_core);

    // ---- Constraint 1: weights in PSRAM -------------------------------------
    // param_copy = true  -> parameters are copied out of flash into the heap.
    // internal_size = 0   -> the memory manager reserves no internal SRAM, so
    //                        the copied weights and activations land in PSRAM.
    const int psram_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    const int internal_before = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

    Model *model = new Model("model",
                             0,                                    // model_index
                             fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
                             0,                                    // internal_size = 0
                             MEMORY_MANAGER_GREEDY,
                             nullptr,                              // no decrypt key
                             true);                                // param_copy
    TEST_ASSERT_NOT_NULL(model);

    const int psram_used = psram_before - heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    const int internal_used = internal_before - heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

    // Warm caches/scratch before the silent window.
    model->run();
    model->run();

    // Sanity check + the LAST log line before the probe window. Confirms the
    // weights really went to PSRAM, not internal SRAM.
    ESP_LOGI(TAG,
             "core %d running, core %d halted | psram_used=%d B, internal_used=%d B | baseline %d s then continuous inference (silent)",
             run_core,
             other_core,
             psram_used,
             internal_used,
             POWER_BASELINE_S);
    TEST_ASSERT_TRUE_MESSAGE(psram_used > internal_used, "weights did not land in PSRAM");

    // ---- Constraint 3: all GPIOs inactive -----------------------------------
    // No GPIO is configured or driven anywhere in this test. Console UART is the
    // only active pad pair; nothing is logged below, so it stays idle. For an
    // absolutely pad-quiet probe, build with CONFIG_ESP_CONSOLE_NONE=y.

    // Baseline window: one core idle, other halted, no inference.
    vTaskDelay(pdMS_TO_TICKS(POWER_BASELINE_S * 1000));

    // Probe window: continuous, gap-free inference forever. Reset the board
    // (Ctrl-]) once enough current samples are captured.
    while (true) {
        model->run();
    }

    // not reached
    delete model;
    esp_cpu_unstall(other_core);
}
