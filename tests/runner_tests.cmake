if(NOT DEFINED EVOBRAINBOT_EXECUTABLE)
    message(FATAL_ERROR "EVOBRAINBOT_EXECUTABLE was not provided")
endif()

# Executes one runner case and checks its exit code and exact output streams.
function(expect_command name expected_result expected_stdout expected_stderr)
    execute_process(
        COMMAND "${EVOBRAINBOT_EXECUTABLE}" ${ARGN}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        RESULT_VARIABLE actual_result
        OUTPUT_VARIABLE actual_stdout
        ERROR_VARIABLE actual_stderr
    )

    # Windows text-mode output may contain CRLF, so normalize before comparing.
    string(REPLACE "\r\n" "\n" actual_stdout "${actual_stdout}")
    string(REPLACE "\r\n" "\n" actual_stderr "${actual_stderr}")

    if(NOT "${actual_result}" STREQUAL "${expected_result}")
        message(FATAL_ERROR
            "${name}: expected exit ${expected_result}, got ${actual_result}")
    endif()
    if(NOT "${actual_stdout}" STREQUAL "${expected_stdout}")
        message(FATAL_ERROR
            "${name}: unexpected stdout\nExpected: [${expected_stdout}]\n"
            "Actual: [${actual_stdout}]")
    endif()
    if(NOT "${actual_stderr}" STREQUAL "${expected_stderr}")
        message(FATAL_ERROR
            "${name}: unexpected stderr\nExpected: [${expected_stderr}]\n"
            "Actual: [${actual_stderr}]")
    endif()
endfunction()

# Accepts either a real CUDA run or the required explicit unavailable error.
function(expect_gpu_backend_behavior)
    set(gpu_checkpoint "${CMAKE_CURRENT_BINARY_DIR}/gpu-backend-test.evo")
    file(REMOVE "${gpu_checkpoint}")
    execute_process(
        COMMAND "${EVOBRAINBOT_EXECUTABLE}" run "${gpu_checkpoint}"
            --seed 1 --ticks 0 --brain-backend gpu
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        RESULT_VARIABLE actual_result
        OUTPUT_VARIABLE actual_stdout
        ERROR_VARIABLE actual_stderr
    )
    string(REPLACE "\r\n" "\n" actual_stdout "${actual_stdout}")
    string(REPLACE "\r\n" "\n" actual_stderr "${actual_stderr}")
    if("${actual_result}" STREQUAL "0")
        if(NOT "${actual_stderr}" STREQUAL ""
            OR NOT actual_stdout MATCHES "Brain backend: gpu\n"
            OR NOT EXISTS "${gpu_checkpoint}")
            message(FATAL_ERROR "available GPU backend did not complete a valid CLI run")
        endif()
    elseif("${actual_result}" STREQUAL "1")
        if(NOT "${actual_stdout}" STREQUAL ""
            OR NOT "${actual_stderr}" STREQUAL
                "Error: requested brain backend is unavailable\n")
            message(FATAL_ERROR "unavailable GPU backend did not fail explicitly")
        endif()
    else()
        message(FATAL_ERROR "GPU backend CLI test returned ${actual_result}")
    endif()
    file(REMOVE "${gpu_checkpoint}")
endfunction()

# Checks a successful run whose automatically selected seed is intentionally variable.
function(expect_generated_seed name expected_stdout)
    execute_process(
        COMMAND "${EVOBRAINBOT_EXECUTABLE}" ${ARGN}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        RESULT_VARIABLE actual_result
        OUTPUT_VARIABLE actual_stdout
        ERROR_VARIABLE actual_stderr
    )
    string(REPLACE "\r\n" "\n" actual_stdout "${actual_stdout}")
    string(REPLACE "\r\n" "\n" actual_stderr "${actual_stderr}")

    if(NOT "${actual_result}" STREQUAL "0")
        message(FATAL_ERROR "${name}: expected exit 0, got ${actual_result}")
    endif()
    if(NOT "${actual_stderr}" STREQUAL "")
        message(FATAL_ERROR "${name}: unexpected stderr [${actual_stderr}]")
    endif()
    string(REGEX MATCH "Seed: ([0-9]+)" seed_match "${actual_stdout}")
    set(actual_seed "${CMAKE_MATCH_1}")
    if("${seed_match}" STREQUAL ""
        OR actual_seed LESS 1
        OR actual_seed GREATER 999)
        message(FATAL_ERROR
            "${name}: expected an automatically generated seed from 1 to 999")
    endif()
    string(REGEX REPLACE "Seed: [0-9]+" "Seed: <generated>"
        normalized_stdout "${actual_stdout}")
    if(NOT "${normalized_stdout}" STREQUAL "${expected_stdout}")
        message(FATAL_ERROR
            "${name}: unexpected stdout\nExpected: [${expected_stdout}]\n"
            "Actual: [${normalized_stdout}]")
    endif()
endfunction()

string(CONCAT top_level_help
    "Usage:\n"
    "  EvoBrainBot [--help]\n"
    "  EvoBrainBot run [<checkpoint>] [--seed <seed>] [--ticks <ticks>] [--brain-backend cpu|gpu]\n"
    "  EvoBrainBot resume <checkpoint.evo> [--ticks <ticks>] [--brain-backend cpu|gpu]\n"
    "\n"
    "Run starts a new simulation. Seed defaults to a random integer from 1 to 999.\n"
    "Without ticks, training continues until Q, q, SIGINT, or SIGTERM requests a stop.\n"
    "The checkpoint defaults to autosave.evo; .evo is appended when its filename\n"
    "contains no dot. The checkpoint is saved when training stops.\n"
    "\n"
    "Resume continues the exact checkpoint path supplied. Without ticks, it also\n"
    "continues until Q, q, SIGINT, or SIGTERM requests a stop, then atomically replaces\n"
    "the input checkpoint. Seed and ticks are decimal unsigned 64-bit integers.\n")

set(default_checkpoint_path "${CMAKE_CURRENT_BINARY_DIR}/autosave.evo")
set(checkpoint_path "${CMAKE_CURRENT_BINARY_DIR}/runner-test-checkpoint.evo")
set(no_extension_path "${CMAKE_CURRENT_BINARY_DIR}/runner-test-no-extension")
set(normalized_checkpoint_path "${no_extension_path}.evo")
set(custom_extension_path "${CMAKE_CURRENT_BINARY_DIR}/runner-test-checkpoint.data")
set(dotted_directory "${CMAKE_CURRENT_BINARY_DIR}/runner-tests.v1")
set(dotted_parent_input "${dotted_directory}/checkpoint")
set(dotted_parent_output "${dotted_parent_input}.evo")
set(missing_checkpoint_path
    "${CMAKE_CURRENT_BINARY_DIR}/runner-test-missing-checkpoint.evo")
set(invalid_checkpoint_path
    "${CMAKE_CURRENT_BINARY_DIR}/runner-test-invalid-checkpoint.evo")

file(REMOVE
    "${default_checkpoint_path}"
    "${checkpoint_path}"
    "${no_extension_path}"
    "${normalized_checkpoint_path}"
    "${custom_extension_path}"
    "${missing_checkpoint_path}"
    "${invalid_checkpoint_path}")
file(REMOVE_RECURSE "${dotted_directory}")
file(MAKE_DIRECTORY "${dotted_directory}")

string(CONCAT default_one_tick_status
    "File: autosave.evo\n"
    "Brain backend: cpu\n"
    "Seed: 1234\n"
    "Tick: 1\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
string(CONCAT default_two_tick_status
    "File: ${default_checkpoint_path}\n"
    "Brain backend: cpu\n"
    "Seed: 1234\n"
    "Tick: 2\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")

expect_command(no_arguments 0 "${top_level_help}" "")
expect_command(top_level_help 0 "${top_level_help}" "" --help)
expect_command(run_help_removed 2 "" "Error: unknown option\n" run --help)
expect_command(resume_help_removed 2 "" "Error: unknown option\n" resume --help)

expect_command(valid_run 0 "${default_one_tick_status}" ""
    run --seed 1234 --ticks 1)
if(NOT EXISTS "${default_checkpoint_path}")
    message(FATAL_ERROR "valid_run: default checkpoint was not created")
endif()
expect_command(resume_default_checkpoint 0 "${default_two_tick_status}" ""
    resume "${default_checkpoint_path}" --ticks 1)
expect_command(read_resaved_default_checkpoint 0 "${default_two_tick_status}" ""
    resume "${default_checkpoint_path}" --ticks 0)

string(CONCAT generated_seed_status
    "File: ${checkpoint_path}\n"
    "Brain backend: cpu\n"
    "Seed: <generated>\n"
    "Tick: 0\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
expect_generated_seed(generated_seed "${generated_seed_status}"
    run "${checkpoint_path}" --ticks 0)

string(CONCAT normalized_path_status
    "File: ${normalized_checkpoint_path}\n"
    "Brain backend: cpu\n"
    "Seed: 1234\n"
    "Tick: 0\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
expect_command(checkpoint_before_options 0 "${normalized_path_status}" ""
    run "${no_extension_path}" --ticks 0 --seed 1234)
if(NOT EXISTS "${normalized_checkpoint_path}")
    message(FATAL_ERROR "checkpoint_before_options: .evo was not appended")
endif()
expect_command(checkpoint_between_options 0 "${normalized_path_status}" ""
    run --seed 1234 "${no_extension_path}" --ticks 0)

string(REPLACE "${normalized_checkpoint_path}" "${custom_extension_path}"
    custom_extension_status "${normalized_path_status}")
expect_command(custom_extension_unchanged 0 "${custom_extension_status}" ""
    run --seed 1234 --ticks 0 "${custom_extension_path}")
if(NOT EXISTS "${custom_extension_path}")
    message(FATAL_ERROR "custom_extension_unchanged: checkpoint was not created")
endif()

string(REPLACE "${normalized_checkpoint_path}" "${dotted_parent_output}"
    dotted_parent_status "${normalized_path_status}")
expect_command(dot_in_parent_still_appends_extension 0 "${dotted_parent_status}" ""
    run --ticks 0 "${dotted_parent_input}" --seed 1234)
if(NOT EXISTS "${dotted_parent_output}")
    message(FATAL_ERROR
        "dot_in_parent_still_appends_extension: .evo was not appended")
endif()

string(CONCAT maximum_seed_status
    "File: ${checkpoint_path}\n"
    "Brain backend: cpu\n"
    "Seed: 18446744073709551615\n"
    "Tick: 0\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
expect_command(maximum_seed 0 "${maximum_seed_status}" ""
    run "${checkpoint_path}" --seed 18446744073709551615 --ticks 0)

string(CONCAT checkpoint_one_tick_status
    "File: ${checkpoint_path}\n"
    "Brain backend: cpu\n"
    "Seed: 1234\n"
    "Tick: 1\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
expect_command(overwrite_checkpoint 0 "${checkpoint_one_tick_status}" ""
    run --ticks 1 "${checkpoint_path}" --seed 1234)
expect_command(resume_requires_exact_path 1 ""
    "Error: unable to open checkpoint input\n"
    resume "${no_extension_path}" --ticks 0)

file(WRITE "${invalid_checkpoint_path}" "not a checkpoint")
expect_command(invalid_checkpoint 1 ""
    "Error: invalid EvoBrainBot checkpoint identifier\n"
    resume "${invalid_checkpoint_path}" --ticks 0)
expect_command(missing_checkpoint 1 ""
    "Error: unable to open checkpoint input\n"
    resume "${missing_checkpoint_path}" --ticks 0)
expect_command(invalid_checkpoint_output 1 ""
    "Error: unable to open checkpoint output\n"
    run --seed 1 "${checkpoint_path}/child.evo" --ticks 0)

expect_command(unknown_command 2 "" "Error: unknown command\n" unknown)
expect_command(missing_value 2 "" "Error: missing option value\n"
    run --seed --ticks 1)
expect_command(missing_ticks_value 2 "" "Error: missing option value\n"
    run --seed 1 --ticks)
expect_command(duplicate_seed 2 "" "Error: duplicate option\n"
    run --seed 1 --seed 2 --ticks 3)
expect_command(duplicate_ticks 2 "" "Error: duplicate option\n"
    run --ticks 1 --ticks 2 --seed 3)
expect_command(negative_seed 2 "" "Error: invalid unsigned integer value\n"
    run --seed -1 --ticks 1)
expect_command(negative_ticks 2 "" "Error: invalid unsigned integer value\n"
    run --seed 1 --ticks -1)
expect_command(malformed_seed 2 "" "Error: invalid unsigned integer value\n"
    run --seed 1x --ticks 1)
expect_command(malformed_ticks 2 "" "Error: invalid unsigned integer value\n"
    run --seed 1 --ticks 12x)
expect_command(overflow_seed 2 "" "Error: invalid unsigned integer value\n"
    run --seed 18446744073709551616 --ticks 1)
expect_command(overflow_ticks 2 "" "Error: invalid unsigned integer value\n"
    run --seed 1 --ticks 18446744073709551616)
expect_command(unknown_option 2 "" "Error: unknown option\n"
    run --seed 1 --ticks 1 --extra 2)
expect_command(invalid_brain_backend 2 "" "Error: brain backend must be cpu or gpu\n"
    run --seed 1 --ticks 0 --brain-backend other)
expect_command(duplicate_brain_backend 2 "" "Error: duplicate option\n"
    run --brain-backend cpu --brain-backend cpu --ticks 0)
expect_gpu_backend_behavior()
expect_command(removed_checkpoint_out 2 "" "Error: unknown option\n"
    run --seed 1 --ticks 1 --checkpoint-out other.evo)
expect_command(duplicate_checkpoint 2 "" "Error: unexpected positional argument\n"
    run one.evo --seed 1 two.evo --ticks 0)

expect_command(resume_missing_input 2 ""
    "Error: missing checkpoint path\n"
    resume --ticks 1)
expect_command(resume_rejects_seed 2 "" "Error: unknown option\n"
    resume "${checkpoint_path}" --ticks 1 --seed 2)
expect_command(resume_rejects_checkpoint_input 2 "" "Error: unknown option\n"
    resume "${checkpoint_path}" --checkpoint-in other.evo)
expect_command(resume_rejects_checkpoint_output 2 "" "Error: unknown option\n"
    resume "${checkpoint_path}" --checkpoint-out other.evo)
expect_command(resume_duplicate_ticks 2 "" "Error: duplicate option\n"
    resume "${checkpoint_path}" --ticks 1 --ticks 2)
expect_command(resume_invalid_ticks 2 ""
    "Error: invalid unsigned integer value\n"
    resume "${checkpoint_path}" --ticks nope)
expect_command(resume_missing_ticks_value 2 "" "Error: missing option value\n"
    resume "${checkpoint_path}" --ticks)
expect_command(resume_rejects_extra_position 2 ""
    "Error: unexpected positional argument\n"
    resume "${checkpoint_path}" other.evo)
expect_command(run_rejects_input 2 "" "Error: unknown option\n"
    run --seed 1 --ticks 1 --checkpoint-in "${checkpoint_path}")

file(REMOVE
    "${default_checkpoint_path}"
    "${checkpoint_path}"
    "${no_extension_path}"
    "${normalized_checkpoint_path}"
    "${custom_extension_path}"
    "${invalid_checkpoint_path}")
file(REMOVE_RECURSE "${dotted_directory}")
