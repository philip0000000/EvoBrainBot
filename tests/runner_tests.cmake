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

string(CONCAT top_level_help
    "Usage:\n"
    "  EvoBrainBot --help\n"
    "  EvoBrainBot run --help\n"
    "  EvoBrainBot run --seed <seed> --ticks <ticks> [--checkpoint-out <path>]\n"
    "  EvoBrainBot resume --help\n"
    "  EvoBrainBot resume <checkpoint.evo> [--ticks <ticks>]\n")
string(CONCAT run_help
    "Usage: EvoBrainBot run --seed <seed> --ticks <ticks> [--checkpoint-out <path>]\n"
    "\nSeed and ticks are required decimal unsigned 64-bit integers.\n"
    "If checkpoint-out is omitted, the final state is saved to autosave.evo.\n")
string(CONCAT resume_help
    "Usage: EvoBrainBot resume <checkpoint.evo> [--ticks <ticks>]\n"
    "\nTicks is an optional decimal unsigned 64-bit limit for this command.\n"
    "Without ticks, training continues until Q, SIGINT, or SIGTERM requests a stop.\n"
    "The completed state atomically replaces the input checkpoint.\n")
string(CONCAT zero_tick_summary
    "Seed: 1234\n"
    "Completed ticks: 0\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
string(CONCAT one_tick_summary
    "Seed: 1234\n"
    "Completed ticks: 1\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
string(CONCAT two_tick_summary
    "Seed: 1234\n"
    "Completed ticks: 2\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")

set(default_checkpoint_path
    "${CMAKE_CURRENT_BINARY_DIR}/autosave.evo")
file(REMOVE "${default_checkpoint_path}")

string(CONCAT default_two_tick_status
    "File: ${default_checkpoint_path}\n"
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

expect_command(top_level_help 0 "${top_level_help}" "" --help)
expect_command(run_help 0 "${run_help}" "" run --help)
expect_command(resume_help 0 "${resume_help}" "" resume --help)
expect_command(valid_run 0 "${one_tick_summary}" ""
    run --seed 1234 --ticks 1)
if(NOT EXISTS "${default_checkpoint_path}")
    message(FATAL_ERROR "valid_run: default checkpoint was not created")
endif()
expect_command(resume_default_checkpoint 0 "${default_two_tick_status}" ""
    resume "${default_checkpoint_path}" --ticks 1)
expect_command(read_resaved_default_checkpoint 0 "${default_two_tick_status}" ""
    resume "${default_checkpoint_path}" --ticks 0)
expect_command(reversed_options 0 "${zero_tick_summary}" ""
    run --ticks 0 --seed 1234)

string(CONCAT maximum_seed_summary
    "Seed: 18446744073709551615\n"
    "Completed ticks: 0\n"
    "Population: 30\n"
    "Herbivores: 30\n"
    "Carnivores: 0\n"
    "Food: 1000\n"
    "Births: 0\n"
    "Introduced agents: 0\n"
    "Deaths: 0\n"
    "Agents eaten: 0\n")
expect_command(maximum_seed 0 "${maximum_seed_summary}" ""
    run --seed 18446744073709551615 --ticks 0)

set(checkpoint_path "${CMAKE_CURRENT_BINARY_DIR}/runner-test-checkpoint.evo")
set(missing_checkpoint_path
    "${CMAKE_CURRENT_BINARY_DIR}/runner-test-missing-checkpoint.evo")
set(invalid_checkpoint_path
    "${CMAKE_CURRENT_BINARY_DIR}/runner-test-invalid-checkpoint.evo")
file(REMOVE
    "${default_checkpoint_path}"
    "${checkpoint_path}"
    "${missing_checkpoint_path}"
    "${invalid_checkpoint_path}")

expect_command(save_checkpoint 0 "${zero_tick_summary}" ""
    run --seed 1234 --ticks 0 --checkpoint-out "${checkpoint_path}")
if(NOT EXISTS "${checkpoint_path}")
    message(FATAL_ERROR "save_checkpoint: checkpoint file was not created")
endif()

string(CONCAT checkpoint_one_tick_status
    "File: ${checkpoint_path}\n"
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
expect_command(resume_checkpoint 0 "${checkpoint_one_tick_status}" ""
    resume "${checkpoint_path}" --ticks 1)
if(NOT EXISTS "${checkpoint_path}")
    message(FATAL_ERROR "resume_checkpoint: input checkpoint was not replaced")
endif()

file(WRITE "${invalid_checkpoint_path}" "not a checkpoint")
expect_command(invalid_checkpoint 1 ""
    "Error: invalid EvoBrainBot checkpoint identifier\n"
    resume "${invalid_checkpoint_path}" --ticks 0)
expect_command(missing_checkpoint 1 ""
    "Error: unable to open checkpoint input\n"
    resume "${missing_checkpoint_path}" --ticks 0)
expect_command(invalid_checkpoint_output 1 ""
    "Error: unable to open checkpoint output\n"
    run --seed 1 --ticks 0 --checkpoint-out "${checkpoint_path}/child.evo")

expect_command(missing_command 2 "" "Error: missing command\n")
expect_command(unknown_command 2 "" "Error: unknown command\n" unknown)
expect_command(missing_seed 2 "" "Error: missing required option '--seed'\n"
    run --ticks 1)
expect_command(missing_ticks 2 "" "Error: missing required option '--ticks'\n"
    run --seed 1)
expect_command(missing_value 2 "" "Error: missing option value\n"
    run --seed --ticks 1)
expect_command(missing_ticks_value 2 "" "Error: missing option value\n"
    run --seed 1 --ticks)
expect_command(duplicate_seed 2 "" "Error: duplicate option\n"
    run --seed 1 --seed 2 --ticks 3)
expect_command(duplicate_ticks 2 "" "Error: duplicate option\n"
    run --ticks 1 --ticks 2 --seed 3)
expect_command(duplicate_output 2 "" "Error: duplicate option\n"
    run --seed 1 --ticks 0 --checkpoint-out one --checkpoint-out two)
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
expect_command(unexpected_position 2 "" "Error: unexpected positional argument\n"
    run --seed 1 --ticks 1 extra)

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
    "${checkpoint_path}"
    "${invalid_checkpoint_path}")
