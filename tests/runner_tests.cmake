if(NOT DEFINED EVOBRAINBOT_EXECUTABLE)
    message(FATAL_ERROR "EVOBRAINBOT_EXECUTABLE was not provided")
endif()

# Executes one runner case and checks its exit code and exact output streams.
function(expect_command name expected_result expected_stdout expected_stderr)
    execute_process(
        COMMAND "${EVOBRAINBOT_EXECUTABLE}" ${ARGN}
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

set(top_level_help
    "Usage:\n  EvoBrainBot --help\n  EvoBrainBot run --help\n  EvoBrainBot run --seed <seed> --ticks <ticks>\n")
set(run_help
    "Usage: EvoBrainBot run --seed <seed> --ticks <ticks>\n\nBoth values are required decimal unsigned 64-bit integers.\n")

expect_command(top_level_help 0 "${top_level_help}" "" --help)
expect_command(run_help 0 "${run_help}" "" run --help)
expect_command(valid_run 0 "Seed: 1234\nCompleted ticks: 10\n" ""
    run --seed 1234 --ticks 10)
expect_command(reversed_options 0 "Seed: 7\nCompleted ticks: 3\n" ""
    run --ticks 3 --seed 7)
expect_command(zero_values 0 "Seed: 0\nCompleted ticks: 0\n" ""
    run --seed 0 --ticks 0)
expect_command(maximum_seed 0 "Seed: 18446744073709551615\nCompleted ticks: 0\n" ""
    run --seed 18446744073709551615 --ticks 0)

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
