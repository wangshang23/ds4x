/* Thin server aggregation unit; implementation lives in server/parts/. */
#include "server/parts/00_http_json_core.inc"
#include "server/parts/01_request_types_and_tools.inc"
#include "server/parts/02_prompt_rendering.inc"
#include "server/parts/03_request_parsers.inc"
#include "server/parts/04_response_encoding_http.inc"
#include "server/parts/05_openai_streaming.inc"
#include "server/parts/06_responses_streaming.inc"
#include "server/parts/07_anthropic_streaming.inc"
#include "server/parts/08_runtime_state_and_cache.inc"
#include "server/parts/09_trace_and_live_state.inc"
#include "server/parts/10_request_execution.inc"
#include "server/parts/11_scheduler_http_options.inc"

#ifndef DS4_SERVER_TEST
#include "server/parts/12_server_main.inc"
#else
#include "server/parts/test_00_protocol_and_parser.inc"
#include "server/parts/test_01_rendering_and_streams.inc"
#include "server/parts/test_02_cache_and_cancellation.inc"
#include "server/parts/test_03_thinking_and_runner.inc"
#endif
