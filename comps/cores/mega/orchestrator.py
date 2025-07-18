# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import copy
import json
import os
import re
import threading
import time
from typing import Dict, List

import aiohttp
import requests
from fastapi.responses import StreamingResponse
from prometheus_client import Gauge, Histogram
from pydantic import BaseModel

from ..proto.docarray import LLMParams
from ..telemetry.opea_telemetry import opea_telemetry, tracer
from .constants import ServiceType
from .dag import DAG
from .logger import CustomLogger

logger = CustomLogger("comps-core-orchestrator")
LOGFLAG = os.getenv("LOGFLAG", False)
ENABLE_OPEA_TELEMETRY = bool(os.environ.get("TELEMETRY_ENDPOINT"))


class OrchestratorMetrics:
    def __init__(self) -> None:
        # locking for latency metric creation / method change
        self._lock = threading.Lock()

        # Metrics related to token processing are created on demand,
        # to avoid bogus ones for services that never handle tokens
        self.first_token_latency = None
        self.inter_token_latency = None
        self.request_latency = None
        self.request_pending = None

        # initial methods to create the metrics
        self.token_update = self._token_update_create
        self.request_update = self._request_update_create
        self.pending_update = self._pending_update_create

    def _token_update_create(self, token_start: float, is_first: bool) -> float:
        with self._lock:
            # in case another thread already got here
            if self.token_update == self._token_update_create:
                self.first_token_latency = Histogram(
                    "megaservice_first_token_latency", "First token latency (histogram)"
                )
                self.inter_token_latency = Histogram(
                    "megaservice_inter_token_latency", "Inter-token latency (histogram)"
                )
                self.token_update = self._token_update_real
        return self.token_update(token_start, is_first)

    def _request_update_create(self, req_start: float) -> None:
        with self._lock:
            # in case another thread already got here
            if self.request_update == self._request_update_create:
                self.request_latency = Histogram(
                    "megaservice_request_latency", "Whole LLM request/reply latency (histogram)"
                )
                self.request_update = self._request_update_real
        self.request_update(req_start)

    def _pending_update_create(self, increase: bool) -> None:
        with self._lock:
            # in case another thread already got here
            if self.pending_update == self._pending_update_create:
                self.request_pending = Gauge(
                    "megaservice_request_pending", "Count of currently pending requests (gauge)"
                )
                self.pending_update = self._pending_update_real
        self.pending_update(increase)

    def _token_update_real(self, token_start: float, is_first: bool) -> float:
        now = time.monotonic()
        if is_first:
            self.first_token_latency.observe(now - token_start)
        else:
            self.inter_token_latency.observe(now - token_start)
        return now

    def _request_update_real(self, req_start: float) -> None:
        self.request_latency.observe(time.monotonic() - req_start)

    def _pending_update_real(self, increase: bool) -> None:
        if increase:
            self.request_pending.inc()
        else:
            self.request_pending.dec()


# Prometheus metrics need to be singletons, not per Orchestrator
_metrics = OrchestratorMetrics()


class ServiceOrchestrator(DAG):
    """Manage 1 or N micro services in a DAG through Python API."""

    def __init__(self) -> None:
        self.metrics = _metrics
        self.services = {}  # all services, id -> service
        super().__init__()

    def add(self, service):
        if service.name not in self.services:
            self.services[service.name] = service
            self.add_node_if_not_exists(service.name)
        else:
            raise Exception(f"Service {service.name} already exists!")
        return self

    def flow_to(self, from_service, to_service):
        try:
            self.add_edge(from_service.name, to_service.name)
            return True
        except Exception as e:
            logger.error(e)
            return False

    @opea_telemetry
    async def schedule(self, initial_inputs: Dict | BaseModel, llm_parameters: LLMParams = LLMParams(), **kwargs):
        req_start = time.monotonic()
        self.metrics.pending_update(True)

        result_dict = {}
        runtime_graph = DAG()
        runtime_graph.graph = copy.deepcopy(self.graph)
        if LOGFLAG:
            logger.info(initial_inputs)

        timeout = aiohttp.ClientTimeout(total=2000)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
            pending = {
                asyncio.create_task(
                    self.execute(session, req_start, node, initial_inputs, runtime_graph, llm_parameters, **kwargs)
                )
                for node in self.ind_nodes()
            }
            ind_nodes = self.ind_nodes()

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for done_task in done:
                    response, node = await done_task
                    result_dict[node] = response

                    # traverse the current node's downstream nodes and execute if all one's predecessors are finished
                    downstreams = runtime_graph.downstream(node)

                    # remove all the black nodes that are skipped to be forwarded to
                    if not isinstance(response, StreamingResponse) and "downstream_black_list" in response:
                        for black_node in response["downstream_black_list"]:
                            for downstream in reversed(downstreams):
                                try:
                                    if re.findall(black_node, downstream):
                                        if LOGFLAG:
                                            logger.info(f"skip forwardding to {downstream}...")
                                        runtime_graph.delete_edge(node, downstream)
                                        downstreams.remove(downstream)
                                except re.error as e:
                                    logger.error("Pattern invalid! Operation cancelled.")
                            if len(downstreams) == 0 and llm_parameters.stream:
                                # turn the response to a StreamingResponse
                                # to make the response uniform to UI
                                def fake_stream(text):
                                    yield "data: b'" + text + "'\n\n"
                                    yield "data: [DONE]\n\n"

                                result_dict[node] = StreamingResponse(
                                    fake_stream(response["text"]), media_type="text/event-stream"
                                )

                    for d_node in downstreams:
                        if all(i in result_dict for i in runtime_graph.predecessors(d_node)):
                            inputs = self.process_outputs(runtime_graph.predecessors(d_node), result_dict)
                            pending.add(
                                asyncio.create_task(
                                    self.execute(
                                        session, req_start, d_node, inputs, runtime_graph, llm_parameters, **kwargs
                                    )
                                )
                            )
        nodes_to_keep = []
        for i in ind_nodes:
            nodes_to_keep.append(i)
            nodes_to_keep.extend(runtime_graph.all_downstreams(i))

        all_nodes = list(runtime_graph.graph.keys())

        for node in all_nodes:
            if node not in nodes_to_keep:
                runtime_graph.delete_node_if_exists(node)

        if not llm_parameters.stream:
            self.metrics.pending_update(False)

        return result_dict, runtime_graph

    def process_outputs(self, prev_nodes: List, result_dict: Dict) -> Dict:
        all_outputs = {}

        # assume all prev_nodes outputs' keys are not duplicated
        for prev_node in prev_nodes:
            all_outputs.update(result_dict[prev_node])
        return all_outputs

    def wrap_iterable(self, iterable, is_first=True):

        with tracer.start_as_current_span("llm_generate_stream") if ENABLE_OPEA_TELEMETRY else contextlib.nullcontext():
            while True:
                with (
                    tracer.start_as_current_span("llm_generate_stream_first_token")
                    if is_first and ENABLE_OPEA_TELEMETRY
                    else contextlib.nullcontext()
                ):  #  else tracer.start_as_current_span(f"llm_generate_stream_next_token")
                    try:
                        token = next(iterable)
                        yield token
                        is_first = False
                    except StopIteration:
                        # Exiting the iterable loop cleanly
                        break
                    except Exception as e:
                        raise e

    @opea_telemetry
    async def execute(
        self,
        session: aiohttp.client.ClientSession,
        req_start: float,
        cur_node: str,
        inputs: Dict,
        runtime_graph: DAG,
        llm_parameters: LLMParams = LLMParams(),
        **kwargs,
    ):
        # send the cur_node request/reply

        llm_parameters_dict = llm_parameters.dict()

        is_llm_vlm = self.services[cur_node].service_type in (ServiceType.LLM, ServiceType.LVM)

        if is_llm_vlm:
            for field, value in llm_parameters_dict.items():
                if inputs.get(field) != value:
                    inputs[field] = value
        # pre-process
        inputs = self.align_inputs(inputs, cur_node, runtime_graph, llm_parameters_dict, **kwargs)
        access_token = self.services[cur_node].api_key_value
        if access_token:
            endpoint = self.services[cur_node].endpoint_path(inputs["model"])
        else:
            endpoint = self.services[cur_node].endpoint_path(None)
        if is_llm_vlm and llm_parameters.stream:
            # Still leave to sync requests.post for StreamingResponse
            if LOGFLAG:
                logger.info(inputs)
            with (
                tracer.start_as_current_span(f"{cur_node}_asyn_generate")
                if ENABLE_OPEA_TELEMETRY
                else contextlib.nullcontext()
            ):
                if access_token:
                    response = requests.post(
                        url=endpoint,
                        data=json.dumps(inputs),
                        headers={"Content-type": "application/json", "Authorization": f"Bearer {access_token}"},
                        proxies={"http": None},
                        stream=True,
                        timeout=2000,
                    )

                else:
                    response = requests.post(
                        url=endpoint,
                        data=json.dumps(inputs),
                        headers={
                            "Content-type": "application/json",
                        },
                        proxies={"http": None},
                        stream=True,
                        timeout=2000,
                    )

            downstream = runtime_graph.downstream(cur_node)
            if downstream:
                assert len(downstream) == 1, "Not supported multiple stream downstreams yet!"
                cur_node = downstream[0]
                hitted_ends = [".", "?", "!", "。", "，", "！"]
                downstream_endpoint = self.services[downstream[0]].endpoint_path()

            def generate():
                token_start = req_start
                if response:
                    # response.elapsed = time until first headers received
                    buffered_chunk_str = ""
                    is_first = True
                    for chunk in self.wrap_iterable(response.iter_content(chunk_size=None)):
                        if chunk:
                            if downstream:
                                chunk = chunk.decode("utf-8")
                                buffered_chunk_str += self.extract_chunk_str(chunk)
                                is_last = chunk.endswith("[DONE]\n\n")
                                if (buffered_chunk_str and buffered_chunk_str[-1] in hitted_ends) or is_last:
                                    if access_token:
                                        res = requests.post(
                                            url=downstream_endpoint,
                                            data=json.dumps({"text": buffered_chunk_str}),
                                            headers={
                                                "Content-type": "application/json",
                                                "Authorization": f"Bearer {access_token}",
                                            },
                                            proxies={"http": None},
                                            timeout=2000,
                                        )
                                    else:
                                        res = requests.post(
                                            url=downstream_endpoint,
                                            data=json.dumps({"text": buffered_chunk_str}),
                                            headers={
                                                "Content-type": "application/json",
                                            },
                                            proxies={"http": None},
                                            timeout=2000,
                                        )
                                    res_json = res.json()
                                    if "text" in res_json:
                                        res_txt = res_json["text"]
                                    else:
                                        raise Exception("Other response types not supported yet!")
                                    buffered_chunk_str = ""  # clear
                                    yield from self.token_generator(
                                        res_txt, token_start, is_first=is_first, is_last=is_last
                                    )
                                    token_start = time.monotonic()
                                    is_first = False
                            else:
                                token_start = self.metrics.token_update(token_start, is_first)
                                is_first = False
                                yield chunk

                    self.metrics.request_update(req_start)
                    self.metrics.pending_update(False)

            return (
                StreamingResponse(self.align_generator(generate(), **kwargs), media_type="text/event-stream"),
                cur_node,
            )
        else:
            if LOGFLAG:
                logger.info(inputs)
            if not isinstance(inputs, dict):
                input_data = inputs.dict()
                # remove null
                input_data = {k: v for k, v in input_data.items() if v is not None}
            else:
                input_data = inputs

            with (
                tracer.start_as_current_span(f"{cur_node}_generate")
                if ENABLE_OPEA_TELEMETRY
                else contextlib.nullcontext()
            ):
                response = await session.post(
                    endpoint,
                    json=input_data,
                    headers={"Content-type": "application/json", "Authorization": f"Bearer {access_token}"},
                )

            if response.content_type == "audio/wav":
                audio_data = await response.read()
                data = self.align_outputs(audio_data, cur_node, inputs, runtime_graph, llm_parameters_dict, **kwargs)
            else:
                # Parse as JSON
                data = await response.json()
                # post process
                data = self.align_outputs(data, cur_node, inputs, runtime_graph, llm_parameters_dict, **kwargs)

            return data, cur_node

    def align_inputs(self, inputs, *args, **kwargs):
        """Override this method in megaservice definition."""
        return inputs

    def align_outputs(self, data, *args, **kwargs):
        """Override this method in megaservice definition."""
        return data

    def align_generator(self, gen, *args, **kwargs):
        """Override this method in megaservice definition."""
        return gen

    def get_all_final_outputs(self, result_dict, runtime_graph):
        final_output_dict = {}
        for leaf in runtime_graph.all_leaves():
            final_output_dict[leaf] = result_dict[leaf]
        return final_output_dict

    def extract_chunk_str(self, chunk_str):
        if chunk_str == "data: [DONE]\n\n":
            return ""
        prefix = "data: b'"
        prefix_2 = 'data: b"'
        suffix = "'\n\n"
        suffix_2 = '"\n\n'
        if chunk_str.startswith(prefix) or chunk_str.startswith(prefix_2):
            chunk_str = chunk_str[len(prefix) :]
        if chunk_str.endswith(suffix) or chunk_str.endswith(suffix_2):
            chunk_str = chunk_str[: -len(suffix)]
        return chunk_str

    def token_generator(self, sentence: str, token_start: float, is_first: bool, is_last: bool) -> str:
        prefix = "data: "
        suffix = "\n\n"
        tokens = re.findall(r"\s?\S+\s?", sentence, re.UNICODE)
        for token in tokens:
            token_start = self.metrics.token_update(token_start, is_first)
            yield prefix + repr(token.replace("\\n", "\n").encode("utf-8")) + suffix
            is_first = False
        if is_last:
            yield "data: [DONE]\n\n"
