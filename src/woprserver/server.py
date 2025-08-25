import asyncio
import signal
import logging

from typing import Optional, List

from mlserver.repository.factory import ModelRepositoryFactory

from mlserver.settings import Settings
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.metrics import MetricsServer
from mlserver.kafka import KafkaServer
from mlserver.server import MLServer
from .logging import configure_logger
from .parallel.registry import InferencePoolRegistry
from .rest.server import WRESTServer
from .grpc.server import GRPCServer

HANDLED_SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]


class WOPRserver(MLServer):
    """
    Minimal drop-in MLServer replacement.

    - Inherits full init/start/stop behavior
    - You can override pieces if needed
    """
    def __init__(self, settings: Settings):
        self._settings = settings

        # ---- 1) Configure logging FIRST so *all* subsequent logs are captured ----
        # This installs a single stdout handler on ROOT + a LogRecordFactory that
        # rewrites mlserver* → woprserver and injects [PID][scope].
        self._logger = configure_logger(self._settings)

        # ---- 2) proceed with normal startup ----
        self._add_signal_handlers()

        self._metrics_server = None
        if self._settings.metrics_endpoint:
            self._metrics_server = MetricsServer(self._settings)

        self._inference_pool_registry = None
        if self._settings.parallel_workers:
            on_worker_stop = []
            if self._metrics_server:
                on_worker_stop = [self._metrics_server.on_worker_stop]
            self._inference_pool_registry = InferencePoolRegistry(
                self._settings, on_worker_stop=on_worker_stop  # type: ignore
            )

        self._model_registry = self._create_model_registry()
        self._model_repository = ModelRepositoryFactory.resolve_model_repository(
            self._settings
        )
        self._data_plane = DataPlane(
            settings=self._settings, model_registry=self._model_registry
        )
        self._model_repository_handlers = ModelRepositoryHandlers(
            repository=self._model_repository, model_registry=self._model_registry
        )

        self._create_servers()
        self._logger.info("WOPRserver initialized 🚀")


    def _configure_logger(self):
        self._logger = configure_logger(self._settings)

    def _create_servers(self):
        self._rest_server = WRESTServer(
            self._settings, self._data_plane, self._model_repository_handlers
        )
        self._grpc_server = GRPCServer(
            self._settings, self._data_plane, self._model_repository_handlers
        )

        self._kafka_server = None
        if self._settings.kafka_enabled:
            self._kafka_server = KafkaServer(self._settings, self._data_plane)
