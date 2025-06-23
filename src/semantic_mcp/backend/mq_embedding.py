import zmq 
import multiprocessing as mp 

import signal 

import json 
import pickle 

import time 

from typing import List, Tuple, Dict, Any, Optional, Generator
from operator import itemgetter, attrgetter

from contextlib import contextmanager, ExitStack, suppress
from sentence_transformers import SentenceTransformer

from semantic_mcp.log import logger


class MQEmbedding:
    INNER_ROUTER_ADDR:str = "ipc:///tmp/router2worker.ipc"
    OUTER_ROUTER_ADDR:str = "ipc:///tmp/worker2router.ipc"
    
    def __init__(self, embedding_model_name:str, cache_folder:str, device:str="cpu", nb_workers:int=1):
        self.embedding_model_name = embedding_model_name
        self.cache_folder = cache_folder
        self.nb_workers = nb_workers
        self.device = device
    
    @contextmanager
    def _build_socket(self, ctx:zmq.Context, socket_type:str, method:str, addr:str) -> Generator[zmq.Socket, None, None]:
        socket = ctx.socket(socket_type=socket_type)
        try:
            attrgetter(method)(socket)(addr=addr)
            yield socket
        except Exception as e:
            logger.error(f"Error building socket: {e}")
        finally:
            socket.close()
    
    def worker(self, worker_id:int):
        
        try:
            model = SentenceTransformer(
                model_name_or_path=self.embedding_model_name,
                cache_folder=self.cache_folder,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            return 
        
        signal.signal(
            signalnum=signal.SIGTERM, 
            handler=lambda signame, frame: signal.raise_signal(signal.SIGINT)
        )
        
        ctx = zmq.Context()
        socket_builder = self._build_socket(ctx, zmq.DEALER, "connect", self.INNER_ROUTER_ADDR)
        logger.info(f"Worker {worker_id} started...")
        with socket_builder as dealer_socket:
            dealer_socket.send_multipart([b"", b"AVAILABLE", b"", b""])
            while True:
                try:
                    socket_status = dealer_socket.poll(timeout=1000)
                    if socket_status != zmq.POLLIN:
                        continue 
                    
                    incoming_task = dealer_socket.recv_multipart()
                    _, source_client_id, encoded_message_data = incoming_task
                    sentences:List[str] = pickle.loads(encoded_message_data)
                    embeddings = model.encode(sentences=sentences)
                    encoded_response_data = pickle.dumps(embeddings)
                    dealer_socket.send_multipart([b"", b"MESSAGE_CONSUMED", source_client_id, encoded_response_data])
                    dealer_socket.send_multipart([b"", b"AVAILABLE", b"", b""])
                    
                except KeyboardInterrupt:
                    logger.warning(f"Worker {worker_id} received keyboard interrupt, shutting down...")
                    break 
                except Exception as e:
                    logger.error(f"Worker {worker_id} received error: {e}")
                    break 
        
        del model 
        ctx.term()
                    
    def broker(self): 

        signal.signal(
            signalnum=signal.SIGTERM, 
            handler=lambda signame, frame: signal.raise_signal(signal.SIGINT)
        )

        resources_manager = ExitStack()
        ctx = zmq.Context()

        inner_router = resources_manager.enter_context(self._build_socket(ctx, zmq.ROUTER, "bind", self.INNER_ROUTER_ADDR))
        outer_router = resources_manager.enter_context(self._build_socket(ctx, zmq.ROUTER, "bind", self.OUTER_ROUTER_ADDR))

        poller = zmq.Poller()
        poller.register(inner_router, zmq.POLLIN)
        poller.register(outer_router, zmq.POLLIN)

        worker_socket_ids:List[bytes] = []
        logger.info(f"Broker started...")
        start_time = time.time()
        while True:
            try:
                duration = time.time() - start_time
                if duration > 5:
                    logger.info(f"Broker has been running on the background.")
                    logger.info(f"Number of registered workers: {len(worker_socket_ids)}")
                    start_time = time.time()

                socket_states_hmap:Dict[zmq.Socket, int] = dict(poller.poll(timeout=1000))
                if outer_router in socket_states_hmap:
                    if socket_states_hmap[outer_router] == zmq.POLLIN and len(worker_socket_ids) > 0:
                        client_incoming_message = outer_router.recv_multipart()
                        source_client_id, _, encoded_message_data = client_incoming_message
                        tarrget_worker_id = worker_socket_ids.pop(0)
                        inner_router.send_multipart([tarrget_worker_id, b"", source_client_id, encoded_message_data])
                
                if not inner_router in socket_states_hmap:
                    continue 

                if socket_states_hmap[inner_router] != zmq.POLLIN:
                    continue 
                
                worker_incoming_message = inner_router.recv_multipart()
                source_worker_id, _, flag, target_client_id, encoded_response_data = worker_incoming_message

                match flag:
                    case b"AVAILABLE":
                        worker_socket_ids.append(source_worker_id)
                    case b"MESSAGE_CONSUMED":
                        outer_router.send_multipart([target_client_id, b"", encoded_response_data])
                    case _:
                        logger.warning(f"Broker received unknown flag: {flag}")
                        continue 

            except KeyboardInterrupt:
                logger.warning(f"Broker received keyboard interrupt, shutting down...")
                break 
            except Exception as e:
                logger.error(f"Broker received error: {e}")
                break 
        
        logger.info(f"Broker shutting down...")
        poller.unregister(inner_router)
        poller.unregister(outer_router)
        resources_manager.close()
        ctx.term()
    
    def run_loop(self):
        logger.info(f"Starting embedding service...")
        signal.signal(
            signalnum=signal.SIGTERM, 
            handler=lambda signame, frame: signal.raise_signal(signal.SIGINT)
        )

        background_processes = []

        background_processes.append(mp.Process(target=self.broker))
        background_processes[0].start()

        for worker_id in range(self.nb_workers):
            background_processes.append(mp.Process(target=self.worker, args=(worker_id,)))
            background_processes[-1].start() 

        try:
            for process in background_processes:
                process.join()
        except KeyboardInterrupt:
            for process in background_processes:
                process.join() # wait for the process to finish
        
        for process in background_processes:
            if process.is_alive():
                process.terminate()
                process.join()

        logger.info(f"All background processes have finished")

        
        
if __name__ == "__main__":
    embedding_model_name = "all-mpnet-base-v2"
    cache_folder = "/Users/milkymap/Models"
    device = "cpu"
    nb_workers = 4

    mq_embedding = MQEmbedding(
        embedding_model_name=embedding_model_name, 
        cache_folder=cache_folder, device=device, nb_workers=nb_workers
    )
    mq_embedding.run_loop()