"""
Launcher

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import logging
import re
import random
import time
import subprocess
from datetime import timedelta
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from larfm.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=60)
logging.basicConfig(level=logging.INFO)


def _is_port_available(port, addr="127.0.0.1"):
    """Check if a port is available on the specified address."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.bind((addr, port))
        sock.close()
        return True
    except (socket.error, OSError):
        return False


def _find_free_port(start_port=20000, end_port=30000, max_tries=100):
    """Find a free port in the given range with multiple attempts."""
    for _ in range(max_tries):
        port = random.randint(start_port, end_port)
        if _is_port_available(port):
            return port
    
    # Sequential scan as a fallback
    for port in range(start_port, end_port):
        if _is_port_available(port):
            return port
    
    raise RuntimeError(f"Could not find an available port in range {start_port}-{end_port}")


def resolve_root_node_address(nodes):
    """Resolve the root node address from SLURM_NODELIST format.
    
    Args:
        nodes (str): Node list in SLURM format (e.g., 'host[5-9]' or 'host0,host1')
        
    Returns:
        str: First node address
    """
    # Extract the base hostname for node ranges like 'prefix[1-5,7,9-10]'
    if '[' in nodes and ']' in nodes:
        base = nodes.split('[')[0]
        indices = nodes.split('[')[1].split(']')[0]
        
        # Parse the first index
        if ',' in indices:
            first_range = indices.split(',')[0]
        else:
            first_range = indices
            
        if '-' in first_range:
            first_index = first_range.split('-')[0]
        else:
            first_index = first_range
            
        return f"{base}{first_index}"
    elif ',' in nodes:
        # For comma-separated hostnames without brackets: 'host1,host2,host3'
        return nodes.split(',')[0]
    else:
        # Single hostname
        return nodes


def is_slurm_run():
    """Check if current job is running under SLURM."""
    return "SLURM_JOB_ID" in os.environ and "SLURM_NTASKS" in os.environ


def get_slurm_node_info():
    """Get node information from SLURM environment variables."""
    if not is_slurm_run():
        return None
    
    node_info = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "ntasks": int(os.environ.get("SLURM_NTASKS", 1)),
        "ntasks_per_node": int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
        "nnodes": int(os.environ.get("SLURM_NNODES", 1)),
        "node_id": int(os.environ.get("SLURM_NODEID", 0)),
        "proc_id": int(os.environ.get("SLURM_PROCID", 0)),
        "local_id": int(os.environ.get("SLURM_LOCALID", 0)),
        "nodelist": os.environ.get("SLURM_NODELIST", "127.0.0.1"),
    }
    return node_info


def get_network_interfaces():
    """Get a list of available network interfaces except loopback."""
    interfaces = []
    try:
        # Try to use netifaces if available
        import netifaces
        for iface in netifaces.interfaces():
            if iface != 'lo':  # Skip loopback
                addresses = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addresses:
                    for addr_info in addresses[netifaces.AF_INET]:
                        if 'addr' in addr_info:
                            interfaces.append((iface, addr_info['addr']))
    except ImportError:
        # Fallback to using ip command
        try:
            output = subprocess.check_output(["ip", "-o", "-4", "addr", "show"]).decode("utf-8")
            for line in output.split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] != "lo":
                        iface = parts[1]
                        addr = parts[3].split("/")[0]
                        interfaces.append((iface, addr))
        except (subprocess.SubprocessError, FileNotFoundError):
            # Last resort - assume common interface names
            pass
            
    return interfaces


def get_suitable_interface():
    """Find a suitable network interface for distributed training."""
    interfaces = get_network_interfaces()
    
    # Prefer high-speed interfaces typically used in HPC
    preferred_prefixes = ['ib', 'en', 'eth', 'em', 'eno', 'enp']
    
    for prefix in preferred_prefixes:
        for iface, addr in interfaces:
            if iface.startswith(prefix) and not addr.startswith('127.'):
                return iface
                
    # If no preferred interface, return first non-loopback
    for iface, addr in interfaces:
        if not addr.startswith('127.'):
            return iface
            
    # Fallback to just excluding loopback
    return "^lo"


def get_master_address():
    """Get master address from environment or SLURM configuration."""
    if "MASTER_ADDR" in os.environ:
        return os.environ["MASTER_ADDR"]
    
    if is_slurm_run():
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        root_node = resolve_root_node_address(nodelist)
        os.environ["MASTER_ADDR"] = root_node
        return root_node
    
    # Default fallback
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    return "127.0.0.1"


def get_master_port():
    """Get master port from environment or generate one based on job information."""
    if "MASTER_PORT" in os.environ:
        return int(os.environ["MASTER_PORT"])
    
    # If running under SLURM, use job ID to derive a port
    if is_slurm_run():
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is not None and len(job_id) >= 4:
            # Use last 4 digits of job ID + random offset to avoid immediate reuse collisions
            base_port = int(job_id[-4:]) + 15000
            port_range = 1000  # Try up to 1000 ports from the base
            
            # Check if the port is actually available
            for offset in range(port_range):
                port = base_port + offset
                if _is_port_available(port):
                    break
            else:
                # If we couldn't find a port in that range, use completely random port
                port = _find_free_port()
        else:
            # Fallback to random port in a safe range
            port = _find_free_port()
    else:
        # Not on SLURM, use a random port
        port = _find_free_port()
    
    os.environ["MASTER_PORT"] = str(port)
    return port


def setup_nccl_env_vars():
    """Set up necessary environment variables for NCCL."""
    # Only set variables that aren't already set
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "^lo"  # Exclude loopback
    
    if "NCCL_IB_TIMEOUT" not in os.environ:
        os.environ["NCCL_IB_TIMEOUT"] = "30"
    
    if "NCCL_SOCKET_NTHREADS" not in os.environ:
        os.environ["NCCL_SOCKET_NTHREADS"] = "8"
    
    # Pytorch specific environment variables        
    if "TORCH_DISTRIBUTED_TIMEOUT" not in os.environ:
        os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1200"  # 20 minutes timeout


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    cfg=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
                       or "slurm" to use SLURM's node assignment and port selection.
                       or "env" to use environment variables MASTER_ADDR and MASTER_PORT.
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    # Set up NCCL environment variables
    setup_nccl_env_vars()
    
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == "auto":
            if num_machines > 1:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "dist_url=auto not recommended for multi-machine jobs. Consider using 'slurm' if on a SLURM cluster."
                )
            
            port = _find_free_port()
            master_addr = get_master_address()
            assert _is_port_available(port, master_addr), f"Port {port} is not available on {master_addr}"
                
            dist_url = f"tcp://{master_addr}:{port}"
            
        elif dist_url == "slurm" or dist_url == "env":
            master_addr = get_master_address()
            master_port = get_master_port()
            dist_url = f"tcp://{master_addr}:{master_port}"
            
            logger = logging.getLogger(__name__)
            logger.info(f"Using distributed URL: {dist_url}")
            logger.info(f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")
            
        elif dist_url is None:
            master_addr = get_master_address()
            master_port = get_master_port()
            dist_url = f"tcp://{master_addr}:{master_port}"
            
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        logger.info(f"Launching {num_gpus_per_machine} processes on machine {machine_rank}")
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                cfg,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*cfg)


def initialize_process_group_with_retry(backend, init_method, world_size, rank, timeout, max_retries=3, retry_interval=5):
    """Initialize process group with retry logic."""
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Process group initialization attempt {attempt+1}/{max_retries}")
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                timeout=timeout,
            )
            logger.info(f"Successfully initialized process group on attempt {attempt+1}")
            return True
        except Exception as e:
            if "address already in use" in str(e).lower():
                logger.warning(f"Port is already in use. Retrying in {retry_interval} seconds...")
                if "MASTER_PORT" in os.environ:
                    # Try a different port
                    new_port = _find_free_port()
                    os.environ["MASTER_PORT"] = str(new_port)
                    # Update init_method with new port
                    parts = init_method.split(":")
                    if len(parts) == 3 and parts[0] == "tcp":
                        init_method = f"{parts[0]}:{parts[1]}:{new_port}"
            elif attempt < max_retries - 1:
                logger.warning(f"Error initializing process group: {e}. Retrying in {retry_interval} seconds...")
            else:
                logger.error(f"Failed to initialize process group after {max_retries} attempts: {e}")
                if "NCCL_DEBUG" not in os.environ or os.environ["NCCL_DEBUG"] != "INFO":
                    logger.error("Consider setting NCCL_DEBUG=INFO for more detailed error messages")
                raise e
            logger.info(f"Sleeping for {retry_interval} seconds")
            time.sleep(retry_interval)
    
    return False


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    cfg,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank

    if dist_url.startswith("tcp://"):
        parts = dist_url.split(":")
        if len(parts) == 3:
            master_addr = parts[1].replace("//", "")
            master_port = parts[2]
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
    
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing process group: rank {global_rank}/{world_size} - {dist_url}")
    logger.info(f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")
    
    # Make multiple attempts to initialize the process group
    try:
        success = initialize_process_group_with_retry(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
            max_retries=3
        )
        
        if not success:
            raise RuntimeError("Failed to initialize process group after multiple attempts")
            
    except Exception as e:
        logger.error(f"Process group URL: {dist_url}")
        logger.error(f"Error initializing process group. MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        
        # Log network information to help debugging
        try:
            logger.error("Network interfaces on this machine:")
            for iface, addr in get_network_interfaces():
                logger.error(f"  - {iface}: {addr}")
        except Exception:
            pass
            
        raise e

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count(), f"num_gpus_per_machine {num_gpus_per_machine} is greater than the number of GPUs on this machine {torch.cuda.device_count()}"
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    logger.info(f"Process successfully initialized: global rank {global_rank}, local rank {local_rank}")
    main_func(*cfg)
