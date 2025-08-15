#!/usr/bin/env python3
"""
Fake Ollama API Server with Supabase License Management
Mimics Ollama API endpoints for compatibility with various tools
"""

import asyncio
import atexit
import copy
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from supabase import Client, create_client

# Load environment variables
load_dotenv()

# Configuration
class Config:
    """Application configuration"""
    # Supabase
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://vstrhspqurvrqdmfamtp.supabase.co")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzdHJoc3BxdXJ2cnFkbWZhbXRwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk2MjAwNjUsImV4cCI6MjA2NTE5NjA2NX0.1vrH-tcQLw3EttC9bA7kqeF6vvhhW1B6ByMk9u8W6PI")
    
    # License
    LICENSE_KEY = os.environ.get("LICENSE_KEY", "")
    MAX_IPS_PER_KEY = int(os.environ.get("MAX_IPS_PER_KEY", "10"))
    
    # Server
    SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("SERVER_PORT", "14444"))
    
    # Features
    ENABLE_NVIDIA_SMI = os.environ.get("ENABLE_NVIDIA_SMI", "true").lower() == "true"
    GPU_COUNT = int(os.environ.get("GPU_COUNT", "6"))
    
    # Response settings
    MIN_TOKENS = int(os.environ.get("MIN_TOKENS", "500"))
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2500"))
    MIN_TPS = float(os.environ.get("MIN_TPS", "130"))
    MAX_TPS = float(os.environ.get("MAX_TPS", "200"))

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fake-ollama.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for logger_name in ["httpx", "httpcore", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Global variables
supabase: Optional[Client] = None
client_ip: Optional[str] = None
key_id: Optional[str] = None

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = Field(default="llama3.1:8b-instruct-q4_K_M")
    max_tokens: Optional[int] = Field(default=150)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(default=False)
    n: Optional[int] = Field(default=1, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0)
    stream_options: Optional[Dict[str, Any]] = None

class PullRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    insecure: Optional[bool] = False

# Available models
AVAILABLE_MODELS = [
    {
        "name": "llama3.1:8b-instruct-q4_K_M",
        "model": "llama3.1:8b-instruct-q4_K_M",
        "modified_at": "2025-01-21T19:26:37.037Z",
        "size": 4920753328,
        "digest": "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8.0B",
            "quantization_level": "Q4_K_M"
        }
    },
    {
        "name": "deepseek-r1:1.5b",
        "model": "deepseek-r1:1.5b",
        "modified_at": "2025-01-12T22:50:21.591Z",
        "size": 1117322599,
        "digest": "a42b25d8c10a841bd24724309898ae851466696a7d7f3a0a408b895538ccbc96",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "qwen2",
            "families": ["qwen2"],
            "parameter_size": "1.8B",
            "quantization_level": "Q4_K_M"
        }
    },
    {
        "name": "llama3.2:3b-instruct-fp16",
        "model": "llama3.2:3b-instruct-fp16",
        "modified_at": "2025-01-07T10:00:00.000Z",
        "size": 3870000000,
        "digest": "8a7b0e31a9d72f2c3d8c7f1f82aef5d9812cbbf85e97b2a310f3a77e219b9423",
        "details": {
            "parent_model": "llama-3.2",
            "format": "fp16",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "3.2B",
            "quantization_level": "FP16"
        }
    },
    {
        "name": "qwen2.5:7b-instruct",
        "model": "qwen2.5:7b-instruct",
        "modified_at": "2025-01-15T14:30:00.000Z",
        "size": 7342886912,
        "digest": "7c2a8f9d3e4b5c6d8e9f0a1b2c3d4e5f6789abcdef1234567890abcdef123456",
        "details": {
            "parent_model": "qwen2.5",
            "format": "gguf",
            "family": "qwen2",
            "families": ["qwen2"],
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M"
        }
    }
]

# Default response template
DEFAULT_RESPONSE = """I understand you're asking about AMD. AMD (Advanced Micro Devices) is a major semiconductor company that competes with Intel in processors and NVIDIA in graphics cards. Their Ryzen CPUs offer excellent performance for gaming and productivity tasks, while their Radeon GPUs provide competitive graphics solutions. AMD's focus on multi-core performance and competitive pricing has made them popular among enthusiasts and professionals. Their recent innovations in chiplet design and 3D V-Cache technology have pushed the boundaries of processor performance."""

def get_public_ip() -> str:
    """Get the public IP address of this server"""
    ip_services = [
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://checkip.amazonaws.com",
        "https://ifconfig.me/ip",
    ]
    
    for service in ip_services:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(service)
                response.raise_for_status()
                ip = response.text.strip()
                if ip and len(ip) <= 45:  # Basic IP validation
                    logger.info(f"Public IP obtained from {service}: {ip}")
                    return ip
        except Exception as e:
            logger.debug(f"Failed to get IP from {service}: {e}")
            continue
    
    logger.error("Failed to obtain public IP from all services")
    return "127.0.0.1"

def setup_nvidia_smi():
    """Setup fake nvidia-smi binary"""
    if not Config.ENABLE_NVIDIA_SMI:
        logger.info("nvidia-smi setup skipped (disabled in config)")
        return True
        
    try:
        # Generate GPU data
        gpu_data = []
        gpu_models = [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 4080",
            "NVIDIA A100-SXM4-40GB",
            "NVIDIA Tesla V100-SXM2-32GB",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA GeForce RTX 4070 Ti"
        ]
        
        for i in range(Config.GPU_COUNT):
            gpu_uuid = f"GPU-{uuid.uuid4()}"
            gpu_data.append({
                "index": i,
                "uuid": gpu_uuid,
                "name": gpu_models[i % len(gpu_models)],
                "driver": "545.29.06",
                "cuda": "12.3",
                "memory": 24576 if "4090" in gpu_models[i % len(gpu_models)] else 16384,
                "memory_used": random.randint(1000, 8000),
                "power_usage": 50 + i * 10,
                "power_cap": 450,
                "temp": 45 + i * 2,
                "fan": 30 + i * 3,
                "util": random.randint(0, 100)
            })

        # Build query output
        query_output = "\n".join([
            f'{gpu["uuid"]}, {gpu["driver"]}, {gpu["name"]}, {gpu["memory"]} MiB'
            for gpu in gpu_data
        ])

        # Build nvidia-smi script
        script = f'''#!/bin/bash
# Fake nvidia-smi for testing purposes

if [[ "$*" == *"--query-gpu="* ]]; then
    if [[ "$*" == *"--format=csv,noheader,nounits"* ]]; then
        echo "{query_output}"
    else
        echo "uuid, driver_version, name, memory.total"
        echo "{query_output}"
    fi
    exit 0
fi

# Default nvidia-smi output
cat << 'EOF'
{generate_nvidia_smi_output(gpu_data)}
EOF
'''

        # Write script to file
        nvidia_smi_path = Path("/usr/local/bin/nvidia-smi")
        nvidia_smi_path.parent.mkdir(parents=True, exist_ok=True)
        nvidia_smi_path.write_text(script)
        nvidia_smi_path.chmod(0o755)
        
        logger.info(f"Fake nvidia-smi set up with {Config.GPU_COUNT} GPUs")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup nvidia-smi: {e}")
        return False

def generate_nvidia_smi_output(gpu_data: List[Dict]) -> str:
    """Generate realistic nvidia-smi output"""
    output = [
        "+-----------------------------------------------------------------------------+",
        f"| NVIDIA-SMI {gpu_data[0]['driver']}    Driver Version: {gpu_data[0]['driver']}    CUDA Version: {gpu_data[0]['cuda']} |",
        "+-------------------------------+----------------------+----------------------+",
        "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |",
        "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |",
        "|                               |                      |               MIG M. |",
        "|===============================+======================+======================|"
    ]
    
    for i, gpu in enumerate(gpu_data):
        output.extend([
            f"| {gpu['index']:>3}  {gpu['name'][:21]:<21} Off  | 00000000:{i:02X}:00.0 Off |                  N/A |",
            f"| {gpu['fan']:>3}%   {gpu['temp']:>2}C    P2    {gpu['power_usage']:>3}W / {gpu['power_cap']:>3}W |  {gpu['memory_used']:>5}MiB / {gpu['memory']:>5}MiB |    {gpu['util']:>3}%      Default |",
            f"|                               |                      |                  N/A |",
            "+-------------------------------+----------------------+----------------------+"
        ])
    
    output.extend([
        "                                                                               ",
        "+-----------------------------------------------------------------------------+",
        "| Processes:                                                                  |",
        "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |",
        "|        ID   ID                                                   Usage      |",
        "|=============================================================================|",
        "|  No running processes found                                                 |",
        "+-----------------------------------------------------------------------------+"
    ])
    
    return "\n".join(output)

async def validate_license(key: str, ip: str) -> bool:
    """Validate license key and register IP (with backward compatibility)"""
    global key_id
    
    try:
        if not supabase:
            logger.error("Supabase client not initialized")
            return False
        
        # First, check what columns exist in the database
        # Try to get max_ips, but fall back if it doesn't exist
        try:
            # Try query with max_ips column
            key_response = supabase.table("serial_keys") \
                .select("id, max_ips") \
                .eq("key", key) \
                .eq("is_active", True) \
                .single() \
                .execute()
                
            if key_response.data:
                key_data = key_response.data
                key_id = key_data["id"]
                max_ips = key_data.get("max_ips", Config.MAX_IPS_PER_KEY)
            else:
                logger.error(f"License key not found or inactive: {key[:8]}...")
                return False
                
        except Exception as e:
            # If max_ips column doesn't exist, try without it
            logger.debug(f"Trying query without max_ips column: {e}")
            
            key_response = supabase.table("serial_keys") \
                .select("id") \
                .eq("key", key) \
                .eq("is_active", True) \
                .single() \
                .execute()
                
            if key_response.data:
                key_id = key_response.data["id"]
                max_ips = Config.MAX_IPS_PER_KEY  # Use default from config
                logger.info(f"Using default max_ips value: {max_ips}")
            else:
                logger.error(f"License key not found or inactive: {key[:8]}...")
                return False
        
        # Get current IPs for this key
        ip_response = supabase.table("key_ip_mappings") \
            .select("ip_address") \
            .eq("key_id", key_id) \
            .execute()
            
        existing_ips = [entry["ip_address"] for entry in ip_response.data]
        
        if ip in existing_ips:
            # Update last_used timestamp
            try:
                # Try to update - trigger will handle last_used if it exists
                supabase.table("key_ip_mappings") \
                    .update({"last_used": "now()"}) \
                    .eq("key_id", key_id) \
                    .eq("ip_address", ip) \
                    .execute()
            except Exception as e:
                # If trigger doesn't exist, update might fail but that's OK
                logger.debug(f"last_used update note: {e}")
                
            logger.info(f"Existing IP {ip} validated for key {key[:8]}...")
        else:
            # Check if we can add new IP
            if len(existing_ips) >= max_ips:
                logger.error(f"Max IP limit ({max_ips}) reached for key {key[:8]}...")
                return False
            
            # Add new IP
            supabase.table("key_ip_mappings") \
                .insert({"key_id": key_id, "ip_address": ip}) \
                .execute()
            logger.info(f"New IP {ip} registered for key {key[:8]}... ({len(existing_ips) + 1}/{max_ips})")
        
        return True
        
    except Exception as e:
        logger.error(f"License validation error: {e}")
        return False

def cleanup_license_ip():
    """Clean up IP mapping on shutdown"""
    global key_id, client_ip
    
    try:
        if key_id and client_ip and supabase:
            supabase.table("key_ip_mappings") \
                .delete() \
                .eq("key_id", key_id) \
                .eq("ip_address", client_ip) \
                .execute()
            logger.info(f"Cleaned up IP {client_ip} for key_id {key_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup IP: {e}")

# Register cleanup handlers
atexit.register(cleanup_license_ip)

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    cleanup_license_ip()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global supabase, client_ip
    
    # Startup
    logger.info("=" * 50)
    logger.info("Starting Fake Ollama API Server")
    logger.info("=" * 50)
    
    # Validate configuration
    if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set!")
        sys.exit(1)
    
    if not Config.LICENSE_KEY:
        logger.error("LICENSE_KEY must be set!")
        sys.exit(1)
    
    # Initialize Supabase
    try:
        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
        sys.exit(1)
    
    # Get public IP
    client_ip = get_public_ip()
    
    # Validate license
    if not await validate_license(Config.LICENSE_KEY, client_ip):
        logger.error("License validation failed!")
        sys.exit(1)
    
    # Setup nvidia-smi if enabled
    if Config.ENABLE_NVIDIA_SMI:
        setup_nvidia_smi()
    
    logger.info(f"Server ready on {Config.SERVER_HOST}:{Config.SERVER_PORT}")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fake Ollama API Server...")
    cleanup_license_ip()

app = FastAPI(
    title="Fake Ollama API",
    description="Ollama-compatible API server for testing",
    version="2.0.0",
    lifespan=lifespan
)

# Helper functions
def generate_completion_id() -> str:
    """Generate unique completion ID"""
    return f"chatcmpl-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"

def generate_response_content(token_count: int) -> str:
    """Generate response content with specified token count"""
    base_tokens = DEFAULT_RESPONSE.split()
    if token_count <= len(base_tokens):
        return " ".join(base_tokens[:token_count])
    
    # Repeat and trim to exact token count
    repeated = (base_tokens * ((token_count // len(base_tokens)) + 1))
    return " ".join(repeated[:token_count])

async def stream_chat_response(request: ChatRequest):
    """Stream chat completion response"""
    # Generate metrics
    completion_tokens = random.randint(Config.MIN_TOKENS, Config.MAX_TOKENS)
    tps = round(random.uniform(Config.MIN_TPS, Config.MAX_TPS), 2)
    total_duration = completion_tokens / tps
    
    # Generate content
    content = generate_response_content(completion_tokens)
    completion_id = generate_completion_id()
    created_time = int(time.time())
    
    # Start timing
    start_time = time.time()
    
    # Send role chunk
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"
    
    # Send content chunk
    content_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(content_chunk)}\n\n"
    
    # Simulate processing time
    elapsed = time.time() - start_time
    if total_duration > elapsed:
        await asyncio.sleep(total_duration - elapsed)
    
    # Send final chunk with usage
    usage = {
        "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
        "completion_tokens": completion_tokens,
        "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + completion_tokens,
    }
    
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": usage,
        "system_fingerprint": f"fp_{uuid.uuid4().hex[:10]}"
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# Routes
@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root endpoint"""
    return "Ollama is running"

@app.get("/api/version")
async def get_version():
    """Get version information"""
    return {
        "version": "0.1.32",
        "commit": uuid.uuid4().hex[:7],
        "llama_version": "b2724"
    }

@app.get("/api/tags")
async def list_models():
    """List available models"""
    return {"models": AVAILABLE_MODELS}

@app.post("/api/pull")
async def pull_model(request: PullRequest):
    """Pull model endpoint"""
    model_name = request.model or request.name
    
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Model name is required"
        )
    
    # Find model
    model = next(
        (m for m in AVAILABLE_MODELS if m["name"] == model_name or m["model"] == model_name),
        None
    )
    
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    logger.info(f"Model pull request: {model_name}")
    
    return {
        "status": "success",
        "message": f"Model '{model_name}' is available",
        "model": model
    }

@app.post("/api/generate")
async def generate(request: Request):
    """Generate endpoint (Ollama native format)"""
    try:
        body = await request.json()
        model = body.get("model", "llama3.1:8b-instruct-q4_K_M")
        prompt = body.get("prompt", "")
        stream = body.get("stream", False)
        
        # Convert to chat format
        chat_request = ChatRequest(
            messages=[Message(role="user", content=prompt)],
            model=model,
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                stream_chat_response(chat_request),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            completion_tokens = random.randint(Config.MIN_TOKENS, Config.MAX_TOKENS)
            content = generate_response_content(completion_tokens)
            
            return {
                "model": model,
                "created_at": time.time(),
                "response": content,
                "done": True,
                "context": [],
                "total_duration": int(random.uniform(1, 3) * 1e9),
                "load_duration": int(random.uniform(0.1, 0.5) * 1e9),
                "prompt_eval_duration": int(random.uniform(0.1, 0.3) * 1e9),
                "eval_count": completion_tokens,
                "eval_duration": int(random.uniform(1, 2) * 1e9)
            }
            
    except Exception as e:
        logger.error(f"Generate error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    completion_tokens = random.randint(Config.MIN_TOKENS, Config.MAX_TOKENS)
    tps = round(random.uniform(Config.MIN_TPS, Config.MAX_TPS), 2)
    total_duration = completion_tokens / tps
    
    content = generate_response_content(completion_tokens)
    completion_id = generate_completion_id()
    
    # Simulate processing
    await asyncio.sleep(total_duration)
    
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
            "completion_tokens": completion_tokens,
            "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + completion_tokens
        },
        "system_fingerprint": f"fp_{uuid.uuid4().hex[:10]}"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with backward compatibility"""
    health_data = {
        "status": "healthy",
        "server_time": int(time.time()),
        "server_ip": client_ip,
        "license_status": "valid" if key_id else "invalid",
        "models_available": len(AVAILABLE_MODELS),
        "gpu_count": Config.GPU_COUNT if Config.ENABLE_NVIDIA_SMI else 0
    }
    
    # Try to get license usage info if view exists
    if key_id and supabase:
        try:
            usage_response = supabase.table("active_license_connections") \
                .select("*") \
                .eq("license_key", LICENSE_KEY) \
                .single() \
                .execute()
                
            if usage_response.data:
                health_data["license_usage"] = {
                    "used_ips": usage_response.data.get("used_ips", 0),
                    "max_ips": usage_response.data.get("max_ips", Config.MAX_IPS_PER_KEY),
                    "available_slots": usage_response.data.get("available_slots", 0)
                }
        except Exception as e:
            # View might not exist, that's OK
            logger.debug(f"Could not get license usage from view: {e}")
    
    return health_data

@app.get("/api/ps")
async def list_running_models():
    """List running models (always empty for fake server)"""
    return {"models": []}

# Main entry point
if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "main:app",
        host=Config.SERVER_HOST,
        port=Config.SERVER_PORT,
        log_level="info",
        access_log=False
    )