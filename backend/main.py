from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime, timezone
from routes import router
from config import settings
from middleware import SecurityHeadersMiddleware, RateLimitMiddleware
import logging
import json
import asyncio
from pathlib import Path

# Apply startup optimizations early
from startup_config import apply_startup_optimizations

apply_startup_optimizations()

# Import circuit breaker utilities
try:
    from utils.circuit_breaker import get_all_circuit_breaker_stats
except ImportError:

    def get_all_circuit_breaker_stats():
        return {}


# Configure logging
log_file = None
if settings.log_file:
    # Create log directory if it doesn't exist
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_file

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file,
)
logger = logging.getLogger(__name__)

# Create FastAPI app with environment-specific settings
app = FastAPI(
    title="StudyMate API",
    description="StudyMate v2 backend API for audio processing and study material generation",
    version="2.0.0",
    debug=settings.debug,
    # Hide docs in production unless explicitly enabled
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)


# Add debug middleware to log upload requests
@app.middleware("http")
async def debug_upload_requests(request: Request, call_next):
    if request.url.path == "/upload":
        logger.debug(f"Upload request to {request.url.path}")
        logger.debug(f"Method: {request.method}")
        logger.debug(f"Headers: {dict(request.headers)}")
        logger.debug(f"Content-Type: {request.headers.get('content-type', 'Not set')}")

    response = await call_next(request)
    return response


# Log startup information
logger.info(f"Starting StudyMate API in {settings.environment} mode")
logger.info(f"Debug mode: {settings.debug}")
logger.info(f"CORS origins: {settings.allowed_origins}")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Cache-Control",
        "Pragma",
    ],
    expose_headers=["*"],
    max_age=settings.cors_max_age,
)

# Production-specific middleware
if settings.is_production:
    logger.info("Applying production middleware")

    # Add trusted host middleware for production
    if settings.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts,
        )
        logger.info(f"Trusted hosts configured: {settings.trusted_hosts}")

    # Add security headers middleware
    if settings.secure_headers:
        app.add_middleware(SecurityHeadersMiddleware)
        logger.info("Security headers middleware enabled")

    # Add rate limiting middleware with production settings
    app.add_middleware(
        RateLimitMiddleware,
        calls=settings.rate_limit_calls,
        period=settings.rate_limit_period,
    )
    logger.info(
        f"Rate limiting: {settings.rate_limit_calls} calls per {settings.rate_limit_period} seconds"
    )

# Development-specific middleware
elif settings.is_development:
    logger.info("Running in development mode")

    # Add rate limiting middleware with development settings (more lenient)
    app.add_middleware(
        RateLimitMiddleware,
        calls=settings.rate_limit_calls,
        period=settings.rate_limit_period,
    )
    logger.info(
        f"Development rate limiting: {settings.rate_limit_calls} calls per {settings.rate_limit_period} seconds"
    )

app.include_router(router)

# Global background task storage
background_tasks = []


def create_background_task(coro, task_name: str = "background_task"):
    """
    Create a background task with proper exception handling

    Args:
        coro: The coroutine to run as a background task
        task_name: A descriptive name for the task (for logging)

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro)
    background_tasks.append(task)

    def handle_task_exception(task):
        """Handle exceptions from background tasks"""
        try:
            task.result()  # This will raise any exception that occurred
            logger.info(f"✅ Background task '{task_name}' completed successfully")
        except asyncio.CancelledError:
            logger.info(f"🔄 Background task '{task_name}' was cancelled")
        except Exception as e:
            logger.error(f"❌ Background task '{task_name}' failed: {e}")
        finally:
            # Remove completed/failed task from active tasks
            if task in background_tasks:
                background_tasks.remove(task)

    task.add_done_callback(handle_task_exception)
    return task


# Startup and shutdown events for cleanup service
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logger.info("Application startup - initializing services...")

    # Start the optional cleanup service
    try:
        from utils.cleanup_service import start_cleanup_service

        start_cleanup_service()
    except Exception as e:
        logger.error(f"Failed to start cleanup service: {e}")
        # Don't fail startup if cleanup service fails

    # Initialize S3 storage in background for better upload performance
    try:
        from utils.s3_storage import init_storage_background

        # Start S3 initialization in background - don't await to avoid blocking startup
        # Use helper function for proper exception handling
        create_background_task(init_storage_background(), task_name="S3_initialization")
        logger.info("🚀 S3 background initialization started")
    except Exception as e:
        logger.error(f"Failed to start S3 background initialization: {e}")
        # Don't fail startup if S3 background init fails


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on application shutdown"""
    logger.info("Application shutdown - cleaning up services...")

    # Cancel any remaining background tasks
    if background_tasks:
        logger.info(f"Cancelling {len(background_tasks)} background tasks...")
        for task in background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to be cancelled (with timeout)
        if background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*background_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Some background tasks did not shut down gracefully")
            except Exception as e:
                logger.error(f"Error during background task cleanup: {e}")

        background_tasks.clear()

    # Stop the cleanup service
    try:
        from utils.cleanup_service import stop_cleanup_service

        stop_cleanup_service()
    except Exception as e:
        logger.error(f"Error stopping cleanup service: {e}")


@app.get("/")
def read_root():
    return {
        "message": "StudyMate v2 backend is live!",
        "environment": settings.environment,
        "version": "2.0.0",
        "debug": settings.debug,
        "cors_origins": settings.allowed_origins if settings.debug else "configured",
        "docs_url": "/docs" if settings.debug else "disabled",
    }


@app.get("/health/background-tasks")
def get_background_task_status():
    """Get status of background tasks (for monitoring/debugging)"""
    task_info = []
    for i, task in enumerate(background_tasks):
        task_info.append(
            {
                "task_id": i,
                "done": task.done(),
                "cancelled": task.cancelled(),
                "exception": (
                    str(task.exception()) if task.done() and task.exception() else None
                ),
            }
        )

    return {
        "active_tasks": len(background_tasks),
        "tasks": task_info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "debug": settings.debug,
        "version": "2.0.0",
    }


@app.get("/health/detailed")
def detailed_health_check():
    """Detailed health check including dependencies and circuit breaker status"""
    health_status = {
        "status": "healthy",
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "dependencies": {},
        "circuit_breakers": get_all_circuit_breaker_stats(),
    }

    overall_healthy = True

    # Check Redis connection if configured
    if settings.redis_url:
        try:
            import redis

            redis_client = redis.from_url(settings.redis_url, socket_timeout=5)
            try:
                redis_client.ping()
            finally:
                redis_client.close()
            health_status["dependencies"]["redis"] = {
                "status": "healthy",
                "latency_ms": None,  # Could measure actual latency
            }
        except Exception as e:
            health_status["dependencies"]["redis"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            overall_healthy = False
    # Check OpenAI API key configuration
    if settings.openai_api_key:
        health_status["dependencies"]["openai"] = {
            "status": "configured",
            "key_format": (
                "valid" if settings.openai_api_key.startswith("sk-") else "invalid"
            ),
        }
        if not settings.openai_api_key.startswith("sk-"):
            overall_healthy = False
    else:
        health_status["dependencies"]["openai"] = {"status": "not_configured"}
        if settings.is_production:
            overall_healthy = False

    # Check file system permissions
    try:
        import os

        # Ensure upload directory exists before testing write permissions
        upload_dir = settings.upload_directory
        os.makedirs(upload_dir, exist_ok=True)

        # Test write permissions in configured upload directory
        # Test write permissions in configured upload directory
        test_file = os.path.join(upload_dir, ".health_check")
        try:
            with open(test_file, "w") as f:
                f.write("test")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

        health_status["dependencies"]["filesystem"] = {
            "status": "healthy",
            "upload_dir_writable": True,
            "upload_dir_path": upload_dir,
        }
    except Exception as e:
        health_status["dependencies"]["filesystem"] = {
            "status": "unhealthy",
            "error": str(e),
            "upload_dir_writable": False,
            "upload_dir_path": settings.upload_directory,
        }
        overall_healthy = False

    # Set overall status
    health_status["status"] = "healthy" if overall_healthy else "degraded"

    # Return appropriate HTTP status
    status_code = 200 if overall_healthy else 503

    return Response(
        content=json.dumps(health_status, indent=2),
        status_code=status_code,
        media_type="application/json",
    )


@app.get("/config")
def get_config():
    """Get current configuration (debug info only available in development)"""
    base_config = {
        "environment": settings.environment,
        "version": "2.0.0",
        "debug": settings.debug,
    }

    # Only expose detailed config in development
    if settings.is_development:
        base_config.update(
            {
                "cors_origins": settings.allowed_origins,
                "host": settings.host,
                "port": settings.port,
                "reload": settings.reload,
                "log_level": settings.log_level,
                "rate_limit": {
                    "calls": settings.rate_limit_calls,
                    "period": settings.rate_limit_period,
                },
                "upload_settings": {
                    "max_size": settings.upload_max_size,
                    "directory": settings.upload_directory,
                    "temp_directory": settings.temp_directory,
                },
            }
        )

    return base_config


# Global exception handlers for better error responses
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with consistent error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions"""
    logger.error(f"ValueError on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": f"Invalid input: {str(exc)}",
            "status_code": 400,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(
        f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True
    )

    # Don't expose internal error details in production
    error_message = (
        "An internal server error occurred" if settings.is_production else str(exc)
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": error_message,
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
        },
    )
