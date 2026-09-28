import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voltguard.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["Alerts"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected.")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to websocket: {e}")


manager = ConnectionManager()


@router.websocket("/alerts")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time event stream for vehicle alerts.
    Clients connect here to stream real-time fault predictions or self-healing events.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # In a real setup, we might listen to an internal Event Bus here and
            # push to the websocket independently of user incoming messages.
            # Here we just echo back acknowledgements.
            await manager.broadcast({"status": "received", "data": json.loads(data)})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
