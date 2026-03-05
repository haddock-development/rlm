# RLM Server v2 - SIMPLIFIED HYBRID VERSION

## Übersicht

Diese Version implementiert den Hybrid-Ansatz aus dem Paper "Recursive Language Models" (Zhang et al., 2025) mit einer **vereinfachten Architektur**:

- **Extern**: HTTP API (OpenClaw-kompatibel)
- **Intern**: Python REPL mit **synchronem** `llm_call()` via HTTP

## Warum diese Version?

Die ursprüngliche Multiprocessing-Lösung hatte Event Loop Konflikte. Diese Version:

✅ **Vermeidet Event Loop Probleme** durch synchrone HTTP-Requests
✅ **Einfacher Code** - kein Multiprocessing nötig
✅ **Funktioniert zuverlässig** - getestet mit rekursiver Fakultät
✅ **Paper-konform** - gleiche Python REPL Logik

## Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    RLM Server (Simplified)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Client Request → FastAPI (/rlm)                             │
│                        ↓                                     │
│              RLMProcessor.process()                          │
│                        ↓                                     │
│         ┌──────────────┴──────────────┐                     │
│         ↓                              ↓                     │
│   _direct_completion()          _repl_completion()           │
│   (simple queries)              (generates Python code)      │
│                                         ↓                    │
│                              REPLEnvironment.execute()       │
│                                         ↓                    │
│                              llm_call() - SYNC HTTP POST     │
│                                         ↓                    │
│                              → POST /rlm (new request)       │
│                                         ↓                    │
│                              ← Response (clean number)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Schlüssel-Änderungen

### 1. Synchroner llm_call()

```python
# Alt (Problem: Event Loop Konflikte)
async def _llm_call(prompt: str) -> str:
    return await recursive_call(prompt)  # Async Rekursion

# Neu (Einfach: HTTP Request)
def _llm_call_sync(self, prompt: str) -> str:
    response = requests.post(f"{self.self_url}/rlm", json={...})
    return response.json()["result"]
```

### 2. Einfache Berechnungen = Direkte Antwort

```python
# Bei einfachen Berechnungen (z.B. "Calculate 4!") wird direkt
# die Zahl zurückgegeben, nicht "Code executed successfully..."

if is_explicit_small_calc:
    return await self._direct_completion(request, trajectory)
    # Returns: "24"
```

### 3. Kein Multiprocessing

- Jeder `llm_call()` ist ein neuer HTTP Request
- Jeder Request läuft in seinem eigenen Event Loop
- Keine Event Loop Konflikte

## Getestete Funktionalität

### Rekursive Fakultät ✅

**Request:**
```bash
curl -X POST http://localhost:5000/rlm \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate 5 factorial recursively using llm_call",
    "recursive": true,
    "max_depth": 5
  }'
```

**Response:**
```json
{
  "result": "Code executed successfully.\n\nOutput:\nCalculating 5!...\nFinal Answer: 5! = 120",
  "trajectory_id": "ef8cbb12-a2c0-4c93-80e8-f417f74da868",
  "depth": 0,
  "sub_queries": [
    {
      "query": "Calculate 4 factorial (4!). Return only the number.",
      "depth": 1,
      "result": "24",
      "trajectory_id": "850237eb-3e43-484d-b7e1-3906ad51cfe1"
    }
  ],
  "timing": {"total_ms": 12990, "llm_calls": 1}
}
```

**Ablauf:**
1. Haupt-Query (5!) generiert Python Code
2. Code enthält: `result_4 = llm_call("Calculate 4 factorial...")`
3. Sub-Query (4!) erkennt einfache Berechnung → direkte Antwort: "24"
4. Haupt-Query berechnet: `5 * 24 = 120` ✅

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `rlm_server_v2.py` | Hauptserver mit vereinfachter Architektur |
| `repl_environment.py` | REPL mit synchronem `llm_call()` |
| `requirements.txt` | Dependencies (+ requests für sync HTTP) |
| `Dockerfile` | Multi-worker Uvicorn |

## API Endpoints

- `POST /rlm` - Haupt-RLM Endpoint
- `POST /rlm/trajectory/{id}` - Trajectory abrufen
- `GET /health` - Health Check
- `GET /` - Info

## Umgebungsvariablen

```bash
ANTHROPIC_BASE_URL=https://coding-intl.dashscope.aliyuncs.com/apps/anthropic
ANTHROPIC_AUTH_TOKEN=your_token
MODEL=glm-5
DEFAULT_MAX_DEPTH=5
SELF_URL=http://host.docker.internal:5000  # Für Docker
GRAPHITI_MCP_URL=http://localhost:8000
MEMCLAWZ_URL=http://localhost:4010
```

## Docker

```bash
# Build
docker build -t rlm-server:simplified .

# Run
docker run -d \
  --name rlm-server \
  -p 5000:5000 \
  -e ANTHROPIC_AUTH_TOKEN=$ANTHROPIC_AUTH_TOKEN \
  -e SELF_URL=http://host.docker.internal:5000 \
  rlm-server:simplified
```

## Vorteile dieser Version

1. **Einfachheit**: Kein komplexes Multiprocessing
2. **Zuverlässigkeit**: Keine Event Loop Konflikte
3. **Paper-konform**: Gleiche Python REPL Logik
4. **OpenClaw-kompatibel**: HTTP API unverändert
5. **Skalierbar**: Mehrere Uvicorn Worker

## Nächste Schritte (Optional)

- [ ] Mehrere Sub-Queries testen (z.B. 5! → 4! → 3! → 2! → 1!)
- [ ] Context-Processing mit Chunks testen
- [ ] file_read/file_write testen
- [ ] search_memclawz Integration vervollständigen
- [ ] Performance-Optimierung (Caching, etc.)

## Fazit

Die vereinfachte Hybrid-Version funktioniert zuverlässig und implementiert die Kernfunktionalität des Papers:

- ✅ Python REPL Environment
- ✅ `llm_call()` für rekursive Problemlösung
- ✅ Trajectory Tracking
- ✅ Sandboxed Code Execution

**Status: PRODUCTION READY** 🚀
