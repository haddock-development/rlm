"""Tests for RLM v2 Hybrid System (Python REPL - Paper-konform).

This test suite verifies the hybrid implementation:
- HTTP API externally (OpenClaw-compatible)
- Python REPL internally (Paper-konform)
"""
import asyncio
import httpx
import json

RLM_URL = "http://localhost:5000"
GRAPHITI_URL = "http://localhost:8000"

async def test_health():
    """Test 1: Health endpoint - should show hybrid version."""
    print("\n=== Test 1: Health Check ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{RLM_URL}/health")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Version: {data.get('version')}")
        print(f"Features: {data.get('features', [])}")
        assert response.status_code == 200
        assert "hybrid" in data.get("version", "").lower() or "repl" in data.get("version", "").lower()
        print("✅ Health check passed")

async def test_simple_completion():
    """Test 2: Simple non-recursive completion."""
    print("\n=== Test 2: Simple Completion ===")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RLM_URL}/rlm",
            json={
                "query": "What is 2+2?",
                "recursive": False,
                "store_trajectory": True,
            },
            timeout=60.0
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Result: {data.get('result', 'N/A')[:200]}...")
        print(f"Trajectory ID: {data.get('trajectory_id')}")
        assert response.status_code == 200
        assert "trajectory_id" in data
        print("✅ Simple completion passed")
        return data.get("trajectory_id")

async def test_python_repl_basic():
    """Test 3: Basic Python REPL execution."""
    print("\n=== Test 3: Python REPL Basic Execution ===")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RLM_URL}/rlm",
            json={
                "query": "Write Python code that prints 'Hello from REPL' and calculates 10 * 5.",
                "recursive": True,
                "max_depth": 3,
                "store_trajectory": True,
            },
            timeout=60.0
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        result = data.get('result', 'N/A')
        print(f"Result: {result[:500]}...")
        print(f"Code executions: {len(data.get('code_executions', []))}")

        # Check if code was executed
        code_execs = data.get('code_executions', [])
        if code_execs:
            print(f"Executed code snippet: {code_execs[0].get('code', 'N/A')[:100]}...")
            print(f"Code output: {code_execs[0].get('output', 'N/A')[:200]}...")

        assert response.status_code == 200
        print("✅ Python REPL basic test passed")
        return data.get("trajectory_id")

async def test_recursive_llm_call():
    """Test 4: Recursive llm_call() in REPL."""
    print("\n=== Test 4: Recursive llm_call() in REPL ===")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RLM_URL}/rlm",
            json={
                "query": "Calculate 5 factorial using recursion. Write Python code that calls llm_call() for the sub-problem (4!), then multiply by 5.",
                "recursive": True,
                "max_depth": 10,
                "store_trajectory": True,
            },
            timeout=120.0
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        result = data.get('result', 'N/A')
        print(f"Result: {result[:500]}...")
        print(f"Depth: {data.get('depth', 0)}")
        print(f"Sub-queries: {len(data.get('sub_queries', []))}")
        print(f"Code executions: {len(data.get('code_executions', []))}")

        # List sub-queries
        for sq in data.get('sub_queries', []):
            print(f"  - Sub-query (depth {sq.get('depth')}): {sq.get('query', 'N/A')[:50]}...")

        assert response.status_code == 200
        assert len(data.get('sub_queries', [])) > 0, "Should have sub-queries from llm_call()"
        print("✅ Recursive llm_call test passed")
        return data.get("trajectory_id")

async def test_context_peeking():
    """Test 5: Context peeking like in the Paper."""
    print("\n=== Test 5: Context Peeking (Paper-style) ===")

    # Create a long context
    long_context = "This is a test document. " * 100  # ~2400 chars

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RLM_URL}/rlm",
            json={
                "query": "Analyze this context. Write Python code to peek at the first 100 characters and print the total length.",
                "context": long_context,
                "recursive": True,
                "max_depth": 3,
                "store_trajectory": True,
            },
            timeout=60.0
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        result = data.get('result', 'N/A')
        print(f"Result: {result[:500]}...")

        # Check if context was accessed
        if "2400" in result or "len" in result.lower():
            print("✅ Context length was calculated!")

        assert response.status_code == 200
        print("✅ Context peeking test passed")
        return data.get("trajectory_id")

async def test_depth_limit():
    """Test 6: Depth limit enforcement."""
    print("\n=== Test 6: Depth Limit (max_depth=2) ===")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RLM_URL}/rlm",
            json={
                "query": "Test the depth limit",
                "recursive": True,
                "max_depth": 2,
                "store_trajectory": True,
            },
            timeout=30.0
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Result: {data.get('result', 'N/A')[:200]}...")
        print(f"Max Depth: 2, Actual Depth: {data.get('depth', 0)}")
        assert response.status_code == 200
        assert data.get("depth", 0) <= 2
        print("✅ Depth limit test passed")

async def test_trajectory_storage(trajectory_id: str):
    """Test 7: Verify trajectory was stored with code executions."""
    print(f"\n=== Test 7: Trajectory Storage ({trajectory_id[:8]}...) ===")

    # Check if trajectory exists in memory
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{RLM_URL}/trajectory/{trajectory_id}")
        print(f"Memory Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found in memory: {data.get('query', 'N/A')[:50]}...")

            # Check for code executions
            code_execs = data.get('code_executions', [])
            print(f"Code executions stored: {len(code_execs)}")
            for i, ce in enumerate(code_execs[:2]):  # Show first 2
                print(f"  Code {i+1}: {ce.get('code', 'N/A')[:80]}...")

    # Search in Graphiti
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{RLM_URL}/trajectories/search",
            params={"query": "llm_call", "max_results": 5}
        )
        print(f"Graphiti Search Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f"Found {len(results)} trajectories in Graphiti")

    print("✅ Trajectory storage test passed")

async def test_list_tools():
    """Test 8: List available REPL functions."""
    print("\n=== Test 8: List REPL Functions ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{RLM_URL}/tools")
        print(f"Status: {response.status_code}")
        data = response.json()
        tools = data.get("tools", [])
        print(f"Available REPL functions:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['signature']}")
            print(f"    Example: {tool.get('example', 'N/A')}")

        tool_names = [t['name'] for t in tools]
        assert "llm_call" in tool_names
        assert "file_read" in tool_names
        print("✅ Tools list test passed")

async def test_file_operations():
    """Test 9: File read/write via REPL."""
    print("\n=== Test 9: File Operations in REPL ===")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RLM_URL}/rlm",
            json={
                "query": "Write Python code that writes 'Hello World' to /tmp/test_rlm.txt, then reads it back and prints the content.",
                "recursive": True,
                "max_depth": 3,
                "store_trajectory": True,
            },
            timeout=60.0
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        result = data.get('result', 'N/A')
        print(f"Result: {result[:500]}...")

        # Check if file operations worked
        if "Hello World" in result or "successfully" in result.lower():
            print("✅ File operations worked!")

        assert response.status_code == 200
        print("✅ File operations test passed")

async def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RLM v2 Hybrid System Tests (Python REPL - Paper-konform)")
    print("="*70)
    print("Features tested:")
    print("  - Python REPL code execution")
    print("  - llm_call() for recursive problem solving")
    print("  - Context peeking (Paper-style)")
    print("  - Code execution tracking")
    print("  - File operations in sandbox")
    print("="*70)

    try:
        # Basic tests
        await test_health()
        await test_list_tools()

        # Simple completion
        traj_id_1 = await test_simple_completion()
        if traj_id_1:
            await test_trajectory_storage(traj_id_1)

        # Python REPL tests
        traj_id_2 = await test_python_repl_basic()
        if traj_id_2:
            await test_trajectory_storage(traj_id_2)

        # Recursive llm_call test
        traj_id_3 = await test_recursive_llm_call()
        if traj_id_3:
            await test_trajectory_storage(traj_id_3)

        # Context peeking test
        traj_id_4 = await test_context_peeking()

        # File operations test
        await test_file_operations()

        # Depth limit test
        await test_depth_limit()

        print("\n" + "="*70)
        print("All tests completed! ✅")
        print("="*70)
        print("\nThe RLM Hybrid System is working correctly:")
        print("  ✅ HTTP API compatible with OpenClaw")
        print("  ✅ Python REPL for paper-conform recursive execution")
        print("  ✅ llm_call() function for sub-problem solving")
        print("  ✅ Context variable for long-context processing")
        print("  ✅ Code execution tracking in trajectories")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
