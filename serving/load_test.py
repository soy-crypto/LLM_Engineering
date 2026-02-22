import asyncio
import time
import httpx
import statistics

URL = "http://localhost:8000/generate"
CONCURRENT_USERS = 20
REQUESTS_PER_USER = 5

async def send_request(client, prompt):
    start = time.time()
    response = await client.post(URL, json={"prompt": prompt})
    latency = time.time() - start
    return latency

async def user_simulation(user_id):
    async with httpx.AsyncClient(timeout=60.0) as client:
        latencies = []
        for _ in range(REQUESTS_PER_USER):
            latency = await send_request(client, f"User {user_id} test prompt.")
            latencies.append(latency)
        return latencies

async def main():
    print(f"Running load test with {CONCURRENT_USERS} users...")

    tasks = [user_simulation(i) for i in range(CONCURRENT_USERS)]
    results = await asyncio.gather(*tasks)

    all_latencies = [lat for user in results for lat in user]

    print(f"Total requests: {len(all_latencies)}")
    print(f"Average latency: {statistics.mean(all_latencies):.3f} sec")
    print(f"P50 latency: {statistics.median(all_latencies):.3f} sec")
    print(f"P95 latency: {sorted(all_latencies)[int(len(all_latencies)*0.95)]:.3f} sec")
    print(f"Max latency: {max(all_latencies):.3f} sec")

if __name__ == "__main__":
    asyncio.run(main())
