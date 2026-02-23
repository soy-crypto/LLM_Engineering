import asyncio
import time
import statistics
import argparse
import aiohttp

# ==============================
# Load test client
# ==============================

class LoadTester:

    def __init__(self, url, prompt, num_requests, concurrency):

        self.url = url
        self.prompt = prompt
        self.num_requests = num_requests
        self.concurrency = concurrency

        self.latencies = []

    async def worker(self, session, request_id):

        start = time.time()

        async with session.post(
            self.url,
            json={"prompt": self.prompt}
        ) as response:

            await response.json()

        latency = time.time() - start

        self.latencies.append(latency)

    async def run(self):

        connector = aiohttp.TCPConnector(limit=self.concurrency)

        async with aiohttp.ClientSession(connector=connector) as session:

            tasks = []

            start_time = time.time()

            for i in range(self.num_requests):

                task = asyncio.create_task(
                    self.worker(session, i)
                )

                tasks.append(task)

                # control concurrency
                if len(tasks) >= self.concurrency:

                    await asyncio.gather(*tasks)
                    tasks = []

            if tasks:
                await asyncio.gather(*tasks)

            end_time = time.time()

        total_time = end_time - start_time

        return total_time

    def report(self, total_time):

        latencies_ms = [l * 1000 for l in self.latencies]

        print("\n========== RESULTS ==========")

        print(f"Requests: {len(latencies_ms)}")
        print(f"Total time: {total_time:.2f} sec")

        print(f"Throughput: {len(latencies_ms)/total_time:.2f} req/sec")

        print(f"Latency p50: {statistics.median(latencies_ms):.2f} ms")

        print(f"Latency p95: {percentile(latencies_ms, 95):.2f} ms")

        print(f"Latency p99: {percentile(latencies_ms, 99):.2f} ms")

        print(f"Mean latency: {statistics.mean(latencies_ms):.2f} ms")

def percentile(data, p):

    data_sorted = sorted(data)

    k = int(len(data_sorted) * p / 100)

    return data_sorted[k]

# ==============================
# Main
# ==============================

async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://localhost:8001/generate"
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=8
    )

    parser.add_argument(
        "--prompt",
        default="Explain transformer inference."
    )

    args = parser.parse_args()

    tester = LoadTester(
        args.url,
        args.prompt,
        args.requests,
        args.concurrency
    )

    total_time = await tester.run()

    tester.report(total_time)

if __name__ == "__main__":
    asyncio.run(main())