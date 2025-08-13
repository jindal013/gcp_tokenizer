# from ray.util.multiprocessing import Pool
# import time

# def f(index):
#     return index

# pool = Pool()
# for result in pool.map(f, range(1000000)):
#     print(result)

from ray.util.multiprocessing import Pool
import time

def f(index):
    return index

pool = Pool()

start = time.perf_counter()
results = list(pool.map(f, range(10000)))
end = time.perf_counter()
elapsed = end - start

print(f"Processed {len(results):,} items in {elapsed:.2f} s")
