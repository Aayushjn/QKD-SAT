from concurrent.futures.thread import ThreadPoolExecutor


def limit_function_execution(timeout: float, func, *args, **kwargs):
    pool = ThreadPoolExecutor()
    future = pool.submit(func, *args, **kwargs)
    try:
        return future.result(timeout)
    except TimeoutError as e:
        future.cancel()
        raise e
    finally:
        pool.shutdown(wait=False)
