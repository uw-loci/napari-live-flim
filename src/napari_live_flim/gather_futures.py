from concurrent.futures import Future
from threading import RLock

def gather_futures(*futures : Future):
    """
    Return a new future that completes when all the given futures complete.

    The value of the returned future is the tuple containing the values of all
    the given futures.

    If any of the futures complete with an exception, the returned future also
    completes with an exception (which is arbitrarily chosen among the given
    futures' exceptions).

    If the returned future is canceled, it has no effect on the given futures
    themselves.
    """

    ret = Future()

    # Immediately mark as running, because we may finish upon calling
    # add_done_callback()
    not_canceled = ret.set_running_or_notify_cancel()
    assert not_canceled

    # We need a reentrant lock because done_callback() may be called
    # synchronously inside add_done_callback()
    lock = RLock() 

    with lock:
        unfinished = set(futures)
        if len(unfinished) < len(futures):
            raise ValueError("Futures must be distinct")

        results = [None] * len(futures)
        finished = [False]

        for i, fut in enumerate(futures):
            def done_callback(f : Future):
                finished_results = None
                finished_exception = None
                with lock:
                    if finished[0]:
                        return
                    unfinished.remove(f)
                    try:
                        results[i] = f.result()
                        if not unfinished:
                            finished_results = tuple(results)
                            finished[0] = True
                    except Exception as e:
                        finished_exception = e
                        finished[0] = True
                    need_to_set_result = finished[0]

                if need_to_set_result:
                    if finished_results:
                        ret.set_result(finished_results)
                    else:
                        ret.set_exception(finished_exception)

            fut.add_done_callback(done_callback)

    return ret