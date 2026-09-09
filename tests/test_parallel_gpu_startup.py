from types import SimpleNamespace

from annotate import _start_gpu_servers


def test_all_gpu_servers_start_before_waiting_for_readiness():
    started = []
    waited = []
    servers = [SimpleNamespace(gpu_id=i, start=lambda i=i: started.append(i)) for i in range(3)]

    def ready(i):
        assert started == [0, 1, 2]
        waited.append(i)

    events = [SimpleNamespace(wait=lambda i=i: ready(i)) for i in range(3)]
    _start_gpu_servers(servers, events)
    assert waited == [0, 1, 2]
