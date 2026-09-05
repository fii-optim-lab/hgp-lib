def pytest_addoption(parser):
    parser.addoption("--hgp-scenario")


def pytest_collection_modifyitems(config, items):
    scenario = config.getoption("--hgp-scenario")
    if scenario is None:
        return

    selected = []
    deselected = []
    for item in items:
        marker = item.get_closest_marker("scenario")
        if marker is not None and marker.args[0] == scenario:
            selected.append(item)
        else:
            deselected.append(item)

    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
