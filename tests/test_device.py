from medical_scribe.device import pick_device


def test_pick_device_prefers_cuda(mocker):
    mocker.patch("medical_scribe.device.torch.cuda.is_available", return_value=True)
    mocker.patch("medical_scribe.device.torch.backends.mps.is_available", return_value=False)
    assert pick_device() == "cuda"


def test_pick_device_falls_back_to_mps(mocker):
    mocker.patch("medical_scribe.device.torch.cuda.is_available", return_value=False)
    mocker.patch("medical_scribe.device.torch.backends.mps.is_available", return_value=True)
    assert pick_device() == "mps"


def test_pick_device_falls_back_to_cpu(mocker):
    mocker.patch("medical_scribe.device.torch.cuda.is_available", return_value=False)
    mocker.patch("medical_scribe.device.torch.backends.mps.is_available", return_value=False)
    assert pick_device() == "cpu"
