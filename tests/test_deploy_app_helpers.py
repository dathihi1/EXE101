from vsl_mvp.deploy_app import quality_hints, stable_sign_id


def test_vietnamese_sign_ids_are_stable():
    assert stable_sign_id("Ngày Nhà giáo Việt Nam") == "ngay-nha-giao-viet-nam"
    assert stable_sign_id("Quạt (đứng)") == "quat-dung"
    assert stable_sign_id("Ướt") == "uot"


def test_quality_hints_cover_camera_failure_states():
    assert any("bàn tay" in hint for hint in quality_hints("no_hand", {"hand_frame_ratio": 0.0}))
    assert any("hai giây" in hint for hint in quality_hints("too_short", {}))
    assert any("video mẫu" in hint for hint in quality_hints("wrong_target", {}))
    assert quality_hints("ok", {"hand_frame_ratio": 1.0, "both_hands_ratio": 1.0}) == []
