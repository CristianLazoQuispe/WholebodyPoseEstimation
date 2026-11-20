# Parts of rtmlib 0.0.16 (https://github.com/Tau-J/rtmlib), Apache-2.0, vendored and modified by the
# WholeBodyPose authors; see the header of each module, THIRD_PARTY.md and LICENSES/Apache-2.0.txt.
"""ONNX Runtime + NumPy implementation of rtmlib's YOLOX, RTMPose and ``Wholebody`` (no rtmlib needed)."""

from wholebodypose.models.rtmpose._onnx.rtmw import RTMPose
from wholebodypose.models.rtmpose._onnx.wholebody import MODE, Wholebody
from wholebodypose.models.rtmpose._onnx.yolox import YOLOX

__all__ = ["MODE", "RTMPose", "Wholebody", "YOLOX"]
