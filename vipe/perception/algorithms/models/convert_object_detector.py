import sys

sys.path.append("/workspace/perception/algorithms")
from mobilesamv2 import ObjectAwareModel  # noqa: E402

obj_model_path = "/workspace/perception/algorithms/models/weight/ObjectAwareModel.pt"
ObjAwareModel = ObjectAwareModel(obj_model_path)
ObjAwareModel.export(format="engine", device="0")
