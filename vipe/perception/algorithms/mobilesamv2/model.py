import sys

from vipe.perception.algorithms.ultralytics.yolo.cfg import get_cfg  # noqa: E402
from vipe.perception.algorithms.ultralytics.yolo.engine.exporter import Exporter  # noqa: E402
from vipe.perception.algorithms.ultralytics.yolo.engine.model import YOLO  # noqa: E402
from vipe.perception.algorithms.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, ROOT, is_git_dir  # noqa: E402

from vipe.perception.algorithms.ultralytics.yolo.utils.torch_utils import (  # noqa: E402
    model_info,
    smart_inference_mode,
)
from .predict import PromptModelPredictor  # noqa: E402


class ObjectAwareModel(YOLO):
    def init(self, **kwargs):
        overrides = self.overrides.copy()
        overrides["conf"] = 0.25
        overrides["imgsz"] = 640
        overrides["verbose"] = kwargs.get("verbose", False)
        overrides.update(kwargs)  # prefer kwargs
        overrides["mode"] = kwargs.get("mode", "predict")
        assert overrides["mode"] in ["track", "predict"]
        overrides["save"] = kwargs.get(
            "save", False
        )  # do not save by default if called in Python
        self.predictor = PromptModelPredictor(overrides=overrides)

        self.predictor.setup_model(model=self.model, verbose=False)
        return None

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = (
                ROOT / "assets"
                if is_git_dir()
                else "https://ultralytics.com/images/bus.jpg"
            )
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        return self.predictor(source, stream=stream)

    def train(self, **kwargs):
        raise NotImplementedError("Currently, the training codes are on the way.")

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        overrides = dict(task="detect")
        overrides.update(kwargs)
        overrides["mode"] = "export"
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        # if args.imgsz == DEFAULT_CFG.imgsz:
        #    args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = 640
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args)(model=self.model)

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}"
        )
