import logging
from seg.utils.register import MODEL_REGISTRY


def get_model(cfg):
    logger = logging.getLogger("seg")
    try:
        model = MODEL_REGISTRY.get(f"build_{cfg.MODEL.NAME}")(cfg)
        logger.info(f"Model name ================ {cfg.MODEL.NAME} ===================")
    except KeyError:
        raise NotImplementedError(f"The model {cfg.MODEL.NAME} has not implemented!")
    return model