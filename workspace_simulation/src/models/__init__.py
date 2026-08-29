from .le_net5_net import LeNet5
from .mobilenetv2_net import (
    MobileNetV2Alpha035,
    MobileNetV2Alpha035GroupNorm,
    MobileNetV2Alpha035LayerNorm,
    MobileNetV2Baseline,
    MobileNetV2BaselineGroupNorm,
    MobileNetV2BaselineLayerNorm,
    MobileNetV2Small,
    MobileNetV2SmallGroupNorm,
    MobileNetV2SmallLayerNorm,
)
from .mobilenetv4_net import (
    MobileNetV4Small,
    MobileNetV4SmallGroupNorm,
    MobileNetV4SmallLayerNorm,
)
from .convolutional_net import ConvolutionalNet
from .fcn_net import FCN
from .deeper_fc_net import DeepFC
from .basic_conv_net import BasicConv
from .mobilenetv1_net import (
    MobileNetV1Small,
    MobileNetV1SmallGroupNorm,
    MobileNetV1SmallLayerNorm,
    MobileNetV1SmallNoBN,
)
from .fomo_net import (
    FOMOMNv2Alpha035,
    FOMOMNv2Alpha035GroupNorm,
    FOMOMNv2Alpha035LayerNorm,
    FOMOMNv2Baseline,
    FOMOMNv2BaselineGroupNorm,
    FOMOMNv2BaselineLayerNorm,
)
from .micro_yolo_net import MircoYOLO, MircoYOLOGroupNorm, MircoYOLOLayerNorm

from .fomo_net import FOMOLossPerson, FOMOLossVehicle, FOMOLossVehicleBinary
from .fomo_net import FOMOMetricsPerson, FOMOMetricsVehicle, FOMOMetricsVehicleBinary
from .micro_yolo_net import YoLoLossPerson, YoLoLossVehicle, YoLoLossVehicleBinary
from .micro_yolo_net import YoLoMAPPerson, YoLoMAPVehicle, YoLoMAPVehicleBinary
from .mobile_blocks import (
    ConvBN,
    DepthwiseSeparableConv,
    InvertedBottleNeck,
    LayerNorm2d,
    SEBlock,
    SeparableConvBn,
    UniversalInvertedBottleneck,
)
from .definitions import CRITERIA_REGISTRATION_NAMES, MODEL_REGISTRATION_NAMES
from .model_registry import CRITERIA_REGISTRY, MODEL_REGISTRY, CriteriaRegistry, ModelRegistry
from .parameter_vector import FederatedModelMixin, ParameterVectorMixin


for registry_name, class_name in MODEL_REGISTRATION_NAMES.items():
    MODEL_REGISTRY.register(registry_name, globals()[class_name])

for registry_name, class_name in CRITERIA_REGISTRATION_NAMES.items():
    CRITERIA_REGISTRY.register(registry_name, globals()[class_name])

NETWORKS = MODEL_REGISTRY
Criteria = CRITERIA_REGISTRY

__all__ = [
    "NETWORKS",
    "Criteria",
    "MODEL_REGISTRY", "CRITERIA_REGISTRY", "ModelRegistry", "CriteriaRegistry",
    "FederatedModelMixin", "ParameterVectorMixin",
    "FCN", "DeepFC", "BasicConv", "LeNet5",
    "ConvolutionalNet", "MobileNetV1Small", "MobileNetV1SmallNoBN",
    "MobileNetV1SmallGroupNorm", "MobileNetV1SmallLayerNorm",
    "MobileNetV2Small", "MobileNetV2SmallGroupNorm",
    "MobileNetV2SmallLayerNorm", "MobileNetV2Baseline",
    "MobileNetV2BaselineGroupNorm", "MobileNetV2BaselineLayerNorm",
    "MobileNetV2Alpha035", "MobileNetV2Alpha035GroupNorm",
    "MobileNetV2Alpha035LayerNorm", "MobileNetV4Small",
    "MobileNetV4SmallGroupNorm", "MobileNetV4SmallLayerNorm",
    "FOMOMNv2Baseline", "FOMOMNv2BaselineGroupNorm",
    "FOMOMNv2BaselineLayerNorm", "FOMOMNv2Alpha035",
    "FOMOMNv2Alpha035GroupNorm", "FOMOMNv2Alpha035LayerNorm",
    "MircoYOLO", "MircoYOLOGroupNorm", "MircoYOLOLayerNorm",
    "ConvBN", "DepthwiseSeparableConv", "InvertedBottleNeck", "LayerNorm2d",
    "SEBlock", "SeparableConvBn", "UniversalInvertedBottleneck",

    "FOMOLossPerson", "FOMOLossVehicle", "FOMOLossVehicleBinary",
    "FOMOMetricsPerson", "FOMOMetricsVehicle", "FOMOMetricsVehicleBinary",
    "YoLoLossPerson", "YoLoLossVehicle", "YoLoLossVehicleBinary",
    "YoLoMAPPerson", "YoLoMAPVehicle", "YoLoMAPVehicleBinary",
]
