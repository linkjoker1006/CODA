from .basic_controller import BasicMAC
from .n_controller import NMAC
from .n_controller_adhoctd import NMACAdHocTD
from .n_controller_cons import NMACCONS
from .n_controller_coda import NMACCODA

REGISTRY = {}

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["n_mac_adhoctd"] = NMACAdHocTD
REGISTRY["n_mac_cons"] = NMACCONS
REGISTRY["n_mac_coda"] = NMACCODA
