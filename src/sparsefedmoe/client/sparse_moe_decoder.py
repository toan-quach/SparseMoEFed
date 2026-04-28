"""SparseMoEDecoder — NVFlare Filter for server-side decompression (arch §3.2).

Paired with ``SparseMoEEncoder``. Reads ``compression_metadata`` from the DXO,
upcasts FP16 back to FP32 and dequantises INT8 packages into their original
shapes. SKIPped params are absent from the incoming dict and stay absent; the
aggregator handles them by treating the client as a non-contributor for those
experts (arch §3.3).
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

logger = logging.getLogger(__name__)


class SparseMoEDecoder(Filter):
    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Union[Shareable, None]:
        try:
            dxo = DXO.from_shareable(shareable)
        except Exception as e:  # noqa: BLE001
            logger.warning("SparseMoEDecoder: cannot parse DXO (%s); passing through", e)
            return shareable

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS):
            return shareable

        meta = dxo.get_meta_props() or {}
        comp_metadata = meta.get("compression_metadata")
        if comp_metadata is None:
            return shareable  # encoder didn't run (e.g., baseline job)

        compression_map = comp_metadata.get("compression_map", {})
        original_shapes = comp_metadata.get("original_shapes", {})

        out = {}
        for name, value in dxo.data.items():
            tier = compression_map.get(name, "none")
            if tier == "skipped":
                continue  # client skipped; leave it out
            if tier == "none":
                out[name] = value
            elif tier == "fp16":
                arr = np.asarray(value)
                out[name] = arr.astype(np.float32, copy=False)
            elif tier == "int8":
                if isinstance(value, dict) and "quantized" in value:
                    q = np.asarray(value["quantized"], dtype=np.int8)
                    scale = float(value["scale"])
                    shape = original_shapes.get(name)
                    dequantized = (q.astype(np.float32) * scale)
                    out[name] = dequantized.reshape(shape) if shape else dequantized
                else:
                    out[name] = value  # malformed; hand it through
            else:
                out[name] = value

        new_meta = dict(meta)
        new_meta["decompressed"] = True
        logger.info(
            "SparseMoEDecoder: %d params decompressed, %d skipped",
            len(out), len(meta.get("skipped_experts", [])),
        )
        return DXO(data_kind=dxo.data_kind, data=out, meta=new_meta).to_shareable()
