"""Audited logical arithmetic ledger; special functions are reported separately."""

from collections import Counter
from math import prod

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def tensors(value):
    return [x for x in tree_flatten(value)[0] if isinstance(x, torch.Tensor)]


def shape_description(value):
    return [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in tensors(value)]


class ArithmeticLedger(TorchDispatchMode):
    """Count each dispatched operation once, with explicit fused-op formulas."""

    ZERO = set("alias view reshape _unsafe_view as_strided permute transpose t numpy_T slice select expand "
        "unsqueeze squeeze clone copy_ detach detach_ split split_with_sizes chunk unbind cat stack "
        "repeat repeat_interleave gather scatter scatter_ index index_select index_copy index_copy_ embedding "
        "empty empty_like empty_strided new_empty new_zeros new_full zeros zeros_like ones ones_like "
        "full full_like fill_ zero_ _to_copy to lift_fresh lift_fresh_copy _local_scalar_dense "
        "pad constant_pad_nd slice_scatter masked_fill masked_fill_ where gt ge lt le eq ne "
        "clamp clamp_min clamp_max amax amin max min maximum minimum relu relu_ all any isnan isinf isfinite "
        "sort argsort topk masked_select "
        "bitwise_not logical_not arange resolve_conj resolve_neg _assert_async _assert_scalar "
        "_reshape_alias narrow expand_as view_as resize_ _pin_memory is_pinned contiguous flatten unflatten".split())

    def __init__(self, module_stack=None, bucket="all"):
        super().__init__()
        self.module_stack = module_stack if module_stack is not None else []
        self.bucket = bucket
        self.rows = {}
        self.observed = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        name = func._schema.name
        short = name.split("::")[-1]
        self.observed.add(str(func))
        ins, outs = tensors(args), tensors(output)
        out = outs[0] if outs else None
        floating = any(x.is_floating_point() for x in ins + outs)
        matrix = other = 0
        special = Counter()
        non_arithmetic = Counter()
        if short in ("sort", "argsort", "topk", "amax", "amin", "max", "min"):
            non_arithmetic["selection_input_elements"] = ins[0].numel() if ins else 0
        elif short in ("gather", "index", "index_select", "scatter", "scatter_", "scatter_add_"):
            non_arithmetic["indexing_output_elements"] = sum(t.numel() for t in outs)
        elif short in ("gt", "ge", "lt", "le", "eq", "ne", "maximum", "minimum"):
            non_arithmetic["comparison_output_elements"] = sum(t.numel() for t in outs)
        if not floating:
            rule = "integer/boolean/administrative: no floating arithmetic"
        elif short == "dropout":
            assert not args[2], "Training dropout is outside this inference benchmark"
            rule = "evaluation dropout: identity"
        elif name.startswith("fbgemm::") and any(s in short for s in
                ("jagged_to_padded_dense", "dense_to_jagged")):
            rule = "jagged layout conversion: data movement"
        elif name.startswith("aten::") and short in self.ZERO:
            rule = "data movement, comparison, cast, or tensor construction"
        elif short == "linear":
            matrix = 2 * out.numel() * args[0].shape[-1]
            other = out.numel() if len(args) > 2 and args[2] is not None else 0
            rule = "fused linear: 2 MAC FLOPs per input/output contraction, plus bias"
        elif short == "matmul":
            matrix = 2 * out.numel() * args[0].shape[-1]
            rule = "matmul: 2 MAC FLOPs per contraction"
        elif short == "einsum":
            left, right = args[0].replace(" ", "").split("->")
            terms = left.split(",")
            ellipsis_rank = max((t.ndim - len(term.replace("...", ""))
                                 if "..." in term else 0)
                                for term, t in zip(terms, args[1]))
            labels = "".join(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in left + right)[:ellipsis_rank]
            assert len(labels) == ellipsis_rank
            expanded = []
            for term, tensor in zip(terms, args[1]):
                rank = tensor.ndim - len(term.replace("...", "")) if "..." in term else 0
                expanded.append(term.replace("...", labels[-rank:] if rank else ""))
            right = right.replace("...", labels)
            sizes = {}
            for letters, tensor in zip(expanded, args[1]):
                assert len(letters) == tensor.ndim
                for letter, size in zip(letters, tensor.shape):
                    assert letter not in sizes or sizes[letter] == size or sizes[letter] == 1 or size == 1
                    sizes[letter] = max(sizes.get(letter, 1), size)
            contracted = [size for letter, size in sizes.items() if letter not in right]
            assert contracted, "Only contraction einsums are defined"
            matrix = 2 * out.numel() * prod(contracted)
            rule = "einsum: 2*output_elements*contracted_dimension_product"
        elif short in ("mm", "bmm", "addmm", "baddbmm"):
            a, b = (args[1], args[2]) if short in ("addmm", "baddbmm") else (args[0], args[1])
            matrix = 2 * out.numel() * a.shape[-1]
            if short in ("addmm", "baddbmm"):
                if kwargs.get("beta", 1) != 0:
                    other += out.numel()
                    if kwargs.get("beta", 1) != 1:
                        other += out.numel()
                if kwargs.get("alpha", 1) != 1:
                    other += out.numel()
            rule = "2 MAC FLOPs per contraction; fused bias/scales counted separately"
        elif short in ("convolution", "_convolution", "conv1d"):
            if short != "conv1d":
                assert not args[6], "Transposed convolution requires its own rule"
            weight = args[1]
            matrix = 2 * out.numel() * prod(weight.shape[1:])
            other = out.numel() if args[2] is not None else 0
            rule = "2*output_elements*input_channels_per_group*kernel_volume, plus bias"
        elif short in ("native_layer_norm", "layer_norm"):
            d = prod(args[1])
            rows = args[0].numel() // d
            other = rows * (5 * d + 1)
            special["rsqrt"] = rows
            if len(args) > 2 and args[2] is not None:
                other += args[0].numel()
            if len(args) > 3 and args[3] is not None:
                other += args[0].numel()
            rule = "two-pass centered LayerNorm: rows*(5D+1), rsqrt; affine if present"
        elif short in ("linalg_vector_norm", "linalg_norm"):
            assert len(args) < 2 or args[1] in (None, 2), "Only L2 norm is defined"
            if short == "linalg_norm":
                dims = args[2] if len(args) > 2 else kwargs.get("dim")
                assert isinstance(dims, (list, tuple)) and len(dims) == 1, "Only vector L2 norm is defined"
            other = 2 * args[0].numel() - out.numel()
            special["sqrt"] = out.numel()
            rule = "L2 norm: squares plus reduction, separate sqrt"
        elif short in ("_softmax", "softmax", "_safe_softmax"):
            dim = args[1] % args[0].ndim
            rows = args[0].numel() // args[0].shape[dim]
            other = 3 * args[0].numel() - rows
            special["exp"] = args[0].numel()
            rule = "stable softmax: subtract, sum, divide; separate exp; comparisons excluded"
        elif short in ("silu", "silu_"):
            other = out.numel()
            special["sigmoid"] = out.numel()
            rule = "SiLU: one multiply and one sigmoid per element"
        elif short in ("exp", "log", "sqrt", "rsqrt", "sigmoid", "tanh", "gelu"):
            special[short] = out.numel()
            rule = "special-function evaluations, not priced as arithmetic FLOPs"
        elif short in ("add", "add_", "sub", "sub_", "rsub", "mul", "mul_", "div", "div_", "neg", "reciprocal"):
            other = out.numel() if out.is_floating_point() else 0
            if short in ("add", "add_", "sub", "sub_") and kwargs.get("alpha", 1) != 1:
                other += out.numel()
            rule = "one arithmetic operation per broadcast output element"
        elif short in ("sum", "mean"):
            other = args[0].numel() - out.numel()
            if short == "mean":
                other += out.numel()
            rule = "reduction additions; mean also divides each output"
        elif short == "square" or (short == "pow" and not isinstance(args[1], torch.Tensor) and args[1] == 2):
            other = out.numel()
            rule = "square: one multiply per element"
        else:
            raise RuntimeError(f"Unclassified floating operation: {func}; inputs={shape_description(args)}")
        module = self.module_stack[-1] if self.module_stack else "adapter"
        key = (self.bucket, module, str(func), str(shape_description(args)),
               str(shape_description(output)), rule, tuple(sorted(special.keys())))
        if key not in self.rows:
            self.rows[key] = {"bucket": self.bucket, "module": module, "operator": str(func),
                "inputs": shape_description(args), "outputs": shape_description(output),
                "rule": rule, "calls": 0, "matrix_flops": 0, "other_arithmetic_flops": 0,
                "special_functions": {}, "non_arithmetic_operations": {}}
        row = self.rows[key]
        row["calls"] += 1
        row["matrix_flops"] += int(matrix)
        row["other_arithmetic_flops"] += int(other)
        row["special_functions"] = dict(Counter(row["special_functions"]) + special)
        row["non_arithmetic_operations"] = dict(Counter(row["non_arithmetic_operations"]) + non_arithmetic)
        return output

    def totals(self):
        special = Counter()
        non_arithmetic = Counter()
        matrix = other = 0
        for row in self.rows.values():
            matrix += row["matrix_flops"]
            other += row["other_arithmetic_flops"]
            special.update(row["special_functions"])
            non_arithmetic.update(row["non_arithmetic_operations"])
        return {"matrix_flops": matrix, "other_arithmetic_flops": other,
                "arithmetic_flops": matrix + other, "special_functions": dict(special),
                "non_arithmetic_operations": dict(non_arithmetic)}


def sum_totals(*values):
    special = Counter()
    non_arithmetic = Counter()
    for value in values:
        special.update(value["special_functions"])
        non_arithmetic.update(value.get("non_arithmetic_operations", {}))
    return {key: sum(value[key] for value in values) for key in
            ("matrix_flops", "other_arithmetic_flops", "arithmetic_flops")} | {
                "special_functions": dict(special), "non_arithmetic_operations": dict(non_arithmetic)}
