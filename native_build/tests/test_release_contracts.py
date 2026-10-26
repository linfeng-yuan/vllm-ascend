"""Regression checks for the published DS41 source and default AOT contracts."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(module, name):
    path = ROOT / "samples" / module / (module + ".py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec")


class ReleaseContracts(unittest.TestCase):
    def profiles(self, module, function):
        calls = []
        def record(layout):
            return lambda *args: calls.append((layout, args))
        scope = {"os": SimpleNamespace(environ={}), "clear_caches": lambda: None,
                 "clear_tnd_caches": lambda: None,
                 "_get_compiled_fused_runner": record("PA_BBND"),
                 "_get_tnd_compiled_fused_runner": record("TND"), "TILE_N": 128}
        exec(extract(module, function), scope)
        scope[function]()
        return calls

    def test_qli_preserves_true_and_adds_false(self):
        calls = self.profiles("quant_lightning_indexer_dsl", "export_qli")
        self.assertEqual(len(calls), 36)
        # has_seqused_q is the seventh compiled-runner static argument.
        self.assertEqual({args[6] for _, args in calls}, {False, True})
        for layout, args in calls:
            other = list(args); other[6] = not args[6]
            self.assertIn((layout, tuple(other)), calls)
        base = [args for layout,args in calls if layout == "PA_BBND" and args[0] == 512 and not args[6] and args[3] == 3]
        self.assertTrue(any(not a[1] and a[2] == 1 and a[4] == 2 and a[9] == 64 for a in base))
        self.assertTrue(any(a[1] and a[2] == 2048 and a[4] == 1 and a[9] == 128 for a in base))

    def test_qsli_preserves_true_and_adds_false(self):
        calls = self.profiles("quant_sparse_lightning_indexer_dsl", "export_qsli")
        self.assertEqual(len(calls), 36)
        self.assertEqual({args[6] for _, args in calls}, {False, True})
        for layout, args in calls:
            other = list(args); other[6] = not args[6]
            self.assertIn((layout, tuple(other)), calls)
        self.assertTrue(any(layout == "PA_BBND" and a[0] == 16384 and a[1] == 512 and a[2] == 2048 and a[3] == 3 and a[4] == 1 and not a[6] and not a[9] and a[10] == 128 for layout,a in calls))

    def test_core_count_comes_from_input_device(self):
        for module in ["quant_lightning_indexer_dsl", "quant_sparse_lightning_indexer_dsl"]:
            for count in [1, 28, 32, 36, 0, 37]:
                devices = []
                def properties(device):
                    devices.append(device)
                    return SimpleNamespace(cube_core_num=count)
                scope = {"torch": SimpleNamespace(npu=SimpleNamespace(get_device_properties=properties)), "LD_RECORD_START": 36}
                exec(extract(module, "_get_cube_core_num"), scope)
                if 1 <= count <= 36:
                    self.assertEqual(scope["_get_cube_core_num"]("npu:3"), count)
                else:
                    with self.assertRaises(RuntimeError): scope["_get_cube_core_num"]("npu:3")
                self.assertEqual(devices, ["npu:3"])

    def test_metadata_no_longer_overrides_workers_to_32(self):
        for module in ["quant_lightning_indexer_metadata_dsl", "quant_sparse_lightning_indexer_metadata_dsl"]:
            path = ROOT / "samples" / module / (module + ".py")
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "workers" for t in node.targets):
                    self.assertFalse(isinstance(node.value, ast.Constant) and node.value.value == 32)
            self.assertIn("get_device_properties(q.device).cube_core_num", path.read_text())



class BuildHeaderContracts(unittest.TestCase):
    def test_uses_installed_hcc_version_and_keeps_existing_include(self):
        import os, tempfile, types, sys
        from unittest.mock import patch
        builder = Path(__file__).parents[1] / "net/native_package/build.py"
        tree = ast.parse(builder.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_aicpu_headers")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++"
            for version in ["7.3.0", "14.3.0"]:
                include = base/version
                (include/"aarch64-target-linux-gnu").mkdir(parents=True)
                (include/"backward").mkdir()
                (include/"cstdint").write_text("// fixture")
            fake = types.ModuleType("cannbotdsl.aicpu.toolchain")
            fake.AicpuToolchain = lambda **kwargs: types.SimpleNamespace(toolkit=str(root))
            scope = {"Path":Path,"os":os}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(builder),"exec"),scope)
            with patch.dict(sys.modules,{"cannbotdsl.aicpu.toolchain":fake}), patch.dict(os.environ,{"CPLUS_INCLUDE_PATH":"/existing/custom/include"}):
                selected = scope["configure_aicpu_headers"]("dav-3510")
                self.assertEqual(selected[0],str(base/"14.3.0"))
                self.assertEqual(os.environ["CPLUS_INCLUDE_PATH"].split(":")[-1],"/existing/custom/include")
                self.assertNotIn(str(base/"7.3.0"),os.environ["CPLUS_INCLUDE_PATH"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
