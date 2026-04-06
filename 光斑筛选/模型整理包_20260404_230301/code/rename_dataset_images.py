import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class RenamePlan:
    old_path: Path
    new_path: Path
    index: int


def parse_extensions(raw: str) -> Set[str]:
    extensions = {ext.strip().lower() for ext in raw.split(",") if ext.strip()}
    normalized = set()
    for ext in extensions:
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return normalized


def natural_numeric_key(path: Path) -> Tuple[int, int, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name.lower())
    return (1, 0, path.name.lower())


def list_image_files(directory: Path, extensions: Set[str]) -> List[Path]:
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=natural_numeric_key)


def build_rename_plans(files: Sequence[Path], start_index: int) -> List[RenamePlan]:
    plans: List[RenamePlan] = []
    current = start_index
    for old_path in files:
        new_name = f"{current}{old_path.suffix.lower()}"
        plans.append(RenamePlan(old_path=old_path, new_path=old_path.with_name(new_name), index=current))
        current += 1
    return plans


def ensure_no_target_conflict(plans: Iterable[RenamePlan]) -> None:
    targets = [str(plan.new_path.resolve()) for plan in plans]
    if len(targets) != len(set(targets)):
        raise RuntimeError("检测到重复目标文件名，已中止。")


def execute_rename(plans: Sequence[RenamePlan], dry_run: bool) -> None:
    if dry_run:
        return

    temp_paths: List[Tuple[Path, Path]] = []
    for order, plan in enumerate(plans, start=1):
        temp_name = f"__tmp_rename_{uuid.uuid4().hex}_{order}{plan.old_path.suffix.lower()}"
        temp_path = plan.old_path.with_name(temp_name)
        plan.old_path.rename(temp_path)
        temp_paths.append((temp_path, plan.new_path))

    for temp_path, final_path in temp_paths:
        temp_path.rename(final_path)


def write_manifest(manifest_path: Path, light_plans: Sequence[RenamePlan], without_light_plans: Sequence[RenamePlan]) -> None:
    payload = {
        "light": [
            {
                "index": plan.index,
                "old_name": plan.old_path.name,
                "new_name": plan.new_path.name,
                "old_path": str(plan.old_path.resolve()),
                "new_path": str(plan.new_path.resolve()),
            }
            for plan in light_plans
        ],
        "without_light": [
            {
                "index": plan.index,
                "old_name": plan.old_path.name,
                "new_name": plan.new_path.name,
                "old_path": str(plan.old_path.resolve()),
                "new_path": str(plan.new_path.resolve()),
            }
            for plan in without_light_plans
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按顺序批量重命名数据集图片：light->1..n，without_light->n+1.."
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--light-dir", type=str, default="light")
    parser.add_argument("--without-light-dir", type=str, default="without_light")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.bmp,.webp")
    parser.add_argument("--without-start", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("重命名映射.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    light_dir = data_root / args.light_dir
    without_light_dir = data_root / args.without_light_dir
    extensions = parse_extensions(args.extensions)

    if not light_dir.exists() or not light_dir.is_dir():
        raise FileNotFoundError(f"未找到目录: {light_dir}")
    if not without_light_dir.exists() or not without_light_dir.is_dir():
        raise FileNotFoundError(f"未找到目录: {without_light_dir}")

    light_files = list_image_files(light_dir, extensions)
    without_light_files = list_image_files(without_light_dir, extensions)

    light_plans = build_rename_plans(light_files, start_index=1)
    without_start = args.without_start if args.without_start is not None else (len(light_plans) + 1)
    if without_start <= 0:
        raise ValueError("--without-start 必须是正整数。")
    without_light_plans = build_rename_plans(without_light_files, start_index=without_start)
    all_plans = light_plans + without_light_plans
    ensure_no_target_conflict(all_plans)

    execute_rename(all_plans, dry_run=args.dry_run)
    write_manifest(args.manifest.resolve(), light_plans, without_light_plans)

    light_range = f"1..{len(light_plans)}" if light_plans else "空"
    if without_light_plans:
        start = without_light_plans[0].index
        end = without_light_plans[-1].index
        without_light_range = f"{start}..{end}"
    else:
        without_light_range = "空"

    mode = "预演(dry-run)" if args.dry_run else "已执行"
    print(f"{mode}：light 共 {len(light_plans)} 张，重命名区间 {light_range}")
    print(f"{mode}：without_light 共 {len(without_light_plans)} 张，重命名区间 {without_light_range}")
    print(f"映射文件已输出：{args.manifest.resolve()}")


if __name__ == "__main__":
    main()
