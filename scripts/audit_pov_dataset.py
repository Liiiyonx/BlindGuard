# -*- coding: utf-8 -*-
"""Validate a first-person dataset manifest without modifying media files."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_MANIFEST = 'pov_manifest.json'
REQUIRED_FIELDS = (
    'sample_id',
    'media_path',
    'consent_id',
    'consent_scope',
    'privacy_status',
    'viewpoint',
    'split',
)
MEDIA_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp',
    '.mp4', '.mov', '.avi', '.mkv',
}
CONSENT_SCOPES = {'research', 'training', 'evaluation', 'internal_demo'}
PRIVACY_STATUSES = {'redacted', 'no_personal_data'}
VIEWPOINTS = {'first_person', 'head_mounted', 'chest_mounted', 'pov'}
SPLITS = {'train', 'val', 'test'}
REVIEW_STATUSES = {'pending', 'reviewed', 'approved', 'rejected'}
PLACEHOLDERS = {'', 'none', 'null', 'todo', 'tbd', 'unknown', 'placeholder'}


class Audit:
    def __init__(self, strict=False):
        self.strict = strict
        self.errors = []
        self.warnings = []
        self.split_counts = Counter()
        self.weather_counts = Counter()
        self.viewpoint_counts = Counter()

    def error(self, sample_id, message):
        self.errors.append(f'{sample_id}: {message}')

    def warning(self, sample_id, message):
        self.warnings.append(f'{sample_id}: {message}')


def _load_manifest(path):
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {'schema_version': 1, 'records': payload}
    if not isinstance(payload, dict):
        raise ValueError('清单根节点必须是对象或数组')
    records = payload.get('records')
    if not isinstance(records, list):
        raise ValueError('清单缺少 records 数组')
    return payload


def _clean(value):
    return str(value or '').strip()


def _is_placeholder(value):
    return _clean(value).lower() in PLACEHOLDERS


def _consent_scopes(value):
    if isinstance(value, str):
        return {part.strip() for part in value.split(',') if part.strip()}
    if isinstance(value, list):
        return {_clean(part) for part in value if _clean(part)}
    return set()


def _safe_media_path(root, value):
    media = (root / value).resolve()
    try:
        media.relative_to(root)
    except ValueError:
        return None
    return media


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root, manifest_path, strict=False, include_hash=False):
    audit = Audit(strict=strict)
    manifest = _load_manifest(manifest_path)
    records = manifest['records']
    if not records:
        audit.error('manifest', 'records 为空')

    sample_ids = set()
    media_paths = set()
    scene_splits = {}
    content_hashes = {}

    for index, record in enumerate(records, start=1):
        label = f'第 {index} 条'
        if not isinstance(record, dict):
            audit.error(label, '记录必须是 JSON 对象')
            continue
        sample_id = _clean(record.get('sample_id')) or label
        if _is_placeholder(sample_id) or sample_id == label:
            audit.error(sample_id, 'sample_id 缺失或为占位值')
        elif sample_id in sample_ids:
            audit.error(sample_id, 'sample_id 重复')
        else:
            sample_ids.add(sample_id)

        for field in REQUIRED_FIELDS:
            if field not in record:
                audit.error(sample_id, f'缺少必填字段 {field}')

        consent_id = _clean(record.get('consent_id'))
        if _is_placeholder(consent_id):
            audit.error(sample_id, 'consent_id 缺失或为占位值')

        scopes = _consent_scopes(record.get('consent_scope'))
        invalid_scopes = scopes - CONSENT_SCOPES
        if invalid_scopes:
            audit.error(sample_id,
                        f"未知 consent_scope: {', '.join(sorted(invalid_scopes))}")
        if not scopes:
            audit.error(sample_id, 'consent_scope 不能为空')

        split = _clean(record.get('split')).lower()
        if split not in SPLITS:
            audit.error(sample_id, f'split 非法: {split or "空"}')
        else:
            audit.split_counts[split] += 1
            required_scope = 'training' if split in {'train', 'val'} else 'evaluation'
            if scopes and required_scope not in scopes:
                audit.error(sample_id, f'{split} 数据缺少 {required_scope} 授权')

        privacy = _clean(record.get('privacy_status')).lower()
        if privacy not in PRIVACY_STATUSES:
            audit.error(sample_id,
                        f'privacy_status 非法: {privacy or "空"}')

        viewpoint = _clean(record.get('viewpoint')).lower()
        if viewpoint not in VIEWPOINTS:
            audit.error(sample_id, f'viewpoint 非法: {viewpoint or "空"}')
        else:
            audit.viewpoint_counts[viewpoint] += 1

        review_status = _clean(record.get('review_status')).lower()
        if review_status and review_status not in REVIEW_STATUSES:
            audit.error(sample_id, f'review_status 非法: {review_status}')
        elif not review_status:
            audit.warning(sample_id, '缺少 review_status，建议进入训练前补齐')

        media_value = _clean(record.get('media_path'))
        if _is_placeholder(media_value):
            audit.error(sample_id, 'media_path 缺失或为占位值')
        else:
            media = _safe_media_path(root, media_value)
            if media is None:
                audit.error(sample_id, 'media_path 越出数据根目录')
            else:
                if media_value in media_paths:
                    audit.error(sample_id, f'media_path 重复: {media_value}')
                media_paths.add(media_value)
                if media.suffix.lower() not in MEDIA_EXTENSIONS:
                    audit.error(sample_id, f'不支持的媒体扩展名: {media.suffix}')
                if not media.is_file():
                    audit.error(sample_id, f'媒体文件不存在: {media_value}')
                elif include_hash:
                    digest = _sha256(media)
                    if digest in content_hashes:
                        audit.error(
                            sample_id,
                            f'内容与 {content_hashes[digest]} 重复（SHA256）',
                        )
                    else:
                        content_hashes[digest] = sample_id

        scene_group = _clean(record.get('scene_group'))
        if scene_group:
            previous = scene_splits.get(scene_group)
            if previous and split and previous != split:
                audit.error(
                    sample_id,
                    f'scene_group {scene_group} 同时出现在 {previous} 和 {split}',
                )
            elif split:
                scene_splits[scene_group] = split
        else:
            audit.warning(sample_id, '缺少 scene_group，无法自动检查场景泄漏')

        weather = _clean(record.get('weather'))
        if weather:
            audit.weather_counts[weather] += 1
        else:
            audit.warning(sample_id, '缺少 weather，采样覆盖无法统计')

        if not _clean(record.get('route_type')):
            audit.warning(sample_id, '缺少 route_type，路线覆盖无法统计')
        if not _clean(record.get('device_id')):
            audit.warning(sample_id, '缺少 device_id，设备覆盖无法统计')

    return audit, manifest


def _print_report(audit, manifest, manifest_path):
    print(f'清单: {manifest_path}')
    print(f"Schema: {manifest.get('schema_version', '未声明')}")
    print(f'记录数: {sum(audit.split_counts.values())}')
    print(f'划分: {dict(sorted(audit.split_counts.items()))}')
    print(f'视角: {dict(sorted(audit.viewpoint_counts.items()))}')
    print(f'天气: {dict(sorted(audit.weather_counts.items()))}')
    print('')
    print(f'错误: {len(audit.errors)}')
    for item in audit.errors:
        print(f'  [ERROR] {item}')
    print(f'警告: {len(audit.warnings)}')
    for item in audit.warnings:
        print(f'  [WARN] {item}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='只读审核 POV 数据清单，不修改或上传媒体文件。')
    parser.add_argument('--root', required=True,
                        help='数据根目录')
    parser.add_argument('--manifest', default=None,
                        help=f'清单路径（默认: ROOT/{DEFAULT_MANIFEST}）')
    parser.add_argument('--strict', action='store_true',
                        help='把警告也视为失败')
    parser.add_argument('--hash', action='store_true',
                        help='计算 SHA256 并检查重复内容')
    parser.add_argument('--json', action='store_true',
                        help='以 JSON 输出机器可读结果')
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f'错误: 数据根目录不存在: {root}', file=sys.stderr)
        return 1
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest else root / DEFAULT_MANIFEST
    )
    if not manifest_path.is_file():
        print(f'错误: 清单不存在: {manifest_path}', file=sys.stderr)
        return 1

    try:
        audit, manifest = validate(
            root,
            manifest_path,
            strict=args.strict,
            include_hash=args.hash,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'错误: 无法审核清单: {exc}', file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            'manifest': str(manifest_path),
            'schema_version': manifest.get('schema_version'),
            'records': sum(audit.split_counts.values()),
            'split_counts': dict(audit.split_counts),
            'weather_counts': dict(audit.weather_counts),
            'viewpoint_counts': dict(audit.viewpoint_counts),
            'errors': audit.errors,
            'warnings': audit.warnings,
        }, ensure_ascii=False, indent=2))
    else:
        _print_report(audit, manifest, manifest_path)

    failed = bool(audit.errors) or (args.strict and bool(audit.warnings))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
