# -*- coding: utf-8 -*-
"""Audit large Git objects without modifying repository history."""

import argparse
import shutil
import subprocess
import sys


def _run_git(args):
    return subprocess.run(
        ['git', *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )


def _human_size(size):
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024.0 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} TiB'


def audit(top, min_bytes, path_filter):
    if shutil.which('git') is None:
        raise RuntimeError('未找到 git 命令')

    root_check = _run_git(['rev-parse', '--show-toplevel'])
    if root_check.returncode != 0:
        raise RuntimeError(root_check.stderr.strip() or '当前目录不是 Git 仓库')

    rev_args = ['rev-list', '--objects', '--all']
    if path_filter:
        rev_args.extend(['--', path_filter])

    rev_list = subprocess.Popen(
        ['git', *rev_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    batch_check = subprocess.Popen(
        [
            'git',
            'cat-file',
            '--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)',
        ],
        stdin=rev_list.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    rev_list.stdout.close()

    objects = 0
    blobs = 0
    blob_bytes = 0
    large = []
    for raw_line in batch_check.stdout:
        parts = raw_line.rstrip('\n').split(' ', 3)
        if len(parts) < 4:
            continue
        object_type, object_id, size_text, object_path = parts
        objects += 1
        try:
            size = int(size_text)
        except ValueError:
            continue
        if object_type != 'blob':
            continue
        blobs += 1
        blob_bytes += size
        if size >= min_bytes:
            large.append((size, object_path, object_id))

    batch_check.wait()
    rev_list.wait()
    if batch_check.returncode or rev_list.returncode:
        error = batch_check.stderr.read() or rev_list.stderr.read()
        raise RuntimeError(error.strip() or 'Git 对象审计失败')

    large.sort(reverse=True)
    scope = f'，路径前缀 `{path_filter}`' if path_filter else ''
    print(f'Git 对象审计完成{scope}')
    print(f'对象总数: {objects:,}')
    print(f'Blob 数: {blobs:,}')
    print(f'Blob 总大小（去重对象）: {_human_size(blob_bytes)}')
    print(f'大于等于 {_human_size(min_bytes)} 的 Blob: {len(large):,}')
    print('')
    if not large:
        print('未发现达到阈值的大对象。')
        return

    print(f"{'大小':>12}  {'对象':<12}  路径")
    for size, object_path, object_id in large[:top]:
        print(f'{_human_size(size):>12}  {object_id[:12]:<12}  {object_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='只读审计 Git 历史中的大对象，不会重写历史。')
    parser.add_argument('--top', type=int, default=30,
                        help='最多显示多少个对象（默认: 30）')
    parser.add_argument('--min-mb', type=float, default=1.0,
                        help='大对象阈值，单位 MiB（默认: 1）')
    parser.add_argument('--path', default=None,
                        help='只审计指定路径，例如 downloads')
    args = parser.parse_args(argv)

    if args.top < 1:
        parser.error('--top 必须大于 0')
    if args.min_mb < 0:
        parser.error('--min-mb 不能为负数')

    try:
        audit(args.top, int(args.min_mb * 1024 * 1024), args.path)
    except RuntimeError as exc:
        print(f'错误: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
