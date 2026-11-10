#!/usr/bin/env python3
"""Read-only checks for files a git add would include; no simulator or staging."""
import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DOCS = ['README.md', 'docs/README.md', 'docs/repository.md',
              'docs/reproduction.md', 'scripts/README.md', 'reports/README.md',
              'src/humanoid_imitation/README.md']
SECRET_PATTERNS = {
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
    'aws_access_id': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'openai_token': r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b',
}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    paths = sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''})
    paths = [p for p in paths if (ROOT / p).is_file()]
    candidates = set(paths)
    errors = []
    scope_path = ROOT / 'publication_manifest.json'
    if scope_path.exists():
        allowed = set(json.loads(scope_path.read_text())['files'])
        for rel in sorted(candidates - allowed):
            errors.append({'file': rel, 'reason': 'outside final publication scope; remove from Git index, not disk'})
        for rel in sorted(allowed - candidates):
            errors.append({'file': rel, 'reason': 'allowlisted file missing or ignored'})
        # Resolve project-local imports without importing simulator modules.
        modules = {p.stem: p for p in (ROOT / 'scripts').glob('*.py')}
        for p in (ROOT / 'src').rglob('*.py'):
            parts = list(p.relative_to(ROOT / 'src').with_suffix('').parts)
            if parts[-1] == '__init__':
                parts.pop()
            modules['.'.join(parts)] = p
        for rel in paths:
            if not rel.endswith('.py'):
                continue
            tree = ast.parse((ROOT / rel).read_text(), filename=rel)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [x.name for x in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ''
                    if node.level and rel.startswith('src/'):
                        package = list(Path(rel).relative_to('src').parts[:-1])
                        base = '.'.join(package[:len(package)-node.level+1] + ([base] if base else []))
                    names = [base] + [base + '.' + x.name for x in node.names]
                for name in names:
                    if name in modules:
                        dependency = str(modules[name].relative_to(ROOT))
                        if dependency not in candidates:
                            errors.append({'file': rel, 'reason': 'excluded local Python import', 'dependency': dependency})
    source_count = 0
    local_path_files = 0
    for rel in paths:
        path = ROOT / rel
        if path.stat().st_size >= 50 * 1024 * 1024:
            errors.append({'file': rel, 'reason': 'file >=50MiB; review before publishing'})
        if path.suffix in {'.pt', '.pth', '.ckpt', '.pkl', '.fbx', '.bvh', '.npz', '.npy', '.pem', '.key'}:
            errors.append({'file': rel, 'reason': 'model/data/key in publication candidates'})
        if path.name.startswith('.env') and path.name != '.env.example':
            errors.append({'file': rel, 'reason': 'local environment file'})
        if rel.startswith(('tmp/', '.aws/', '.codex/', '.agents/')):
            errors.append({'file': rel, 'reason': 'local-only directory'})
        if path.suffix.lower() not in {'.py', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.html', '.cjs', '.sh', ''}:
            continue
        try:
            text = path.read_text()
        except (UnicodeError, OSError):
            continue
        if '/home/an/' in text:
            local_path_files += 1
        if path.suffix == '.py':
            source_count += 1
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as e:
                errors.append({'file': rel, 'line': e.lineno, 'reason': 'Python syntax: ' + e.msg})
        for kind, pattern in SECRET_PATTERNS.items():
            for match in re.finditer(pattern, text):
                # Never echo credential values.
                errors.append({'file': rel, 'line': text.count('\n', 0, match.start()) + 1, 'reason': kind})
    links_checked = 0
    for rel in sorted(p for p in paths if p.endswith('.md')):
        path = ROOT / rel
        if not path.exists():
            continue
        for target in re.findall(r'\]\(([^)\s]+)(?:\s+[^)]*)?\)', path.read_text()):
            if target.startswith(('http:', 'https:', 'mailto:', '#')):
                continue
            target = unquote(target.split('#')[0])
            resolved = (path.parent / target).resolve()
            links_checked += 1
            if not resolved.exists():
                errors.append({'file': rel, 'reason': 'broken relative link', 'target': target})
            elif resolved.is_file() and resolved.is_relative_to(ROOT) and str(resolved.relative_to(ROOT)) not in candidates:
                errors.append({'file': rel, 'reason': 'entry link points to excluded local file', 'target': target})
    sizes = [(p, (ROOT / p).stat().st_size) for p in paths]
    result = {
        'candidate_files': len(paths), 'candidate_bytes': sum(n for _, n in sizes),
        'python_syntax_checked': source_count, 'entry_links_checked': links_checked,
        'files_with_historical_local_paths': local_path_files,
        'largest_files': sorted(sizes, key=lambda x: x[1], reverse=True)[:5],
        'top_level_counts': dict(Counter(p.split('/')[0] for p in paths)),
        'errors': errors,
        'scope': 'working tree candidates, not a full secret audit or clean-machine reproduction',
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
